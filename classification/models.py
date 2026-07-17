# from visualizer import get_local
import torch
import torchinfo
import torch.nn as nn
from spikingjelly.clock_driven.neuron import (
    MultiStepParametricLIFNode,
    MultiStepLIFNode,
)
from spikingjelly.clock_driven import layer
from timm.models.layers import to_2tuple, trunc_normal_, DropPath
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg
from einops.layers.torch import Rearrange
import torch.nn.functional as F
from functools import partial
import os


class BNAndPadLayer(nn.Module):
    def __init__(
        self,
        pad_pixels,
        num_features,
        eps=1e-5,
        momentum=0.1,
        affine=True,
        track_running_stats=True,
    ):
        super(BNAndPadLayer, self).__init__()
        self.bn = nn.BatchNorm2d(
            num_features, eps, momentum, affine, track_running_stats
        )
        self.pad_pixels = pad_pixels

    def forward(self, input):
        output = self.bn(input)
        if self.pad_pixels > 0:
            if self.bn.affine:
                pad_values = (
                    self.bn.bias.detach()
                    - self.bn.running_mean
                    * self.bn.weight.detach()
                    / torch.sqrt(self.bn.running_var + self.bn.eps)
                )
            else:
                pad_values = -self.bn.running_mean / torch.sqrt(
                    self.bn.running_var + self.bn.eps
                )
            output = F.pad(output, [self.pad_pixels] * 4)
            pad_values = pad_values.view(1, -1, 1, 1)
            output[:, :, 0 : self.pad_pixels, :] = pad_values
            output[:, :, -self.pad_pixels :, :] = pad_values
            output[:, :, :, 0 : self.pad_pixels] = pad_values
            output[:, :, :, -self.pad_pixels :] = pad_values
        return output

    @property
    def weight(self):
        return self.bn.weight

    @property
    def bias(self):
        return self.bn.bias

    @property
    def running_mean(self):
        return self.bn.running_mean

    @property
    def running_var(self):
        return self.bn.running_var

    @property
    def eps(self):
        return self.bn.eps


class RepConv(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        bias=False,
    ):
        super().__init__()
        # hidden_channel = in_channel
        conv1x1 = nn.Conv2d(in_channel, in_channel, 1, 1, 0, bias=False, groups=1)
        bn = BNAndPadLayer(pad_pixels=1, num_features=in_channel)
        conv3x3 = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, 3, 1, 0, groups=in_channel, bias=False),
            nn.Conv2d(in_channel, out_channel, 1, 1, 0, groups=1, bias=False),
            nn.BatchNorm2d(out_channel),
        )

        self.body = nn.Sequential(conv1x1, bn, conv3x3)

    def forward(self, x):
        return self.body(x)


class SepConv(nn.Module):
    r"""
    Inverted separable convolution from MobileNetV2: https://arxiv.org/abs/1801.04381.
    """

    def __init__(
        self,
        dim,
        expansion_ratio=2,
        act2_layer=nn.Identity,
        bias=False,
        kernel_size=7,
        padding=3,
    ):
        super().__init__()
        med_channels = int(expansion_ratio * dim)
        self.lif1 = MultiStepLIFNode(tau=2.0, detach_reset=True, backend="cupy")
        self.pwconv1 = nn.Conv2d(dim, med_channels, kernel_size=1, stride=1, bias=bias)
        self.bn1 = nn.BatchNorm2d(med_channels)
        self.lif2 = MultiStepLIFNode(tau=2.0, detach_reset=True, backend="cupy")
        self.dwconv = nn.Conv2d(
            med_channels,
            med_channels,
            kernel_size=kernel_size,
            padding=padding,
            groups=med_channels,
            bias=bias,
        )  # depthwise conv
        self.pwconv2 = nn.Conv2d(med_channels, dim, kernel_size=1, stride=1, bias=bias)
        self.bn2 = nn.BatchNorm2d(dim)

    def forward(self, x):
        T, B, C, H, W = x.shape
        x = self.lif1(x)
        x = self.bn1(self.pwconv1(x.flatten(0, 1))).reshape(T, B, -1, H, W)
        x = self.lif2(x)
        x = self.dwconv(x.flatten(0, 1))
        x = self.bn2(self.pwconv2(x)).reshape(T, B, -1, H, W)
        return x


class MS_ConvBlock(nn.Module):
    def __init__(
        self,
        dim,
        mlp_ratio=4.0,
    ):
        super().__init__()

        self.Conv = SepConv(dim=dim)
        # self.Conv = MHMC(dim=dim)

        self.lif1 = MultiStepLIFNode(tau=2.0, detach_reset=True, backend="cupy")
        self.conv1 = nn.Conv2d(
            dim, dim * mlp_ratio, kernel_size=3, padding=1, groups=1, bias=False
        )
        # self.conv1 = RepConv(dim, dim*mlp_ratio)
        self.bn1 = nn.BatchNorm2d(dim * mlp_ratio)  # 这里可以进行改进
        self.lif2 = MultiStepLIFNode(tau=2.0, detach_reset=True, backend="cupy")
        self.conv2 = nn.Conv2d(
            dim * mlp_ratio, dim, kernel_size=3, padding=1, groups=1, bias=False
        )
        # self.conv2 = RepConv(dim*mlp_ratio, dim)
        self.bn2 = nn.BatchNorm2d(dim)  # 这里可以进行改进

    def forward(self, x):
        T, B, C, H, W = x.shape

        x = self.Conv(x) + x
        x_feat = x
        x = self.bn1(self.conv1(self.lif1(x).flatten(0, 1))).reshape(T, B, 4 * C, H, W)
        x = self.bn2(self.conv2(self.lif2(x).flatten(0, 1))).reshape(T, B, C, H, W)
        x = x_feat + x

        return x


class MS_MLP(nn.Module):
    def __init__(
        self, in_features, hidden_features=None, out_features=None, drop=0.0, layer=0
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        # self.fc1 = linear_unit(in_features, hidden_features)
        self.fc1_conv = nn.Conv1d(in_features, hidden_features, kernel_size=1, stride=1)
        self.fc1_bn = nn.BatchNorm1d(hidden_features)
        self.fc1_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend="cupy")

        # self.fc2 = linear_unit(hidden_features, out_features)
        self.fc2_conv = nn.Conv1d(
            hidden_features, out_features, kernel_size=1, stride=1
        )
        self.fc2_bn = nn.BatchNorm1d(out_features)
        self.fc2_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend="cupy")
        # self.drop = nn.Dropout(0.1)

        self.c_hidden = hidden_features
        self.c_output = out_features

    def forward(self, x):
        T, B, C, H, W = x.shape
        N = H * W
        x = x.flatten(3)
        x = self.fc1_lif(x)
        x = self.fc1_conv(x.flatten(0, 1))
        x = self.fc1_bn(x).reshape(T, B, self.c_hidden, N).contiguous()

        x = self.fc2_lif(x)
        x = self.fc2_conv(x.flatten(0, 1))
        x = self.fc2_bn(x).reshape(T, B, C, H, W).contiguous()

        return x


class MS_Attention_RepConv_qkv_id(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        sr_ratio=1,
    ):
        super().__init__()
        assert (
            dim % num_heads == 0
        ), f"dim {dim} should be divided by num_heads {num_heads}."
        self.dim = dim
        self.num_heads = num_heads
        self.scale = 0.125

        self.head_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend="cupy")

        self.q_conv = nn.Sequential(RepConv(dim, dim, bias=False), nn.BatchNorm2d(dim))

        self.k_conv = nn.Sequential(RepConv(dim, dim, bias=False), nn.BatchNorm2d(dim))

        self.v_conv = nn.Sequential(RepConv(dim, dim, bias=False), nn.BatchNorm2d(dim))

        self.q_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend="cupy")

        self.k_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend="cupy")

        self.v_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend="cupy")

        self.attn_lif = MultiStepLIFNode(
            tau=2.0, v_threshold=0.5, detach_reset=True, backend="cupy"
        )

        self.proj_conv = nn.Sequential(
            RepConv(dim, dim, bias=False), nn.BatchNorm2d(dim)
        )

    def forward(self, x):
        T, B, C, H, W = x.shape
        N = H * W

        x = self.head_lif(x)
        # Measure firing rate of head_lif output: fraction of non-zero (spikes)
        try:
            firing_rate = (x != 0).float().mean()
            # store last firing rate as a CPU tensor and append to simple log list
            self.last_head_lif_firing_rate = firing_rate.detach().cpu()
            if not hasattr(self, "head_lif_firing_rate_log"):
                self.head_lif_firing_rate_log = []
            # append scalar float
            self.head_lif_firing_rate_log.append(float(self.last_head_lif_firing_rate.item()))
            # quick console output for inspection
            print(f"[MS_Attention] head_lif firing rate (fraction): {self.last_head_lif_firing_rate.item():.4f}")
            with torch.no_grad():
                print(torch.unique(x))
        except Exception:
            # don't break forward pass if logging fails
            pass

        q = self.q_conv(x.flatten(0, 1)).reshape(T, B, C, H, W)
        k = self.k_conv(x.flatten(0, 1)).reshape(T, B, C, H, W)
        v = self.v_conv(x.flatten(0, 1)).reshape(T, B, C, H, W)

        q = self.q_lif(q).flatten(3)
        q = (
            q.transpose(-1, -2)
            .reshape(T, B, N, self.num_heads, C // self.num_heads)
            .permute(0, 1, 3, 2, 4)
            .contiguous()
        )

        k = self.k_lif(k).flatten(3)
        k = (
            k.transpose(-1, -2)
            .reshape(T, B, N, self.num_heads, C // self.num_heads)
            .permute(0, 1, 3, 2, 4)
            .contiguous()
        )

        v = self.v_lif(v).flatten(3)
        v = (
            v.transpose(-1, -2)
            .reshape(T, B, N, self.num_heads, C // self.num_heads)
            .permute(0, 1, 3, 2, 4)
            .contiguous()
        )

        x = k.transpose(-2, -1) @ v
        x = (q @ x) * self.scale

        x = x.transpose(3, 4).reshape(T, B, C, N).contiguous()
        x = self.attn_lif(x).reshape(T, B, C, H, W)
        x = x.reshape(T, B, C, H, W)
        x = x.flatten(0, 1)
        x = self.proj_conv(x).reshape(T, B, C, H, W)

        return x


class MS_Block(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        norm_layer=nn.LayerNorm,
        sr_ratio=1,
    ):
        super().__init__()

        self.attn = MS_Attention_RepConv_qkv_id(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            sr_ratio=sr_ratio,
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MS_MLP(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)

    def forward(self, x):
        x = x + self.attn(x)
        x = x + self.mlp(x)

        return x


class MS_DownSampling(nn.Module):
    def __init__(
        self,
        in_channels=2,
        embed_dims=256,
        kernel_size=3,
        stride=2,
        padding=1,
        first_layer=True,
    ):
        super().__init__()

        self.encode_conv = nn.Conv2d(
            in_channels,
            embed_dims,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )

        self.encode_bn = nn.BatchNorm2d(embed_dims)
        if not first_layer:
            self.encode_lif = MultiStepLIFNode(
                tau=2.0, detach_reset=True, backend="cupy"
            )

    def forward(self, x):
        T, B, _, _, _ = x.shape

        if hasattr(self, "encode_lif"):
            x = self.encode_lif(x)
        x = self.encode_conv(x.flatten(0, 1))
        _, _, H, W = x.shape
        x = self.encode_bn(x).reshape(T, B, -1, H, W).contiguous()

        return x


class Spiking_vit_MetaFormer(nn.Module):
    def __init__(
        self,
        img_size_h=128,
        img_size_w=128,
        patch_size=16,
        in_channels=2,
        num_classes=11,
        embed_dim=[64, 128, 256],
        num_heads=[1, 2, 4],
        mlp_ratios=[4, 4, 4],
        qkv_bias=False,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        depths=[6, 8, 6],
        sr_ratios=[8, 4, 2],
        kd=False,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths
        self.T = 1
        # embed_dim = [64, 128, 256, 512]

        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, depths)
        ]  # stochastic depth decay rule

        self.downsample1_1 = MS_DownSampling(
            in_channels=in_channels,
            embed_dims=embed_dim[0] // 2,
            kernel_size=7,
            stride=2,
            padding=3,
            first_layer=True,
        )

        self.ConvBlock1_1 = nn.ModuleList(
            [MS_ConvBlock(dim=embed_dim[0] // 2, mlp_ratio=mlp_ratios)]
        )

        self.downsample1_2 = MS_DownSampling(
            in_channels=embed_dim[0] // 2,
            embed_dims=embed_dim[0],
            kernel_size=3,
            stride=2,
            padding=1,
            first_layer=False,
        )

        self.ConvBlock1_2 = nn.ModuleList(
            [MS_ConvBlock(dim=embed_dim[0], mlp_ratio=mlp_ratios)]
        )

        self.downsample2 = MS_DownSampling(
            in_channels=embed_dim[0],
            embed_dims=embed_dim[1],
            kernel_size=3,
            stride=2,
            padding=1,
            first_layer=False,
        )

        self.ConvBlock2_1 = nn.ModuleList(
            [MS_ConvBlock(dim=embed_dim[1], mlp_ratio=mlp_ratios)]
        )

        self.ConvBlock2_2 = nn.ModuleList(
            [MS_ConvBlock(dim=embed_dim[1], mlp_ratio=mlp_ratios)]
        )

        self.downsample3 = MS_DownSampling(
            in_channels=embed_dim[1],
            embed_dims=embed_dim[2],
            kernel_size=3,
            stride=2,
            padding=1,
            first_layer=False,
        )

        self.block3 = nn.ModuleList(
            [
                MS_Block(
                    dim=embed_dim[2],
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratios,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[j],
                    norm_layer=norm_layer,
                    sr_ratio=sr_ratios,
                )
                for j in range(6)
            ]
        )

        self.downsample4 = MS_DownSampling(
            in_channels=embed_dim[2],
            embed_dims=embed_dim[3],
            kernel_size=3,
            stride=1,
            padding=1,
            first_layer=False,
        )

        self.block4 = nn.ModuleList(
            [
                MS_Block(
                    dim=embed_dim[3],
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratios,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[j],
                    norm_layer=norm_layer,
                    sr_ratio=sr_ratios,
                )
                for j in range(2)
            ]
        )

        self.lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend="cupy")
        self.head = (
            nn.Linear(embed_dim[3], num_classes) if num_classes > 0 else nn.Identity()
        )

        self.kd = kd
        if self.kd:
            self.head_kd = (
                nn.Linear(embed_dim[3], num_classes)
                if num_classes > 0
                else nn.Identity()
            )
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):
        x = self.downsample1_1(x)
        for blk in self.ConvBlock1_1:
            x = blk(x)
        x = self.downsample1_2(x)
        for blk in self.ConvBlock1_2:
            x = blk(x)

        x = self.downsample2(x)
        for blk in self.ConvBlock2_1:
            x = blk(x)
        for blk in self.ConvBlock2_2:
            x = blk(x)

        x = self.downsample3(x)
        for blk in self.block3:
            x = blk(x)

        x = self.downsample4(x)
        for blk in self.block4:
            x = blk(x)
        return x  # T,B,C,N

    def forward(self, x):
        x = (x.unsqueeze(0)).repeat(self.T, 1, 1, 1, 1)
        x = self.forward_features(x)
        x = x.flatten(3).mean(3)
        x_lif = self.lif(x)
        x = self.head(x_lif).mean(0)
        if self.kd:
            x_kd = self.head_kd(x_lif).mean(0)
            if self.training:
                return x, x_kd
            else:
                return (x + x_kd) / 2
        return x


def metaspikformer_8_384(**kwargs):
    num_classes = kwargs.pop("num_classes", 1000)
    model = Spiking_vit_MetaFormer(
        img_size_h=224,
        img_size_w=224,
        patch_size=16,
        embed_dim=[96, 192, 384, 480],
        num_heads=8,
        mlp_ratios=4,
        in_channels=3,
        num_classes=num_classes,
        qkv_bias=False,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        depths=8,
        sr_ratios=1,
        **kwargs,
    )
    return model


def metaspikformer_8_512(**kwargs):
    num_classes = kwargs.pop("num_classes", 1000)
    model = Spiking_vit_MetaFormer(
        img_size_h=224,
        img_size_w=224,
        patch_size=16,
        embed_dim=[128, 256, 512, 640],
        num_heads=8,
        mlp_ratios=4,
        in_channels=3,
        num_classes=num_classes,
        qkv_bias=False,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        depths=8,
        sr_ratios=1,
        **kwargs,
    )
    return model


def metaspikformer_8_768(**kwargs):
    num_classes = kwargs.pop("num_classes", 1000)
    model = Spiking_vit_MetaFormer(
        img_size_h=224,
        img_size_w=224,
        patch_size=16,
        embed_dim=[192, 384, 768, 960],
        num_heads=8,
        mlp_ratios=4,
        in_channels=3,
        num_classes=num_classes,
        qkv_bias=False,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        depths=8,
        sr_ratios=1,
        **kwargs,
    )
    return model


from timm.models import create_model


def plot_firing_rate_by_block(model, out_dir="graphics", filename="firing_rate_by_block.png"):
    """Collect average head_lif firing rate from all MS_Attention_RepConv_qkv_id
    modules in `model` and save a bar plot to `out_dir/filename` with values
    printed above each bar. Matplotlib is imported lazily.
    """
    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        print("plot_firing_rate_by_block: matplotlib not available:", e)
        return None

    # gather attention modules
    attn_modules = [m for m in model.modules() if isinstance(m, MS_Attention_RepConv_qkv_id)]
    if len(attn_modules) == 0:
        print("plot_firing_rate_by_block: no MS_Attention_RepConv_qkv_id modules found in model")
        return None

    import statistics

    means = []
    stds = []
    labels = []
    for idx, m in enumerate(attn_modules):
        logs = getattr(m, "head_lif_firing_rate_log", None)
        if logs and len(logs) > 0:
            try:
                mean_val = float(statistics.mean(logs))
                std_val = float(statistics.pstdev(logs)) if len(logs) > 1 else 0.0
            except Exception:
                mean_val = float(sum(logs) / len(logs))
                # fallback population std
                std_val = float((sum((x - mean_val) ** 2 for x in logs) / len(logs)) ** 0.5)
        else:
            last = getattr(m, "last_head_lif_firing_rate", None)
            try:
                mean_val = float(last) if last is not None else 0.0
            except Exception:
                mean_val = 0.0
            std_val = 0.0

        means.append(mean_val)
        stds.append(std_val)
        labels.append(f"attn_{idx}")

    os.makedirs(out_dir, exist_ok=True)

    plt.figure(figsize=(max(6, len(means) * 0.6), 4))
    bars = plt.bar(labels, means, color="tab:blue", yerr=stds, capsize=4, error_kw={"elinewidth":1, "alpha":0.8})
    plt.ylabel("Firing rate (fraction)")
    plt.xlabel("Attention module")
    plt.title("Average head_lif firing rate per attention module")
    plt.ylim(0.0, 1.0)
    plt.xticks(rotation=45, ha="right")

    # annotate values above bars with mean ± std
    for bar, mean_val, std_val in zip(bars, means, stds):
        height = bar.get_height()
        label = f"{mean_val:.3f}±{std_val:.3f}" if std_val > 0 else f"{mean_val:.3f}"
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height + max(0.01, 0.02 * (1.0 if height == 0 else height)),
            label,
            ha="center",
            va="bottom",
            fontsize=8,
        )

    plt.tight_layout()

    out_path = os.path.join(out_dir, filename)
    plt.savefig(out_path)
    plt.close()

    print(f"Saved firing rate plot to {out_path}")
    return out_path

