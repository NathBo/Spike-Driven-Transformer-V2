import argparse
import json
import os
import time
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, SequentialSampler

import models
from engine_finetune import evaluate
from spikingjelly.clock_driven import functional
from util.datasets import build_dataset


def log(message):
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def build_args():
    parser = argparse.ArgumentParser(description="Measure dense/event-driven SDT metrics")
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model", default="metaspikformer_8_512")
    parser.add_argument("--nb_classes", type=int, default=1000)
    parser.add_argument("--input_size", type=int, default=224)
    parser.add_argument("--time_steps", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--max_eval_batches", type=int, default=0)
    parser.add_argument("--output", default="metrics.json")
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def dataset_args(args):
    return SimpleNamespace(
        data_path=args.data_path,
        input_size=args.input_size,
        color_jitter=0.3,
        aa="rand-m9-mstd0.5-inc1",
        reprob=0.25,
        remode="pixel",
        recount=1,
    )


def load_model(args, mode, device):
    log(f"Chargement du modele: {mode}")
    model = models.__dict__[args.model](
        kd=False,
        num_classes=args.nb_classes,
        event_pointwise=mode == "spatial_sparse",
        channel_sparse=mode == "channel_sparse",
    )
    model.T = args.time_steps
    # Training checkpoints contain argparse.Namespace metadata in addition to weights.
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("model", checkpoint)
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device).eval()
    log(f"Modele pret: {mode}")
    return model


def reset_model(model):
    functional.reset_net(model)


def collect_firing_rate_hooks(model):
    values = {}
    handles = []
    for index, module in enumerate(model.modules()):
        if isinstance(module, models.MS_Attention_RepConv_qkv_id):
            name = f"attn_{index}"
            values[name] = []

            def hook(_module, _inputs, output, key=name):
                with torch.no_grad():
                    values[key].append(float((output != 0).float().mean().item()))

            handles.append(module.head_lif.register_forward_hook(hook))
    return values, handles


def collect_event_hooks(model):
    errors = {}
    handles = []
    for index, module in enumerate(model.modules()):
        if isinstance(module, (models.EventPointwiseConv, models.ChannelSparsePointwiseConv)):
            name = f"event_conv_{index}"
            errors[name] = {"max_error": 0.0, "mean_error_sum": 0.0, "samples": 0}

            def hook(event_module, inputs, output, key=name):
                x = inputs[0]
                with torch.no_grad():
                    dense = F.conv2d(
                        x,
                        event_module.weight,
                        event_module.bias,
                        event_module.stride,
                        event_module.padding,
                        event_module.dilation,
                        event_module.groups,
                    )
                    error = (dense.float() - output.float()).abs()
                    values = errors[key]
                    values["max_error"] = max(values["max_error"], float(error.max().item()))
                    values["mean_error_sum"] += float(error.mean().item())
                    values["samples"] += 1
                    values.setdefault("allclose", True)
                    values["allclose"] = values["allclose"] and torch.allclose(
                        dense, output, atol=1e-5, rtol=1e-5
                    )
                    values.setdefault("allclose_relaxed", True)
                    values["allclose_relaxed"] = values["allclose_relaxed"] and torch.allclose(
                        dense, output, atol=1e-3, rtol=1e-3
                    )

            handles.append(module.register_forward_hook(hook))
    return errors, handles


def accuracy_pass(model, loader, device, max_batches, label):
    model.eval()
    correct1 = 0
    correct5 = 0
    total = 0
    total_batches = len(loader) if not max_batches else min(len(loader), max_batches)
    log(f"Debut accuracy {label}: {total_batches} batches")
    for batch_index, (images, targets) in enumerate(loader):
        if max_batches and batch_index >= max_batches:
            break
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        with torch.no_grad():
            output = model(images)
        topk = output.topk(min(5, output.shape[1]), dim=1).indices
        correct1 += int((topk[:, :1] == targets[:, None]).any(dim=1).sum().item())
        correct5 += int((topk == targets[:, None]).any(dim=1).sum().item())
        total += targets.numel()
        reset_model(model)
        log(f"Accuracy {label}: batch {batch_index + 1}/{total_batches}")
    log(f"Fin accuracy {label}: Top-1={100.0 * correct1 / total:.3f}% Top-5={100.0 * correct5 / total:.3f}%")
    return {"top1": 100.0 * correct1 / total, "top5": 100.0 * correct5 / total, "samples": total}


def timed_forward(model, sample, args, label):
    log(f"Debut warmup {label}: {args.warmup} forwards")
    for iteration in range(args.warmup):
        with torch.no_grad():
            model(sample)
        reset_model(model)
        log(f"Warmup {label}: {iteration + 1}/{args.warmup}")
    if sample.is_cuda:
        torch.cuda.synchronize(sample.device)
    log(f"Debut timing {label}: {args.iters} forwards")
    start = time.perf_counter()
    for iteration in range(args.iters):
        with torch.no_grad():
            model(sample)
        reset_model(model)
        log(f"Timing {label}: {iteration + 1}/{args.iters}")
    if sample.is_cuda:
        torch.cuda.synchronize(sample.device)
    average_ms = (time.perf_counter() - start) * 1000.0 / args.iters
    log(f"Fin timing {label}: {average_ms:.3f} ms/forward")
    return average_ms


def event_metrics(model):
    result = {}
    for index, module in enumerate(model.modules()):
        if isinstance(module, models.EventPointwiseConv):
            total = module.event_total_positions
            active = module.event_active_positions
            dense_ops = total * module.in_channels * module.out_channels
            result[f"event_conv_{index}"] = {
                "in_channels": module.in_channels,
                "out_channels": module.out_channels,
                "total_positions": total,
                "active_positions": active,
                "active_fraction": active / total if total else 0.0,
                "skipped_fraction": 1.0 - active / total if total else 0.0,
                "dense_macs_estimate": dense_ops,
                "active_macs_estimate": active * module.in_channels * module.out_channels,
            }
        elif isinstance(module, models.ChannelSparsePointwiseConv):
            total = module.channel_total
            active = module.channel_active
            positions = module.channel_positions
            dense_ops = total * module.out_channels
            result[f"event_conv_{index}"] = {
                "in_channels": module.in_channels,
                "out_channels": module.out_channels,
                "total_channel_values": total,
                "active_channel_values": active,
                "positions": positions,
                "mean_active_channels_per_position": active / positions if positions else 0.0,
                "active_fraction": active / total if total else 0.0,
                "skipped_fraction": 1.0 - active / total if total else 0.0,
                "dense_macs_estimate": dense_ops,
                "active_macs_estimate": active * module.out_channels,
            }
    return result


def global_firing_rate(rates):
    values = [item["mean"] for item in rates.values() if item["samples"]]
    return sum(values) / len(values) if values else 0.0


def main():
    args = build_args()
    log("Demarrage de la mesure")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    log("Chargement du dataset de validation")
    data = build_dataset(False, dataset_args(args))
    loader = DataLoader(
        data,
        sampler=SequentialSampler(data),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    log(f"Dataset pret: {len(data)} images, {len(loader)} batches")

    results = {"configuration": vars(args)}
    dense_model = load_model(args, "dense", device)
    spatial_model = load_model(args, "spatial_sparse", device)
    channel_model = load_model(args, "channel_sparse", device)

    dense_rates, dense_rate_handles = collect_firing_rate_hooks(dense_model)
    spatial_rates, spatial_rate_handles = collect_firing_rate_hooks(spatial_model)
    channel_rates, channel_rate_handles = collect_firing_rate_hooks(channel_model)
    spatial_errors, spatial_error_handles = collect_event_hooks(spatial_model)
    channel_errors, channel_error_handles = collect_event_hooks(channel_model)

    results["accuracy_dense"] = accuracy_pass(
        dense_model, loader, device, args.max_eval_batches, "dense"
    )
    results["accuracy_spatial_sparse"] = accuracy_pass(
        spatial_model, loader, device, args.max_eval_batches, "spatial_sparse"
    )
    results["accuracy_channel_sparse"] = accuracy_pass(
        channel_model, loader, device, args.max_eval_batches, "channel_sparse"
    )

    results["firing_rate_dense_by_attention"] = {
        key: {"mean": sum(values) / len(values), "samples": len(values)}
        for key, values in dense_rates.items()
        if values
    }
    results["firing_rate_spatial_by_attention"] = {
        key: {"mean": sum(values) / len(values), "samples": len(values)}
        for key, values in spatial_rates.items()
        if values
    }
    results["firing_rate_channel_by_attention"] = {
        key: {"mean": sum(values) / len(values), "samples": len(values)}
        for key, values in channel_rates.items()
        if values
    }
    results["firing_rate_dense_global"] = global_firing_rate(
        results["firing_rate_dense_by_attention"]
    )
    results["firing_rate_spatial_global"] = global_firing_rate(
        results["firing_rate_spatial_by_attention"]
    )
    results["firing_rate_channel_global"] = global_firing_rate(
        results["firing_rate_channel_by_attention"]
    )
    results["spatial_sparse_stats"] = event_metrics(spatial_model)
    results["channel_sparse_stats"] = event_metrics(channel_model)

    for handle in spatial_error_handles + channel_error_handles:
        handle.remove()

    sample, _ = next(iter(loader))
    sample = sample.to(device, non_blocking=True)
    results["time_ms_dense"] = timed_forward(dense_model, sample, args, "dense")
    results["time_ms_spatial_sparse"] = timed_forward(
        spatial_model, sample, args, "spatial_sparse"
    )
    results["time_ms_channel_sparse"] = timed_forward(
        channel_model, sample, args, "channel_sparse"
    )

    for handle in (
        dense_rate_handles + spatial_rate_handles + channel_rate_handles
        + spatial_error_handles + channel_error_handles
    ):
        handle.remove()

    def error_report(errors):
        return {
            key: {
                "max_error": values["max_error"],
                "mean_error": values["mean_error_sum"] / values["samples"]
                if values["samples"] else 0.0,
                "allclose": values.get("allclose", False),
                "allclose_relaxed": values.get("allclose_relaxed", False),
            }
            for key, values in errors.items()
        }

    results["spatial_sparse_validation"] = error_report(spatial_errors)
    results["channel_sparse_validation"] = error_report(channel_errors)

    with open(args.output, "w", encoding="utf-8") as output_file:
        json.dump(results, output_file, indent=2)
    log(f"Mesure terminee, resultats ecrits dans {args.output}")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()