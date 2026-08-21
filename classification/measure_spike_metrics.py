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


def load_model(args, event_pointwise, device):
    model = models.__dict__[args.model](
        kd=False,
        num_classes=args.nb_classes,
        event_pointwise=event_pointwise,
    )
    model.T = args.time_steps
    # Training checkpoints contain argparse.Namespace metadata in addition to weights.
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("model", checkpoint)
    model.load_state_dict(state_dict, strict=False)
    return model.to(device).eval()


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
        if isinstance(module, models.EventPointwiseConv):
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

            handles.append(module.register_forward_hook(hook))
    return errors, handles


def accuracy_pass(model, loader, device, max_batches):
    model.eval()
    correct1 = 0
    correct5 = 0
    total = 0
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
    return {"top1": 100.0 * correct1 / total, "top5": 100.0 * correct5 / total, "samples": total}


def timed_forward(model, sample, args):
    for _ in range(args.warmup):
        with torch.no_grad():
            model(sample)
        reset_model(model)
    if sample.is_cuda:
        torch.cuda.synchronize(sample.device)
    start = time.perf_counter()
    for _ in range(args.iters):
        with torch.no_grad():
            model(sample)
        reset_model(model)
    if sample.is_cuda:
        torch.cuda.synchronize(sample.device)
    return (time.perf_counter() - start) * 1000.0 / args.iters


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
    return result


def global_firing_rate(rates):
    values = [item["mean"] for item in rates.values() if item["samples"]]
    return sum(values) / len(values) if values else 0.0


def main():
    args = build_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    data = build_dataset(False, dataset_args(args))
    loader = DataLoader(
        data,
        sampler=SequentialSampler(data),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    results = {"configuration": vars(args)}
    dense_model = load_model(args, False, device)
    event_model = load_model(args, True, device)

    dense_rates, dense_rate_handles = collect_firing_rate_hooks(dense_model)
    event_rates, event_rate_handles = collect_firing_rate_hooks(event_model)
    event_errors, event_error_handles = collect_event_hooks(event_model)

    results["accuracy_dense"] = accuracy_pass(
        dense_model, loader, device, args.max_eval_batches
    )
    results["accuracy_event_pointwise"] = accuracy_pass(
        event_model, loader, device, args.max_eval_batches
    )

    results["firing_rate_dense_by_attention"] = {
        key: {"mean": sum(values) / len(values), "samples": len(values)}
        for key, values in dense_rates.items()
        if values
    }
    results["firing_rate_event_by_attention"] = {
        key: {"mean": sum(values) / len(values), "samples": len(values)}
        for key, values in event_rates.items()
        if values
    }
    results["firing_rate_dense_global"] = global_firing_rate(
        results["firing_rate_dense_by_attention"]
    )
    results["firing_rate_event_global"] = global_firing_rate(
        results["firing_rate_event_by_attention"]
    )
    results["event_pointwise_sparsity"] = event_metrics(event_model)

    sample, _ = next(iter(loader))
    sample = sample.to(device, non_blocking=True)
    results["time_ms_dense"] = timed_forward(dense_model, sample, args)
    results["time_ms_event_pointwise"] = timed_forward(event_model, sample, args)

    for handle in dense_rate_handles + event_rate_handles + event_error_handles:
        handle.remove()

    event_error_report = {}
    for key, values in event_errors.items():
        samples = values["samples"]
        event_error_report[key] = {
            "max_error": values["max_error"],
            "mean_error": values["mean_error_sum"] / samples if samples else 0.0,
            "allclose": values.get("allclose", False),
        }
    results["event_pointwise_validation"] = event_error_report

    with open(args.output, "w", encoding="utf-8") as output_file:
        json.dump(results, output_file, indent=2)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()