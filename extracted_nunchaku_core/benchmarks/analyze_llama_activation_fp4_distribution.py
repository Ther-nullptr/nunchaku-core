from __future__ import annotations

import argparse
import gc
import json
import os
from collections import defaultdict
from datetime import datetime

import torch
from torch import nn

from benchmark_hf_llama_fp4_inference import (
    ensure_model_downloaded,
    get_dtype,
    load_model,
    load_tokenizer,
    load_wikitext_token_stream,
    set_hf_mirror,
)
from native_fp4.layout import compute_fp4_logical_scales, quantize_fp4_codes_from_dense


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze activation distributions for FP4-sensitive Llama modules.")
    parser.add_argument("--model-id", type=str, default="NousResearch/Llama-2-7b-hf")
    parser.add_argument(
        "--model-dir",
        type=str,
        default="/home/wyj24/projects/nunchaku/extracted_nunchaku_core/models/Llama-2-7b-hf",
    )
    parser.add_argument("--dtype", choices=["fp16", "bf16"], default="fp16")
    parser.add_argument(
        "--sensitivity-json",
        type=str,
        default="/home/wyj24/projects/nunchaku/extracted_nunchaku_core/results/llama_module_fp4_sensitivity_20260321_202421.json",
    )
    parser.add_argument("--num-sensitive", type=int, default=6)
    parser.add_argument("--num-insensitive", type=int, default=6)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--dataset-name", type=str, default="wikitext")
    parser.add_argument("--dataset-config", type=str, default="wikitext-2-raw-v1")
    parser.add_argument("--dataset-split", type=str, default="test")
    parser.add_argument("--dataset-max-docs", type=int, default=0)
    parser.add_argument("--hf-endpoint", type=str, default="https://hf-mirror.com")
    parser.add_argument("--results-dir", type=str, default="results")
    return parser.parse_args()


def cleanup_model(model) -> None:
    if model is not None:
        del model
    gc.collect()
    torch.cuda.empty_cache()


def choose_modules(sensitivity_json: str, num_sensitive: int, num_insensitive: int) -> dict[str, str]:
    data = json.load(open(sensitivity_json, "r", encoding="utf-8"))
    chosen = {}
    for record in data["summary"]["most_sensitive_modules"][:num_sensitive]:
        chosen[record["module"]] = "sensitive"
    for record in data["summary"]["least_sensitive_modules"][:num_insensitive]:
        chosen[record["module"]] = "insensitive"
    return chosen


def capture_inputs(model: nn.Module, module_names: list[str], input_ids: torch.Tensor, attention_mask: torch.Tensor) -> dict[str, torch.Tensor]:
    captured: dict[str, torch.Tensor] = {}
    hooks = []
    targets = set(module_names)

    for name, module in model.named_modules():
        if name in targets and isinstance(module, nn.Linear):
            def _hook(mod, args, full_name=name):
                captured[full_name] = args[0].detach().cpu()

            hooks.append(module.register_forward_pre_hook(_hook))

    with torch.inference_mode():
        _ = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).logits

    for hook in hooks:
        hook.remove()
    return captured


def summarize_tensor(x: torch.Tensor) -> dict:
    x2d = x.reshape(-1, x.shape[-1]).to(device="cuda")
    rows, cols = x2d.shape
    cols_pad = (cols + 127) // 128 * 128
    pad = torch.zeros((rows, cols_pad), dtype=x2d.dtype, device=x2d.device)
    pad[:, :cols] = x2d

    logical = compute_fp4_logical_scales(pad).float()
    expanded = logical.to(x2d.dtype).repeat_interleave(16, dim=1)
    denom = expanded * 6.0
    ratio = torch.zeros_like(pad, dtype=torch.float32)
    nz = denom > 0
    ratio[nz] = pad.abs()[nz].float() / denom[nz].float()

    codes = quantize_fp4_codes_from_dense(pad, logical.to(x2d.dtype))
    mag = (codes & 0x7).float()

    group = pad.abs().reshape(rows, cols_pad // 16, 16).float()
    group_max = group.amax(dim=2)
    group_mean = group.mean(dim=2)
    group_p95 = torch.quantile(group, 0.95, dim=2)

    abs_x = x2d.abs().float().reshape(-1)
    scales_flat = logical.reshape(-1)

    result = {
        "input_shape": list(x.shape),
        "num_groups": int(scales_flat.numel()),
        "abs_q50": float(torch.quantile(abs_x, 0.50).item()),
        "abs_q90": float(torch.quantile(abs_x, 0.90).item()),
        "abs_q99": float(torch.quantile(abs_x, 0.99).item()),
        "abs_q999": float(torch.quantile(abs_x, 0.999).item()),
        "abs_max": float(abs_x.max().item()),
        "scale_mean": float(scales_flat.mean().item()),
        "scale_median": float(torch.quantile(scales_flat, 0.50).item()),
        "scale_q99": float(torch.quantile(scales_flat, 0.99).item()),
        "scale_q999": float(torch.quantile(scales_flat, 0.999).item()),
        "scale_max": float(scales_flat.max().item()),
        "frac_scale_gt_1": float((scales_flat > 1).float().mean().item()),
        "frac_scale_gt_10": float((scales_flat > 10).float().mean().item()),
        "frac_scale_gt_100": float((scales_flat > 100).float().mean().item()),
        "top_code_frac": float((mag[:, :cols] == 7).float().mean().item()),
        "max_ratio_to_fp4_limit": float(ratio[:, :cols].max().item()),
        "true_overflow_count": int((ratio[:, :cols] > 1.00001).sum().item()),
        "mean_group_max_over_mean": float((group_max / group_mean.clamp_min(1e-12)).mean().item()),
        "mean_group_max_over_p95": float((group_max / group_p95.clamp_min(1e-12)).mean().item()),
        "max_group_max_over_mean": float((group_max / group_mean.clamp_min(1e-12)).max().item()),
    }

    del x2d, pad, logical, expanded, denom, ratio, codes, mag, group, group_max, group_mean, group_p95, abs_x, scales_flat
    torch.cuda.empty_cache()
    return result


def summarize_by_bucket(records: list[dict]) -> dict[str, dict]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        grouped[record["bucket"]].append(record)
    summary = {}
    metrics = [
        "abs_q999",
        "abs_max",
        "scale_q999",
        "scale_max",
        "frac_scale_gt_1",
        "frac_scale_gt_10",
        "frac_scale_gt_100",
        "top_code_frac",
        "mean_group_max_over_mean",
        "mean_group_max_over_p95",
    ]
    for bucket, bucket_records in grouped.items():
        summary[bucket] = {"count": len(bucket_records)}
        for metric in metrics:
            values = [record[metric] for record in bucket_records]
            summary[bucket][f"{metric}_mean"] = sum(values) / len(values)
            summary[bucket][f"{metric}_max"] = max(values)
    return summary


def main() -> None:
    args = parse_args()
    set_hf_mirror(args.hf_endpoint)
    os.makedirs(args.results_dir, exist_ok=True)

    chosen = choose_modules(args.sensitivity_json, args.num_sensitive, args.num_insensitive)
    dtype = get_dtype(args.dtype)
    model_dir = ensure_model_downloaded(args.model_id, args.model_dir)
    tokenizer = load_tokenizer(model_dir)
    token_stream = load_wikitext_token_stream(
        tokenizer=tokenizer,
        dataset_name=args.dataset_name,
        dataset_config=args.dataset_config,
        dataset_split=args.dataset_split,
        dataset_max_docs=args.dataset_max_docs,
    )

    model = load_model(model_dir, dtype=dtype)
    input_ids = token_stream[args.offset : args.offset + args.seq_len].unsqueeze(0).to("cuda")
    attention_mask = torch.ones_like(input_ids)
    captured = capture_inputs(model, list(chosen.keys()), input_ids=input_ids, attention_mask=attention_mask)
    cleanup_model(model)

    records = []
    for module_name, bucket in chosen.items():
        stats = summarize_tensor(captured[module_name].to(dtype))
        stats["module"] = module_name
        stats["bucket"] = bucket
        stats["kind"] = module_name.split(".")[-1]
        records.append(stats)

    result = {
        "model_id": args.model_id,
        "model_dir": model_dir,
        "dtype": args.dtype,
        "sensitivity_json": args.sensitivity_json,
        "dataset": {
            "name": args.dataset_name,
            "config": args.dataset_config,
            "split": args.dataset_split,
            "seq_len": args.seq_len,
            "offset": args.offset,
        },
        "module_records": records,
        "summary": summarize_by_bucket(records),
    }

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(args.results_dir, f"llama_activation_fp4_distribution_{timestamp}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(out_path)


if __name__ == "__main__":
    main()
