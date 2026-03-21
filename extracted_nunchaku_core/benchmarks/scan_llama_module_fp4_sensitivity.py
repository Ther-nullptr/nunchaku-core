from __future__ import annotations

import argparse
import gc
import json
import math
import os
from collections import defaultdict
from datetime import datetime

import torch
from torch import nn

from benchmark_hf_llama_fp4_inference import (
    compute_perplexity_sum,
    ensure_model_downloaded,
    get_dtype,
    load_model,
    load_tokenizer,
    load_wikitext_token_stream,
    set_hf_mirror,
)
from native_fp4.operators import NunchakuFP4GemmOp


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scan Llama module-wise FP4 sensitivity.")
    parser.add_argument("--model-id", type=str, default="NousResearch/Llama-2-7b-hf")
    parser.add_argument(
        "--model-dir",
        type=str,
        default="/home/wyj24/projects/nunchaku/extracted_nunchaku_core/models/Llama-2-7b-hf",
    )
    parser.add_argument("--dtype", choices=["fp16", "bf16"], default="fp16")
    parser.add_argument("--dataset-name", type=str, default="wikitext")
    parser.add_argument("--dataset-config", type=str, default="wikitext-2-raw-v1")
    parser.add_argument("--dataset-split", type=str, default="test")
    parser.add_argument("--dataset-max-docs", type=int, default=0)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--num-seqs", type=int, default=8)
    parser.add_argument("--offset", type=int, default=8192)
    parser.add_argument("--linear-prefix", type=str, default="model.layers.")
    parser.add_argument("--include-lm-head", action="store_true")
    parser.add_argument("--hf-endpoint", type=str, default="https://hf-mirror.com")
    parser.add_argument("--results-dir", type=str, default="results")
    return parser.parse_args()


def cleanup_model(model) -> None:
    if model is not None:
        del model
    gc.collect()
    torch.cuda.empty_cache()


def should_scan_module(name: str, include_lm_head: bool, linear_prefix: str) -> bool:
    return name.startswith(linear_prefix) or (include_lm_head and name == "lm_head")


def split_parent(root: nn.Module, full_name: str) -> tuple[nn.Module, str]:
    parts = full_name.split(".")
    parent = root
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]


def infer_module_kind(full_name: str) -> str:
    return full_name.split(".")[-1]


def collect_linear_modules(model: nn.Module, linear_prefix: str, include_lm_head: bool) -> list[str]:
    names = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and should_scan_module(name, include_lm_head=include_lm_head, linear_prefix=linear_prefix):
            names.append(name)
    return names


def evaluate_module(
    model: nn.Module,
    module_name: str,
    dtype: torch.dtype,
    token_stream: torch.Tensor,
    seq_len: int,
    num_seqs: int,
    offset: int,
) -> dict:
    parent, attr = split_parent(model, module_name)
    orig_module = getattr(parent, attr)
    assert isinstance(orig_module, nn.Linear)

    weight = orig_module.weight.detach().to(device="cuda", dtype=dtype).contiguous()
    bias = None
    if orig_module.bias is not None:
        bias = orig_module.bias.detach().to(device="cuda", dtype=dtype).contiguous()
    replacement = NunchakuFP4GemmOp(weight=weight, bias=bias)
    setattr(parent, attr, replacement)

    try:
        nll_sum, total_tokens = compute_perplexity_sum(
            model=model,
            token_stream=token_stream,
            seq_len=seq_len,
            num_seqs=num_seqs,
            offset_tokens=offset,
        )
        ppl = math.exp(nll_sum / max(total_tokens, 1))
    finally:
        setattr(parent, attr, orig_module)
        del replacement, weight, bias
        torch.cuda.empty_cache()

    return {
        "module": module_name,
        "kind": infer_module_kind(module_name),
        "nll_sum": nll_sum,
        "total_tokens": total_tokens,
        "perplexity": ppl,
        "num_params": orig_module.weight.numel() + (0 if orig_module.bias is None else orig_module.bias.numel()),
    }


def summarize_by_kind(records: list[dict]) -> dict[str, dict]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for record in records:
        grouped[record["kind"]].append(record["perplexity_ratio_vs_fp16"])
    summary = {}
    for kind, values in grouped.items():
        values_sorted = sorted(values)
        summary[kind] = {
            "count": len(values),
            "min_ratio": values_sorted[0],
            "median_ratio": values_sorted[len(values_sorted) // 2],
            "max_ratio": values_sorted[-1],
            "mean_ratio": sum(values_sorted) / len(values_sorted),
        }
    return summary


def main() -> None:
    args = parse_args()
    set_hf_mirror(args.hf_endpoint)
    os.makedirs(args.results_dir, exist_ok=True)

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
    module_names = collect_linear_modules(
        model,
        linear_prefix=args.linear_prefix,
        include_lm_head=args.include_lm_head,
    )

    baseline_nll_sum, baseline_total_tokens = compute_perplexity_sum(
        model=model,
        token_stream=token_stream,
        seq_len=args.seq_len,
        num_seqs=args.num_seqs,
        offset_tokens=args.offset,
    )
    baseline_ppl = math.exp(baseline_nll_sum / max(baseline_total_tokens, 1))

    records = []
    for module_name in module_names:
        record = evaluate_module(
            model=model,
            module_name=module_name,
            dtype=dtype,
            token_stream=token_stream,
            seq_len=args.seq_len,
            num_seqs=args.num_seqs,
            offset=args.offset,
        )
        record["perplexity_ratio_vs_fp16"] = record["perplexity"] / baseline_ppl
        records.append(record)

    cleanup_model(model)

    ranked = sorted(records, key=lambda item: item["perplexity"], reverse=True)
    insensitive = sorted(records, key=lambda item: item["perplexity"])

    result = {
        "model_id": args.model_id,
        "model_dir": model_dir,
        "dtype": args.dtype,
        "baseline_fp16": {
            "perplexity": baseline_ppl,
            "nll_sum": baseline_nll_sum,
            "total_tokens": baseline_total_tokens,
        },
        "dataset": {
            "name": args.dataset_name,
            "config": args.dataset_config,
            "split": args.dataset_split,
            "seq_len": args.seq_len,
            "num_seqs": args.num_seqs,
            "offset": args.offset,
        },
        "num_modules": len(records),
        "module_records": records,
        "summary": {
            "most_sensitive_modules": ranked[:10],
            "least_sensitive_modules": insensitive[:10],
            "by_kind": summarize_by_kind(records),
        },
    }

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(args.results_dir, f"llama_module_fp4_sensitivity_{timestamp}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(out_path)


if __name__ == "__main__":
    main()
