from __future__ import annotations

import argparse
import gc
import json
import math
import os
from datetime import datetime

import torch

from benchmark_hf_llama_fp4_inference import (
    ensure_model_downloaded,
    get_dtype,
    load_model,
    load_tokenizer,
    load_wikitext_token_stream,
    replace_linear_modules,
    set_hf_mirror,
    compute_perplexity_sum,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scan Llama layer-wise FP4 sensitivity.")
    parser.add_argument("--model-id", type=str, default="NousResearch/Llama-2-7b-hf")
    parser.add_argument(
        "--model-dir",
        type=str,
        default="/home/wyj24/projects/nunchaku/extracted_nunchaku_core/models/Llama-2-7b-hf",
    )
    parser.add_argument("--dtype", choices=["fp16", "bf16"], default="fp16")
    parser.add_argument("--variant", choices=["fp4", "fp4_hybrid"], default="fp4")
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--num-layers", type=int, default=32)
    parser.add_argument("--dataset-name", type=str, default="wikitext")
    parser.add_argument("--dataset-config", type=str, default="wikitext-2-raw-v1")
    parser.add_argument("--dataset-split", type=str, default="test")
    parser.add_argument("--dataset-max-docs", type=int, default=0)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--num-seqs", type=int, default=8)
    parser.add_argument("--offset", type=int, default=8192)
    parser.add_argument("--hf-endpoint", type=str, default="https://hf-mirror.com")
    parser.add_argument("--results-dir", type=str, default="results")
    return parser.parse_args()


def cleanup_model(model) -> None:
    if model is not None:
        del model
    gc.collect()
    torch.cuda.empty_cache()


def evaluate_layer(
    model_dir: str,
    dtype: torch.dtype,
    token_stream: torch.Tensor,
    variant: str,
    rank: int,
    seq_len: int,
    num_seqs: int,
    offset: int,
    layer_idx: int,
) -> dict:
    model = load_model(model_dir, dtype=dtype)
    replaced_modules, replaced_params = replace_linear_modules(
        module=model,
        variant=variant,
        dtype=dtype,
        rank=rank,
        linear_prefix="model.layers.",
        include_lm_head=False,
        layer_start=layer_idx,
        layer_end=layer_idx + 1,
        name_substrings=None,
        factor_mode="svd_lowrank",
        svd_lowrank_oversample=8,
        svd_lowrank_niter=2,
    )
    nll_sum, total_tokens = compute_perplexity_sum(
        model=model,
        token_stream=token_stream,
        seq_len=seq_len,
        num_seqs=num_seqs,
        offset_tokens=offset,
    )
    ppl = math.exp(nll_sum / max(total_tokens, 1))
    cleanup_model(model)
    return {
        "layer_idx": layer_idx,
        "replaced_modules": replaced_modules,
        "replaced_params": replaced_params,
        "nll_sum": nll_sum,
        "total_tokens": total_tokens,
        "perplexity": ppl,
    }


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

    baseline_model = load_model(model_dir, dtype=dtype)
    baseline_nll_sum, baseline_total_tokens = compute_perplexity_sum(
        model=baseline_model,
        token_stream=token_stream,
        seq_len=args.seq_len,
        num_seqs=args.num_seqs,
        offset_tokens=args.offset,
    )
    baseline_ppl = math.exp(baseline_nll_sum / max(baseline_total_tokens, 1))
    cleanup_model(baseline_model)

    records = []
    for layer_idx in range(args.num_layers):
        record = evaluate_layer(
            model_dir=model_dir,
            dtype=dtype,
            token_stream=token_stream,
            variant=args.variant,
            rank=args.rank,
            seq_len=args.seq_len,
            num_seqs=args.num_seqs,
            offset=args.offset,
            layer_idx=layer_idx,
        )
        record["perplexity_ratio_vs_fp16"] = record["perplexity"] / baseline_ppl
        records.append(record)

    ranked = sorted(records, key=lambda item: item["perplexity"], reverse=True)
    insensitive = sorted(records, key=lambda item: item["perplexity"])

    result = {
        "model_id": args.model_id,
        "model_dir": model_dir,
        "dtype": args.dtype,
        "variant": args.variant,
        "rank": args.rank,
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
        "layer_records": records,
        "summary": {
            "most_sensitive_layers": ranked[:5],
            "least_sensitive_layers": insensitive[:5],
        },
    }

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(args.results_dir, f"llama_layer_fp4_sensitivity_{args.variant}_{timestamp}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(out_path)


if __name__ == "__main__":
    main()
