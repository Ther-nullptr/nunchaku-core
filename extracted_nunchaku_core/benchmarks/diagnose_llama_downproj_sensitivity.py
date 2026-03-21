from __future__ import annotations

import argparse
import gc
import json
import math
import os
import re
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
    replace_linear_modules,
    set_hf_mirror,
)
from native_fp4.operators import NunchakuFP4GemmOp


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose Llama down_proj FP4-sensitive layers.")
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
    parser.add_argument("--operator-seq-len", type=int, default=256)
    parser.add_argument("--operator-offset", type=int, default=0)
    parser.add_argument("--ppl-seq-len", type=int, default=256)
    parser.add_argument("--ppl-num-seqs", type=int, default=8)
    parser.add_argument("--ppl-offset", type=int, default=8192)
    parser.add_argument("--topk", type=int, default=6)
    parser.add_argument("--hf-endpoint", type=str, default="https://hf-mirror.com")
    parser.add_argument("--results-dir", type=str, default="results")
    return parser.parse_args()


def parse_layer_index(name: str) -> int:
    match = re.match(r"^model\.layers\.(\d+)\.mlp\.down_proj$", name)
    if match is None:
        raise ValueError(f"Unexpected module name: {name}")
    return int(match.group(1))


def cleanup_model(model: nn.Module | None) -> None:
    if model is not None:
        del model
    gc.collect()
    torch.cuda.empty_cache()


def capture_downproj_inputs(model: nn.Module, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> dict[str, torch.Tensor]:
    captured: dict[str, torch.Tensor] = {}
    hooks = []

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and name.startswith("model.layers.") and name.endswith("mlp.down_proj"):
            def _hook(mod, args, full_name=name):
                captured[full_name] = args[0].detach().cpu()

            hooks.append(module.register_forward_pre_hook(_hook))

    with torch.inference_mode():
        _ = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).logits

    for hook in hooks:
        hook.remove()
    return captured


def compute_operator_metrics(model: nn.Module, captured_inputs: dict[str, torch.Tensor], dtype: torch.dtype) -> list[dict]:
    records: list[dict] = []

    for name, module in model.named_modules():
        if not (isinstance(module, nn.Linear) and name in captured_inputs):
            continue

        x = captured_inputs[name].to(device="cuda", dtype=dtype)
        weight = module.weight.detach().to(device="cuda", dtype=dtype).contiguous()
        bias = None if module.bias is None else module.bias.detach().to(device="cuda", dtype=dtype).contiguous()

        op = NunchakuFP4GemmOp(weight=weight, bias=bias)
        with torch.inference_mode():
            y_ref = module(x).float()
            y_fp4 = op(x).float()

        diff = y_fp4 - y_ref
        ref_norm = y_ref.norm().item()
        diff_norm = diff.norm().item()
        record = {
            "module": name,
            "layer_idx": parse_layer_index(name),
            "input_shape": list(x.shape),
            "weight_shape": list(weight.shape),
            "mae": diff.abs().mean().item(),
            "max_abs": diff.abs().max().item(),
            "rel_l2": diff_norm / max(ref_norm, 1e-12),
            "ref_rms": y_ref.pow(2).mean().sqrt().item(),
            "diff_rms": diff.pow(2).mean().sqrt().item(),
        }
        records.append(record)

        del op, x, weight, bias, y_ref, y_fp4, diff
        torch.cuda.empty_cache()

    records.sort(key=lambda item: item["layer_idx"])
    return records


def compute_single_layer_ppl(
    model_dir: str,
    dtype: torch.dtype,
    token_stream: torch.Tensor,
    seq_len: int,
    num_seqs: int,
    offset_tokens: int,
    layer_idx: int,
) -> dict:
    model = load_model(model_dir, dtype=dtype)
    replaced_modules, replaced_params = replace_linear_modules(
        module=model,
        variant="fp4",
        dtype=dtype,
        rank=32,
        linear_prefix="model.layers.",
        include_lm_head=False,
        layer_start=layer_idx,
        layer_end=layer_idx + 1,
        name_substrings=["mlp.down_proj"],
        factor_mode="svd_lowrank",
        svd_lowrank_oversample=8,
        svd_lowrank_niter=2,
    )
    nll_sum, total_tokens = compute_perplexity_sum(
        model=model,
        token_stream=token_stream,
        seq_len=seq_len,
        num_seqs=num_seqs,
        offset_tokens=offset_tokens,
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
    input_ids = token_stream[args.operator_offset : args.operator_offset + args.operator_seq_len].unsqueeze(0).to("cuda")
    attention_mask = torch.ones_like(input_ids)
    captured_inputs = capture_downproj_inputs(baseline_model, input_ids=input_ids, attention_mask=attention_mask)
    operator_records = compute_operator_metrics(baseline_model, captured_inputs=captured_inputs, dtype=dtype)

    baseline_nll_sum, baseline_total_tokens = compute_perplexity_sum(
        model=baseline_model,
        token_stream=token_stream,
        seq_len=args.ppl_seq_len,
        num_seqs=args.ppl_num_seqs,
        offset_tokens=args.ppl_offset,
    )
    baseline_ppl = math.exp(baseline_nll_sum / max(baseline_total_tokens, 1))
    cleanup_model(baseline_model)

    ranked = sorted(operator_records, key=lambda item: item["rel_l2"], reverse=True)
    selected_layers = []
    for record in ranked[: args.topk]:
        selected_layers.append(record["layer_idx"])
    for control_layer in (0, len(operator_records) - 1):
        if control_layer not in selected_layers:
            selected_layers.append(control_layer)
    selected_layers = sorted(set(selected_layers))

    ppl_records = []
    for layer_idx in selected_layers:
        ppl_record = compute_single_layer_ppl(
            model_dir=model_dir,
            dtype=dtype,
            token_stream=token_stream,
            seq_len=args.ppl_seq_len,
            num_seqs=args.ppl_num_seqs,
            offset_tokens=args.ppl_offset,
            layer_idx=layer_idx,
        )
        ppl_record["perplexity_ratio_vs_fp16"] = ppl_record["perplexity"] / baseline_ppl
        ppl_records.append(ppl_record)

    summary = {
        "most_sensitive_by_rel_l2": ranked[: min(args.topk, len(ranked))],
        "least_sensitive_by_rel_l2": ranked[-min(3, len(ranked)) :],
        "highest_ppl_layers": sorted(ppl_records, key=lambda item: item["perplexity"], reverse=True)[: min(5, len(ppl_records))],
    }

    result = {
        "model_id": args.model_id,
        "model_dir": model_dir,
        "dtype": args.dtype,
        "dataset": {
            "name": args.dataset_name,
            "config": args.dataset_config,
            "split": args.dataset_split,
        },
        "operator_probe": {
            "seq_len": args.operator_seq_len,
            "offset": args.operator_offset,
            "num_layers": len(operator_records),
        },
        "ppl_probe": {
            "seq_len": args.ppl_seq_len,
            "num_seqs": args.ppl_num_seqs,
            "offset": args.ppl_offset,
        },
        "baseline_fp16": {
            "perplexity": baseline_ppl,
            "nll_sum": baseline_nll_sum,
            "total_tokens": baseline_total_tokens,
        },
        "down_proj_operator_records": operator_records,
        "down_proj_single_layer_ppl": ppl_records,
        "summary": summary,
    }

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(args.results_dir, f"llama_downproj_sensitivity_{timestamp}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(out_path)


if __name__ == "__main__":
    main()
