from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any

import torch

BENCHMARK_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BENCHMARK_DIR)
if BENCHMARK_DIR not in sys.path:
    sys.path.insert(0, BENCHMARK_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from benchmark_hf_llama_fp4_lora_finetuning import (  # noqa: E402
    DEFAULT_EXCLUDE_MODULES,
    DEFAULT_MODEL_DIR,
    DEFAULT_MODEL_ID,
    DEFAULT_FP4_LORA_TARGET_MODULES,
    FP4_LORA_TARGET_POLICY_MODULES,
    INCLUDE_REUSE_VARIANTS,
    REUSE_DY_UP_VARIANTS,
    _is_summary_row_dominated,
    _summary_accuracy_metric,
    _summary_row,
    build_batch_from_stream,
    dtype_from_name,
    effective_exclude_modules,
    ensure_model_downloaded,
    fp4_lora_target_modules_for_policy,
    load_tokenizer,
    load_wikitext_token_stream,
    run_dense_lora_variant,
    run_fp4_variant,
)


FP4_VARIANTS = (
    "fp4_accuracy",
    "fp4_balanced",
    "fp4_throughput",
    "fp4_memory_saving",
    "fp4_memory_saving_dequant",
)
POLICIES = (
    "dense_lora",
    "fp4_base",
    "fp4_task_rank_bump",
    "fp4_residual_only",
    "fp4_residual_plus_task",
    "fp4_keep_dense",
)
OUTLIER_POLICIES = tuple(policy for policy in POLICIES if policy not in {"dense_lora", "fp4_base"})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a same-batch HF/LLaMA FP4 LoRA outlier-policy sweep. "
            "This wraps benchmark_hf_llama_fp4_lora_finetuning.py so each policy uses the same replacement path."
        )
    )
    parser.add_argument("--model-id", type=str, default=DEFAULT_MODEL_ID)
    parser.add_argument("--model-dir", type=str, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--hf-endpoint", type=str, default="https://hf-mirror.com")
    parser.add_argument("--dtype", choices=["fp16", "bf16"], default="bf16")
    parser.add_argument("--lowrank-dtype", choices=["fp16", "bf16"], default="bf16")
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--lora-alpha", type=float, default=None)
    parser.add_argument("--init", choices=["zero", "gaussian", "residual_svd", "pissa"], default="zero")
    parser.add_argument("--fp4-variant", choices=FP4_VARIANTS, default="fp4_balanced")
    parser.add_argument("--policies", nargs="+", choices=POLICIES, default=list(POLICIES))
    parser.add_argument("--target-policy", choices=tuple(FP4_LORA_TARGET_POLICY_MODULES), default=None)
    parser.add_argument("--target-modules", nargs="+", default=list(DEFAULT_FP4_LORA_TARGET_MODULES))
    parser.add_argument("--exclude-modules", nargs="+", default=list(DEFAULT_EXCLUDE_MODULES))
    parser.add_argument("--linear-prefix", type=str, default="model.layers.")
    parser.add_argument("--include-lm-head", action="store_true")
    parser.add_argument("--replace-layer-start", type=int, default=None)
    parser.add_argument("--replace-layer-end", type=int, default=None)
    parser.add_argument("--replace-name-substrings", nargs="*", default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--dataset-name", type=str, default="wikitext")
    parser.add_argument("--dataset-config", type=str, default="wikitext-2-raw-v1")
    parser.add_argument("--dataset-split", type=str, default="test")
    parser.add_argument("--dataset-max-docs", type=int, default=0)
    parser.add_argument("--dataset-offset-tokens", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--adam-eps", type=float, default=1e-4)
    parser.add_argument("--lora-weight-decay", type=float, default=0.0)
    parser.add_argument("--bias-weight-decay", type=float, default=0.0)
    parser.add_argument("--train-bias", action="store_true")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--prime-steps", type=int, default=1)
    parser.add_argument("--no-frozen-residual", action="store_true")
    parser.add_argument("--frozen-residual-rank", type=int, default=None)
    parser.add_argument("--residual-svd-method", choices=["full_svd", "svd_lowrank"], default=None)
    parser.add_argument("--residual-svd-lowrank-oversample", type=int, default=8)
    parser.add_argument("--residual-svd-lowrank-niter", type=int, default=2)
    parser.add_argument("--no-cache-lora-act", action="store_true")
    parser.add_argument("--activation-checkpoint", action="store_true")
    parser.add_argument("--model-gradient-checkpointing", action="store_true")
    parser.add_argument("--backward-weight-policy", choices=["repack", "cache"], default="repack")
    parser.add_argument("--reuse-fused-dy-up-for-d-lora-down", action="store_true")
    parser.add_argument("--include-reuse-policies", action="store_true")
    parser.add_argument("--overlap-lora-grad-min-rows", type=int, default=4096)
    parser.add_argument("--fp4-activation-cache-min-rows", type=int, default=0)
    parser.add_argument("--fp4-activation-cache-d-lora-down-backend", choices=["fused", "dequant_gemm"], default="fused")
    parser.add_argument("--no-zero-lora-up-fast-path", action="store_true")
    parser.add_argument("--outlier-report", type=str, default=None)
    parser.add_argument("--outlier-rank-field", type=str, default="suggested_rank")
    parser.add_argument("--outlier-rank-multiple", type=int, default=16)
    parser.add_argument("--outlier-min-rank", type=int, default=None)
    parser.add_argument("--outlier-max-rank", type=int, default=None)
    parser.add_argument("--outlier-keep-dense-candidates-key", type=str, default="keep_dense_candidates")
    parser.add_argument("--outlier-frozen-residual-rank-field", type=str, default=None)
    parser.add_argument("--outlier-disable-fuse-frozen-residual-dx", action="store_true")
    parser.add_argument("--sensitivity-report", type=str, default=None)
    parser.add_argument("--sensitivity-ratio-field", type=str, default="perplexity_ratio_vs_fp16")
    parser.add_argument("--sensitivity-rank-bump-ratio", type=float, default=1.05)
    parser.add_argument("--sensitivity-exclude-ratio", type=float, default=None)
    parser.add_argument("--sensitivity-rank-scale", type=float, default=2.0)
    parser.add_argument("--sensitivity-rank-multiple", type=int, default=16)
    parser.add_argument("--sensitivity-min-rank", type=int, default=None)
    parser.add_argument("--sensitivity-max-rank", type=int, default=None)
    parser.add_argument("--sensitivity-disable-fuse-frozen-residual-dx", action="store_true")
    parser.add_argument("--attn-implementation", type=str, default=None)
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    requested_policies = set(args.policies)
    if requested_policies & set(OUTLIER_POLICIES):
        if not args.outlier_report:
            parser.error(
                "--outlier-report is required for policies: "
                + ", ".join(policy for policy in OUTLIER_POLICIES if policy in requested_policies)
            )
        if not os.path.exists(args.outlier_report):
            parser.error(f"--outlier-report does not exist: {args.outlier_report}")
    return args


def policy_args(base_args: argparse.Namespace, policy: str) -> argparse.Namespace:
    args = argparse.Namespace(**vars(base_args))
    args.variants = [base_args.fp4_variant]
    args.outlier_keep_dense = False
    args.outlier_no_task_rank_bump = False
    args.outlier_bump_frozen_residual = False

    if policy == "fp4_base":
        args.outlier_report = None
        args.outlier_no_task_rank_bump = True
    elif policy == "fp4_task_rank_bump":
        args.outlier_report = base_args.outlier_report
    elif policy == "fp4_residual_only":
        args.outlier_report = base_args.outlier_report
        args.outlier_no_task_rank_bump = True
        args.outlier_bump_frozen_residual = True
    elif policy == "fp4_residual_plus_task":
        args.outlier_report = base_args.outlier_report
        args.outlier_bump_frozen_residual = True
    elif policy == "fp4_keep_dense":
        args.outlier_report = base_args.outlier_report
        args.outlier_keep_dense = True
        args.outlier_no_task_rank_bump = True
    else:
        raise ValueError(f"Unsupported FP4 policy: {policy}")
    return args


def policy_record_plan(
    policies: list[str],
    fp4_variant: str,
    *,
    include_reuse_policies: bool,
) -> list[tuple[str, str, str | None]]:
    plan: list[tuple[str, str, str | None]] = []
    reuse_variant = INCLUDE_REUSE_VARIANTS.get(fp4_variant) if include_reuse_policies else None
    for policy in policies:
        if policy == "dense_lora":
            plan.append(("dense_lora", policy, None))
            continue
        plan.append((policy, policy, fp4_variant))
        if reuse_variant is not None:
            plan.append((f"{policy}_reuse_dy_up", policy, reuse_variant))
    return plan


def policy_metadata(args: argparse.Namespace, policy: str, fp4_variant: str | None = None) -> dict[str, Any]:
    if policy == "dense_lora":
        return {"dense_lora_baseline": True}
    p_args = policy_args(args, policy)
    variant = fp4_variant or args.fp4_variant
    return {
        "fp4_variant": variant,
        "base_fp4_variant": REUSE_DY_UP_VARIANTS.get(variant, variant),
        "reuse_fused_dy_up_for_d_lora_down": bool(
            p_args.reuse_fused_dy_up_for_d_lora_down or variant in REUSE_DY_UP_VARIANTS
        ),
        "init": p_args.init,
        "outlier_report": p_args.outlier_report,
        "outlier_bump_task_rank": not p_args.outlier_no_task_rank_bump,
        "outlier_bump_frozen_residual": p_args.outlier_bump_frozen_residual,
        "outlier_keep_dense": p_args.outlier_keep_dense,
        "outlier_rank_field": p_args.outlier_rank_field,
        "outlier_rank_multiple": p_args.outlier_rank_multiple,
        "outlier_min_rank": p_args.outlier_min_rank,
        "outlier_max_rank": p_args.outlier_max_rank,
        "outlier_frozen_residual_rank_field": p_args.outlier_frozen_residual_rank_field,
        "outlier_disable_fuse_frozen_residual_dx": p_args.outlier_disable_fuse_frozen_residual_dx,
        "sensitivity_disable_fuse_frozen_residual_dx": p_args.sensitivity_disable_fuse_frozen_residual_dx,
        "fp4_activation_cache_min_rows": p_args.fp4_activation_cache_min_rows,
    }


def build_policy_summary(records: dict[str, dict[str, Any]]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for record_name, record in records.items():
        row = _summary_row(record_name, record)
        row["policy"] = record.get("policy")
        row["policy_record"] = record.get("policy_record", record_name)
        row["outlier_policy"] = record.get("outlier_policy")
        rows.append(row)

    dense_rows = [row for row in rows if row["record"] == "dense_lora" or row["variant"] == "dense_lora"]
    fp4_rows = [row for row in rows if row not in dense_rows]
    pareto_rows = [
        row
        for row in fp4_rows
        if not any(_is_summary_row_dominated(row, other) for other in fp4_rows if other["record"] != row["record"])
    ]
    return {
        "dense_lora": dense_rows[0] if dense_rows else None,
        "records_by_train_step_ms": sorted(fp4_rows, key=lambda row: row["train_step_ms"]),
        "records_by_peak_delta_bytes": sorted(fp4_rows, key=lambda row: row["peak_delta_bytes"]),
        "records_by_accuracy_metric": sorted(
            fp4_rows,
            key=lambda row: (_summary_accuracy_metric(row), row["train_step_ms"]),
        ),
        "pareto_frontier": sorted(
            pareto_rows,
            key=lambda row: (row["train_step_ms"], row["peak_delta_bytes"], _summary_accuracy_metric(row)),
        ),
    }


def train_step_latency(record: dict[str, Any]) -> float:
    return float(record["latency_ms"]["train_step_with_optimizer"])


def peak_delta(record: dict[str, Any]) -> int:
    return int(record["peak_memory_bytes"]["train_step_delta"])


def logits_rel_l2(record: dict[str, Any]) -> float | None:
    error = record.get("initial_logits_vs_dense_lora")
    if not error:
        return None
    return float(error["rel_l2"])


def add_relative_to_base(results: dict[str, Any]) -> None:
    base = results["records"].get("fp4_base")
    if not base:
        return
    base_latency = train_step_latency(base)
    base_peak = peak_delta(base)
    base_rel_l2 = logits_rel_l2(base)
    base_loss = float(base["initial_loss"])

    for policy, record in results["records"].items():
        if policy == "fp4_base" or policy == "dense_lora":
            continue
        latency = train_step_latency(record)
        peak = peak_delta(record)
        rel_l2 = logits_rel_l2(record)
        record["relative_to_fp4_base"] = {
            "speedup_vs_fp4_base": base_latency / latency,
            "latency_ratio_vs_fp4_base": latency / base_latency,
            "peak_delta_ratio_vs_fp4_base": None if base_peak == 0 else peak / base_peak,
            "initial_loss_delta_vs_fp4_base": float(record["initial_loss"]) - base_loss,
            "logits_rel_l2_ratio_vs_fp4_base": None
            if base_rel_l2 in (None, 0.0) or rel_l2 is None
            else rel_l2 / base_rel_l2,
        }


def main() -> None:
    args = parse_args()
    if args.target_policy is not None:
        args.target_modules = list(fp4_lora_target_modules_for_policy(args.target_policy))
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.iters <= 0:
        raise ValueError("--iters must be positive")
    if args.warmup < 0 or args.prime_steps < 0:
        raise ValueError("--warmup and --prime-steps must be non-negative")

    policies = list(dict.fromkeys(args.policies))
    if args.include_reuse_policies and args.reuse_fused_dy_up_for_d_lora_down:
        raise ValueError("--include-reuse-policies should not be combined with global --reuse-fused-dy-up-for-d-lora-down")
    if args.include_reuse_policies and args.fp4_variant not in INCLUDE_REUSE_VARIANTS:
        raise ValueError("--include-reuse-policies only supports fp4_balanced or fp4_throughput")
    torch.manual_seed(args.seed)
    os.makedirs(args.results_dir, exist_ok=True)

    from benchmark_hf_llama_fp4_lora_finetuning import set_hf_mirror  # noqa: E402

    set_hf_mirror(args.hf_endpoint)
    model_dir = ensure_model_downloaded(args.model_id, args.model_dir)
    dtype = dtype_from_name(args.dtype)
    lowrank_dtype = dtype_from_name(args.lowrank_dtype)
    if args.include_reuse_policies and dtype != lowrank_dtype:
        raise ValueError("--include-reuse-policies requires --dtype to match --lowrank-dtype")

    tokenizer = load_tokenizer(model_dir)
    token_stream = load_wikitext_token_stream(
        tokenizer=tokenizer,
        dataset_name=args.dataset_name,
        dataset_config=args.dataset_config,
        dataset_split=args.dataset_split,
        dataset_max_docs=args.dataset_max_docs,
    )
    batch = build_batch_from_stream(
        token_stream,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        offset_tokens=args.dataset_offset_tokens,
    )
    record_plan = policy_record_plan(
        policies,
        args.fp4_variant,
        include_reuse_policies=args.include_reuse_policies,
    )

    results: dict[str, Any] = {
        "experiment": "hf_causal_lm_fp4_lora_outlier_policy_sweep",
        "model_id": args.model_id,
        "model_dir": model_dir,
        "dtype": args.dtype,
        "lowrank_dtype": args.lowrank_dtype,
        "rank": args.rank,
        "lora_alpha": args.lora_alpha,
        "fp4_variant": args.fp4_variant,
        "policies_requested": policies,
        "policy_records_requested": [
            {"record": record_name, "policy": policy, "fp4_variant": variant}
            for record_name, policy, variant in record_plan
        ],
        "policy_metadata": {
            record_name: policy_metadata(args, policy, variant)
            for record_name, policy, variant in record_plan
        },
        "selection": {
            "target_policy": args.target_policy,
            "target_modules": args.target_modules,
            "exclude_modules": list(effective_exclude_modules(args)),
            "linear_prefix": args.linear_prefix,
            "include_lm_head": args.include_lm_head,
            "replace_layer_start": args.replace_layer_start,
            "replace_layer_end": args.replace_layer_end,
            "replace_name_substrings": args.replace_name_substrings,
        },
        "train": {
            "batch_size": args.batch_size,
            "seq_len": args.seq_len,
            "lr": args.lr,
            "adam_eps": args.adam_eps,
            "lora_weight_decay": args.lora_weight_decay,
            "bias_weight_decay": args.bias_weight_decay,
            "train_bias": args.train_bias,
            "warmup": args.warmup,
            "iters": args.iters,
            "prime_steps": args.prime_steps,
            "model_gradient_checkpointing": args.model_gradient_checkpointing,
        },
        "dataset": {
            "name": args.dataset_name,
            "config": args.dataset_config,
            "split": args.dataset_split,
            "max_docs": args.dataset_max_docs,
            "offset_tokens": args.dataset_offset_tokens,
            "token_stream_tokens": int(token_stream.numel()),
        },
        "fp4_options": {
            "use_frozen_residual": not args.no_frozen_residual,
            "init": args.init,
            "frozen_residual_rank": args.frozen_residual_rank,
            "residual_svd_method": args.residual_svd_method,
            "residual_svd_lowrank_oversample": args.residual_svd_lowrank_oversample,
            "residual_svd_lowrank_niter": args.residual_svd_lowrank_niter,
            "cache_lora_act": not args.no_cache_lora_act,
            "activation_checkpoint": args.activation_checkpoint,
            "backward_weight_policy": args.backward_weight_policy,
            "reuse_fused_dy_up_for_d_lora_down": args.reuse_fused_dy_up_for_d_lora_down,
            "include_reuse_policies": args.include_reuse_policies,
            "overlap_lora_grad_min_rows": args.overlap_lora_grad_min_rows,
            "fp4_activation_cache_min_rows": args.fp4_activation_cache_min_rows,
            "fp4_activation_cache_d_lora_down_backend": args.fp4_activation_cache_d_lora_down_backend,
            "zero_lora_up_fast_path": not args.no_zero_lora_up_fast_path,
            "sensitivity_report": args.sensitivity_report,
            "sensitivity_ratio_field": args.sensitivity_ratio_field,
            "sensitivity_rank_bump_ratio": args.sensitivity_rank_bump_ratio,
            "sensitivity_exclude_ratio": args.sensitivity_exclude_ratio,
            "sensitivity_rank_scale": args.sensitivity_rank_scale,
            "sensitivity_rank_multiple": args.sensitivity_rank_multiple,
            "sensitivity_min_rank": args.sensitivity_min_rank,
            "sensitivity_max_rank": args.sensitivity_max_rank,
            "sensitivity_disable_fuse_frozen_residual_dx": args.sensitivity_disable_fuse_frozen_residual_dx,
        },
        "records": {},
        "all_passed": False,
    }

    dense_initial_logits: torch.Tensor | None = None
    dense_latency_ms: float | None = None
    dense_peak_delta: int | None = None
    if "dense_lora" in policies:
        dense_record, dense_initial_logits = run_dense_lora_variant(
            args,
            model_dir=model_dir,
            dtype=dtype,
            lowrank_dtype=lowrank_dtype,
            batch=batch,
        )
        dense_record["policy"] = "dense_lora"
        results["records"]["dense_lora"] = dense_record
        dense_latency_ms = dense_record["latency_ms"]["train_step_with_optimizer"]
        dense_peak_delta = dense_record["peak_memory_bytes"]["train_step_delta"]

    for record_name, policy, variant in record_plan:
        if policy == "dense_lora":
            continue
        if variant is None:
            raise RuntimeError(f"Missing FP4 variant for policy record {record_name}")
        p_args = policy_args(args, policy)
        record = run_fp4_variant(
            p_args,
            variant=variant,
            model_dir=model_dir,
            dtype=dtype,
            lowrank_dtype=lowrank_dtype,
            batch=batch,
            dense_initial_logits=dense_initial_logits,
            dense_latency_ms=dense_latency_ms,
            dense_peak_delta=dense_peak_delta,
        )
        record["policy"] = policy
        record["policy_record"] = record_name
        record["fp4_variant"] = variant
        record["outlier_policy"] = policy_metadata(args, policy, variant)
        results["records"][record_name] = record

    add_relative_to_base(results)
    results["all_passed"] = bool(all(record["all_passed"] for record in results["records"].values()))
    results["policy_summary"] = build_policy_summary(results["records"])

    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(args.results_dir, f"hf_llama_fp4_lora_policy_sweep_{stamp}.json")
    latest_path = os.path.join(args.results_dir, "latest_hf_llama_fp4_lora_policy_sweep.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    with open(latest_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(json.dumps(results, indent=2))
    print(f"Saved benchmark to: {out_path}")
    print(f"Saved latest to: {latest_path}")
    if not results["all_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
