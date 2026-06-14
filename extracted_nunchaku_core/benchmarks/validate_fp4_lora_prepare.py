from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import replace
from datetime import datetime
from typing import Any

import torch

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from native_fp4 import (  # noqa: E402
    NunchakuFP4LoRALinear,
    fp4_lora_finetune_config,
    fp4_lora_state_dict,
    iter_fp4_lora_modules,
    iter_fp4_lora_named_parameters,
    prepare_fp4_lora_finetuning,
)


class TinyBlock(torch.nn.Module):
    def __init__(self, hidden: int, dtype: torch.dtype):
        super().__init__()
        self.q_proj = torch.nn.Linear(hidden, hidden, bias=False, dtype=dtype)
        self.k_proj = torch.nn.Linear(hidden, hidden, bias=False, dtype=dtype)
        self.down_proj = torch.nn.Linear(hidden, hidden, bias=True, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(torch.nn.functional.silu(self.q_proj(x)) + self.k_proj(x))


class TinyModel(torch.nn.Module):
    def __init__(self, hidden: int, dtype: torch.dtype):
        super().__init__()
        self.layers = torch.nn.ModuleList([TinyBlock(hidden, dtype), TinyBlock(hidden, dtype)])
        self.lm_head = torch.nn.Linear(hidden, hidden, bias=False, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return self.lm_head(x)


def dtype_from_name(name: str) -> torch.dtype:
    if name == "fp16":
        return torch.float16
    if name == "bf16":
        return torch.bfloat16
    raise ValueError(f"Unsupported dtype: {name}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Validate high-level FP4 LoRA fine-tuning preparation API.")
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--rank", type=int, default=32)
    p.add_argument("--override-rank", type=int, default=64)
    p.add_argument("--auto-rank", type=int, default=48)
    p.add_argument("--dtype", choices=["fp16", "bf16"], default="bf16")
    p.add_argument("--lowrank-dtype", choices=["fp16", "bf16"], default="bf16")
    p.add_argument("--mode", choices=["accuracy", "balanced", "throughput", "memory_saving"], default="balanced")
    p.add_argument("--no-frozen-residual", action="store_true")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--adam-eps", type=float, default=1e-4)
    p.add_argument("--backward-weight-policy", choices=["repack", "cache"], default="repack")
    p.add_argument("--reuse-fused-dy-up-for-d-lora-down", action="store_true")
    p.add_argument("--fp4-activation-cache-min-rows", type=int, default=0)
    p.add_argument("--fp4-activation-cache-d-lora-down-backend", choices=["fused", "dequant_gemm"], default="fused")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--results-dir", type=str, default="results")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.manual_seed(args.seed)

    dtype = dtype_from_name(args.dtype)
    lowrank_dtype = dtype_from_name(args.lowrank_dtype)
    model = TinyModel(args.hidden, dtype).cuda()
    base_cfg = fp4_lora_finetune_config(
        mode=args.mode,
        rank=args.rank,
        dtype=dtype,
        lowrank_dtype=lowrank_dtype,
        use_frozen_residual=not args.no_frozen_residual,
        backward_weight_policy=args.backward_weight_policy,
        reuse_fused_dy_up_for_d_lora_down=args.reuse_fused_dy_up_for_d_lora_down,
        fp4_activation_cache_min_rows=args.fp4_activation_cache_min_rows,
        fp4_activation_cache_d_lora_down_backend=args.fp4_activation_cache_d_lora_down_backend,
    )
    manual_override = replace(base_cfg, rank=args.override_rank)
    outlier_report = {
        "summary": {
            "rank_bump_candidates": [
                {
                    "module": "layers.1.down_proj",
                    "suggested_rank": args.auto_rank,
                }
            ],
            "keep_dense_candidates": [
                {
                    "module": "layers.0.q_proj",
                },
                {
                    "module": "layers.1.down_proj",
                }
            ]
        }
    }
    sensitivity_report = {
        "module_records": [
            {
                "module": "layers.0.q_proj",
                "kind": "q_proj",
                "perplexity_ratio_vs_fp16": 1.10,
            },
            {
                "module": "model.layers.0.down_proj",
                "kind": "down_proj",
                "perplexity_ratio_vs_fp16": 1.30,
            },
            {
                "module": "layers.1.down_proj",
                "kind": "down_proj",
                "perplexity_ratio_vs_fp16": 1.40,
            },
        ]
    }

    result = prepare_fp4_lora_finetuning(
        model,
        mode=args.mode,
        rank=args.rank,
        dtype=dtype,
        lowrank_dtype=lowrank_dtype,
        use_frozen_residual=not args.no_frozen_residual,
        backward_weight_policy=args.backward_weight_policy,
        reuse_fused_dy_up_for_d_lora_down=args.reuse_fused_dy_up_for_d_lora_down,
        fp4_activation_cache_min_rows=args.fp4_activation_cache_min_rows,
        fp4_activation_cache_d_lora_down_backend=args.fp4_activation_cache_d_lora_down_backend,
        target_modules=("q_proj", "down_proj"),
        exclude_modules=("lm_head",),
        config_overrides={"layers.1.down_proj": manual_override},
        outlier_report=outlier_report,
        sensitivity_report=sensitivity_report,
        sensitivity_rank_bump_ratio=1.05,
        sensitivity_exclude_ratio=1.25,
        sensitivity_rank_scale=2.0,
        lr=args.lr,
    )
    outlier_exclude_model = TinyModel(args.hidden, dtype).cuda()
    outlier_exclude_result = prepare_fp4_lora_finetuning(
        outlier_exclude_model,
        mode=args.mode,
        rank=args.rank,
        dtype=dtype,
        lowrank_dtype=lowrank_dtype,
        use_frozen_residual=not args.no_frozen_residual,
        backward_weight_policy=args.backward_weight_policy,
        reuse_fused_dy_up_for_d_lora_down=args.reuse_fused_dy_up_for_d_lora_down,
        fp4_activation_cache_min_rows=args.fp4_activation_cache_min_rows,
        fp4_activation_cache_d_lora_down_backend=args.fp4_activation_cache_d_lora_down_backend,
        target_modules=("q_proj", "down_proj"),
        exclude_modules=("lm_head",),
        config_overrides={"layers.1.down_proj": manual_override},
        outlier_report=outlier_report,
        outlier_exclude_keep_dense=True,
        lr=args.lr,
    )
    outlier_exclude_modules = dict(iter_fp4_lora_modules(outlier_exclude_result.model))
    expected_outlier_exclude_replaced = {
        "layers.0.down_proj",
        "layers.1.q_proj",
        "layers.1.down_proj",
    }
    outlier_exclude_no_manual_model = TinyModel(args.hidden, dtype).cuda()
    outlier_exclude_no_manual_result = prepare_fp4_lora_finetuning(
        outlier_exclude_no_manual_model,
        mode=args.mode,
        rank=args.rank,
        dtype=dtype,
        lowrank_dtype=lowrank_dtype,
        use_frozen_residual=not args.no_frozen_residual,
        backward_weight_policy=args.backward_weight_policy,
        reuse_fused_dy_up_for_d_lora_down=args.reuse_fused_dy_up_for_d_lora_down,
        fp4_activation_cache_min_rows=args.fp4_activation_cache_min_rows,
        fp4_activation_cache_d_lora_down_backend=args.fp4_activation_cache_d_lora_down_backend,
        target_modules=("q_proj", "down_proj"),
        exclude_modules=("lm_head",),
        outlier_report=outlier_report,
        outlier_exclude_keep_dense=True,
        lr=args.lr,
    )
    outlier_exclude_no_manual_modules = dict(iter_fp4_lora_modules(outlier_exclude_no_manual_result.model))
    expected_outlier_exclude_no_manual_replaced = {
        "layers.0.down_proj",
        "layers.1.q_proj",
    }
    outlier_residual_model = TinyModel(args.hidden, dtype).cuda()
    outlier_residual_result = prepare_fp4_lora_finetuning(
        outlier_residual_model,
        mode=args.mode,
        rank=args.rank,
        dtype=dtype,
        lowrank_dtype=lowrank_dtype,
        use_frozen_residual=not args.no_frozen_residual,
        backward_weight_policy=args.backward_weight_policy,
        reuse_fused_dy_up_for_d_lora_down=args.reuse_fused_dy_up_for_d_lora_down,
        fp4_activation_cache_min_rows=args.fp4_activation_cache_min_rows,
        fp4_activation_cache_d_lora_down_backend=args.fp4_activation_cache_d_lora_down_backend,
        target_modules=("q_proj", "down_proj"),
        exclude_modules=("lm_head",),
        outlier_report=outlier_report,
        outlier_bump_frozen_residual=True,
        lr=args.lr,
    )
    outlier_residual_modules = dict(iter_fp4_lora_modules(outlier_residual_result.model))
    expected_outlier_residual_rank = max(
        base_cfg.frozen_residual_rank,
        ((args.auto_rank + 15) // 16) * 16,
    )
    outlier_residual_only_model = TinyModel(args.hidden, dtype).cuda()
    outlier_residual_only_result = prepare_fp4_lora_finetuning(
        outlier_residual_only_model,
        mode=args.mode,
        rank=args.rank,
        dtype=dtype,
        lowrank_dtype=lowrank_dtype,
        use_frozen_residual=not args.no_frozen_residual,
        backward_weight_policy=args.backward_weight_policy,
        reuse_fused_dy_up_for_d_lora_down=args.reuse_fused_dy_up_for_d_lora_down,
        fp4_activation_cache_min_rows=args.fp4_activation_cache_min_rows,
        fp4_activation_cache_d_lora_down_backend=args.fp4_activation_cache_d_lora_down_backend,
        target_modules=("q_proj", "down_proj"),
        exclude_modules=("lm_head",),
        outlier_report=outlier_report,
        outlier_bump_task_rank=False,
        outlier_bump_frozen_residual=True,
        lr=args.lr,
    )
    outlier_residual_only_modules = dict(iter_fp4_lora_modules(outlier_residual_only_result.model))
    prepared = result.model
    fp4_modules = dict(iter_fp4_lora_modules(prepared))
    expected_replaced = {
        "layers.0.q_proj",
        "layers.1.q_proj",
        "layers.1.down_proj",
    }
    expected_excludes = ("lm_head", "model.layers.0.down_proj", "layers.0.down_proj")
    expected_sensitivity_rank = max(
        args.rank,
        ((int(math.ceil(args.rank * 2.0)) + 15) // 16) * 16,
    )
    expected_auto_fuse_frozen_residual_dx = base_cfg.fuse_frozen_residual_dx

    x = torch.randn(args.batch, args.hidden, device="cuda", dtype=dtype, requires_grad=True)
    y = prepared(x)
    loss = y.float().square().mean()
    loss.backward()

    named_lora_params = dict(iter_fp4_lora_named_parameters(prepared, train_bias=base_cfg.train_bias))
    optimizer = torch.optim.AdamW(result.optimizer_param_groups, eps=args.adam_eps)
    hook = result.register_cache_refresh_hook(optimizer)
    optimizer.step()
    hook_refresh_count = hook.last_refresh_count
    hook_forward_refresh_count = hook.last_fused_lora_forward_refresh_count
    hook_dx_refresh_count = hook.last_fused_lora_dx_refresh_count
    hook_backward_weight_cache_count = hook.last_backward_weight_cache_count
    hook.remove()

    adapter_state = fp4_lora_state_dict(prepared)
    optimizer_param_ids = {id(param) for group in result.optimizer_param_groups for param in group["params"]}
    expected_param_ids = {id(param) for param in named_lora_params.values()}
    caches_current = all(
        child._cached_lora_down_version == child.lora_down._version
        and child._cached_lora_up_version == child.lora_up._version
        for _, child in fp4_modules.items()
        if child.fuse_lora_dx and child.cache_fused_lora_dx
    )
    expected_cache_count = (
        len(expected_replaced) if base_cfg.fuse_lora_dx and base_cfg.cache_fused_lora_dx else 0
    )
    expected_forward_cache_count = sum(
        1
        for name in expected_replaced
        if (
            fp4_modules[name].fuse_lowrank_forward
            and fp4_modules[name].lowrank_dtype == fp4_modules[name].fp4_forward.compute_dtype
        )
    )
    zero_fast_path_initial = bool(base_cfg.init == "zero" and base_cfg.zero_lora_up_fast_path)
    expected_initial_cache_count = 0 if zero_fast_path_initial else expected_cache_count
    expected_initial_forward_cache_count = 0 if zero_fast_path_initial else expected_forward_cache_count
    expected_backward_weight_count = len(expected_replaced) if base_cfg.backward_weight_policy == "cache" else 0
    expected_cache_summary = result.cache_summary
    expected_adapter_keys = {
        f"{name}.{param}" for name in expected_replaced for param in ("lora_down", "lora_up")
    }
    trainable_named = {name for name, param in prepared.named_parameters() if param.requires_grad}
    all_non_lora_frozen = all(
        (name in named_lora_params) == param.requires_grad for name, param in prepared.named_parameters()
    )
    grads_present = all(param.grad is not None for param in named_lora_params.values())
    groups_have_lr = all(group.get("lr") == args.lr for group in result.optimizer_param_groups)
    trainable_param_count = sum(param.numel() for param in named_lora_params.values())

    checks = {
        "result_model_is_input_model": result.model is model,
        "replaced_expected_modules": set(result.replaced_modules) == expected_replaced,
        "all_replaced_are_fp4_lora": all(isinstance(fp4_modules[name], NunchakuFP4LoRALinear) for name in expected_replaced),
        "result_config_backend_matches": (
            result.config.fp4_activation_cache_d_lora_down_backend
            == args.fp4_activation_cache_d_lora_down_backend
        ),
        "result_config_activation_cache_min_rows_matches": (
            result.config.fp4_activation_cache_min_rows == args.fp4_activation_cache_min_rows
        ),
        "result_config_reuse_flag_matches": (
            result.config.reuse_fused_dy_up_for_d_lora_down
            == args.reuse_fused_dy_up_for_d_lora_down
        ),
        "result_config_backward_weight_policy_matches": (
            result.config.backward_weight_policy == args.backward_weight_policy
        ),
        "all_replaced_backend_matches": all(
            fp4_modules[name].fp4_activation_cache_d_lora_down_backend
            == args.fp4_activation_cache_d_lora_down_backend
            for name in expected_replaced
        ),
        "all_replaced_activation_cache_min_rows_matches": all(
            fp4_modules[name].fp4_activation_cache_min_rows == args.fp4_activation_cache_min_rows
            for name in expected_replaced
        ),
        "all_replaced_reuse_flag_matches": all(
            fp4_modules[name].reuse_fused_dy_up_for_d_lora_down
            == args.reuse_fused_dy_up_for_d_lora_down
            for name in expected_replaced
        ),
        "all_replaced_backward_weight_policy_matches": all(
            fp4_modules[name].backward_weight_policy == args.backward_weight_policy for name in expected_replaced
        ),
        "lm_head_not_replaced": not isinstance(prepared.lm_head, NunchakuFP4LoRALinear),
        "sensitivity_exclude_keeps_layer0_down_proj_dense": (
            "layers.0.down_proj" not in fp4_modules
            and isinstance(prepared.layers[0].down_proj, torch.nn.Linear)
        ),
        "sensitivity_rank_bump_applied": (
            fp4_modules["layers.0.q_proj"].requested_rank == expected_sensitivity_rank
        ),
        "sensitivity_rank_bump_preserves_fused_residual_dx": (
            fp4_modules["layers.0.q_proj"].fuse_frozen_residual_dx == expected_auto_fuse_frozen_residual_dx
        ),
        "manual_override_wins_over_outlier_report": (
            fp4_modules["layers.1.down_proj"].requested_rank == args.override_rank
        ),
        "manual_override_wins_over_sensitivity_exclude": (
            "layers.1.down_proj" in fp4_modules and "layers.1.down_proj" not in result.exclude_modules
        ),
        "outlier_keep_dense_default_is_opt_in": "layers.0.q_proj" in fp4_modules,
        "outlier_keep_dense_excludes_candidate": (
            "layers.0.q_proj" not in outlier_exclude_modules
            and isinstance(outlier_exclude_result.model.layers[0].q_proj, torch.nn.Linear)
        ),
        "outlier_keep_dense_manual_override_wins": (
            "layers.1.down_proj" in outlier_exclude_modules
            and "layers.1.down_proj" not in outlier_exclude_result.exclude_modules
        ),
        "outlier_keep_dense_replaced_expected_modules": (
            set(outlier_exclude_result.replaced_modules) == expected_outlier_exclude_replaced
        ),
        "outlier_keep_dense_exclude_modules_recorded": (
            outlier_exclude_result.exclude_modules == ("lm_head", "layers.0.q_proj")
        ),
        "outlier_keep_dense_no_manual_excludes_all_candidates": (
            "layers.0.q_proj" not in outlier_exclude_no_manual_modules
            and "layers.1.down_proj" not in outlier_exclude_no_manual_modules
        ),
        "outlier_keep_dense_no_manual_replaced_expected_modules": (
            set(outlier_exclude_no_manual_result.replaced_modules) == expected_outlier_exclude_no_manual_replaced
        ),
        "outlier_keep_dense_no_manual_exclude_modules_recorded": (
            outlier_exclude_no_manual_result.exclude_modules
            == ("lm_head", "layers.0.q_proj", "layers.1.down_proj")
        ),
        "outlier_bump_frozen_residual_rank_applied": (
            outlier_residual_modules["layers.1.down_proj"].requested_frozen_residual_rank
            == expected_outlier_residual_rank
            and outlier_residual_modules["layers.1.down_proj"].frozen_residual_rank
            == expected_outlier_residual_rank
        ),
        "outlier_bump_frozen_residual_init_applied": (
            outlier_residual_modules["layers.1.down_proj"].frozen_residual_init == "residual_svd"
            and outlier_residual_modules["layers.1.down_proj"].has_frozen_residual
        ),
        "outlier_bump_frozen_residual_preserves_fused_residual_dx": (
            outlier_residual_modules["layers.1.down_proj"].fuse_frozen_residual_dx
            == expected_auto_fuse_frozen_residual_dx
        ),
        "outlier_bump_frozen_residual_does_not_change_non_candidate_rank": (
            outlier_residual_modules["layers.0.q_proj"].requested_frozen_residual_rank
            == base_cfg.frozen_residual_rank
        ),
        "outlier_residual_only_keeps_task_rank": (
            outlier_residual_only_modules["layers.1.down_proj"].requested_rank == args.rank
        ),
        "outlier_residual_only_bumps_frozen_residual_rank": (
            outlier_residual_only_modules["layers.1.down_proj"].requested_frozen_residual_rank
            == expected_outlier_residual_rank
            and outlier_residual_only_modules["layers.1.down_proj"].has_frozen_residual
        ),
        "outlier_residual_only_preserves_fused_residual_dx": (
            outlier_residual_only_modules["layers.1.down_proj"].fuse_frozen_residual_dx
            == expected_auto_fuse_frozen_residual_dx
        ),
        "trainable_names_match_lora_params": set(result.trainable_names) == set(named_lora_params),
        "all_non_lora_frozen": all_non_lora_frozen,
        "optimizer_param_groups_match_lora": optimizer_param_ids == expected_param_ids,
        "optimizer_param_groups_have_lr": groups_have_lr,
        "trainable_param_count_matches": result.trainable_param_count == trainable_param_count,
        "forward_refresh_count_matches": result.refreshed_forward_cache_count == expected_initial_forward_cache_count,
        "refresh_count_matches": result.refreshed_cache_count == expected_initial_cache_count,
        "backward_weight_refresh_count_matches": (
            result.refreshed_backward_weight_count == expected_backward_weight_count
        ),
        "cache_summary_module_count_matches": expected_cache_summary.module_count == len(expected_replaced),
        "cache_summary_fused_lora_forward_count_matches": (
            expected_cache_summary.fused_lora_forward_cache_count == expected_initial_forward_cache_count
        ),
        "cache_summary_fused_lora_dx_count_matches": (
            expected_cache_summary.fused_lora_dx_cache_count == expected_initial_cache_count
        ),
        "cache_summary_backward_weight_count_matches": (
            expected_cache_summary.backward_weight_cache_count == expected_backward_weight_count
        ),
        "cache_summary_total_bytes_consistent": (
            expected_cache_summary.total_cache_bytes
            == expected_cache_summary.fused_lora_forward_cache_bytes
            + expected_cache_summary.fused_lora_dx_cache_bytes
            + expected_cache_summary.backward_weight_cache_bytes
        ),
        "cache_summary_dense_weight_bytes_positive": expected_cache_summary.dense_weight_bytes > 0,
        "backward_weight_cache_state_matches": all(
            (fp4_modules[name].fp4_backward._cached_qweight_bwd is not None)
            == (args.backward_weight_policy == "cache")
            for name in expected_replaced
        ),
        "cache_hook_forward_refresh_count_matches": hook_forward_refresh_count == expected_forward_cache_count,
        "cache_hook_dx_refresh_count_matches": hook_dx_refresh_count == expected_cache_count,
        "cache_hook_refresh_count_matches": hook_refresh_count == expected_forward_cache_count + expected_cache_count,
        "cache_hook_backward_weight_cache_count_matches": (
            hook_backward_weight_cache_count == expected_backward_weight_count
        ),
        "cache_hook_caches_current": caches_current,
        "forward_output_finite": bool(torch.isfinite(y).all()),
        "x_grad_finite": bool(x.grad is not None and torch.isfinite(x.grad).all()),
        "lora_grads_present": grads_present,
        "adapter_state_keys_match": set(adapter_state) == expected_adapter_keys,
        "target_modules_recorded": result.target_modules == ("q_proj", "down_proj"),
        "exclude_modules_recorded": result.exclude_modules == expected_excludes,
    }

    payload: dict[str, Any] = {
        "shape": {
            "batch": args.batch,
            "hidden": args.hidden,
            "rank": args.rank,
            "override_rank": args.override_rank,
            "auto_rank": args.auto_rank,
            "sensitivity_rank": expected_sensitivity_rank,
            "dtype": args.dtype,
            "lowrank_dtype": args.lowrank_dtype,
            "mode": args.mode,
            "use_frozen_residual": not args.no_frozen_residual,
            "backward_weight_policy": args.backward_weight_policy,
            "reuse_fused_dy_up_for_d_lora_down": args.reuse_fused_dy_up_for_d_lora_down,
            "fp4_activation_cache_min_rows": args.fp4_activation_cache_min_rows,
            "fp4_activation_cache_d_lora_down_backend": args.fp4_activation_cache_d_lora_down_backend,
            "zero_lora_up_fast_path": base_cfg.zero_lora_up_fast_path,
            "zero_fast_path_initial": zero_fast_path_initial,
            "base_fuse_frozen_residual_dx": base_cfg.fuse_frozen_residual_dx,
        },
        "replaced": result.replaced_modules,
        "outlier_keep_dense_replaced": outlier_exclude_result.replaced_modules,
        "outlier_keep_dense_exclude_modules": outlier_exclude_result.exclude_modules,
        "outlier_keep_dense_no_manual_replaced": outlier_exclude_no_manual_result.replaced_modules,
        "outlier_keep_dense_no_manual_exclude_modules": outlier_exclude_no_manual_result.exclude_modules,
        "outlier_bump_frozen_residual_replaced": outlier_residual_result.replaced_modules,
        "outlier_residual_only_replaced": outlier_residual_only_result.replaced_modules,
        "trainable": result.trainable_names,
        "trainable_param_count": result.trainable_param_count,
        "refreshed_forward_cache_count": result.refreshed_forward_cache_count,
        "refreshed_cache_count": result.refreshed_cache_count,
        "refreshed_backward_weight_count": result.refreshed_backward_weight_count,
        "cache_summary": {
            "module_count": result.cache_summary.module_count,
            "fused_lora_forward_cache_count": result.cache_summary.fused_lora_forward_cache_count,
            "fused_lora_forward_cache_bytes": result.cache_summary.fused_lora_forward_cache_bytes,
            "fused_lora_dx_cache_count": result.cache_summary.fused_lora_dx_cache_count,
            "fused_lora_dx_cache_bytes": result.cache_summary.fused_lora_dx_cache_bytes,
            "backward_weight_cache_count": result.cache_summary.backward_weight_cache_count,
            "backward_weight_cache_bytes": result.cache_summary.backward_weight_cache_bytes,
            "fp4_forward_qweight_bytes": result.cache_summary.fp4_forward_qweight_bytes,
            "dense_weight_bytes": result.cache_summary.dense_weight_bytes,
            "total_cache_bytes": result.cache_summary.total_cache_bytes,
            "fused_lora_forward_cache_vs_dense_weight": (
                result.cache_summary.fused_lora_forward_cache_vs_dense_weight
            ),
            "fused_lora_dx_cache_vs_dense_weight": (
                result.cache_summary.fused_lora_dx_cache_vs_dense_weight
            ),
            "backward_weight_cache_vs_dense_weight": (
                result.cache_summary.backward_weight_cache_vs_dense_weight
            ),
            "total_cache_vs_dense_weight": result.cache_summary.total_cache_vs_dense_weight,
        },
        "hook_refresh_count": hook_refresh_count,
        "hook_forward_refresh_count": hook_forward_refresh_count,
        "hook_dx_refresh_count": hook_dx_refresh_count,
        "hook_backward_weight_cache_count": hook_backward_weight_cache_count,
        "exclude_modules": result.exclude_modules,
        "checks": checks,
        "all_passed": bool(all(checks.values())),
    }

    os.makedirs(args.results_dir, exist_ok=True)
    latest = os.path.join(args.results_dir, "latest_fp4_lora_prepare_validation.json")
    stamped = os.path.join(
        args.results_dir,
        f"fp4_lora_prepare_validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
    )
    for path in (latest, stamped):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    print(json.dumps(payload, indent=2))
    print(f"Wrote {latest}")
    if not payload["all_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
