from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict
from datetime import datetime
from typing import Any

import torch
import torch.nn.functional as F

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from native_fp4 import (  # noqa: E402
    FP4LoRAConfig,
    NunchakuFP4LoRALinear,
    fp4_lora_finetune_config,
    fp4_lora_parameter_groups,
    register_fp4_lora_cache_refresh_hook,
)


MODES = ("accuracy", "balanced", "throughput", "memory_saving")


def dtype_from_name(name: str) -> torch.dtype:
    if name == "fp16":
        return torch.float16
    if name == "bf16":
        return torch.bfloat16
    raise ValueError(f"Unsupported dtype: {name}")


def tensor_error(a: torch.Tensor, b: torch.Tensor) -> dict[str, float]:
    da = a.float()
    db = b.float()
    diff = (da - db).abs()
    return {
        "max_abs": float(diff.max().item()),
        "mae": float(diff.mean().item()),
        "rel_l2": float((da - db).norm().item() / (db.norm().item() + 1e-12)),
    }


def jsonable_config(cfg: FP4LoRAConfig) -> dict[str, Any]:
    data = asdict(cfg)
    data["lowrank_dtype"] = str(cfg.lowrank_dtype).replace("torch.", "")
    return data


def make_module(
    cfg: FP4LoRAConfig,
    weight: torch.Tensor,
    bias: torch.Tensor,
) -> NunchakuFP4LoRALinear:
    return NunchakuFP4LoRALinear(
        weight=weight,
        bias=bias,
        rank=cfg.rank,
        lora_alpha=cfg.lora_alpha,
        lowrank_dtype=cfg.lowrank_dtype,
        init=cfg.init,
        frozen_residual_rank=cfg.frozen_residual_rank,
        frozen_residual_init=cfg.frozen_residual_init,
        residual_svd_method=cfg.residual_svd_method,
        residual_svd_lowrank_oversample=cfg.residual_svd_lowrank_oversample,
        residual_svd_lowrank_niter=cfg.residual_svd_lowrank_niter,
        train_bias=cfg.train_bias,
        cache_lora_act=cfg.cache_lora_act,
        activation_checkpoint=cfg.activation_checkpoint,
        fuse_lowrank_forward=cfg.fuse_lowrank_forward,
        fuse_lora_dx=cfg.fuse_lora_dx,
        fuse_frozen_residual_dx=cfg.fuse_frozen_residual_dx,
        cache_fused_lora_dx=cfg.cache_fused_lora_dx,
        reuse_fused_dy_up_for_d_lora_down=cfg.reuse_fused_dy_up_for_d_lora_down,
        overlap_lora_grad=cfg.overlap_lora_grad,
        overlap_lora_grad_min_rows=cfg.overlap_lora_grad_min_rows,
        fp4_activation_cache_d_lora_down=cfg.fp4_activation_cache_d_lora_down,
        fp4_activation_cache_d_lora_down_backend=cfg.fp4_activation_cache_d_lora_down_backend,
    )


def mode_expectations(mode: str, dtype: torch.dtype, lowrank_dtype: torch.dtype, backend: str) -> dict[str, Any]:
    fuse_frozen_residual_dx = mode == "throughput" and dtype == torch.float16 and lowrank_dtype == torch.float16
    return {
        "fuse_lora_dx": mode != "accuracy",
        "cache_fused_lora_dx": mode != "accuracy",
        "overlap_lora_grad": mode in ("balanced", "throughput") and not fuse_frozen_residual_dx,
        "fp4_activation_cache_d_lora_down": mode == "memory_saving",
        "fp4_activation_cache_d_lora_down_backend": backend,
        "fuse_lowrank_forward": mode == "throughput",
        "fuse_frozen_residual_dx": fuse_frozen_residual_dx,
    }


def run_policy(
    *,
    mode: str,
    dtype: torch.dtype,
    dtype_name: str,
    lowrank_dtype: torch.dtype,
    lowrank_dtype_name: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    torch.manual_seed(args.seed)
    x = torch.randn(args.m, args.in_features, device="cuda", dtype=dtype).mul(float(args.input_std))
    target = torch.randn(args.m, args.out_features, device="cuda", dtype=dtype).mul(float(args.target_std))
    weight = torch.randn(args.out_features, args.in_features, device="cuda", dtype=dtype).mul(float(args.weight_std))
    bias = torch.randn(args.out_features, device="cuda", dtype=dtype).mul(float(args.bias_std))

    cfg = fp4_lora_finetune_config(
        mode=mode,
        rank=args.rank,
        dtype=dtype,
        lowrank_dtype=lowrank_dtype,
        use_frozen_residual=not args.no_frozen_residual,
        frozen_residual_rank=None if args.frozen_residual_rank is None else args.frozen_residual_rank,
        train_bias=args.train_bias,
        overlap_lora_grad_min_rows=args.overlap_lora_grad_min_rows,
        fp4_activation_cache_d_lora_down_backend=args.fp4_activation_cache_d_lora_down_backend,
    )
    module = make_module(cfg, weight=weight, bias=bias).train()
    if cfg.cache_fused_lora_dx:
        module.refresh_fused_lora_dx_cache()

    frozen_down_before = (
        module.frozen_residual_down.detach().clone() if module.has_frozen_residual else torch.empty(0, device="cuda")
    )
    frozen_up_before = (
        module.frozen_residual_up.detach().clone() if module.has_frozen_residual else torch.empty(0, device="cuda")
    )
    lora_down_before = module.lora_down.detach().clone()
    lora_up_before = module.lora_up.detach().clone()

    x = x.detach().clone().requires_grad_(True)
    param_groups = fp4_lora_parameter_groups(module, train_bias=cfg.train_bias)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, eps=args.adam_eps)
    hook = register_fp4_lora_cache_refresh_hook(optimizer, module) if cfg.cache_fused_lora_dx else None

    y = None
    initial_loss = None
    for step in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        x.grad = None
        y = module(x)
        loss = F.mse_loss(y.float(), target.float())
        if step == 0:
            initial_loss = loss.detach()
        loss.backward()
        optimizer.step()
    torch.cuda.synchronize()
    assert y is not None and initial_loss is not None

    with torch.no_grad():
        final_y = module(x.detach())
        final_loss = F.mse_loss(final_y.float(), target.float())
        frozen_down_after = module.frozen_residual_down.detach() if module.has_frozen_residual else frozen_down_before
        frozen_up_after = module.frozen_residual_up.detach() if module.has_frozen_residual else frozen_up_before
        frozen_unchanged = bool(
            torch.equal(frozen_down_before, frozen_down_after)
            and torch.equal(frozen_up_before, frozen_up_after)
        )
        lora_down_changed = bool((module.lora_down.detach().float() - lora_down_before.float()).norm().item() > 0.0)
        lora_up_changed = bool((module.lora_up.detach().float() - lora_up_before.float()).norm().item() > 0.0)

    trainable_names = [name for name, param in module.named_parameters() if param.requires_grad]
    expected_trainable = {"lora_down", "lora_up"}
    if cfg.train_bias:
        expected_trainable.add("bias")
    expected_flags = mode_expectations(mode, dtype, lowrank_dtype, args.fp4_activation_cache_d_lora_down_backend)
    actual_flags = {
        "fuse_lora_dx": module.fuse_lora_dx,
        "cache_fused_lora_dx": module.cache_fused_lora_dx,
        "overlap_lora_grad": module.overlap_lora_grad,
        "fp4_activation_cache_d_lora_down": module.fp4_activation_cache_d_lora_down,
        "fp4_activation_cache_d_lora_down_backend": module.fp4_activation_cache_d_lora_down_backend,
        "fuse_lowrank_forward": module.fuse_lowrank_forward,
        "fuse_frozen_residual_dx": module.fuse_frozen_residual_dx,
    }
    grads_finite = bool(
        x.grad is not None
        and torch.isfinite(x.grad).all()
        and module.lora_down.grad is not None
        and torch.isfinite(module.lora_down.grad).all()
        and module.lora_up.grad is not None
        and torch.isfinite(module.lora_up.grad).all()
    )
    checks = {
        "expected_flags": actual_flags == expected_flags,
        "loss_finite": bool(torch.isfinite(initial_loss) and torch.isfinite(final_loss)),
        "grads_finite": grads_finite,
        "lora_params_changed": lora_down_changed and lora_up_changed,
        "frozen_residual_unchanged": frozen_unchanged,
        "only_expected_params_trainable": set(trainable_names) == expected_trainable,
        "cache_hook_ran_if_needed": hook is None or hook.last_refresh_count > 0,
    }
    if hook is not None:
        hook.remove()

    return {
        "mode": mode,
        "dtype": dtype_name,
        "lowrank_dtype": lowrank_dtype_name,
        "config": jsonable_config(cfg),
        "expected_flags": expected_flags,
        "actual_flags": actual_flags,
        "loss": {
            "initial": float(initial_loss.detach().item()),
            "after_steps": float(final_loss.detach().item()),
        },
        "output_change_after_step": tensor_error(final_y, y.detach()),
        "trainable_params": trainable_names,
        "checks": checks,
        "all_passed": bool(all(checks.values())),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Validate recommended FP4 LoRA fine-tuning policy configs.")
    p.add_argument("--m", type=int, default=129)
    p.add_argument("--in-features", type=int, default=512)
    p.add_argument("--out-features", type=int, default=768)
    p.add_argument("--rank", type=int, default=32)
    p.add_argument("--frozen-residual-rank", type=int, default=None)
    p.add_argument("--dtype", choices=["fp16", "bf16"], default="bf16")
    p.add_argument("--lowrank-dtype", choices=["fp16", "bf16"], default="bf16")
    p.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--input-std", type=float, default=1.0)
    p.add_argument("--target-std", type=float, default=1.0)
    p.add_argument("--weight-std", type=float, default=0.02)
    p.add_argument("--bias-std", type=float, default=0.02)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--steps", type=int, default=2)
    p.add_argument("--adam-eps", type=float, default=1e-4)
    p.add_argument("--train-bias", action="store_true")
    p.add_argument("--no-frozen-residual", action="store_true")
    p.add_argument("--overlap-lora-grad-min-rows", type=int, default=4096)
    p.add_argument("--fp4-activation-cache-d-lora-down-backend", choices=["fused", "dequant_gemm"], default="fused")
    p.add_argument("--results-dir", type=str, default="results")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    dtype = dtype_from_name(args.dtype)
    lowrank_dtype = dtype_from_name(args.lowrank_dtype)
    records = [
        run_policy(
            mode=mode,
            dtype=dtype,
            dtype_name=args.dtype,
            lowrank_dtype=lowrank_dtype,
            lowrank_dtype_name=args.lowrank_dtype,
            args=args,
        )
        for mode in args.modes
    ]

    payload = {
        "shape": {
            "m": args.m,
            "in_features": args.in_features,
            "out_features": args.out_features,
            "rank": args.rank,
            "dtype": args.dtype,
            "lowrank_dtype": args.lowrank_dtype,
            "modes": args.modes,
            "no_frozen_residual": args.no_frozen_residual,
            "train_bias": args.train_bias,
            "overlap_lora_grad_min_rows": args.overlap_lora_grad_min_rows,
            "fp4_activation_cache_d_lora_down_backend": args.fp4_activation_cache_d_lora_down_backend,
            "steps": args.steps,
        },
        "policies": {record["mode"]: record for record in records},
        "all_passed": bool(all(record["all_passed"] for record in records)),
    }

    os.makedirs(args.results_dir, exist_ok=True)
    latest = os.path.join(args.results_dir, "latest_fp4_lora_training_policies_validation.json")
    stamped = os.path.join(
        args.results_dir,
        f"fp4_lora_training_policies_validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
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
