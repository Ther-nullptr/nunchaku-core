from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections.abc import Callable
from datetime import datetime
from typing import Any

import torch
import torch.nn.functional as F

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from native_fp4 import NunchakuFP4LoRALinear  # noqa: E402


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


def time_cuda(fn: Callable[[], Any], warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    values: list[float] = []
    for _ in range(iters):
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        values.append(start.elapsed_time(end))
    return float(sum(values) / len(values))


def time_construct(fn: Callable[[], NunchakuFP4LoRALinear]) -> tuple[NunchakuFP4LoRALinear, float]:
    torch.cuda.synchronize()
    start = time.perf_counter()
    module = fn()
    torch.cuda.synchronize()
    return module, float(time.perf_counter() - start)


def zero_grads(module: torch.nn.Module, x: torch.Tensor) -> None:
    module.zero_grad(set_to_none=True)
    x.grad = None


def train_step(module: torch.nn.Module, x: torch.Tensor, dy: torch.Tensor) -> None:
    zero_grads(module, x)
    y = module(x)
    loss = (y.float() * dy.float()).sum()
    loss.backward()


def grad_finite(module: NunchakuFP4LoRALinear, x: torch.Tensor) -> bool:
    grads = [x.grad, module.lora_down.grad, module.lora_up.grad]
    if module.bias is not None and isinstance(module.bias, torch.nn.Parameter):
        grads.append(module.bias.grad)
    return all(g is not None and bool(torch.isfinite(g).all()) for g in grads)


def module_record(
    module: NunchakuFP4LoRALinear,
    construct_s: float,
    y: torch.Tensor,
    y_dense: torch.Tensor,
    train_ms: float,
) -> dict[str, Any]:
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return {
        "construct_s": construct_s,
        "forward_vs_dense": tensor_error(y, y_dense),
        "train_step_ms": train_ms,
        "trainable_params": int(trainable),
        "has_frozen_residual": module.has_frozen_residual,
        "effective_rank": module.rank,
        "effective_frozen_residual_rank": module.frozen_residual_rank,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Benchmark FP4 LoRA residual-SVD initialization policies.")
    p.add_argument("--m", type=int, default=1024)
    p.add_argument("--in-features", type=int, default=1024)
    p.add_argument("--out-features", type=int, default=1024)
    p.add_argument("--rank", type=int, default=32)
    p.add_argument("--dtype", choices=["fp16", "bf16"], default="bf16")
    p.add_argument("--lowrank-dtype", choices=["fp16", "bf16"], default="bf16")
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--iters", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--input-std", type=float, default=1.0)
    p.add_argument("--weight-std", type=float, default=0.02)
    p.add_argument("--bias-std", type=float, default=0.02)
    p.add_argument("--dy-std", type=float, default=1.0)
    p.add_argument("--residual-svd-lowrank-oversample", type=int, default=8)
    p.add_argument("--residual-svd-lowrank-niter", type=int, default=2)
    p.add_argument("--results-dir", type=str, default="results")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    torch.manual_seed(args.seed)
    dtype = dtype_from_name(args.dtype)
    lowrank_dtype = dtype_from_name(args.lowrank_dtype)

    x_base = torch.randn(args.m, args.in_features, device="cuda", dtype=dtype).mul(float(args.input_std))
    dy = torch.randn(args.m, args.out_features, device="cuda", dtype=dtype).mul(float(args.dy_std))
    weight = torch.randn(args.out_features, args.in_features, device="cuda", dtype=dtype).mul(float(args.weight_std))
    bias = torch.randn(args.out_features, device="cuda", dtype=dtype).mul(float(args.bias_std))
    y_dense = F.linear(x_base, weight, bias)

    def make_zero() -> NunchakuFP4LoRALinear:
        return NunchakuFP4LoRALinear(
            weight=weight,
            bias=bias,
            rank=args.rank,
            lowrank_dtype=lowrank_dtype,
            init="zero",
            fuse_lora_dx=True,
            cache_fused_lora_dx=True,
        )

    def make_trainable_residual(method: str) -> NunchakuFP4LoRALinear:
        return NunchakuFP4LoRALinear(
            weight=weight,
            bias=bias,
            rank=args.rank,
            lowrank_dtype=lowrank_dtype,
            init="residual_svd",
            residual_svd_method=method,
            residual_svd_lowrank_oversample=args.residual_svd_lowrank_oversample,
            residual_svd_lowrank_niter=args.residual_svd_lowrank_niter,
            fuse_lora_dx=True,
            cache_fused_lora_dx=True,
        )

    def make_frozen_residual(method: str) -> NunchakuFP4LoRALinear:
        return NunchakuFP4LoRALinear(
            weight=weight,
            bias=bias,
            rank=args.rank,
            lowrank_dtype=lowrank_dtype,
            init="zero",
            frozen_residual_rank=args.rank,
            frozen_residual_init="residual_svd",
            residual_svd_method=method,
            residual_svd_lowrank_oversample=args.residual_svd_lowrank_oversample,
            residual_svd_lowrank_niter=args.residual_svd_lowrank_niter,
            fuse_lora_dx=True,
            cache_fused_lora_dx=True,
        )

    zero, zero_construct_s = time_construct(make_zero)
    trainable_residual_full, trainable_residual_full_construct_s = time_construct(
        lambda: make_trainable_residual("full_svd")
    )
    frozen_residual_full, frozen_residual_full_construct_s = time_construct(lambda: make_frozen_residual("full_svd"))
    torch.manual_seed(args.seed + 17)
    trainable_residual_lowrank, trainable_residual_lowrank_construct_s = time_construct(
        lambda: make_trainable_residual("svd_lowrank")
    )
    torch.manual_seed(args.seed + 17)
    frozen_residual_lowrank, frozen_residual_lowrank_construct_s = time_construct(
        lambda: make_frozen_residual("svd_lowrank")
    )

    modules = {
        "fp4_zero_lora": (zero, zero_construct_s),
        "fp4_trainable_residual_svd_lora_full": (
            trainable_residual_full,
            trainable_residual_full_construct_s,
        ),
        "fp4_frozen_residual_svd_zero_lora_full": (
            frozen_residual_full,
            frozen_residual_full_construct_s,
        ),
        "fp4_trainable_residual_svd_lora_lowrank": (
            trainable_residual_lowrank,
            trainable_residual_lowrank_construct_s,
        ),
        "fp4_frozen_residual_svd_zero_lora_lowrank": (
            frozen_residual_lowrank,
            frozen_residual_lowrank_construct_s,
        ),
    }

    outputs: dict[str, torch.Tensor] = {}
    train_ms: dict[str, float] = {}
    grad_checks: dict[str, bool] = {}
    for name, (module, _) in modules.items():
        module.train()
        outputs[name] = module(x_base).detach()
        x_time = x_base.detach().clone().requires_grad_(True)
        train_ms[name] = time_cuda(lambda m=module, x=x_time: train_step(m, x, dy), args.warmup, args.iters)
        grad_checks[name] = grad_finite(module, x_time)

    records = {
        name: module_record(module, construct_s, outputs[name], y_dense, train_ms[name])
        for name, (module, construct_s) in modules.items()
    }

    zero_rel_l2 = records["fp4_zero_lora"]["forward_vs_dense"]["rel_l2"]
    trainable_full_rel_l2 = records["fp4_trainable_residual_svd_lora_full"]["forward_vs_dense"]["rel_l2"]
    frozen_full_rel_l2 = records["fp4_frozen_residual_svd_zero_lora_full"]["forward_vs_dense"]["rel_l2"]
    trainable_lowrank_rel_l2 = records["fp4_trainable_residual_svd_lora_lowrank"]["forward_vs_dense"]["rel_l2"]
    frozen_lowrank_rel_l2 = records["fp4_frozen_residual_svd_zero_lora_lowrank"]["forward_vs_dense"]["rel_l2"]

    payload = {
        "shape": {
            "m": args.m,
            "in_features": args.in_features,
            "out_features": args.out_features,
            "rank": args.rank,
            "dtype": args.dtype,
            "lowrank_dtype": args.lowrank_dtype,
            "input_std": args.input_std,
            "weight_std": args.weight_std,
            "bias_std": args.bias_std,
            "dy_std": args.dy_std,
            "residual_svd_lowrank_oversample": args.residual_svd_lowrank_oversample,
            "residual_svd_lowrank_niter": args.residual_svd_lowrank_niter,
        },
        "policies": records,
        "comparisons": {
            "trainable_residual_svd_full_vs_zero": tensor_error(
                outputs["fp4_trainable_residual_svd_lora_full"],
                outputs["fp4_zero_lora"],
            ),
            "frozen_residual_svd_full_zero_vs_trainable_residual_svd_full": tensor_error(
                outputs["fp4_frozen_residual_svd_zero_lora_full"],
                outputs["fp4_trainable_residual_svd_lora_full"],
            ),
            "frozen_residual_svd_lowrank_zero_vs_trainable_residual_svd_lowrank": tensor_error(
                outputs["fp4_frozen_residual_svd_zero_lora_lowrank"],
                outputs["fp4_trainable_residual_svd_lora_lowrank"],
            ),
            "frozen_residual_svd_lowrank_vs_full": tensor_error(
                outputs["fp4_frozen_residual_svd_zero_lora_lowrank"],
                outputs["fp4_frozen_residual_svd_zero_lora_full"],
            ),
        },
        "derived": {
            "trainable_residual_svd_full_error_reduction_vs_zero": zero_rel_l2 / trainable_full_rel_l2,
            "frozen_residual_svd_full_error_reduction_vs_zero": zero_rel_l2 / frozen_full_rel_l2,
            "trainable_residual_svd_lowrank_error_reduction_vs_zero": zero_rel_l2 / trainable_lowrank_rel_l2,
            "frozen_residual_svd_lowrank_error_reduction_vs_zero": zero_rel_l2 / frozen_lowrank_rel_l2,
            "frozen_lowrank_construct_speedup_vs_full": (
                frozen_residual_full_construct_s / frozen_residual_lowrank_construct_s
            ),
            "recommended_policy": "fp4_frozen_residual_svd_zero_lora_full_or_lowrank_for_large_models",
        },
        "checks": {
            "trainable_residual_svd_full_improves_forward": trainable_full_rel_l2 < zero_rel_l2,
            "frozen_residual_svd_full_improves_forward": frozen_full_rel_l2 < zero_rel_l2,
            "trainable_residual_svd_lowrank_improves_forward": trainable_lowrank_rel_l2 < zero_rel_l2,
            "frozen_residual_svd_lowrank_improves_forward": frozen_lowrank_rel_l2 < zero_rel_l2,
            "frozen_and_trainable_residual_svd_full_close": (
                tensor_error(
                    outputs["fp4_frozen_residual_svd_zero_lora_full"],
                    outputs["fp4_trainable_residual_svd_lora_full"],
                )["rel_l2"]
                < 5e-3
            ),
            "frozen_and_trainable_residual_svd_lowrank_close": (
                tensor_error(
                    outputs["fp4_frozen_residual_svd_zero_lora_lowrank"],
                    outputs["fp4_trainable_residual_svd_lora_lowrank"],
                )["rel_l2"]
                < 5e-3
            ),
            "all_grads_finite": all(grad_checks.values()),
        },
    }
    payload["all_passed"] = bool(all(payload["checks"].values()))

    os.makedirs(args.results_dir, exist_ok=True)
    latest = os.path.join(args.results_dir, "latest_fp4_lora_initialization.json")
    stamped = os.path.join(
        args.results_dir,
        f"fp4_lora_initialization_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
    )
    for path in (latest, stamped):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    print(json.dumps(payload, indent=2))
    print(f"Wrote {latest}")


if __name__ == "__main__":
    main()
