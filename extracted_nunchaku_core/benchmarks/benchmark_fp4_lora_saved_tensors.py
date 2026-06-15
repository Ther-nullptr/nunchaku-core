from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from typing import Any

import torch

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from native_fp4 import NunchakuFP4LoRALinear  # noqa: E402


SAVED_TENSOR_NAMES = (
    "saved_x",
    "qact",
    "ascales",
    "lora_down",
    "lora_up",
    "frozen_residual_down",
    "frozen_residual_up",
    "saved_lora_act",
)


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


def time_cuda(fn, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    times: list[float] = []
    for _ in range(iters):
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    return float(sum(times) / len(times))


def zero_grads(module: torch.nn.Module, x: torch.Tensor) -> None:
    module.zero_grad(set_to_none=True)
    x.grad = None


def tensor_record(index: int, tensor: torch.Tensor) -> dict[str, Any]:
    return {
        "index": index,
        "name": SAVED_TENSOR_NAMES[index] if index < len(SAVED_TENSOR_NAMES) else f"saved_{index}",
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype).removeprefix("torch."),
        "numel": int(tensor.numel()),
        "element_size": int(tensor.element_size()),
        "bytes": int(tensor.numel() * tensor.element_size()),
    }


def capture_saved_tensors(module: torch.nn.Module, x: torch.Tensor, dy: torch.Tensor) -> list[dict[str, Any]]:
    saved: list[torch.Tensor] = []

    def pack_hook(tensor: torch.Tensor) -> torch.Tensor:
        saved.append(tensor)
        return tensor

    def unpack_hook(tensor: torch.Tensor) -> torch.Tensor:
        return tensor

    zero_grads(module, x)
    with torch.autograd.graph.saved_tensors_hooks(pack_hook, unpack_hook):
        y = module(x)
    loss = (y.float() * dy.float()).sum()
    del loss, y
    torch.cuda.synchronize()
    return [tensor_record(i, tensor) for i, tensor in enumerate(saved)]


def summarize_saved(records: list[dict[str, Any]]) -> dict[str, int]:
    by_name = {record["name"]: int(record["bytes"]) for record in records}
    x_or_fp4_cache = by_name.get("saved_x", 0) + by_name.get("qact", 0) + by_name.get("ascales", 0)
    lora_act = by_name.get("saved_lora_act", 0)
    return {
        "x_or_fp4_cache": x_or_fp4_cache,
        "saved_lora_act": lora_act,
        "activation_context": x_or_fp4_cache + lora_act,
        "all_saved_tensors": sum(int(record["bytes"]) for record in records),
    }


def saved_bytes_by_name(records: list[dict[str, Any]]) -> dict[str, int]:
    return {record["name"]: int(record["bytes"]) for record in records}


def safe_ratio(numerator: int | float, denominator: int | float) -> float | None:
    return None if denominator == 0 else float(numerator) / float(denominator)


def make_module(
    weight: torch.Tensor,
    bias: torch.Tensor,
    rank: int,
    lowrank_dtype: torch.dtype,
    fp4_activation_cache_d_lora_down: bool,
    init: str,
    fp4_activation_cache_min_rows: int = 0,
    fp4_activation_cache_d_lora_down_backend: str = "fused",
) -> NunchakuFP4LoRALinear:
    return NunchakuFP4LoRALinear(
        weight=weight,
        bias=bias,
        rank=rank,
        lowrank_dtype=lowrank_dtype,
        init=init,
        cache_lora_act=True,
        fuse_lora_dx=True,
        cache_fused_lora_dx=True,
        fp4_activation_cache_d_lora_down=fp4_activation_cache_d_lora_down,
        fp4_activation_cache_min_rows=fp4_activation_cache_min_rows,
        fp4_activation_cache_d_lora_down_backend=fp4_activation_cache_d_lora_down_backend,
    )


def sync_lora(dst: NunchakuFP4LoRALinear, src: NunchakuFP4LoRALinear) -> None:
    with torch.no_grad():
        dst.lora_down.copy_(src.lora_down)
        dst.lora_up.copy_(src.lora_up)
    if dst.init_mode == "zero" and src.init_mode == "zero":
        dst.mark_lora_up_zero_fast_path()
    dst.clear_fused_lora_dx_cache()
    dst.refresh_fused_lora_dx_cache()


def train_step(module: torch.nn.Module, x: torch.Tensor, dy: torch.Tensor) -> None:
    zero_grads(module, x)
    y = module(x)
    loss = (y.float() * dy.float()).sum()
    loss.backward()


def run_once(module: torch.nn.Module, x: torch.Tensor, dy: torch.Tensor) -> dict[str, torch.Tensor]:
    zero_grads(module, x)
    y = module(x)
    loss = (y.float() * dy.float()).sum()
    loss.backward()
    return {
        "y": y.detach(),
        "dx": x.grad.detach().clone(),
        "d_lora_down": module.lora_down.grad.detach().clone(),
        "d_lora_up": module.lora_up.grad.detach().clone(),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Measure saved tensors for FP4 LoRA activation-cache dA modes.")
    p.add_argument("--m", type=int, default=4096)
    p.add_argument("--in-features", type=int, default=4096)
    p.add_argument("--out-features", type=int, default=4096)
    p.add_argument("--rank", type=int, default=32)
    p.add_argument("--dtype", choices=["fp16", "bf16"], default="bf16")
    p.add_argument("--lowrank-dtype", choices=["fp16", "bf16"], default="bf16")
    p.add_argument("--init", choices=["zero", "gaussian"], default="gaussian")
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--iters", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--fp4-activation-cache-min-rows", type=int, default=0)
    p.add_argument("--fp4-activation-cache-d-lora-down-backend", choices=["fused", "dequant_gemm"], default="fused")
    p.add_argument("--results-dir", type=str, default="results")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.manual_seed(args.seed)

    dtype = dtype_from_name(args.dtype)
    lowrank_dtype = dtype_from_name(args.lowrank_dtype)
    x_base = torch.randn(args.m, args.in_features, device="cuda", dtype=dtype)
    dy = torch.randn(args.m, args.out_features, device="cuda", dtype=dtype)
    weight = torch.randn(args.out_features, args.in_features, device="cuda", dtype=dtype)
    bias = torch.randn(args.out_features, device="cuda", dtype=dtype)

    exact = make_module(
        weight,
        bias,
        args.rank,
        lowrank_dtype,
        fp4_activation_cache_d_lora_down=False,
        init=args.init,
    )
    fp4_cache = make_module(
        weight,
        bias,
        args.rank,
        lowrank_dtype,
        fp4_activation_cache_d_lora_down=True,
        init=args.init,
        fp4_activation_cache_min_rows=args.fp4_activation_cache_min_rows,
        fp4_activation_cache_d_lora_down_backend=args.fp4_activation_cache_d_lora_down_backend,
    )
    sync_lora(fp4_cache, exact)
    exact.refresh_fused_lora_dx_cache()

    exact_saved = capture_saved_tensors(exact, x_base.detach().clone().requires_grad_(True), dy)
    fp4_cache_saved = capture_saved_tensors(fp4_cache, x_base.detach().clone().requires_grad_(True), dy)
    exact_summary = summarize_saved(exact_saved)
    fp4_cache_summary = summarize_saved(fp4_cache_saved)
    fp4_cache_saved_by_name = saved_bytes_by_name(fp4_cache_saved)

    exact_outputs = run_once(exact, x_base.detach().clone().requires_grad_(True), dy)
    fp4_cache_outputs = run_once(fp4_cache, x_base.detach().clone().requires_grad_(True), dy)

    x_exact_time = x_base.detach().clone().requires_grad_(True)
    x_fp4_cache_time = x_base.detach().clone().requires_grad_(True)
    exact_ms = time_cuda(lambda: train_step(exact, x_exact_time, dy), args.warmup, args.iters)
    fp4_cache_ms = time_cuda(lambda: train_step(fp4_cache, x_fp4_cache_time, dy), args.warmup, args.iters)
    fp4_cache_active = bool(args.m >= args.fp4_activation_cache_min_rows)

    errors = {
        "forward_vs_exact": tensor_error(fp4_cache_outputs["y"], exact_outputs["y"]),
        "dx_vs_exact": tensor_error(fp4_cache_outputs["dx"], exact_outputs["dx"]),
        "d_lora_up_vs_exact": tensor_error(fp4_cache_outputs["d_lora_up"], exact_outputs["d_lora_up"]),
        "d_lora_down_vs_exact": tensor_error(
            fp4_cache_outputs["d_lora_down"],
            exact_outputs["d_lora_down"],
        ),
    }
    checks = {
        "latency_positive": exact_ms > 0.0 and fp4_cache_ms > 0.0,
        "saved_tensor_reduction_positive": fp4_cache_summary["activation_context"] > 0
        and exact_summary["activation_context"] >= fp4_cache_summary["activation_context"],
        "fp4_cache_does_not_save_x_when_active": (
            (not fp4_cache_active) or fp4_cache_saved_by_name.get("saved_x", 0) == 0
        ),
        "zero_up_does_not_save_qact_when_active": (
            args.init != "zero" or (not fp4_cache_active) or fp4_cache_saved_by_name.get("qact", 0) == 0
        ),
        "zero_up_does_not_save_ascales_when_active": (
            args.init != "zero" or (not fp4_cache_active) or fp4_cache_saved_by_name.get("ascales", 0) == 0
        ),
        "forward_error_finite": bool(torch.isfinite(torch.tensor(errors["forward_vs_exact"]["rel_l2"]))),
        "dx_error_finite": bool(torch.isfinite(torch.tensor(errors["dx_vs_exact"]["rel_l2"]))),
        "d_lora_up_error_finite": bool(torch.isfinite(torch.tensor(errors["d_lora_up_vs_exact"]["rel_l2"]))),
        "d_lora_down_error_finite": bool(torch.isfinite(torch.tensor(errors["d_lora_down_vs_exact"]["rel_l2"]))),
    }

    payload = {
        "shape": {
            "m": args.m,
            "in_features": args.in_features,
            "out_features": args.out_features,
            "rank": args.rank,
            "effective_rank": exact.rank,
            "dtype": args.dtype,
            "lowrank_dtype": args.lowrank_dtype,
            "init": args.init,
            "fp4_activation_cache_min_rows": args.fp4_activation_cache_min_rows,
            "fp4_activation_cache_active_for_forward": fp4_cache_active,
            "fp4_activation_cache_d_lora_down_backend": args.fp4_activation_cache_d_lora_down_backend,
        },
        "saved_tensors": {
            "exact_cached_pack": exact_saved,
            "fp4_activation_cache_d_lora_down": fp4_cache_saved,
        },
        "saved_bytes": {
            "exact_cached_pack": exact_summary,
            "fp4_activation_cache_d_lora_down": fp4_cache_summary,
            "activation_context_reduction": safe_ratio(
                exact_summary["activation_context"],
                fp4_cache_summary["activation_context"],
            ),
            "x_or_fp4_cache_reduction": safe_ratio(
                exact_summary["x_or_fp4_cache"],
                fp4_cache_summary["x_or_fp4_cache"],
            ),
            "all_saved_tensors_reduction": safe_ratio(
                exact_summary["all_saved_tensors"],
                fp4_cache_summary["all_saved_tensors"],
            ),
        },
        "latency_ms": {
            "exact_cached_pack_train_step": exact_ms,
            "fp4_activation_cache_d_lora_down_train_step": fp4_cache_ms,
        },
        "speedups": {
            "fp4_activation_cache_d_lora_down_vs_exact_cached_pack": exact_ms / fp4_cache_ms,
        },
        "errors": errors,
        "checks": checks,
        "all_passed": bool(all(checks.values())),
    }

    print(json.dumps(payload, indent=2))
    os.makedirs(args.results_dir, exist_ok=True)
    latest = os.path.join(args.results_dir, "latest_fp4_lora_saved_tensors.json")
    stamped = os.path.join(
        args.results_dir,
        f"fp4_lora_saved_tensors_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
    )
    for path in (latest, stamped):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
    print(f"Wrote {latest}")
    if not payload["all_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
