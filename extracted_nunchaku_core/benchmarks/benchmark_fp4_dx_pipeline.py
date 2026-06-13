from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any

import torch

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from native_fp4 import NunchakuFP4BackwardDXOp, NunchakuFP4LoRALinear  # noqa: E402
from native_fp4.operators import ceil_divide, pad_tensor, quantize_fp4_act_with_lora  # noqa: E402
from native_fp4.training import _fused_lora_dx  # noqa: E402


def dtype_from_name(name: str) -> torch.dtype:
    if name == "fp16":
        return torch.float16
    if name == "bf16":
        return torch.bfloat16
    raise ValueError(f"Unsupported dtype: {name}")


def time_cuda(fn, warmup: int, iters: int) -> float:
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


def rel_l2(a: torch.Tensor, b: torch.Tensor) -> float:
    diff = (a.float() - b.float()).norm()
    denom = b.float().norm().clamp_min(1e-12)
    return float((diff / denom).item())


def benchmark_one(
    *,
    m: int,
    in_features: int,
    out_features: int,
    rank: int,
    dtype: torch.dtype,
    lowrank_dtype: torch.dtype,
    warmup: int,
    iters: int,
    seed: int,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    dy = torch.randn(m, out_features, device="cuda", dtype=dtype)
    weight = torch.randn(out_features, in_features, device="cuda", dtype=dtype)

    pure_dx = NunchakuFP4BackwardDXOp(weight=weight, dummy_rank=max(16, rank))
    lora_dx = NunchakuFP4LoRALinear(
        weight=weight,
        bias=None,
        rank=rank,
        lowrank_dtype=lowrank_dtype,
        init="gaussian",
        fuse_lora_dx=True,
        cache_fused_lora_dx=True,
    )
    lora_dx.refresh_fused_lora_dx_cache()
    packed_lora_dx = lora_dx._get_fused_lora_dx_cache()
    if packed_lora_dx is None:
        raise RuntimeError("expected cached packed LoRA dX factors")
    lora_down_bwd_packed, lora_up_bwd_packed = packed_lora_dx
    lora_scales = [1.0] * ceil_divide(lora_down_bwd_packed.shape[1], 16)

    dy2d_src = dy.reshape(-1, out_features)
    dy2d = dy2d_src if pure_dx.n_pad == out_features else pad_tensor(dy2d_src, divisor=pure_dx.n_pad, dim=1)
    qweight_bwd = pure_dx.repack_qweight_for_backward()
    qdy, ascales = pure_dx.quantize_grad(dy2d)
    qdy_lora, ascales_lora, packed_dy_up = quantize_fp4_act_with_lora(
        dy2d,
        lora_down_packed=lora_down_bwd_packed,
        smooth=pure_dx.smooth_bwd,
        pad_size=256,
    )
    torch.cuda.synchronize()

    def quantize_grad_fn() -> tuple[torch.Tensor, torch.Tensor]:
        return pure_dx.quantize_grad(dy2d)

    def repack_fn() -> torch.Tensor:
        return pure_dx.repack_qweight_for_backward()

    def prequantized_gemm_fn() -> torch.Tensor:
        return pure_dx.backward_prequantized(qdy, ascales, qweight_bwd)

    def quantize_then_gemm_cached_qweight_fn() -> torch.Tensor:
        qdy_local, ascales_local = pure_dx.quantize_grad(dy2d)
        return pure_dx.backward_prequantized(qdy_local, ascales_local, qweight_bwd)

    def full_dx_fn() -> torch.Tensor:
        return pure_dx(dy)

    def quantize_grad_with_lora_fn() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return quantize_fp4_act_with_lora(
            dy2d,
            lora_down_packed=lora_down_bwd_packed,
            smooth=pure_dx.smooth_bwd,
            pad_size=256,
        )

    def fused_prequantized_gemm_fn() -> torch.Tensor:
        return pure_dx.backward_prequantized(
            qdy_lora,
            ascales_lora,
            qweight_bwd,
            lora_act=packed_dy_up,
            lora_up=lora_up_bwd_packed,
            lora_scales=lora_scales,
        )

    def fused_quantize_then_gemm_cached_qweight_fn() -> torch.Tensor:
        qdy_local, ascales_local, packed_dy_up_local = quantize_fp4_act_with_lora(
            dy2d,
            lora_down_packed=lora_down_bwd_packed,
            smooth=pure_dx.smooth_bwd,
            pad_size=256,
        )
        return pure_dx.backward_prequantized(
            qdy_local,
            ascales_local,
            qweight_bwd,
            lora_act=packed_dy_up_local,
            lora_up=lora_up_bwd_packed,
            lora_scales=lora_scales,
        )

    def fused_dx_cached_pack_fn() -> torch.Tensor:
        return _fused_lora_dx(
            dy=dy,
            lora_down=lora_dx.lora_down,
            lora_up=lora_dx.lora_up,
            fp4_backward_op=lora_dx.fp4_backward,
            scaling=lora_dx.scaling,
            lowrank_dtype=lora_dx.lowrank_dtype,
            in_features=lora_dx.in_features,
            out_features=lora_dx.out_features,
            packed_lora_dx=packed_lora_dx,
        )

    with torch.no_grad():
        full_dx = full_dx_fn()
        cached_dx_pad = quantize_then_gemm_cached_qweight_fn()
        cached_dx = cached_dx_pad[:m, :in_features].reshape_as(full_dx)
        fused_dx = fused_dx_cached_pack_fn()
        fused_cached_pad = fused_quantize_then_gemm_cached_qweight_fn()
        fused_cached = fused_cached_pad[:m, :in_features].reshape_as(fused_dx)

    latency = {
        "quantize_grad": time_cuda(quantize_grad_fn, warmup, iters),
        "repack_backbone": time_cuda(repack_fn, warmup, iters),
        "prequantized_gemm": time_cuda(prequantized_gemm_fn, warmup, iters),
        "quantize_then_gemm_cached_qweight": time_cuda(quantize_then_gemm_cached_qweight_fn, warmup, iters),
        "full_dx_transient_repack": time_cuda(full_dx_fn, warmup, iters),
        "quantize_grad_with_lora": time_cuda(quantize_grad_with_lora_fn, warmup, iters),
        "fused_prequantized_gemm": time_cuda(fused_prequantized_gemm_fn, warmup, iters),
        "fused_quantize_then_gemm_cached_qweight": time_cuda(
            fused_quantize_then_gemm_cached_qweight_fn,
            warmup,
            iters,
        ),
        "fused_dx_cached_pack_transient_repack": time_cuda(fused_dx_cached_pack_fn, warmup, iters),
    }

    derived = {
        "quantize_share_of_full_dx": latency["quantize_grad"] / latency["full_dx_transient_repack"],
        "repack_share_of_full_dx": latency["repack_backbone"] / latency["full_dx_transient_repack"],
        "prequantized_gemm_share_of_full_dx": latency["prequantized_gemm"] / latency["full_dx_transient_repack"],
        "cached_qweight_speedup_bound": latency["full_dx_transient_repack"]
        / latency["quantize_then_gemm_cached_qweight"],
        "fused_lora_quantize_over_pure_quantize": latency["quantize_grad_with_lora"] / latency["quantize_grad"],
        "fused_lora_gemm_over_pure_gemm": latency["fused_prequantized_gemm"] / latency["prequantized_gemm"],
        "fused_cached_qweight_speedup_bound": latency["fused_dx_cached_pack_transient_repack"]
        / latency["fused_quantize_then_gemm_cached_qweight"],
        "fused_dx_over_pure_dx": latency["fused_dx_cached_pack_transient_repack"]
        / latency["full_dx_transient_repack"],
        "repack_over_fused_dx": latency["repack_backbone"] / latency["fused_dx_cached_pack_transient_repack"],
    }

    return {
        "shape": {
            "m": m,
            "in_features": in_features,
            "out_features": out_features,
            "rank": rank,
            "dtype": str(dtype).removeprefix("torch."),
            "lowrank_dtype": str(lowrank_dtype).removeprefix("torch."),
        },
        "latency_ms": latency,
        "derived": derived,
        "correctness": {
            "cached_qweight_dx_rel_l2_vs_full": rel_l2(cached_dx, full_dx),
            "fused_cached_qweight_dx_rel_l2_vs_full": rel_l2(fused_cached, fused_dx),
        },
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Split native FP4 backward dX into quantize/repack/GEMM stages.")
    p.add_argument("--m", type=int, default=4096)
    p.add_argument("--in-features", type=int, default=4096)
    p.add_argument("--out-features", type=int, default=4096)
    p.add_argument("--rank", type=int, default=32)
    p.add_argument("--dtype", choices=["fp16", "bf16"], default="bf16")
    p.add_argument("--lowrank-dtype", choices=["fp16", "bf16"], default=None)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--iters", type=int, default=30)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--results-dir", type=str, default="results")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    dtype = dtype_from_name(args.dtype)
    lowrank_dtype = dtype_from_name(args.lowrank_dtype or args.dtype)
    payload = {
        "experiment": "fp4_dx_pipeline",
        "notes": {
            "cached_qweight": "Upper-bound ablation only. The training path still uses transient repack and does not pre-store W^T.",
            "fused_lora": "Uses cached packed trainable LoRA factors but still transiently repacks the frozen FP4 backbone.",
        },
        "result": benchmark_one(
            m=args.m,
            in_features=args.in_features,
            out_features=args.out_features,
            rank=args.rank,
            dtype=dtype,
            lowrank_dtype=lowrank_dtype,
            warmup=args.warmup,
            iters=args.iters,
            seed=args.seed,
        ),
    }

    os.makedirs(args.results_dir, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    latest_path = os.path.join(args.results_dir, "latest_fp4_dx_pipeline.json")
    stamped_path = os.path.join(args.results_dir, f"fp4_dx_pipeline_{stamp}.json")
    for path in (latest_path, stamped_path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    print(json.dumps(payload, indent=2))
    print(f"Wrote {latest_path}")


if __name__ == "__main__":
    main()
