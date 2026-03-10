from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from native_fp4 import NunchakuFP4BackwardDXOp, NunchakuFP4LowRankBackwardDXOp  # noqa: E402
from native_fp4.layout import repack_fp4_weight_for_backward  # noqa: E402


def time_cuda(fn, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    ms_list = []
    for _ in range(iters):
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        ms_list.append(start.elapsed_time(end))

    return float(sum(ms_list) / len(ms_list))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, default=4096)
    parser.add_argument("--in-features", type=int, default=4096)
    parser.add_argument("--out-features", type=int, default=4096)
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--dtype", choices=["fp16", "bf16"], default="fp16")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--results-dir", type=str, default="results")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16

    x = torch.randn(args.m, args.in_features, device="cuda", dtype=dtype)
    dy = torch.randn(args.m, args.out_features, device="cuda", dtype=dtype)
    w = torch.randn(args.out_features, args.in_features, device="cuda", dtype=dtype)

    pure_dx = NunchakuFP4BackwardDXOp(weight=w)
    hybrid_dx = NunchakuFP4LowRankBackwardDXOp(weight=w, rank=args.rank, lowrank_dtype=dtype)
    forward_lora_cache = hybrid_dx.build_forward_lowrank_cache(x)

    def fp16_dx_fn() -> torch.Tensor:
        return torch.matmul(dy, w)

    def repack_only_fn() -> torch.Tensor:
        return pure_dx.repack_qweight_for_backward()

    def repack_python_only_fn() -> torch.Tensor:
        return repack_fp4_weight_for_backward(
            qweight=pure_dx.qweight,
            packed_wscales=pure_dx.wscales,
            out_features=pure_dx.n_pad,
            in_features=pure_dx.k_pad,
            logical_scales_t=pure_dx.wscales_bwd_logical,
        )

    def fp4_dx_fn() -> torch.Tensor:
        return pure_dx(dy)

    def fp4_dx_hybrid_unfused_fn() -> torch.Tensor:
        return hybrid_dx.forward_unfused(dy)

    def fp4_dx_hybrid_fused_fn() -> torch.Tensor:
        return hybrid_dx(dy)

    def fp16_full_backward_fn() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        lora_up = hybrid_dx.lora_up_dense[: hybrid_dx.out_features, : hybrid_dx.rank].to(dtype)
        lora_down = hybrid_dx.lora_down_dense[: hybrid_dx.rank, : hybrid_dx.in_features].to(dtype)
        lora_act = torch.matmul(x.to(dtype), lora_down.t())
        dy_up = torch.matmul(dy.to(dtype), lora_up)
        return (
            torch.matmul(dy, w) + torch.matmul(dy_up, lora_down),
            torch.matmul(dy.to(dtype).t(), lora_act),
            torch.matmul(dy_up.t(), x.to(dtype)),
        )

    def fp4_full_backward_unfused_fn() -> dict[str, torch.Tensor]:
        return hybrid_dx.backward_full(x, dy, fused_dx=False)

    def fp4_full_backward_fused_fn() -> dict[str, torch.Tensor]:
        return hybrid_dx.backward_full(x, dy, fused_dx=True)

    def fp4_full_backward_shared_recompute_fn() -> dict[str, torch.Tensor]:
        return hybrid_dx.backward_full_shared(x, dy)

    def fp4_full_backward_shared_cached_fn() -> dict[str, torch.Tensor]:
        return hybrid_dx.backward_full_shared(x, dy, forward_lora_act=forward_lora_cache)

    def fp4_full_backward_shared_packed_fn() -> dict[str, torch.Tensor]:
        return hybrid_dx.backward_full_shared_packed(x, dy, forward_lora_act=forward_lora_cache)

    def fp4_full_backward_shared_packed_overlap_fn() -> dict[str, torch.Tensor]:
        return hybrid_dx.backward_full_shared_packed_overlap(x, dy, forward_lora_act=forward_lora_cache)

    def fp4_full_backward_shared_dual_fn() -> dict[str, torch.Tensor]:
        return hybrid_dx.backward_full_shared_dual(x, dy, forward_lora_act=forward_lora_cache)

    with torch.no_grad():
        dx_ref = fp16_dx_fn().float()
        dx_fp4 = fp4_dx_fn().float()
        dx_hybrid = fp4_dx_hybrid_fused_fn().float()

    repack_python_ms = time_cuda(repack_python_only_fn, warmup=args.warmup, iters=args.iters)
    repack_ms = time_cuda(repack_only_fn, warmup=args.warmup, iters=args.iters)
    fp16_dx_ms = time_cuda(fp16_dx_fn, warmup=args.warmup, iters=args.iters)
    fp4_dx_ms = time_cuda(fp4_dx_fn, warmup=args.warmup, iters=args.iters)
    fp4_dx_hybrid_unfused_ms = time_cuda(fp4_dx_hybrid_unfused_fn, warmup=args.warmup, iters=args.iters)
    fp4_dx_hybrid_fused_ms = time_cuda(fp4_dx_hybrid_fused_fn, warmup=args.warmup, iters=args.iters)
    fp16_full_backward_ms = time_cuda(fp16_full_backward_fn, warmup=args.warmup, iters=args.iters)
    fp4_full_backward_unfused_ms = time_cuda(fp4_full_backward_unfused_fn, warmup=args.warmup, iters=args.iters)
    fp4_full_backward_fused_ms = time_cuda(fp4_full_backward_fused_fn, warmup=args.warmup, iters=args.iters)
    fp4_full_backward_shared_recompute_ms = time_cuda(
        fp4_full_backward_shared_recompute_fn, warmup=args.warmup, iters=args.iters
    )
    fp4_full_backward_shared_cached_ms = time_cuda(
        fp4_full_backward_shared_cached_fn, warmup=args.warmup, iters=args.iters
    )
    fp4_full_backward_shared_packed_ms = time_cuda(
        fp4_full_backward_shared_packed_fn, warmup=args.warmup, iters=args.iters
    )
    fp4_full_backward_shared_packed_overlap_ms = time_cuda(
        fp4_full_backward_shared_packed_overlap_fn, warmup=args.warmup, iters=args.iters
    )
    fp4_full_backward_shared_dual_ms = time_cuda(
        fp4_full_backward_shared_dual_fn, warmup=args.warmup, iters=args.iters
    )

    payload = {
        "m": args.m,
        "in_features": args.in_features,
        "out_features": args.out_features,
        "rank": args.rank,
        "dtype": args.dtype,
        "repack_python_ms": repack_python_ms,
        "repack_ms": repack_ms,
        "fp16_dx_ms": fp16_dx_ms,
        "fp4_dx_ms": fp4_dx_ms,
        "fp4_dx_hybrid_unfused_ms": fp4_dx_hybrid_unfused_ms,
        "fp4_dx_hybrid_fused_ms": fp4_dx_hybrid_fused_ms,
        "fp16_full_backward_ms": fp16_full_backward_ms,
        "fp4_full_backward_unfused_ms": fp4_full_backward_unfused_ms,
        "fp4_full_backward_fused_ms": fp4_full_backward_fused_ms,
        "fp4_full_backward_shared_recompute_ms": fp4_full_backward_shared_recompute_ms,
        "fp4_full_backward_shared_cached_ms": fp4_full_backward_shared_cached_ms,
        "fp4_full_backward_shared_packed_ms": fp4_full_backward_shared_packed_ms,
        "fp4_full_backward_shared_packed_overlap_ms": fp4_full_backward_shared_packed_overlap_ms,
        "fp4_full_backward_shared_dual_ms": fp4_full_backward_shared_dual_ms,
        "fp4_dx_speedup_vs_fp16": fp16_dx_ms / fp4_dx_ms,
        "fp4_dx_hybrid_unfused_speedup_vs_fp16": fp16_dx_ms / fp4_dx_hybrid_unfused_ms,
        "fp4_dx_hybrid_fused_speedup_vs_fp16": fp16_dx_ms / fp4_dx_hybrid_fused_ms,
        "fp4_full_backward_unfused_speedup_vs_fp16": fp16_full_backward_ms / fp4_full_backward_unfused_ms,
        "fp4_full_backward_fused_speedup_vs_fp16": fp16_full_backward_ms / fp4_full_backward_fused_ms,
        "fp4_full_backward_shared_recompute_speedup_vs_fp16": fp16_full_backward_ms / fp4_full_backward_shared_recompute_ms,
        "fp4_full_backward_shared_cached_speedup_vs_fp16": fp16_full_backward_ms / fp4_full_backward_shared_cached_ms,
        "fp4_full_backward_shared_packed_speedup_vs_fp16": fp16_full_backward_ms / fp4_full_backward_shared_packed_ms,
        "fp4_full_backward_shared_packed_overlap_speedup_vs_fp16": fp16_full_backward_ms
        / fp4_full_backward_shared_packed_overlap_ms,
        "fp4_full_backward_shared_dual_speedup_vs_fp16": fp16_full_backward_ms / fp4_full_backward_shared_dual_ms,
        "repack_speedup_vs_python": repack_python_ms / repack_ms,
        "repack_over_fp4_dx": repack_ms / fp4_dx_ms,
        "repack_over_fp4_dx_hybrid_fused": repack_ms / fp4_dx_hybrid_fused_ms,
        "repack_python_over_fp4_dx": repack_python_ms / fp4_dx_ms,
        "shared_cached_over_fused_full_backward": fp4_full_backward_shared_cached_ms / fp4_full_backward_fused_ms,
        "shared_recompute_over_fused_full_backward": fp4_full_backward_shared_recompute_ms / fp4_full_backward_fused_ms,
        "shared_packed_over_fused_full_backward": fp4_full_backward_shared_packed_ms / fp4_full_backward_fused_ms,
        "shared_packed_over_cached_full_backward": fp4_full_backward_shared_packed_ms / fp4_full_backward_shared_cached_ms,
        "shared_packed_overlap_over_fused_full_backward": fp4_full_backward_shared_packed_overlap_ms
        / fp4_full_backward_fused_ms,
        "shared_packed_overlap_over_packed_full_backward": fp4_full_backward_shared_packed_overlap_ms
        / fp4_full_backward_shared_packed_ms,
        "shared_dual_over_fused_full_backward": fp4_full_backward_shared_dual_ms / fp4_full_backward_fused_ms,
        "shared_dual_over_packed_full_backward": fp4_full_backward_shared_dual_ms / fp4_full_backward_shared_packed_ms,
        "fp4_dx_mae_vs_fp16": float((dx_fp4 - dx_ref).abs().mean().item()),
        "fp4_dx_hybrid_mae_vs_fp16": float((dx_hybrid - dx_ref).abs().mean().item()),
    }

    os.makedirs(args.results_dir, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(args.results_dir, f"native_fp4_backward_{stamp}.json")
    latest_path = os.path.join(args.results_dir, "latest_native_fp4_backward.json")

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    with open(latest_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(json.dumps(payload, indent=2))
    print(f"Saved benchmark to: {out_path}")


if __name__ == "__main__":
    main()
