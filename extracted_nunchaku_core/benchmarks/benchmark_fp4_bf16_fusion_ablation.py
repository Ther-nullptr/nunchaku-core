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

from native_fp4 import NunchakuFP4GemmOp, NunchakuFP4LowRankOp, NunchakuFP4LowRankUnfusedOp  # noqa: E402
from native_fp4.operators import pad_tensor  # noqa: E402


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
    parser.add_argument("--lowrank-dtype", choices=["fp16", "bf16"], default="bf16")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--results-dir", type=str, default="results")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16
    lowrank_dtype = torch.float16 if args.lowrank_dtype == "fp16" else torch.bfloat16

    x = torch.randn(args.m, args.in_features, device="cuda", dtype=dtype)
    w = torch.randn(args.out_features, args.in_features, device="cuda", dtype=dtype)
    b = torch.randn(args.out_features, device="cuda", dtype=dtype)

    fp4_only = NunchakuFP4GemmOp(weight=w, bias=b)
    fused = NunchakuFP4LowRankOp(weight=w, bias=b, rank=args.rank)
    unfused = NunchakuFP4LowRankUnfusedOp(weight=w, bias=b, rank=args.rank, lowrank_dtype=lowrank_dtype)

    def fp16_fn() -> torch.Tensor:
        return torch.matmul(x, w.t()) + b

    def fp4_only_fn() -> torch.Tensor:
        return fp4_only(x)

    def fused_fn() -> torch.Tensor:
        return fused(x)

    def unfused_fn() -> torch.Tensor:
        return unfused(x)

    # low-rank branch only (unfused path)
    x2d_src = x.reshape(-1, unfused.in_features)
    x2d = x2d_src
    if unfused.k_pad != unfused.in_features:
        x2d = pad_tensor(x2d, divisor=unfused.k_pad, dim=1)

    def lowrank_only_fn() -> torch.Tensor:
        return unfused.lowrank_only_padded(x2d)

    # quick correctness context: fused vs unfused functional delta
    with torch.no_grad():
        y_fused = fused_fn().float()
        y_unfused = unfused_fn().float()
        y_ref = fp16_fn().float()

    fp16_ms = time_cuda(fp16_fn, warmup=args.warmup, iters=args.iters)
    fp4_only_ms = time_cuda(fp4_only_fn, warmup=args.warmup, iters=args.iters)
    fused_ms = time_cuda(fused_fn, warmup=args.warmup, iters=args.iters)
    unfused_ms = time_cuda(unfused_fn, warmup=args.warmup, iters=args.iters)
    lowrank_only_ms = time_cuda(lowrank_only_fn, warmup=args.warmup, iters=args.iters)

    payload = {
        "m": args.m,
        "in_features": args.in_features,
        "out_features": args.out_features,
        "rank": args.rank,
        "dtype": args.dtype,
        "lowrank_dtype": args.lowrank_dtype,
        "fp16_ms": fp16_ms,
        "fp4_only_ms": fp4_only_ms,
        "fp4_bf16_fused_ms": fused_ms,
        "fp4_bf16_unfused_ms": unfused_ms,
        "lowrank_only_unfused_ms": lowrank_only_ms,
        "fp4_only_speedup_vs_fp16": fp16_ms / fp4_only_ms,
        "fused_speedup_vs_fp16": fp16_ms / fused_ms,
        "unfused_speedup_vs_fp16": fp16_ms / unfused_ms,
        "unfused_over_fused": unfused_ms / fused_ms,
        "fused_over_fp4_only": fused_ms / fp4_only_ms,
        "unfused_over_fp4_only": unfused_ms / fp4_only_ms,
        "fused_mae_vs_fp16": (y_fused - y_ref).abs().mean().item(),
        "unfused_mae_vs_fp16": (y_unfused - y_ref).abs().mean().item(),
        "unfused_vs_fused_mae": (y_unfused - y_fused).abs().mean().item(),
    }

    os.makedirs(args.results_dir, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(args.results_dir, f"fp4_bf16_fusion_ablation_{stamp}.json")
    latest_path = os.path.join(args.results_dir, "latest_fp4_bf16_fusion_ablation.json")

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    with open(latest_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(json.dumps(payload, indent=2))
    print(f"Saved benchmark to: {out_path}")


if __name__ == "__main__":
    main()
