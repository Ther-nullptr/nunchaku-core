from __future__ import annotations

import argparse
import json
import os
import time

import torch

from nunchaku_core import Int4OnlyLinearCore, fp16_linear_reference


def _time_cuda(fn, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    times = []
    for _ in range(iters):
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))

    return float(sum(times) / len(times))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, default=4096)
    parser.add_argument("--in-features", type=int, default=4096)
    parser.add_argument("--out-features", type=int, default=4096)
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
        raise RuntimeError("CUDA is required for this benchmark")

    dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16
    x = torch.randn(args.m, args.in_features, device="cuda", dtype=dtype)
    w = torch.randn(args.out_features, args.in_features, device="cuda", dtype=dtype)
    b = torch.randn(args.out_features, device="cuda", dtype=dtype)

    int4_only = Int4OnlyLinearCore.from_fp16_weight(w, b)

    def fp16_fn() -> torch.Tensor:
        return fp16_linear_reference(x, w, b)

    def int4_fn() -> torch.Tensor:
        return int4_only(x)

    with torch.no_grad():
        y_ref = fp16_fn().float()
        y_int4 = int4_fn().float()

    fp16_ms = _time_cuda(fp16_fn, warmup=args.warmup, iters=args.iters)
    int4_ms = _time_cuda(int4_fn, warmup=args.warmup, iters=args.iters)

    payload = {
        "m": args.m,
        "in_features": args.in_features,
        "out_features": args.out_features,
        "dtype": args.dtype,
        "fp16_ms": fp16_ms,
        "int4_only_ms": int4_ms,
        "int4_only_speedup_vs_fp16": fp16_ms / int4_ms,
        "int4_only_mae_vs_fp16": (y_int4 - y_ref).abs().mean().item(),
    }

    os.makedirs(args.results_dir, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(args.results_dir, f"int4_only_{stamp}.json")
    latest_path = os.path.join(args.results_dir, "latest_int4_only.json")

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    with open(latest_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(json.dumps(payload, indent=2))
    print(f"Saved benchmark to: {out_path}")


if __name__ == "__main__":
    main()
