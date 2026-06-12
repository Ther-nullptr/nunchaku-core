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

from native_fp4.operators import _OPS, _pack_lowrank_weight_torch, pack_lowrank_weight  # noqa: E402


def time_cuda(fn, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    values = []
    for _ in range(iters):
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        values.append(start.elapsed_time(end))
    return float(sum(values) / len(values))


def parse_shape(text: str) -> tuple[int, int]:
    parts = text.lower().replace("x", ",").split(",")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("shape should be ROWSxCOLS")
    return int(parts[0]), int(parts[1])


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--shape",
        type=parse_shape,
        action="append",
        default=None,
        help="Input shape as ROWSxCOLS. Can be repeated.",
    )
    p.add_argument("--dtype", choices=["fp16", "bf16"], default="bf16")
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--iters", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--results-dir", type=str, default="results")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if not hasattr(_OPS, "pack_lowrank_weight"):
        raise RuntimeError("native pack_lowrank_weight is not available; rebuild the extension first")

    torch.manual_seed(args.seed)
    dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16
    shapes = args.shape or [(32, 4096), (4096, 32), (17, 257), (257, 17)]

    cases = []
    all_passed = True
    for rows, cols in shapes:
        weight = torch.randn(rows, cols, device="cuda", dtype=dtype)
        for down in (False, True):
            ref = _pack_lowrank_weight_torch(weight, down=down)
            out = pack_lowrank_weight(weight, down=down)
            torch.cuda.synchronize()
            matches = bool(torch.equal(out, ref))
            all_passed = all_passed and matches

            def torch_fn() -> torch.Tensor:
                return _pack_lowrank_weight_torch(weight, down=down)

            def native_fn() -> torch.Tensor:
                return pack_lowrank_weight(weight, down=down)

            torch_ms = time_cuda(torch_fn, args.warmup, args.iters)
            native_ms = time_cuda(native_fn, args.warmup, args.iters)
            cases.append(
                {
                    "shape": [rows, cols],
                    "down": down,
                    "dtype": args.dtype,
                    "output_shape": list(out.shape),
                    "matches_torch_reference": matches,
                    "torch_ms": torch_ms,
                    "native_ms": native_ms,
                    "native_speedup_vs_torch": torch_ms / native_ms,
                }
            )

    payload = {
        "cases": cases,
        "all_passed": all_passed,
    }

    os.makedirs(args.results_dir, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(args.results_dir, f"native_fp4_lora_pack_validation_{stamp}.json")
    latest_path = os.path.join(args.results_dir, "latest_native_fp4_lora_pack_validation.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    with open(latest_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(json.dumps(payload, indent=2))
    print(f"Saved validation to: {out_path}")


if __name__ == "__main__":
    main()
