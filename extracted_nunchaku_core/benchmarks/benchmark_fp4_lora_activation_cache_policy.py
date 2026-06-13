from __future__ import annotations

import argparse
import json
import os
import sys

import torch

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from native_fp4 import dequantize_fp4_activation, fp4_activation_cache_lora_down_grad  # noqa: E402
from native_fp4.operators import (  # noqa: E402
    ceil_divide,
    decode_lora_act,
    pack_lowrank_weight,
    pad_tensor,
    quantize_fp4_act_with_lora,
)


def dtype_from_name(name: str) -> torch.dtype:
    return torch.float16 if name == "fp16" else torch.bfloat16


def dtype_bytes(dtype: torch.dtype) -> int:
    if dtype in (torch.float16, torch.bfloat16):
        return 2
    if dtype == torch.float32:
        return 4
    raise ValueError(f"Unsupported dtype: {dtype}")


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
    values = []
    for _ in range(iters):
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        values.append(start.elapsed_time(end))
    return float(sum(values) / len(values))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Benchmark FP4 activation cache as a LoRA dA input policy.")
    p.add_argument("--m", type=int, default=4096)
    p.add_argument("--in-features", type=int, default=4096)
    p.add_argument("--out-features", type=int, default=4096)
    p.add_argument("--rank", type=int, default=32)
    p.add_argument("--dtype", choices=["fp16", "bf16"], default="bf16")
    p.add_argument("--lowrank-dtype", choices=["fp16", "bf16"], default="bf16")
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--iters", type=int, default=30)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--results-dir", type=str, default="results")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.rank % 16 != 0:
        raise ValueError("rank must be divisible by 16 for the current packed LoRA layout")

    torch.manual_seed(args.seed)
    dtype = dtype_from_name(args.dtype)
    lowrank_dtype = dtype_from_name(args.lowrank_dtype)

    m = args.m
    in_features = args.in_features
    out_features = args.out_features
    rank = args.rank
    m_pad = ceil_divide(m, 256) * 256
    k_pad = ceil_divide(in_features, 128) * 128

    x = torch.randn(m, in_features, device="cuda", dtype=dtype)
    dy = torch.randn(m, out_features, device="cuda", dtype=dtype)
    lora_down = torch.randn(rank, in_features, device="cuda", dtype=lowrank_dtype) * 0.02
    lora_up = torch.randn(out_features, rank, device="cuda", dtype=lowrank_dtype) * 0.02

    x_pad = x
    if k_pad != in_features:
        x_pad = pad_tensor(x, divisor=k_pad, dim=1)
    lora_down_pad = pad_tensor(lora_down, divisor=(16, 128), dim=(0, 1))
    lora_down_packed = pack_lowrank_weight(lora_down_pad, down=True).contiguous()
    smooth = torch.ones(k_pad, dtype=lowrank_dtype, device="cuda")

    qact, ascales, packed_lora_act = quantize_fp4_act_with_lora(
        x_pad.to(lowrank_dtype),
        lora_down_packed=lora_down_packed,
        smooth=smooth,
        pad_size=256,
    )
    x_hat_pad, _ = dequantize_fp4_activation(qact, ascales, dtype=lowrank_dtype, return_scales=False)
    x_hat = x_hat_pad[:m, :in_features].contiguous()
    x_lr = x.to(lowrank_dtype).contiguous()
    dy_lr = dy.to(lowrank_dtype).contiguous()
    lora_down_lr = lora_down.to(lowrank_dtype).contiguous()
    lora_up_lr = lora_up.to(lowrank_dtype).contiguous()
    dy_up = (dy_lr @ lora_up_lr).contiguous()

    d_lora_down_ref = dy_up.t() @ x_lr
    d_lora_down_fp4_cache = dy_up.t() @ x_hat
    d_lora_down_fp4_cache_fused = fp4_activation_cache_lora_down_grad(
        qact,
        ascales,
        dy_up,
        in_features=in_features,
    )
    lora_act_ref = x_lr @ lora_down_lr.t()
    lora_act_decoded = decode_lora_act(packed_lora_act, lowrank_dtype)[:m, :rank].contiguous()
    d_lora_up_ref = dy_lr.t() @ lora_act_ref
    d_lora_up_decoded = dy_lr.t() @ lora_act_decoded

    def quantize_cache_fn() -> None:
        q, s, a = quantize_fp4_act_with_lora(
            x_pad.to(lowrank_dtype),
            lora_down_packed=lora_down_packed,
            smooth=smooth,
            pad_size=256,
        )
        _ = q.sum() + s.view(torch.uint8).sum() + a.float().sum()

    def dequant_only_fn() -> None:
        out, _ = dequantize_fp4_activation(qact, ascales, dtype=lowrank_dtype, return_scales=False)
        _ = out[:m, :in_features].float().sum()

    def bf16_d_lora_down_fn() -> None:
        out = dy_up.t() @ x_lr
        _ = out.float().sum()

    def fp4_cache_d_lora_down_fn() -> None:
        out, _ = dequantize_fp4_activation(qact, ascales, dtype=lowrank_dtype, return_scales=False)
        grad = dy_up.t() @ out[:m, :in_features].contiguous()
        _ = grad.float().sum()

    def fp4_cache_fused_d_lora_down_fn() -> None:
        grad = fp4_activation_cache_lora_down_grad(qact, ascales, dy_up, in_features=in_features)
        _ = grad.float().sum()

    def dy_up_fn() -> None:
        out = dy_lr @ lora_up_lr
        _ = out.float().sum()

    quantize_cache_ms = time_cuda(quantize_cache_fn, args.warmup, args.iters)
    dequant_only_ms = time_cuda(dequant_only_fn, args.warmup, args.iters)
    bf16_d_lora_down_ms = time_cuda(bf16_d_lora_down_fn, args.warmup, args.iters)
    fp4_cache_d_lora_down_ms = time_cuda(fp4_cache_d_lora_down_fn, args.warmup, args.iters)
    fp4_cache_fused_d_lora_down_ms = time_cuda(fp4_cache_fused_d_lora_down_fn, args.warmup, args.iters)
    dy_up_ms = time_cuda(dy_up_fn, args.warmup, args.iters)

    bf16_x_cache_bytes = m * in_features * dtype_bytes(dtype)
    bf16_x_cache_padded_bytes = m_pad * k_pad * dtype_bytes(dtype)
    fp4_cache_bytes = qact.numel() + ascales.numel()
    fp4_cache_ideal_bytes = m * in_features // 2 + ceil_divide(m * in_features, 16)
    lora_act_cache_bytes = m_pad * rank * dtype_bytes(lowrank_dtype)

    payload = {
        "shape": {
            "m": m,
            "m_pad": m_pad,
            "in_features": in_features,
            "k_pad": k_pad,
            "out_features": out_features,
            "rank": rank,
            "dtype": args.dtype,
            "lowrank_dtype": args.lowrank_dtype,
        },
        "cache_bytes": {
            "bf16_or_fp16_x_unpadded": bf16_x_cache_bytes,
            "bf16_or_fp16_x_padded": bf16_x_cache_padded_bytes,
            "fp4_qact_plus_fp8_ascales_padded": fp4_cache_bytes,
            "fp4_qact_plus_fp8_ascales_ideal_unpadded": fp4_cache_ideal_bytes,
            "lora_act_padded": lora_act_cache_bytes,
        },
        "derived": {
            "fp4_cache_reduction_vs_unpadded_x": bf16_x_cache_bytes / fp4_cache_bytes,
            "fp4_cache_reduction_vs_padded_x": bf16_x_cache_padded_bytes / fp4_cache_bytes,
            "naive_fp4_cache_materializes_dense_x_hat_in_backward": True,
        },
        "implementation": {
            "fp4_cache_fused_d_lora_down": (
                "cuda_rank_tiled_kvec3_rvec16_rank_le_32"
                if rank <= 32
                else (
                    "cuda_rank_tiled_kvec3_rvec32_threads128_rank_le_256"
                    if rank <= 256
                    else "cuda_rank_tiled_kvec2_rvec16"
                )
            ),
        },
        "latency_ms": {
            "forward_quantize_cache": quantize_cache_ms,
            "dy_up": dy_up_ms,
            "bf16_or_fp16_saved_x_d_lora_down": bf16_d_lora_down_ms,
            "fp4_cache_dequant_only": dequant_only_ms,
            "fp4_cache_dequant_plus_d_lora_down": fp4_cache_d_lora_down_ms,
            "fp4_cache_fused_d_lora_down": fp4_cache_fused_d_lora_down_ms,
        },
        "speedups": {
            "fp4_cache_dequant_plus_d_lora_down_vs_saved_x_d_lora_down": (
                bf16_d_lora_down_ms / fp4_cache_d_lora_down_ms
            ),
            "fp4_cache_fused_d_lora_down_vs_saved_x_d_lora_down": (
                bf16_d_lora_down_ms / fp4_cache_fused_d_lora_down_ms
            ),
            "fp4_cache_fused_d_lora_down_vs_dequant_plus_d_lora_down": (
                fp4_cache_d_lora_down_ms / fp4_cache_fused_d_lora_down_ms
            ),
            "fp4_cache_dequant_only_over_saved_x_d_lora_down": dequant_only_ms / bf16_d_lora_down_ms,
        },
        "errors": {
            "x_hat_vs_x": tensor_error(x_hat, x_lr),
            "d_lora_down_fp4_cache_vs_saved_x": tensor_error(d_lora_down_fp4_cache, d_lora_down_ref),
            "d_lora_down_fp4_cache_fused_vs_saved_x": tensor_error(d_lora_down_fp4_cache_fused, d_lora_down_ref),
            "d_lora_down_fp4_cache_fused_vs_dequant_gemm": tensor_error(
                d_lora_down_fp4_cache_fused,
                d_lora_down_fp4_cache,
            ),
            "lora_act_decoded_vs_dense": tensor_error(lora_act_decoded, lora_act_ref),
            "d_lora_up_decoded_vs_dense": tensor_error(d_lora_up_decoded, d_lora_up_ref),
        },
    }

    os.makedirs(args.results_dir, exist_ok=True)
    out_path = os.path.join(args.results_dir, "latest_fp4_lora_activation_cache_policy.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(json.dumps(payload, indent=2))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
