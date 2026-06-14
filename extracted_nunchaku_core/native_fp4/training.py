from __future__ import annotations

import math
import os
from typing import Literal

import torch
from torch.utils.checkpoint import checkpoint

from .layout import (
    build_backward_scales_from_forward_quant,
    dequantize_fp4_activation,
    dequantize_fp4_weight,
    unpack_fp4_weight_scales,
)
from .operators import (
    FP4BackwardWeightPolicy,
    NunchakuFP4BackwardDXOp,
    NunchakuFP4GemmOp,
    ceil_divide,
    decode_lora_act,
    fp4_activation_cache_lora_down_grad,
    pack_lowrank_weight,
    pad_tensor,
    quantize_fp4_act_with_lora,
    quantize_fp4_act_with_lora_dual,
)

LoRAInitMode = Literal["zero", "gaussian", "residual_svd"]
FrozenResidualInitMode = Literal["none", "residual_svd"]
ResidualSVDMethod = Literal["full_svd", "svd_lowrank"]
FP4ActivationCacheDLoRADownBackend = Literal["fused", "dequant_gemm"]
DEFAULT_OVERLAP_LORA_GRAD_MIN_ROWS = 4096
_ENABLE_OVERLAP_RESOURCE_CACHE = os.environ.get("NUNCHAKU_FP4_LORA_CACHE_OVERLAP_RESOURCES") == "1"


class _FP4LoRAOverlapResources:
    """Reusable CUDA streams/events for one module/device/current-stream pair."""

    def __init__(self, device: torch.device, current_stream_handle: int):
        if device.type != "cuda":
            raise ValueError("overlap resources require a CUDA device")
        device_index = torch.cuda.current_device() if device.index is None else int(device.index)
        self.device_index = device_index
        self.current_stream_handle = int(current_stream_handle)
        self.device = torch.device("cuda", device_index)
        with torch.cuda.device(self.device):
            self.repack_stream = torch.cuda.Stream(device=self.device)
            self.dx_stream = torch.cuda.Stream(device=self.device)
            self.up_stream = torch.cuda.Stream(device=self.device)
            self.down_stream = torch.cuda.Stream(device=self.device)
            self.residual_stream = torch.cuda.Stream(device=self.device)
            self.repack_done = torch.cuda.Event()
            self.quant_done = torch.cuda.Event()


def _get_lora_overlap_resources(
    owner: object,
    device: torch.device,
    current_stream: torch.cuda.Stream,
) -> _FP4LoRAOverlapResources:
    device = torch.device(device)
    device_index = torch.cuda.current_device() if device.index is None else int(device.index)
    stream_handle = int(current_stream.cuda_stream)
    key = (device_index, stream_handle)
    resources_by_key = getattr(owner, "_lora_overlap_resources_by_key", None)
    if resources_by_key is None:
        resources_by_key = {}
        setattr(owner, "_lora_overlap_resources_by_key", resources_by_key)
    resources = resources_by_key.get(key)
    if resources is None:
        resources = _FP4LoRAOverlapResources(torch.device("cuda", device_index), stream_handle)
        resources_by_key[key] = resources
    return resources


def _fp4_activation_cache_lora_down_grad(
    qact: torch.Tensor,
    ascales: torch.Tensor,
    dy_up: torch.Tensor,
    *,
    in_features: int,
    backend: FP4ActivationCacheDLoRADownBackend,
) -> torch.Tensor:
    if backend == "fused":
        return fp4_activation_cache_lora_down_grad(
            qact,
            ascales,
            dy_up.contiguous(),
            in_features=in_features,
        )
    if backend == "dequant_gemm":
        x_hat_pad, _ = dequantize_fp4_activation(qact, ascales, dtype=dy_up.dtype, return_scales=False)
        x_hat = x_hat_pad[: dy_up.shape[0], :in_features].contiguous()
        return torch.matmul(dy_up.t(), x_hat)
    raise RuntimeError(f"Unsupported FP4 activation-cache dA backend: {backend}")


class _FP4LoRALinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        lora_down: torch.Tensor,
        lora_up: torch.Tensor,
        frozen_residual_down: torch.Tensor,
        frozen_residual_up: torch.Tensor,
        bias: torch.Tensor | None,
        fp4_forward_op: NunchakuFP4GemmOp,
        fp4_backward_op: NunchakuFP4BackwardDXOp,
        scaling: float,
        frozen_residual_scaling: float,
        lowrank_dtype: torch.dtype,
        cache_lora_act: bool,
        fuse_lowrank_forward: bool,
        fuse_lora_dx: bool,
        fuse_frozen_residual_dx: bool,
        reuse_fused_dy_up_for_d_lora_down: bool,
        overlap_lora_grad: bool,
        overlap_lora_grad_min_rows: int,
        fp4_activation_cache_d_lora_down: bool,
        fp4_activation_cache_d_lora_down_backend: FP4ActivationCacheDLoRADownBackend,
        lora_up_zero_fast_path_active: bool,
        packed_lora_dx: tuple[torch.Tensor, torch.Tensor] | None,
        packed_lora_forward: tuple[torch.Tensor, torch.Tensor] | None,
    ) -> torch.Tensor:
        if x.shape[-1] != fp4_forward_op.in_features:
            raise ValueError(f"Expected input last dim = {fp4_forward_op.in_features}, got {x.shape[-1]}")

        x2d = x.reshape(-1, fp4_forward_op.in_features)
        x_lr = x2d.to(lowrank_dtype)
        down_lr = lora_down.to(lowrank_dtype)
        up_lr = lora_up.to(lowrank_dtype)
        has_frozen_residual = frozen_residual_down.numel() > 0 and frozen_residual_up.numel() > 0
        use_native_lora_forward = bool(
            fuse_lowrank_forward
            and packed_lora_forward is not None
            and x2d.dtype == fp4_forward_op.compute_dtype
            and lowrank_dtype == fp4_forward_op.compute_dtype
        )

        use_fp4_act_cache = bool(fp4_activation_cache_d_lora_down)
        lora_up_zero_fast_path_active = bool(lora_up_zero_fast_path_active)
        if lora_up_zero_fast_path_active:
            if use_fp4_act_cache:
                x2d_for_fp4 = x2d
                if fp4_forward_op.k_pad != fp4_forward_op.in_features:
                    x2d_for_fp4 = pad_tensor(x2d_for_fp4, divisor=fp4_forward_op.k_pad, dim=1)
                qact, ascales = fp4_forward_op.quantize_input(x2d_for_fp4)
                y_main_pad = fp4_forward_op.forward_prequantized(qact, ascales)
                y = y_main_pad[: x2d.shape[0], : fp4_forward_op.out_features].reshape(
                    *x.shape[:-1],
                    fp4_forward_op.out_features,
                )
            else:
                qact = torch.empty(0, device=x.device, dtype=torch.uint8)
                ascales = torch.empty(0, device=x.device, dtype=torch.float8_e4m3fn)
                y = fp4_forward_op(x)
            if cache_lora_act:
                lora_act = torch.matmul(x_lr, down_lr.t())
            else:
                lora_act = torch.empty(0, device=x.device, dtype=lowrank_dtype)
            if has_frozen_residual:
                residual_act = torch.matmul(x_lr, frozen_residual_down.to(lowrank_dtype).t())
                residual_out = torch.matmul(residual_act, frozen_residual_up.to(lowrank_dtype).t())
                y = y + residual_out.mul(float(frozen_residual_scaling)).to(y.dtype).reshape_as(y)
        elif use_native_lora_forward:
            lora_down_fwd_packed, lora_up_fwd_packed = packed_lora_forward
            x2d_for_fp4 = x2d
            if fp4_forward_op.k_pad != fp4_forward_op.in_features:
                x2d_for_fp4 = pad_tensor(x2d_for_fp4, divisor=fp4_forward_op.k_pad, dim=1)
            if cache_lora_act:
                qact, ascales, packed_lora_act, dense_lora_act = quantize_fp4_act_with_lora_dual(
                    x2d_for_fp4,
                    lora_down_packed=lora_down_fwd_packed,
                    smooth=fp4_forward_op.smooth,
                    pad_size=256,
                    dense_dtype=lowrank_dtype,
                )
                lora_act = dense_lora_act[: x2d.shape[0], : lora_down.shape[0]].contiguous()
            else:
                qact, ascales, packed_lora_act = quantize_fp4_act_with_lora(
                    x2d_for_fp4,
                    lora_down_packed=lora_down_fwd_packed,
                    smooth=fp4_forward_op.smooth,
                    pad_size=256,
                )
                lora_act = torch.empty(0, device=x.device, dtype=lowrank_dtype)
            y_pad = fp4_forward_op.forward_prequantized_with_lora(
                qact,
                ascales,
                packed_lora_act,
                lora_up_fwd_packed,
                [1.0] * ceil_divide(lora_down_fwd_packed.shape[1], 16),
            )
            y = y_pad[: x2d.shape[0], : fp4_forward_op.out_features].reshape(
                *x.shape[:-1],
                fp4_forward_op.out_features,
            )
            if has_frozen_residual and lora_down_fwd_packed.shape[1] == lora_down.shape[0]:
                residual_act = torch.matmul(x_lr, frozen_residual_down.to(lowrank_dtype).t())
                residual_out = torch.matmul(residual_act, frozen_residual_up.to(lowrank_dtype).t())
                y = y + residual_out.mul(float(frozen_residual_scaling)).to(y.dtype).reshape_as(y)
        elif use_fp4_act_cache:
            x2d_for_fp4 = x2d
            if fp4_forward_op.k_pad != fp4_forward_op.in_features:
                x2d_for_fp4 = pad_tensor(x2d_for_fp4, divisor=fp4_forward_op.k_pad, dim=1)
            qact, ascales = fp4_forward_op.quantize_input(x2d_for_fp4)
            y_main_pad = fp4_forward_op.forward_prequantized(qact, ascales)
            y_main = y_main_pad[: x2d.shape[0], : fp4_forward_op.out_features].reshape(
                *x.shape[:-1],
                fp4_forward_op.out_features,
            )
            lora_act = torch.matmul(x_lr, down_lr.t())
            lora_out = torch.matmul(lora_act, up_lr.t()).mul(float(scaling)).to(y_main.dtype)
            lowrank_out = lora_out
            if has_frozen_residual:
                residual_act = torch.matmul(x_lr, frozen_residual_down.to(lowrank_dtype).t())
                residual_out = torch.matmul(residual_act, frozen_residual_up.to(lowrank_dtype).t())
                lowrank_out = lowrank_out + residual_out.mul(float(frozen_residual_scaling)).to(y_main.dtype)
            y = y_main + lowrank_out.reshape(*x.shape[:-1], fp4_forward_op.out_features)
        else:
            qact = torch.empty(0, device=x.device, dtype=torch.uint8)
            ascales = torch.empty(0, device=x.device, dtype=torch.float8_e4m3fn)
            y_main = fp4_forward_op(x)
            if has_frozen_residual and fuse_lowrank_forward:
                residual_down_lr = frozen_residual_down.to(lowrank_dtype)
                residual_up_lr = frozen_residual_up.to(lowrank_dtype)
                combined_down = torch.cat((down_lr, residual_down_lr), dim=0)
                combined_up = torch.cat(
                    (
                        up_lr.mul(float(scaling)),
                        residual_up_lr.mul(float(frozen_residual_scaling)),
                    ),
                    dim=1,
                )
                combined_act = torch.matmul(x_lr, combined_down.t())
                lora_act = combined_act[:, : lora_down.shape[0]]
                lowrank_out = torch.matmul(combined_act, combined_up.t()).to(y_main.dtype)
            else:
                lora_act = torch.matmul(x_lr, down_lr.t())
                lora_out = torch.matmul(lora_act, up_lr.t()).mul(float(scaling)).to(y_main.dtype)
                lowrank_out = lora_out
                if has_frozen_residual:
                    residual_act = torch.matmul(x_lr, frozen_residual_down.to(lowrank_dtype).t())
                    residual_out = torch.matmul(residual_act, frozen_residual_up.to(lowrank_dtype).t())
                    lowrank_out = lowrank_out + residual_out.mul(float(frozen_residual_scaling)).to(y_main.dtype)
            y = y_main + lowrank_out.reshape(*x.shape[:-1], fp4_forward_op.out_features)
        if bias is not None:
            y = y + bias.to(y.dtype)

        if cache_lora_act:
            saved_lora_act = lora_act
        else:
            saved_lora_act = torch.empty(0, device=x.device, dtype=lowrank_dtype)
        saved_x = torch.empty(0, device=x.device, dtype=x.dtype) if use_fp4_act_cache else x
        ctx.save_for_backward(
            saved_x,
            qact,
            ascales,
            lora_down,
            lora_up,
            frozen_residual_down,
            frozen_residual_up,
            saved_lora_act,
        )
        ctx.fp4_backward_op = fp4_backward_op
        ctx.scaling = float(scaling)
        ctx.frozen_residual_scaling = float(frozen_residual_scaling)
        ctx.lowrank_dtype = lowrank_dtype
        ctx.cache_lora_act = bool(cache_lora_act)
        ctx.fuse_lowrank_forward = bool(fuse_lowrank_forward)
        ctx.fuse_lora_dx = bool(fuse_lora_dx)
        ctx.fuse_frozen_residual_dx = bool(fuse_frozen_residual_dx)
        ctx.reuse_fused_dy_up_for_d_lora_down = bool(reuse_fused_dy_up_for_d_lora_down)
        ctx.overlap_lora_grad = bool(overlap_lora_grad)
        ctx.overlap_lora_grad_min_rows = int(overlap_lora_grad_min_rows)
        ctx.fp4_activation_cache_d_lora_down = use_fp4_act_cache
        ctx.fp4_activation_cache_d_lora_down_backend = fp4_activation_cache_d_lora_down_backend
        ctx.lora_up_zero_fast_path_active = lora_up_zero_fast_path_active
        ctx.packed_lora_dx = packed_lora_dx
        ctx.has_frozen_residual = bool(has_frozen_residual)
        ctx.has_bias = bias is not None
        ctx.in_features = fp4_forward_op.in_features
        ctx.out_features = fp4_forward_op.out_features
        ctx.x_shape = tuple(x.shape)
        ctx.x_dtype = x.dtype
        return y

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (
            saved_x,
            qact,
            ascales,
            lora_down,
            lora_up,
            frozen_residual_down,
            frozen_residual_up,
            saved_lora_act,
        ) = ctx.saved_tensors
        dy = grad_output.contiguous()

        if ctx.fp4_activation_cache_d_lora_down:
            x2d = None
            rows = math.prod(ctx.x_shape[:-1]) if len(ctx.x_shape) > 1 else 1
        else:
            x2d = saved_x.reshape(-1, ctx.in_features)
            rows = x2d.shape[0]
        dy2d = dy.reshape(-1, ctx.out_features)

        if ctx.lora_up_zero_fast_path_active:
            if ctx.cache_lora_act:
                lora_act = saved_lora_act
            else:
                if x2d is None:
                    raise RuntimeError("cache_lora_act=True is required for zero-up fast path with FP4 activation cache")
                x_lr_for_act = x2d.to(ctx.lowrank_dtype)
                down_lr_for_act = lora_down.to(ctx.lowrank_dtype)
                lora_act = torch.matmul(x_lr_for_act, down_lr_for_act.t())

            should_overlap_zero_up = (
                ctx.overlap_lora_grad
                and not ctx.fp4_activation_cache_d_lora_down
                and rows >= ctx.overlap_lora_grad_min_rows
                and dy.is_cuda
            )
            if should_overlap_zero_up:
                dx, d_lora_up = _zero_lora_up_backward_overlap(
                    dy=dy,
                    lora_act=lora_act,
                    lora_up=lora_up,
                    fp4_backward_op=ctx.fp4_backward_op,
                    scaling=ctx.scaling,
                    lowrank_dtype=ctx.lowrank_dtype,
                    in_features=ctx.in_features,
                    out_features=ctx.out_features,
                    frozen_residual_down=frozen_residual_down if ctx.has_frozen_residual else None,
                    frozen_residual_up=frozen_residual_up if ctx.has_frozen_residual else None,
                    frozen_residual_scaling=ctx.frozen_residual_scaling,
                )
            else:
                dy_lr = dy2d.to(ctx.lowrank_dtype)
                dx = ctx.fp4_backward_op(dy)
                if ctx.has_frozen_residual:
                    residual_down_lr = frozen_residual_down.to(ctx.lowrank_dtype)
                    residual_up_lr = frozen_residual_up.to(ctx.lowrank_dtype)
                    dy_residual_up = torch.matmul(dy_lr, residual_up_lr)
                    dx_residual = torch.matmul(dy_residual_up, residual_down_lr).mul(ctx.frozen_residual_scaling)
                    dx = dx.to(dx_residual.dtype) + dx_residual.reshape(ctx.x_shape)
                d_lora_up = torch.matmul(dy_lr.t(), lora_act.to(ctx.lowrank_dtype)).mul(ctx.scaling).to(lora_up.dtype)
            d_lora_down = torch.zeros_like(lora_down)
            d_bias = dy2d.sum(dim=0).to(grad_output.dtype) if ctx.has_bias else None
            return (
                dx.to(ctx.x_dtype),
                d_lora_down,
                d_lora_up,
                None,
                None,
                d_bias,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
            )

        should_overlap_lora_grad = (
            ctx.overlap_lora_grad
            and not ctx.fp4_activation_cache_d_lora_down
            and x2d is not None
            and rows >= ctx.overlap_lora_grad_min_rows
        )
        if should_overlap_lora_grad:
            if ctx.cache_lora_act:
                lora_act = saved_lora_act
            else:
                x_lr_for_act = x2d.to(ctx.lowrank_dtype)
                down_lr_for_act = lora_down.to(ctx.lowrank_dtype)
                lora_act = torch.matmul(x_lr_for_act, down_lr_for_act.t())
            if ctx.reuse_fused_dy_up_for_d_lora_down:
                dx, d_lora_down, d_lora_up = _fused_lora_backward_overlap_reuse(
                    dy=dy,
                    x2d=x2d,
                    lora_act=lora_act,
                    lora_down=lora_down,
                    lora_up=lora_up,
                    fp4_backward_op=ctx.fp4_backward_op,
                    scaling=ctx.scaling,
                    lowrank_dtype=ctx.lowrank_dtype,
                    in_features=ctx.in_features,
                    out_features=ctx.out_features,
                    packed_lora_dx=ctx.packed_lora_dx,
                )
            else:
                dx, d_lora_down, d_lora_up = _fused_lora_backward_overlap_exact(
                    dy=dy,
                    x2d=x2d,
                    lora_act=lora_act,
                    lora_down=lora_down,
                    lora_up=lora_up,
                    fp4_backward_op=ctx.fp4_backward_op,
                    scaling=ctx.scaling,
                    lowrank_dtype=ctx.lowrank_dtype,
                    in_features=ctx.in_features,
                    out_features=ctx.out_features,
                    packed_lora_dx=ctx.packed_lora_dx,
                    frozen_residual_down=frozen_residual_down
                    if ctx.has_frozen_residual and not ctx.fuse_frozen_residual_dx
                    else None,
                    frozen_residual_up=frozen_residual_up
                    if ctx.has_frozen_residual and not ctx.fuse_frozen_residual_dx
                    else None,
                    frozen_residual_scaling=ctx.frozen_residual_scaling,
                )
            d_bias = dy2d.sum(dim=0).to(grad_output.dtype) if ctx.has_bias else None
            return (
                dx.to(ctx.x_dtype),
                d_lora_down,
                d_lora_up,
                None,
                None,
                d_bias,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
            )

        x_lr = None if ctx.fp4_activation_cache_d_lora_down else x2d.to(ctx.lowrank_dtype)
        dy_lr = dy2d.to(ctx.lowrank_dtype)
        down_lr = lora_down.to(ctx.lowrank_dtype)
        up_lr = lora_up.to(ctx.lowrank_dtype)

        if ctx.cache_lora_act:
            lora_act = saved_lora_act
        else:
            if x_lr is None:
                raise RuntimeError("cache_lora_act=True is required when using FP4 activation-cache dA")
            lora_act = torch.matmul(x_lr, down_lr.t())

        dy_up = None
        if ctx.fuse_lora_dx:
            fused_dx = _fused_lora_dx(
                dy=dy,
                lora_down=lora_down,
                lora_up=lora_up,
                fp4_backward_op=ctx.fp4_backward_op,
                scaling=ctx.scaling,
                lowrank_dtype=ctx.lowrank_dtype,
                in_features=ctx.in_features,
                out_features=ctx.out_features,
                packed_lora_dx=ctx.packed_lora_dx,
                return_dy_up=ctx.reuse_fused_dy_up_for_d_lora_down,
                frozen_residual_down=frozen_residual_down
                if ctx.has_frozen_residual and ctx.fuse_frozen_residual_dx
                else None,
                frozen_residual_up=frozen_residual_up if ctx.has_frozen_residual and ctx.fuse_frozen_residual_dx else None,
                frozen_residual_scaling=ctx.frozen_residual_scaling,
            )
            if ctx.reuse_fused_dy_up_for_d_lora_down:
                dx, dy_up = fused_dx
            else:
                dx = fused_dx
        else:
            dy_up = torch.matmul(dy_lr, up_lr)
            dx_main = ctx.fp4_backward_op(dy)
            dx_lora = torch.matmul(dy_up, down_lr).mul(ctx.scaling)
            dx = dx_main.to(dx_lora.dtype) + dx_lora.reshape(ctx.x_shape)

        if dy_up is None:
            dy_up = torch.matmul(dy_lr, up_lr)
        d_lora_up = torch.matmul(dy_lr.t(), lora_act).mul(ctx.scaling).to(lora_up.dtype)
        if ctx.fp4_activation_cache_d_lora_down:
            d_lora_down = _fp4_activation_cache_lora_down_grad(
                qact,
                ascales,
                dy_up,
                in_features=ctx.in_features,
                backend=ctx.fp4_activation_cache_d_lora_down_backend,
            )
            d_lora_down = d_lora_down.mul(ctx.scaling).to(lora_down.dtype)
        else:
            d_lora_down = torch.matmul(dy_up.t(), x_lr).mul(ctx.scaling).to(lora_down.dtype)
        if ctx.has_frozen_residual and not (ctx.fuse_lora_dx and ctx.fuse_frozen_residual_dx):
            residual_down_lr = frozen_residual_down.to(ctx.lowrank_dtype)
            residual_up_lr = frozen_residual_up.to(ctx.lowrank_dtype)
            dy_residual_up = torch.matmul(dy_lr, residual_up_lr)
            dx_residual = torch.matmul(dy_residual_up, residual_down_lr).mul(ctx.frozen_residual_scaling)
            dx = dx.to(dx_residual.dtype) + dx_residual.reshape(ctx.x_shape)
        d_bias = dy2d.sum(dim=0).to(grad_output.dtype) if ctx.has_bias else None

        return (
            dx.to(ctx.x_dtype),
            d_lora_down,
            d_lora_up,
            None,
            None,
            d_bias,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


def _pack_lora_dx_factors(
    lora_down: torch.Tensor,
    lora_up: torch.Tensor,
    fp4_backward_op: NunchakuFP4BackwardDXOp,
    scaling: float,
    lowrank_dtype: torch.dtype,
    frozen_residual_down: torch.Tensor | None = None,
    frozen_residual_up: torch.Tensor | None = None,
    frozen_residual_scaling: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    del lowrank_dtype
    packed_dtype = fp4_backward_op.compute_dtype
    b_t = lora_up.t().contiguous().to(packed_dtype)
    a_t = lora_down.t().contiguous().to(packed_dtype).mul(float(scaling))
    if (
        frozen_residual_down is not None
        and frozen_residual_up is not None
        and frozen_residual_down.numel() > 0
        and frozen_residual_up.numel() > 0
    ):
        residual_b_t = frozen_residual_up.t().contiguous().to(packed_dtype)
        residual_a_t = frozen_residual_down.t().contiguous().to(packed_dtype).mul(float(frozen_residual_scaling))
        b_t = torch.cat((b_t, residual_b_t), dim=0)
        a_t = torch.cat((a_t, residual_a_t), dim=1)
    if b_t.shape[1] != fp4_backward_op.n_pad:
        b_t = pad_tensor(b_t, divisor=fp4_backward_op.n_pad, dim=1)
    if a_t.shape[0] != fp4_backward_op.k_pad:
        a_t = pad_tensor(a_t, divisor=fp4_backward_op.k_pad, dim=0)
    return pack_lowrank_weight(b_t, down=True).contiguous(), pack_lowrank_weight(a_t, down=False).contiguous()


def _pack_trainable_lora_for_fused_dx(
    lora_down: torch.Tensor,
    lora_up: torch.Tensor,
    fp4_backward_op: NunchakuFP4BackwardDXOp,
    scaling: float,
    lowrank_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    return _pack_lora_dx_factors(
        lora_down=lora_down,
        lora_up=lora_up,
        fp4_backward_op=fp4_backward_op,
        scaling=scaling,
        lowrank_dtype=lowrank_dtype,
    )


def _pack_lora_forward_factors(
    lora_down: torch.Tensor,
    lora_up: torch.Tensor,
    fp4_forward_op: NunchakuFP4GemmOp,
    scaling: float,
    frozen_residual_down: torch.Tensor | None = None,
    frozen_residual_up: torch.Tensor | None = None,
    frozen_residual_scaling: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    packed_dtype = fp4_forward_op.compute_dtype
    down = lora_down.contiguous().to(packed_dtype)
    up = lora_up.contiguous().to(packed_dtype).mul(float(scaling))
    if (
        frozen_residual_down is not None
        and frozen_residual_up is not None
        and frozen_residual_down.numel() > 0
        and frozen_residual_up.numel() > 0
    ):
        residual_down = frozen_residual_down.contiguous().to(packed_dtype)
        residual_up = frozen_residual_up.contiguous().to(packed_dtype).mul(float(frozen_residual_scaling))
        down = torch.cat((down, residual_down), dim=0)
        up = torch.cat((up, residual_up), dim=1)
    if down.shape[1] != fp4_forward_op.k_pad:
        down = pad_tensor(down, divisor=fp4_forward_op.k_pad, dim=1)
    if up.shape[0] != fp4_forward_op.n_pad:
        up = pad_tensor(up, divisor=fp4_forward_op.n_pad, dim=0)
    return pack_lowrank_weight(down, down=True).contiguous(), pack_lowrank_weight(up, down=False).contiguous()


def _zero_lora_up_backward_overlap(
    dy: torch.Tensor,
    lora_act: torch.Tensor,
    lora_up: torch.Tensor,
    fp4_backward_op: NunchakuFP4BackwardDXOp,
    scaling: float,
    lowrank_dtype: torch.dtype,
    in_features: int,
    out_features: int,
    frozen_residual_down: torch.Tensor | None = None,
    frozen_residual_up: torch.Tensor | None = None,
    frozen_residual_scaling: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not dy.is_cuda:
        raise ValueError("zero-up overlap requires CUDA tensors")
    has_frozen_residual = (
        frozen_residual_down is not None
        and frozen_residual_up is not None
        and frozen_residual_down.numel() > 0
        and frozen_residual_up.numel() > 0
    )

    orig_shape = dy.shape
    dy2d_src = dy.reshape(-1, out_features)

    current_stream = torch.cuda.current_stream(device=dy.device)
    if _ENABLE_OVERLAP_RESOURCE_CACHE:
        resources = _get_lora_overlap_resources(fp4_backward_op, torch.device(dy.device), current_stream)
        dx_stream = resources.dx_stream
        up_stream = resources.up_stream
        residual_stream = resources.residual_stream if has_frozen_residual else None
    else:
        dx_stream = torch.cuda.Stream(device=dy.device)
        up_stream = torch.cuda.Stream(device=dy.device)
        residual_stream = torch.cuda.Stream(device=dy.device) if has_frozen_residual else None

    dx_stream.wait_stream(current_stream)
    with torch.cuda.stream(dx_stream):
        dx = fp4_backward_op(dy)

    up_stream.wait_stream(current_stream)
    with torch.cuda.stream(up_stream):
        d_lora_up = torch.matmul(dy2d_src.to(lowrank_dtype).t(), lora_act.to(lowrank_dtype))
        d_lora_up = d_lora_up.mul(float(scaling)).to(lora_up.dtype)

    if residual_stream is not None:
        residual_stream.wait_stream(current_stream)
        with torch.cuda.stream(residual_stream):
            residual_down_lr = frozen_residual_down.to(lowrank_dtype)
            residual_up_lr = frozen_residual_up.to(lowrank_dtype)
            dy_residual_up = torch.matmul(dy2d_src.to(lowrank_dtype), residual_up_lr)
            dx_residual = torch.matmul(dy_residual_up, residual_down_lr).mul(float(frozen_residual_scaling))

    current_stream.wait_stream(dx_stream)
    current_stream.wait_stream(up_stream)
    if residual_stream is not None:
        current_stream.wait_stream(residual_stream)

    if has_frozen_residual:
        dx = dx.to(dx_residual.dtype) + dx_residual.reshape(*orig_shape[:-1], in_features)
    return dx, d_lora_up


def _fused_lora_dx(
    dy: torch.Tensor,
    lora_down: torch.Tensor,
    lora_up: torch.Tensor,
    fp4_backward_op: NunchakuFP4BackwardDXOp,
    scaling: float,
    lowrank_dtype: torch.dtype,
    in_features: int,
    out_features: int,
    packed_lora_dx: tuple[torch.Tensor, torch.Tensor] | None = None,
    return_dy_up: bool = False,
    frozen_residual_down: torch.Tensor | None = None,
    frozen_residual_up: torch.Tensor | None = None,
    frozen_residual_scaling: float = 1.0,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    orig_shape = dy.shape
    dy2d_src = dy.reshape(-1, out_features)
    dy2d = dy2d_src
    if fp4_backward_op.n_pad != out_features:
        dy2d = pad_tensor(dy2d, divisor=fp4_backward_op.n_pad, dim=1)

    if packed_lora_dx is None:
        lora_down_bwd_packed, lora_up_bwd_packed = _pack_lora_dx_factors(
            lora_down=lora_down,
            lora_up=lora_up,
            fp4_backward_op=fp4_backward_op,
            scaling=scaling,
            lowrank_dtype=lowrank_dtype,
            frozen_residual_down=frozen_residual_down,
            frozen_residual_up=frozen_residual_up,
            frozen_residual_scaling=frozen_residual_scaling,
        )
    else:
        lora_down_bwd_packed, lora_up_bwd_packed = packed_lora_dx
    packed_rank = lora_down_bwd_packed.shape[1]
    dense_dy_up = None
    if return_dy_up and lowrank_dtype == torch.bfloat16:
        qdy, ascales, packed_dy_up, dense_dy_up = quantize_fp4_act_with_lora_dual(
            dy2d,
            lora_down_packed=lora_down_bwd_packed,
            smooth=fp4_backward_op.smooth_bwd,
            pad_size=256,
            dense_dtype=lowrank_dtype,
        )
    else:
        qdy, ascales, packed_dy_up = quantize_fp4_act_with_lora(
            dy2d,
            lora_down_packed=lora_down_bwd_packed,
            smooth=fp4_backward_op.smooth_bwd,
            pad_size=256,
        )
    qweight_bwd = fp4_backward_op.repack_qweight_for_backward()
    dx_pad = fp4_backward_op.backward_prequantized(
        qdy,
        ascales,
        qweight_bwd,
        lora_act=packed_dy_up,
        lora_up=lora_up_bwd_packed,
        lora_scales=[1.0] * ceil_divide(packed_rank, 16),
    )
    dx = dx_pad[: dy2d_src.shape[0], :in_features].reshape(*orig_shape[:-1], in_features)
    if not return_dy_up:
        return dx

    if dense_dy_up is None:
        dy_up = decode_lora_act(packed_dy_up, lowrank_dtype)[: dy2d_src.shape[0], : lora_down.shape[0]]
    else:
        dy_up = dense_dy_up[: dy2d_src.shape[0], : lora_down.shape[0]].to(lowrank_dtype)
    return dx, dy_up


def _fused_lora_backward_overlap_exact(
    dy: torch.Tensor,
    x2d: torch.Tensor,
    lora_act: torch.Tensor,
    lora_down: torch.Tensor,
    lora_up: torch.Tensor,
    fp4_backward_op: NunchakuFP4BackwardDXOp,
    scaling: float,
    lowrank_dtype: torch.dtype,
    in_features: int,
    out_features: int,
    packed_lora_dx: tuple[torch.Tensor, torch.Tensor] | None,
    frozen_residual_down: torch.Tensor | None = None,
    frozen_residual_up: torch.Tensor | None = None,
    frozen_residual_scaling: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if packed_lora_dx is None:
        raise ValueError("overlap_lora_grad requires cached packed LoRA dX factors")
    if not dy.is_cuda:
        raise ValueError("overlap_lora_grad requires CUDA tensors")
    has_frozen_residual = (
        frozen_residual_down is not None
        and frozen_residual_up is not None
        and frozen_residual_down.numel() > 0
        and frozen_residual_up.numel() > 0
    )

    orig_shape = dy.shape
    dy2d_src = dy.reshape(-1, out_features)
    dy2d = dy2d_src
    if fp4_backward_op.n_pad != out_features:
        dy2d = pad_tensor(dy2d, divisor=fp4_backward_op.n_pad, dim=1)

    lora_down_bwd_packed, lora_up_bwd_packed = packed_lora_dx
    packed_rank = lora_down_bwd_packed.shape[1]
    lora_scales = [1.0] * ceil_divide(packed_rank, 16)

    current_stream = torch.cuda.current_stream(device=dy.device)
    if _ENABLE_OVERLAP_RESOURCE_CACHE:
        resources = _get_lora_overlap_resources(fp4_backward_op, torch.device(dy.device), current_stream)
        repack_stream = resources.repack_stream
        dx_stream = resources.dx_stream
        up_stream = resources.up_stream
        down_stream = resources.down_stream
        residual_stream = resources.residual_stream if has_frozen_residual else None
        repack_done = resources.repack_done
        quant_done = resources.quant_done
    else:
        repack_stream = torch.cuda.Stream(device=dy.device)
        dx_stream = torch.cuda.Stream(device=dy.device)
        up_stream = torch.cuda.Stream(device=dy.device)
        down_stream = torch.cuda.Stream(device=dy.device)
        residual_stream = torch.cuda.Stream(device=dy.device) if has_frozen_residual else None
        repack_done = torch.cuda.Event()
        quant_done = torch.cuda.Event()

    repack_stream.wait_stream(current_stream)
    with torch.cuda.stream(repack_stream):
        qweight_bwd = fp4_backward_op.repack_qweight_for_backward()
        repack_done.record(repack_stream)

    up_stream.wait_stream(current_stream)
    with torch.cuda.stream(up_stream):
        d_lora_up = torch.matmul(dy2d_src.to(lowrank_dtype).t(), lora_act.to(lowrank_dtype))
        d_lora_up = d_lora_up.mul(float(scaling)).to(lora_up.dtype)

    down_stream.wait_stream(current_stream)
    with torch.cuda.stream(down_stream):
        dy_up = torch.matmul(dy2d_src.to(lowrank_dtype), lora_up.to(lowrank_dtype))
        d_lora_down = torch.matmul(dy_up.t(), x2d.to(lowrank_dtype))
        d_lora_down = d_lora_down.mul(float(scaling)).to(lora_down.dtype)

    if residual_stream is not None:
        residual_stream.wait_stream(current_stream)
        with torch.cuda.stream(residual_stream):
            residual_down_lr = frozen_residual_down.to(lowrank_dtype)
            residual_up_lr = frozen_residual_up.to(lowrank_dtype)
            dy_residual_up = torch.matmul(dy2d_src.to(lowrank_dtype), residual_up_lr)
            dx_residual = torch.matmul(dy_residual_up, residual_down_lr).mul(float(frozen_residual_scaling))

    qdy, ascales, packed_dy_up = quantize_fp4_act_with_lora(
        dy2d,
        lora_down_packed=lora_down_bwd_packed,
        smooth=fp4_backward_op.smooth_bwd,
        pad_size=256,
    )
    quant_done.record(current_stream)

    with torch.cuda.stream(dx_stream):
        dx_stream.wait_event(quant_done)
        dx_stream.wait_event(repack_done)
        dx_pad = fp4_backward_op.backward_prequantized(
            qdy,
            ascales,
            qweight_bwd,
            lora_act=packed_dy_up,
            lora_up=lora_up_bwd_packed,
            lora_scales=lora_scales,
        )

    current_stream.wait_stream(up_stream)
    current_stream.wait_stream(down_stream)
    current_stream.wait_stream(dx_stream)
    if residual_stream is not None:
        current_stream.wait_stream(residual_stream)

    dx = dx_pad[: dy2d_src.shape[0], :in_features].reshape(*orig_shape[:-1], in_features)
    if has_frozen_residual:
        dx = dx.to(dx_residual.dtype) + dx_residual.reshape(*orig_shape[:-1], in_features)
    return dx, d_lora_down, d_lora_up


def _fused_lora_backward_overlap_reuse(
    dy: torch.Tensor,
    x2d: torch.Tensor,
    lora_act: torch.Tensor,
    lora_down: torch.Tensor,
    lora_up: torch.Tensor,
    fp4_backward_op: NunchakuFP4BackwardDXOp,
    scaling: float,
    lowrank_dtype: torch.dtype,
    in_features: int,
    out_features: int,
    packed_lora_dx: tuple[torch.Tensor, torch.Tensor] | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if packed_lora_dx is None:
        raise ValueError("overlap_lora_grad requires cached packed LoRA dX factors")
    if not dy.is_cuda:
        raise ValueError("overlap_lora_grad requires CUDA tensors")

    orig_shape = dy.shape
    dy2d_src = dy.reshape(-1, out_features)
    dy2d = dy2d_src
    if fp4_backward_op.n_pad != out_features:
        dy2d = pad_tensor(dy2d, divisor=fp4_backward_op.n_pad, dim=1)

    lora_down_bwd_packed, lora_up_bwd_packed = packed_lora_dx
    packed_rank = lora_down_bwd_packed.shape[1]
    lora_scales = [1.0] * ceil_divide(packed_rank, 16)

    current_stream = torch.cuda.current_stream(device=dy.device)
    if _ENABLE_OVERLAP_RESOURCE_CACHE:
        resources = _get_lora_overlap_resources(fp4_backward_op, torch.device(dy.device), current_stream)
        repack_stream = resources.repack_stream
        dx_stream = resources.dx_stream
        up_stream = resources.up_stream
        down_stream = resources.down_stream
        repack_done = resources.repack_done
        quant_done = resources.quant_done
    else:
        repack_stream = torch.cuda.Stream(device=dy.device)
        dx_stream = torch.cuda.Stream(device=dy.device)
        up_stream = torch.cuda.Stream(device=dy.device)
        down_stream = torch.cuda.Stream(device=dy.device)
        repack_done = torch.cuda.Event()
        quant_done = torch.cuda.Event()

    repack_stream.wait_stream(current_stream)
    with torch.cuda.stream(repack_stream):
        qweight_bwd = fp4_backward_op.repack_qweight_for_backward()
        repack_done.record(repack_stream)

    up_stream.wait_stream(current_stream)
    with torch.cuda.stream(up_stream):
        d_lora_up = torch.matmul(dy2d_src.to(lowrank_dtype).t(), lora_act.to(lowrank_dtype))
        d_lora_up = d_lora_up.mul(float(scaling)).to(lora_up.dtype)

    dense_dy_up = None
    if lowrank_dtype == torch.bfloat16:
        qdy, ascales, packed_dy_up, dense_dy_up = quantize_fp4_act_with_lora_dual(
            dy2d,
            lora_down_packed=lora_down_bwd_packed,
            smooth=fp4_backward_op.smooth_bwd,
            pad_size=256,
            dense_dtype=lowrank_dtype,
        )
    else:
        qdy, ascales, packed_dy_up = quantize_fp4_act_with_lora(
            dy2d,
            lora_down_packed=lora_down_bwd_packed,
            smooth=fp4_backward_op.smooth_bwd,
            pad_size=256,
        )
    quant_done.record(current_stream)

    with torch.cuda.stream(down_stream):
        down_stream.wait_event(quant_done)
        if dense_dy_up is None:
            dy_up_for_down = decode_lora_act(packed_dy_up, lowrank_dtype)[: dy2d_src.shape[0], : lora_down.shape[0]]
        else:
            dy_up_for_down = dense_dy_up[: dy2d_src.shape[0], : lora_down.shape[0]].to(lowrank_dtype)
        d_lora_down = torch.matmul(dy_up_for_down.t(), x2d.to(lowrank_dtype))
        d_lora_down = d_lora_down.mul(float(scaling)).to(lora_down.dtype)

    with torch.cuda.stream(dx_stream):
        dx_stream.wait_event(quant_done)
        dx_stream.wait_event(repack_done)
        dx_pad = fp4_backward_op.backward_prequantized(
            qdy,
            ascales,
            qweight_bwd,
            lora_act=packed_dy_up,
            lora_up=lora_up_bwd_packed,
            lora_scales=lora_scales,
        )

    current_stream.wait_stream(up_stream)
    current_stream.wait_stream(down_stream)
    current_stream.wait_stream(dx_stream)

    dx = dx_pad[: dy2d_src.shape[0], :in_features].reshape(*orig_shape[:-1], in_features)
    return dx, d_lora_down, d_lora_up


class NunchakuFP4LoRALinear(torch.nn.Module):
    """Trainable LoRA wrapper over a frozen native FP4 backbone.

    Forward:
        y = FP4(x, W0) + scaling * (x @ A.T) @ B.T + bias

    Backward:
        dX uses the native FP4 backward dX kernel for W0 plus either a dense
        LoRA term or the optional fused LoRA dX epilogue. The fused path can
        cache packed LoRA dX weights with version-based invalidation. dA uses
        exact BF16/FP16 matmul by default, or an opt-in FP4 activation-cache
        approximation when activation memory is the bottleneck. dB uses the
        saved LoRA activation.
    """

    def __init__(
        self,
        weight: torch.Tensor,
        bias: torch.Tensor | None = None,
        rank: int = 32,
        lora_alpha: float | None = None,
        lowrank_dtype: torch.dtype = torch.bfloat16,
        init: LoRAInitMode = "zero",
        frozen_residual_rank: int = 0,
        frozen_residual_init: FrozenResidualInitMode = "none",
        train_bias: bool = False,
        cache_lora_act: bool = True,
        activation_checkpoint: bool = False,
        fuse_lowrank_forward: bool = False,
        fuse_lora_dx: bool = False,
        fuse_frozen_residual_dx: bool = False,
        cache_fused_lora_dx: bool = False,
        backward_weight_policy: FP4BackwardWeightPolicy = "repack",
        reuse_fused_dy_up_for_d_lora_down: bool = False,
        overlap_lora_grad: bool = False,
        overlap_lora_grad_min_rows: int = DEFAULT_OVERLAP_LORA_GRAD_MIN_ROWS,
        fp4_activation_cache_d_lora_down: bool = False,
        fp4_activation_cache_d_lora_down_backend: FP4ActivationCacheDLoRADownBackend = "fused",
        zero_lora_up_fast_path: bool = True,
        residual_svd_method: ResidualSVDMethod = "full_svd",
        residual_svd_lowrank_oversample: int = 8,
        residual_svd_lowrank_niter: int = 2,
    ):
        super().__init__()
        if not weight.is_cuda:
            raise ValueError("weight must be on CUDA")
        if weight.dtype not in (torch.float16, torch.bfloat16):
            raise ValueError("weight dtype must be float16 or bfloat16")
        if weight.dim() != 2:
            raise ValueError("weight should have shape [out_features, in_features]")
        if bias is not None and (not bias.is_cuda or bias.dim() != 1 or bias.shape[0] != weight.shape[0]):
            raise ValueError("bias must be a CUDA tensor with shape [out_features]")
        if rank <= 0:
            raise ValueError("rank must be positive")
        if lowrank_dtype not in (torch.float16, torch.bfloat16):
            raise ValueError("lowrank_dtype must be float16 or bfloat16")
        if init not in ("zero", "gaussian", "residual_svd"):
            raise ValueError("init must be one of: zero, gaussian, residual_svd")
        if frozen_residual_init not in ("none", "residual_svd"):
            raise ValueError("frozen_residual_init must be one of: none, residual_svd")
        if residual_svd_method not in ("full_svd", "svd_lowrank"):
            raise ValueError("residual_svd_method must be one of: full_svd, svd_lowrank")
        if fp4_activation_cache_d_lora_down_backend not in ("fused", "dequant_gemm"):
            raise ValueError("fp4_activation_cache_d_lora_down_backend must be one of: fused, dequant_gemm")
        if residual_svd_lowrank_oversample < 0:
            raise ValueError("residual_svd_lowrank_oversample must be non-negative")
        if residual_svd_lowrank_niter < 0:
            raise ValueError("residual_svd_lowrank_niter must be non-negative")
        if backward_weight_policy not in ("repack", "cache"):
            raise ValueError("backward_weight_policy must be one of: repack, cache")
        if frozen_residual_rank < 0:
            raise ValueError("frozen_residual_rank must be non-negative")
        if frozen_residual_init == "residual_svd" and frozen_residual_rank <= 0:
            raise ValueError("frozen_residual_rank must be positive when frozen_residual_init='residual_svd'")
        if frozen_residual_init == "none" and frozen_residual_rank != 0:
            raise ValueError("frozen_residual_rank must be 0 when frozen_residual_init='none'")
        if fuse_frozen_residual_dx and not fuse_lora_dx:
            raise ValueError("fuse_frozen_residual_dx requires fuse_lora_dx=True")
        if fuse_frozen_residual_dx and frozen_residual_init == "none":
            raise ValueError("fuse_frozen_residual_dx requires frozen_residual_init='residual_svd'")
        if fuse_frozen_residual_dx and weight.dtype != lowrank_dtype:
            raise ValueError("fuse_frozen_residual_dx requires weight dtype to match lowrank_dtype")
        if reuse_fused_dy_up_for_d_lora_down and not fuse_lora_dx:
            raise ValueError("reuse_fused_dy_up_for_d_lora_down requires fuse_lora_dx=True")
        if reuse_fused_dy_up_for_d_lora_down and weight.dtype != lowrank_dtype:
            raise ValueError("reuse_fused_dy_up_for_d_lora_down requires weight dtype to match lowrank_dtype")
        if overlap_lora_grad and not fuse_lora_dx:
            raise ValueError("overlap_lora_grad requires fuse_lora_dx=True")
        if overlap_lora_grad and not cache_fused_lora_dx:
            raise ValueError("overlap_lora_grad requires cache_fused_lora_dx=True")
        if overlap_lora_grad and reuse_fused_dy_up_for_d_lora_down and frozen_residual_init != "none":
            raise ValueError(
                "reuse-based overlap_lora_grad does not currently support frozen residual branches"
            )
        if overlap_lora_grad and fuse_frozen_residual_dx:
            raise ValueError("exact overlap_lora_grad expects frozen residual dX to stay dense")
        if overlap_lora_grad_min_rows < 0:
            raise ValueError("overlap_lora_grad_min_rows must be non-negative")
        if fp4_activation_cache_d_lora_down and not cache_lora_act:
            raise ValueError("fp4_activation_cache_d_lora_down requires cache_lora_act=True for dB")
        if fp4_activation_cache_d_lora_down and overlap_lora_grad:
            raise ValueError("fp4_activation_cache_d_lora_down does not currently support overlap_lora_grad")
        if fp4_activation_cache_d_lora_down and reuse_fused_dy_up_for_d_lora_down:
            raise ValueError("fp4_activation_cache_d_lora_down does not reuse packed dy_up")

        self.out_features, self.in_features = weight.shape
        self.rank = max(16, ceil_divide(rank, 16) * 16)
        self.requested_rank = rank
        self.frozen_residual_rank = 0
        self.requested_frozen_residual_rank = frozen_residual_rank
        self.lowrank_dtype = lowrank_dtype
        self.lora_alpha = float(self.rank if lora_alpha is None else lora_alpha)
        self.scaling = self.lora_alpha / float(self.rank)
        self.frozen_residual_scaling = 1.0
        self.cache_lora_act = bool(cache_lora_act)
        self.activation_checkpoint = bool(activation_checkpoint)
        self.fuse_lowrank_forward = bool(fuse_lowrank_forward)
        self.fuse_lora_dx = bool(fuse_lora_dx)
        self.fuse_frozen_residual_dx = bool(fuse_frozen_residual_dx)
        self.cache_fused_lora_dx = bool(cache_fused_lora_dx)
        self.backward_weight_policy = backward_weight_policy
        self.reuse_fused_dy_up_for_d_lora_down = bool(reuse_fused_dy_up_for_d_lora_down)
        self.overlap_lora_grad = bool(overlap_lora_grad)
        self.overlap_lora_grad_min_rows = int(overlap_lora_grad_min_rows)
        self.fp4_activation_cache_d_lora_down = bool(fp4_activation_cache_d_lora_down)
        self.fp4_activation_cache_d_lora_down_backend = fp4_activation_cache_d_lora_down_backend
        self.zero_lora_up_fast_path = bool(zero_lora_up_fast_path)
        self.init_mode = init
        self.frozen_residual_init = frozen_residual_init
        self.residual_svd_method = residual_svd_method
        self.residual_svd_lowrank_oversample = int(residual_svd_lowrank_oversample)
        self.residual_svd_lowrank_niter = int(residual_svd_lowrank_niter)

        self.fp4_forward = NunchakuFP4GemmOp(weight=weight, bias=None, dummy_rank=self.rank)
        self.fp4_backward = NunchakuFP4BackwardDXOp(
            weight=weight,
            dummy_rank=self.rank,
            backward_weight_policy=backward_weight_policy,
        )
        self._share_forward_backbone_with_backward()
        self.register_buffer("_cached_lora_down_fwd_packed", None, persistent=False)
        self.register_buffer("_cached_lora_up_fwd_packed", None, persistent=False)
        self._cached_lora_down_fwd_version = -1
        self._cached_lora_up_fwd_version = -1
        self._cached_frozen_residual_down_fwd_version = -1
        self._cached_frozen_residual_up_fwd_version = -1
        self._cached_lora_forward_scaling = None
        self._cached_frozen_residual_forward_scaling = None
        self.register_buffer("_cached_lora_down_bwd_packed", None, persistent=False)
        self.register_buffer("_cached_lora_up_bwd_packed", None, persistent=False)
        self._cached_lora_down_version = -1
        self._cached_lora_up_version = -1
        self._lora_up_zero_version = -1
        self._cached_frozen_residual_down_version = -1
        self._cached_frozen_residual_up_version = -1
        self._cached_lora_scaling = None
        self._cached_frozen_residual_scaling = None

        self.lora_down = torch.nn.Parameter(
            torch.empty(self.rank, self.in_features, device=weight.device, dtype=lowrank_dtype)
        )
        self.lora_up = torch.nn.Parameter(
            torch.empty(self.out_features, self.rank, device=weight.device, dtype=lowrank_dtype)
        )
        self.register_buffer("frozen_residual_down", None, persistent=True)
        self.register_buffer("frozen_residual_up", None, persistent=True)
        if frozen_residual_init == "residual_svd":
            self.frozen_residual_rank = max(16, ceil_divide(frozen_residual_rank, 16) * 16)
            down, up = self._residual_svd_factors(weight, self.frozen_residual_rank, scale=1.0)
            self.frozen_residual_down = down.to(lowrank_dtype).contiguous()
            self.frozen_residual_up = up.to(lowrank_dtype).contiguous()

        if bias is None:
            self.register_parameter("bias", None)
        elif train_bias:
            self.bias = torch.nn.Parameter(bias.detach().to(weight.dtype).contiguous())
        else:
            self.register_buffer("bias", bias.detach().to(weight.dtype).contiguous(), persistent=True)

        self.reset_lora_parameters(weight)

    def _share_forward_backbone_with_backward(self) -> None:
        # Keep one resident FP4 packed backbone. Backward still builds W^T transiently.
        self.fp4_backward.qweight = self.fp4_forward.qweight
        self.fp4_backward.wscales = self.fp4_forward.wscales
        fwd_scales = unpack_fp4_weight_scales(
            self.fp4_forward.wscales, self.fp4_forward.n_pad, self.fp4_forward.k_pad
        )
        logical_bwd, packed_bwd = build_backward_scales_from_forward_quant(
            qweight=self.fp4_forward.qweight,
            packed_wscales=self.fp4_forward.wscales,
            out_features=self.fp4_forward.n_pad,
            in_features=self.fp4_forward.k_pad,
        )
        self.fp4_backward.wscales_fwd_logical = fwd_scales.to(torch.float16).contiguous()
        self.fp4_backward.wscales_bwd_logical = logical_bwd.to(torch.float16).contiguous()
        self.fp4_backward.wscales_bwd_packed = packed_bwd.contiguous()
        self.fp4_backward.clear_backward_qweight_cache()

    @classmethod
    def from_linear(
        cls,
        linear: torch.nn.Linear,
        rank: int = 32,
        lora_alpha: float | None = None,
        lowrank_dtype: torch.dtype = torch.bfloat16,
        init: LoRAInitMode = "zero",
        train_bias: bool = False,
        cache_lora_act: bool = True,
        activation_checkpoint: bool = False,
        fuse_lowrank_forward: bool = False,
        fuse_lora_dx: bool = False,
        fuse_frozen_residual_dx: bool = False,
        cache_fused_lora_dx: bool = False,
        backward_weight_policy: FP4BackwardWeightPolicy = "repack",
        reuse_fused_dy_up_for_d_lora_down: bool = False,
        overlap_lora_grad: bool = False,
        overlap_lora_grad_min_rows: int = DEFAULT_OVERLAP_LORA_GRAD_MIN_ROWS,
        fp4_activation_cache_d_lora_down: bool = False,
        fp4_activation_cache_d_lora_down_backend: FP4ActivationCacheDLoRADownBackend = "fused",
        zero_lora_up_fast_path: bool = True,
        frozen_residual_rank: int = 0,
        frozen_residual_init: FrozenResidualInitMode = "none",
        residual_svd_method: ResidualSVDMethod = "full_svd",
        residual_svd_lowrank_oversample: int = 8,
        residual_svd_lowrank_niter: int = 2,
    ) -> "NunchakuFP4LoRALinear":
        return cls(
            weight=linear.weight.detach(),
            bias=None if linear.bias is None else linear.bias.detach(),
            rank=rank,
            lora_alpha=lora_alpha,
            lowrank_dtype=lowrank_dtype,
            init=init,
            frozen_residual_rank=frozen_residual_rank,
            frozen_residual_init=frozen_residual_init,
            train_bias=train_bias,
            cache_lora_act=cache_lora_act,
            activation_checkpoint=activation_checkpoint,
            fuse_lowrank_forward=fuse_lowrank_forward,
            fuse_lora_dx=fuse_lora_dx,
            fuse_frozen_residual_dx=fuse_frozen_residual_dx,
            cache_fused_lora_dx=cache_fused_lora_dx,
            backward_weight_policy=backward_weight_policy,
            reuse_fused_dy_up_for_d_lora_down=reuse_fused_dy_up_for_d_lora_down,
            overlap_lora_grad=overlap_lora_grad,
            overlap_lora_grad_min_rows=overlap_lora_grad_min_rows,
            fp4_activation_cache_d_lora_down=fp4_activation_cache_d_lora_down,
            fp4_activation_cache_d_lora_down_backend=fp4_activation_cache_d_lora_down_backend,
            zero_lora_up_fast_path=zero_lora_up_fast_path,
            residual_svd_method=residual_svd_method,
            residual_svd_lowrank_oversample=residual_svd_lowrank_oversample,
            residual_svd_lowrank_niter=residual_svd_lowrank_niter,
        )

    def clear_fused_lora_forward_cache(self) -> None:
        self._cached_lora_down_fwd_packed = None
        self._cached_lora_up_fwd_packed = None
        self._cached_lora_down_fwd_version = -1
        self._cached_lora_up_fwd_version = -1
        self._cached_frozen_residual_down_fwd_version = -1
        self._cached_frozen_residual_up_fwd_version = -1
        self._cached_lora_forward_scaling = None
        self._cached_frozen_residual_forward_scaling = None

    def clear_fused_lora_dx_cache(self) -> None:
        self._cached_lora_down_bwd_packed = None
        self._cached_lora_up_bwd_packed = None
        self._cached_lora_down_version = -1
        self._cached_lora_up_version = -1
        self._cached_frozen_residual_down_version = -1
        self._cached_frozen_residual_up_version = -1
        self._cached_lora_scaling = None
        self._cached_frozen_residual_scaling = None

    def clear_lora_up_zero_fast_path(self) -> None:
        self._lora_up_zero_version = -1

    def mark_lora_up_zero_fast_path(self) -> None:
        self._lora_up_zero_version = self.lora_up._version

    def _lora_up_zero_fast_path_active(self) -> bool:
        return bool(self.zero_lora_up_fast_path and self._lora_up_zero_version == self.lora_up._version)

    def refresh_fused_lora_forward_cache(self) -> None:
        if self._lora_up_zero_fast_path_active():
            self.clear_fused_lora_forward_cache()
            return
        if self.lowrank_dtype != self.fp4_forward.compute_dtype:
            self.clear_fused_lora_forward_cache()
            return
        with torch.no_grad():
            lora_down_fwd_packed, lora_up_fwd_packed = _pack_lora_forward_factors(
                lora_down=self.lora_down,
                lora_up=self.lora_up,
                fp4_forward_op=self.fp4_forward,
                scaling=self.scaling,
                frozen_residual_down=self.frozen_residual_down if self.has_frozen_residual else None,
                frozen_residual_up=self.frozen_residual_up if self.has_frozen_residual else None,
                frozen_residual_scaling=self.frozen_residual_scaling,
            )
        self._cached_lora_down_fwd_packed = lora_down_fwd_packed.detach()
        self._cached_lora_up_fwd_packed = lora_up_fwd_packed.detach()
        self._cached_lora_down_fwd_version = self.lora_down._version
        self._cached_lora_up_fwd_version = self.lora_up._version
        self._cached_frozen_residual_down_fwd_version = (
            self.frozen_residual_down._version if self.has_frozen_residual else -1
        )
        self._cached_frozen_residual_up_fwd_version = (
            self.frozen_residual_up._version if self.has_frozen_residual else -1
        )
        self._cached_lora_forward_scaling = float(self.scaling)
        self._cached_frozen_residual_forward_scaling = (
            float(self.frozen_residual_scaling) if self.has_frozen_residual else None
        )

    def _get_fused_lora_forward_cache(self) -> tuple[torch.Tensor, torch.Tensor] | None:
        if not self.fuse_lowrank_forward:
            return None
        if self._lora_up_zero_fast_path_active():
            return None
        if self.lowrank_dtype != self.fp4_forward.compute_dtype:
            return None
        frozen_residual_cache_valid = True
        if self.has_frozen_residual:
            frozen_residual_cache_valid = (
                self._cached_frozen_residual_down_fwd_version == self.frozen_residual_down._version
                and self._cached_frozen_residual_up_fwd_version == self.frozen_residual_up._version
                and self._cached_frozen_residual_forward_scaling == float(self.frozen_residual_scaling)
            )
        cache_valid = (
            self._cached_lora_down_fwd_packed is not None
            and self._cached_lora_up_fwd_packed is not None
            and self._cached_lora_down_fwd_version == self.lora_down._version
            and self._cached_lora_up_fwd_version == self.lora_up._version
            and self._cached_lora_forward_scaling == float(self.scaling)
            and frozen_residual_cache_valid
        )
        if not cache_valid:
            self.refresh_fused_lora_forward_cache()
        return self._cached_lora_down_fwd_packed, self._cached_lora_up_fwd_packed

    def clear_backward_weight_cache(self) -> None:
        self.fp4_backward.clear_backward_qweight_cache()

    def refresh_backward_weight_cache(self) -> torch.Tensor | None:
        if self.backward_weight_policy != "cache":
            self.clear_backward_weight_cache()
            return None
        with torch.no_grad():
            return self.fp4_backward.refresh_backward_qweight_cache()

    def refresh_fused_lora_dx_cache(self) -> None:
        if self._lora_up_zero_fast_path_active():
            self.clear_fused_lora_dx_cache()
            return
        with torch.no_grad():
            lora_down_bwd_packed, lora_up_bwd_packed = _pack_lora_dx_factors(
                lora_down=self.lora_down,
                lora_up=self.lora_up,
                fp4_backward_op=self.fp4_backward,
                scaling=self.scaling,
                lowrank_dtype=self.lowrank_dtype,
                frozen_residual_down=self.frozen_residual_down
                if self.has_frozen_residual and self.fuse_frozen_residual_dx
                else None,
                frozen_residual_up=self.frozen_residual_up
                if self.has_frozen_residual and self.fuse_frozen_residual_dx
                else None,
                frozen_residual_scaling=self.frozen_residual_scaling,
            )
        self._cached_lora_down_bwd_packed = lora_down_bwd_packed.detach()
        self._cached_lora_up_bwd_packed = lora_up_bwd_packed.detach()
        self._cached_lora_down_version = self.lora_down._version
        self._cached_lora_up_version = self.lora_up._version
        self._cached_frozen_residual_down_version = (
            self.frozen_residual_down._version if self.has_frozen_residual else -1
        )
        self._cached_frozen_residual_up_version = (
            self.frozen_residual_up._version if self.has_frozen_residual else -1
        )
        self._cached_lora_scaling = float(self.scaling)
        self._cached_frozen_residual_scaling = float(self.frozen_residual_scaling) if self.has_frozen_residual else None

    def _get_fused_lora_dx_cache(self) -> tuple[torch.Tensor, torch.Tensor] | None:
        if not (self.fuse_lora_dx and self.cache_fused_lora_dx):
            return None
        if self._lora_up_zero_fast_path_active():
            return None
        frozen_residual_cache_valid = True
        if self.has_frozen_residual and self.fuse_frozen_residual_dx:
            frozen_residual_cache_valid = (
                self._cached_frozen_residual_down_version == self.frozen_residual_down._version
                and self._cached_frozen_residual_up_version == self.frozen_residual_up._version
                and self._cached_frozen_residual_scaling == float(self.frozen_residual_scaling)
            )
        cache_valid = (
            self._cached_lora_down_bwd_packed is not None
            and self._cached_lora_up_bwd_packed is not None
            and self._cached_lora_down_version == self.lora_down._version
            and self._cached_lora_up_version == self.lora_up._version
            and self._cached_lora_scaling == float(self.scaling)
            and frozen_residual_cache_valid
        )
        if not cache_valid:
            self.refresh_fused_lora_dx_cache()
        return self._cached_lora_down_bwd_packed, self._cached_lora_up_bwd_packed

    def reset_lora_parameters(self, weight: torch.Tensor | None = None) -> None:
        with torch.no_grad():
            if self.init_mode == "zero":
                torch.nn.init.kaiming_uniform_(self.lora_down, a=math.sqrt(5))
                torch.nn.init.zeros_(self.lora_up)
                self.mark_lora_up_zero_fast_path()
            elif self.init_mode == "gaussian":
                torch.nn.init.normal_(self.lora_down, mean=0.0, std=0.02)
                torch.nn.init.normal_(self.lora_up, mean=0.0, std=0.02)
                self.clear_lora_up_zero_fast_path()
            else:
                if weight is None:
                    raise ValueError("weight is required for residual_svd initialization")
                self._init_from_residual_svd(weight)
                self.clear_lora_up_zero_fast_path()
            self.clear_fused_lora_forward_cache()
            self.clear_fused_lora_dx_cache()

    def _residual_svd_factors(
        self,
        weight: torch.Tensor,
        rank: int,
        scale: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        weight_pad = pad_tensor(weight, divisor=(256, 128), dim=(0, 1))
        weight_hat, _ = dequantize_fp4_weight(
            qweight=self.fp4_forward.qweight,
            packed_wscales=self.fp4_forward.wscales,
            out_features=self.fp4_forward.n_pad,
            in_features=self.fp4_forward.k_pad,
            dtype=weight.dtype,
        )
        residual = (weight_pad - weight_hat).float()
        if self.residual_svd_method == "full_svd":
            u, s, vh = torch.linalg.svd(residual, full_matrices=False)
        else:
            q = min(rank + self.residual_svd_lowrank_oversample, min(residual.shape))
            q = max(rank, q)
            u, s, v = torch.svd_lowrank(residual, q=q, niter=self.residual_svd_lowrank_niter)
            vh = v.transpose(0, 1).contiguous()
        eff_rank = min(rank, s.numel())
        safe_scale = scale if abs(scale) > 1e-12 else 1.0

        up = torch.zeros(self.fp4_forward.n_pad, rank, device=weight.device, dtype=weight.dtype)
        down = torch.zeros(rank, self.fp4_forward.k_pad, device=weight.device, dtype=weight.dtype)
        up[:, :eff_rank] = (u[:, :eff_rank] * s[:eff_rank].unsqueeze(0) / safe_scale).to(weight.dtype)
        down[:eff_rank, :] = vh[:eff_rank, :].to(weight.dtype)
        return down[:, : self.in_features], up[: self.out_features, :]

    def _init_from_residual_svd(self, weight: torch.Tensor) -> None:
        down, up = self._residual_svd_factors(weight, self.rank, scale=self.scaling)
        self.lora_up.copy_(up.to(self.lowrank_dtype))
        self.lora_down.copy_(down.to(self.lowrank_dtype))

    @property
    def has_frozen_residual(self) -> bool:
        return (
            self.frozen_residual_down is not None
            and self.frozen_residual_up is not None
            and self.frozen_residual_down.numel() > 0
            and self.frozen_residual_up.numel() > 0
        )

    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        zero_lora_up_fast_path_active = self._lora_up_zero_fast_path_active()
        packed_lora_dx = None if zero_lora_up_fast_path_active else self._get_fused_lora_dx_cache()
        packed_lora_forward = None if zero_lora_up_fast_path_active else self._get_fused_lora_forward_cache()
        return _FP4LoRALinearFunction.apply(
            x,
            self.lora_down,
            self.lora_up,
            self.frozen_residual_down
            if self.has_frozen_residual
            else torch.empty(0, device=x.device, dtype=self.lowrank_dtype),
            self.frozen_residual_up
            if self.has_frozen_residual
            else torch.empty(0, device=x.device, dtype=self.lowrank_dtype),
            self.bias,
            self.fp4_forward,
            self.fp4_backward,
            self.scaling,
            self.frozen_residual_scaling,
            self.lowrank_dtype,
            self.cache_lora_act,
            self.fuse_lowrank_forward,
            self.fuse_lora_dx,
            self.fuse_frozen_residual_dx,
            self.reuse_fused_dy_up_for_d_lora_down,
            self.overlap_lora_grad,
            self.overlap_lora_grad_min_rows,
            self.fp4_activation_cache_d_lora_down,
            self.fp4_activation_cache_d_lora_down_backend,
            zero_lora_up_fast_path_active,
            packed_lora_dx,
            packed_lora_forward,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.activation_checkpoint and self.training and torch.is_grad_enabled():
            return checkpoint(self._forward_impl, x, use_reentrant=False, preserve_rng_state=False)
        return self._forward_impl(x)

    def lora_weight(self) -> torch.Tensor:
        return torch.matmul(self.lora_up.float(), self.lora_down.float()).mul(self.scaling)

    def frozen_residual_weight(self) -> torch.Tensor:
        if not self.has_frozen_residual:
            return torch.zeros(self.out_features, self.in_features, device=self.lora_up.device)
        return torch.matmul(self.frozen_residual_up.float(), self.frozen_residual_down.float()).mul(
            self.frozen_residual_scaling
        )

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"rank={self.rank}, lora_alpha={self.lora_alpha:g}, "
            f"frozen_residual_rank={self.frozen_residual_rank}, "
            f"lowrank_dtype={self.lowrank_dtype}, init={self.init_mode}, "
            f"frozen_residual_init={self.frozen_residual_init}, "
            f"residual_svd_method={self.residual_svd_method}, "
            f"cache_lora_act={self.cache_lora_act}, activation_checkpoint={self.activation_checkpoint}, "
            f"fuse_lowrank_forward={self.fuse_lowrank_forward}, "
            f"fuse_lora_dx={self.fuse_lora_dx}, "
            f"fuse_frozen_residual_dx={self.fuse_frozen_residual_dx}, "
            f"cache_fused_lora_dx={self.cache_fused_lora_dx}, "
            f"backward_weight_policy={self.backward_weight_policy}, "
            f"reuse_fused_dy_up_for_d_lora_down={self.reuse_fused_dy_up_for_d_lora_down}, "
            f"overlap_lora_grad={self.overlap_lora_grad}, "
            f"overlap_lora_grad_min_rows={self.overlap_lora_grad_min_rows}, "
            f"fp4_activation_cache_d_lora_down={self.fp4_activation_cache_d_lora_down}, "
            f"fp4_activation_cache_d_lora_down_backend={self.fp4_activation_cache_d_lora_down_backend}, "
            f"zero_lora_up_fast_path={self.zero_lora_up_fast_path}"
        )
