from __future__ import annotations

import math
from typing import Literal

import torch

from .layout import (
    build_backward_scales_from_forward_quant,
    dequantize_fp4_weight,
    unpack_fp4_weight_scales,
)
from .operators import (
    NunchakuFP4BackwardDXOp,
    NunchakuFP4GemmOp,
    ceil_divide,
    pack_lowrank_weight,
    pad_tensor,
    quantize_fp4_act_with_lora,
)

LoRAInitMode = Literal["zero", "gaussian", "residual_svd"]


class _FP4LoRALinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        lora_down: torch.Tensor,
        lora_up: torch.Tensor,
        bias: torch.Tensor | None,
        fp4_forward_op: NunchakuFP4GemmOp,
        fp4_backward_op: NunchakuFP4BackwardDXOp,
        scaling: float,
        lowrank_dtype: torch.dtype,
        cache_lora_act: bool,
        fuse_lora_dx: bool,
    ) -> torch.Tensor:
        if x.shape[-1] != fp4_forward_op.in_features:
            raise ValueError(f"Expected input last dim = {fp4_forward_op.in_features}, got {x.shape[-1]}")

        x2d = x.reshape(-1, fp4_forward_op.in_features)
        x_lr = x2d.to(lowrank_dtype)
        down_lr = lora_down.to(lowrank_dtype)
        up_lr = lora_up.to(lowrank_dtype)

        y_main = fp4_forward_op(x)
        lora_act = torch.matmul(x_lr, down_lr.t())
        lora_out = torch.matmul(lora_act, up_lr.t()).mul(float(scaling)).to(y_main.dtype)
        y = y_main + lora_out.reshape(*x.shape[:-1], fp4_forward_op.out_features)
        if bias is not None:
            y = y + bias.to(y.dtype)

        if cache_lora_act:
            saved_lora_act = lora_act
        else:
            saved_lora_act = torch.empty(0, device=x.device, dtype=lowrank_dtype)
        ctx.save_for_backward(x, lora_down, lora_up, saved_lora_act)
        ctx.fp4_backward_op = fp4_backward_op
        ctx.scaling = float(scaling)
        ctx.lowrank_dtype = lowrank_dtype
        ctx.cache_lora_act = bool(cache_lora_act)
        ctx.fuse_lora_dx = bool(fuse_lora_dx)
        ctx.has_bias = bias is not None
        ctx.in_features = fp4_forward_op.in_features
        ctx.out_features = fp4_forward_op.out_features
        return y

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        x, lora_down, lora_up, saved_lora_act = ctx.saved_tensors
        dy = grad_output.contiguous()

        x2d = x.reshape(-1, ctx.in_features)
        dy2d = dy.reshape(-1, ctx.out_features)
        x_lr = x2d.to(ctx.lowrank_dtype)
        dy_lr = dy2d.to(ctx.lowrank_dtype)
        down_lr = lora_down.to(ctx.lowrank_dtype)
        up_lr = lora_up.to(ctx.lowrank_dtype)

        if ctx.cache_lora_act:
            lora_act = saved_lora_act
        else:
            lora_act = torch.matmul(x_lr, down_lr.t())

        # Keep LoRA parameter gradients on dense BF16/FP16 matmul. The fused dX path
        # only uses packed dy@B for the epilogue because its dense dual-output variant
        # was not accurate enough for dA on larger shapes.
        dy_up = torch.matmul(dy_lr, up_lr)
        if ctx.fuse_lora_dx:
            dx = _fused_lora_dx(
                dy=dy,
                lora_down=lora_down,
                lora_up=lora_up,
                fp4_backward_op=ctx.fp4_backward_op,
                scaling=ctx.scaling,
                lowrank_dtype=ctx.lowrank_dtype,
                in_features=ctx.in_features,
                out_features=ctx.out_features,
            )
        else:
            dx_main = ctx.fp4_backward_op(dy)
            dx_lora = torch.matmul(dy_up, down_lr).mul(ctx.scaling)
            dx = dx_main.to(dx_lora.dtype) + dx_lora.reshape_as(x)

        d_lora_up = torch.matmul(dy_lr.t(), lora_act).mul(ctx.scaling).to(lora_up.dtype)
        d_lora_down = torch.matmul(dy_up.t(), x_lr).mul(ctx.scaling).to(lora_down.dtype)
        d_bias = dy2d.sum(dim=0).to(grad_output.dtype) if ctx.has_bias else None

        return (
            dx.to(x.dtype),
            d_lora_down,
            d_lora_up,
            d_bias,
            None,
            None,
            None,
            None,
            None,
            None,
        )


def _pack_trainable_lora_for_fused_dx(
    lora_down: torch.Tensor,
    lora_up: torch.Tensor,
    fp4_backward_op: NunchakuFP4BackwardDXOp,
    scaling: float,
    lowrank_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    del lowrank_dtype
    packed_dtype = fp4_backward_op.compute_dtype
    b_t = lora_up.t().contiguous().to(packed_dtype)
    if b_t.shape[1] != fp4_backward_op.n_pad:
        b_t = pad_tensor(b_t, divisor=fp4_backward_op.n_pad, dim=1)
    a_t = lora_down.t().contiguous().to(packed_dtype).mul(float(scaling))
    if a_t.shape[0] != fp4_backward_op.k_pad:
        a_t = pad_tensor(a_t, divisor=fp4_backward_op.k_pad, dim=0)
    return pack_lowrank_weight(b_t, down=True).contiguous(), pack_lowrank_weight(a_t, down=False).contiguous()


def _fused_lora_dx(
    dy: torch.Tensor,
    lora_down: torch.Tensor,
    lora_up: torch.Tensor,
    fp4_backward_op: NunchakuFP4BackwardDXOp,
    scaling: float,
    lowrank_dtype: torch.dtype,
    in_features: int,
    out_features: int,
) -> torch.Tensor:
    orig_shape = dy.shape
    dy2d_src = dy.reshape(-1, out_features)
    dy2d = dy2d_src
    if fp4_backward_op.n_pad != out_features:
        dy2d = pad_tensor(dy2d, divisor=fp4_backward_op.n_pad, dim=1)

    lora_down_bwd_packed, lora_up_bwd_packed = _pack_trainable_lora_for_fused_dx(
        lora_down=lora_down,
        lora_up=lora_up,
        fp4_backward_op=fp4_backward_op,
        scaling=scaling,
        lowrank_dtype=lowrank_dtype,
    )
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
        lora_scales=[1.0] * ceil_divide(lora_down.shape[0], 16),
    )
    return dx_pad[: dy2d_src.shape[0], :in_features].reshape(*orig_shape[:-1], in_features)


class NunchakuFP4LoRALinear(torch.nn.Module):
    """Trainable LoRA wrapper over a frozen native FP4 backbone.

    Forward:
        y = FP4(x, W0) + scaling * (x @ A.T) @ B.T + bias

    Backward:
        dX uses the native FP4 backward dX kernel for W0 plus either a dense
        LoRA term or the optional fused LoRA dX epilogue. dA/dB still use
        torch BF16/FP16 matmul.
    """

    def __init__(
        self,
        weight: torch.Tensor,
        bias: torch.Tensor | None = None,
        rank: int = 32,
        lora_alpha: float | None = None,
        lowrank_dtype: torch.dtype = torch.bfloat16,
        init: LoRAInitMode = "zero",
        train_bias: bool = False,
        cache_lora_act: bool = True,
        fuse_lora_dx: bool = False,
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

        self.out_features, self.in_features = weight.shape
        self.rank = max(16, ceil_divide(rank, 16) * 16)
        self.requested_rank = rank
        self.lowrank_dtype = lowrank_dtype
        self.lora_alpha = float(self.rank if lora_alpha is None else lora_alpha)
        self.scaling = self.lora_alpha / float(self.rank)
        self.cache_lora_act = bool(cache_lora_act)
        self.fuse_lora_dx = bool(fuse_lora_dx)
        self.init_mode = init

        self.fp4_forward = NunchakuFP4GemmOp(weight=weight, bias=None, dummy_rank=self.rank)
        self.fp4_backward = NunchakuFP4BackwardDXOp(weight=weight, dummy_rank=self.rank)
        self._share_forward_backbone_with_backward()

        self.lora_down = torch.nn.Parameter(
            torch.empty(self.rank, self.in_features, device=weight.device, dtype=lowrank_dtype)
        )
        self.lora_up = torch.nn.Parameter(
            torch.empty(self.out_features, self.rank, device=weight.device, dtype=lowrank_dtype)
        )

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
        fuse_lora_dx: bool = False,
    ) -> "NunchakuFP4LoRALinear":
        return cls(
            weight=linear.weight.detach(),
            bias=None if linear.bias is None else linear.bias.detach(),
            rank=rank,
            lora_alpha=lora_alpha,
            lowrank_dtype=lowrank_dtype,
            init=init,
            train_bias=train_bias,
            cache_lora_act=cache_lora_act,
            fuse_lora_dx=fuse_lora_dx,
        )

    def reset_lora_parameters(self, weight: torch.Tensor | None = None) -> None:
        with torch.no_grad():
            if self.init_mode == "zero":
                torch.nn.init.kaiming_uniform_(self.lora_down, a=math.sqrt(5))
                torch.nn.init.zeros_(self.lora_up)
            elif self.init_mode == "gaussian":
                torch.nn.init.normal_(self.lora_down, mean=0.0, std=0.02)
                torch.nn.init.normal_(self.lora_up, mean=0.0, std=0.02)
            else:
                if weight is None:
                    raise ValueError("weight is required for residual_svd initialization")
                self._init_from_residual_svd(weight)

    def _init_from_residual_svd(self, weight: torch.Tensor) -> None:
        weight_pad = pad_tensor(weight, divisor=(256, 128), dim=(0, 1))
        weight_hat, _ = dequantize_fp4_weight(
            qweight=self.fp4_forward.qweight,
            packed_wscales=self.fp4_forward.wscales,
            out_features=self.fp4_forward.n_pad,
            in_features=self.fp4_forward.k_pad,
            dtype=weight.dtype,
        )
        residual = (weight_pad - weight_hat).float()
        u, s, vh = torch.linalg.svd(residual, full_matrices=False)
        eff_rank = min(self.rank, s.numel())
        scale = self.scaling if abs(self.scaling) > 1e-12 else 1.0

        up = torch.zeros(self.fp4_forward.n_pad, self.rank, device=weight.device, dtype=weight.dtype)
        down = torch.zeros(self.rank, self.fp4_forward.k_pad, device=weight.device, dtype=weight.dtype)
        up[:, :eff_rank] = (u[:, :eff_rank] * s[:eff_rank].unsqueeze(0) / scale).to(weight.dtype)
        down[:eff_rank, :] = vh[:eff_rank, :].to(weight.dtype)

        self.lora_up.copy_(up[: self.out_features, :].to(self.lowrank_dtype))
        self.lora_down.copy_(down[:, : self.in_features].to(self.lowrank_dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _FP4LoRALinearFunction.apply(
            x,
            self.lora_down,
            self.lora_up,
            self.bias,
            self.fp4_forward,
            self.fp4_backward,
            self.scaling,
            self.lowrank_dtype,
            self.cache_lora_act,
            self.fuse_lora_dx,
        )

    def lora_weight(self) -> torch.Tensor:
        return torch.matmul(self.lora_up.float(), self.lora_down.float()).mul(self.scaling)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"rank={self.rank}, lora_alpha={self.lora_alpha:g}, "
            f"lowrank_dtype={self.lowrank_dtype}, init={self.init_mode}, "
            f"cache_lora_act={self.cache_lora_act}, fuse_lora_dx={self.fuse_lora_dx}"
        )
