from __future__ import annotations

import torch
from torch import nn

from .ops import quantize_int4_packed, unpack_int4_packed


class Int4LinearCore(nn.Module):
    """INT4 main-branch linear operator with CUDA quantization kernels and torch._int_mm compute."""

    def __init__(self, in_features: int, out_features: int, bias: torch.Tensor | None = None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.register_buffer("qweight_packed", torch.empty(0, dtype=torch.uint8), persistent=True)
        self.register_buffer("w_scales", torch.empty(0, dtype=torch.float32), persistent=True)
        self.register_buffer("w_int8", torch.empty(0, dtype=torch.int8), persistent=True)

        if bias is None:
            self.register_buffer("bias", None, persistent=True)
        else:
            self.register_buffer("bias", bias.detach().contiguous(), persistent=True)

    @classmethod
    def from_fp16_weight(cls, weight: torch.Tensor, bias: torch.Tensor | None = None) -> "Int4LinearCore":
        if weight.dim() != 2:
            raise ValueError(f"weight must be 2D [out_features, in_features], got shape={tuple(weight.shape)}")
        out_features, in_features = weight.shape
        mod = cls(in_features=in_features, out_features=out_features, bias=bias)
        mod.load_fp16_weight(weight)
        return mod

    @torch.no_grad()
    def load_fp16_weight(self, weight: torch.Tensor) -> None:
        if not weight.is_cuda:
            raise ValueError("weight must be CUDA tensor")
        if weight.shape != (self.out_features, self.in_features):
            raise ValueError(
                f"weight shape mismatch: expected {(self.out_features, self.in_features)}, got {tuple(weight.shape)}"
            )
        qweight_packed, w_scales = quantize_int4_packed(weight)
        w_int8 = unpack_int4_packed(qweight_packed)

        self.qweight_packed = qweight_packed
        self.w_scales = w_scales
        self.w_int8 = w_int8.contiguous()

    def _main_branch(self, x2d: torch.Tensor) -> torch.Tensor:
        x_packed, x_scales = quantize_int4_packed(x2d)
        x_int8 = unpack_int4_packed(x_packed)

        acc = torch._int_mm(x_int8, self.w_int8.t().contiguous())
        out = acc.float()
        out = out * x_scales.unsqueeze(1)
        out = out * self.w_scales.unsqueeze(0)
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.in_features:
            raise ValueError(f"Expected input last dim {self.in_features}, got {x.shape[-1]}")

        orig_shape = x.shape
        x2d = x.reshape(-1, self.in_features).contiguous()

        out = self._main_branch(x2d)
        if self.bias is not None:
            out = out + self.bias.float().unsqueeze(0)

        out = out.to(dtype=x.dtype)
        return out.reshape(*orig_shape[:-1], self.out_features)


class SVDQLinearCore(Int4LinearCore):
    """SVDQuant core: INT4 main branch + FP16 low-rank residual branch."""

    def __init__(self, in_features: int, out_features: int, rank: int, bias: torch.Tensor | None = None):
        super().__init__(in_features=in_features, out_features=out_features, bias=bias)
        self.rank = rank

        self.register_buffer("lora_down", torch.empty(in_features, rank, dtype=torch.float16), persistent=True)
        self.register_buffer("lora_up", torch.empty(out_features, rank, dtype=torch.float16), persistent=True)

    @classmethod
    def from_fp16_weight(
        cls,
        weight: torch.Tensor,
        bias: torch.Tensor | None = None,
        rank: int = 32,
    ) -> "SVDQLinearCore":
        if weight.dim() != 2:
            raise ValueError(f"weight must be 2D [out_features, in_features], got shape={tuple(weight.shape)}")

        out_features, in_features = weight.shape
        mod = cls(in_features=in_features, out_features=out_features, rank=rank, bias=bias)
        mod.load_fp16_weight(weight)
        mod.fit_low_rank_residual(weight)
        return mod

    @torch.no_grad()
    def fit_low_rank_residual(self, weight: torch.Tensor) -> None:
        dequant_weight = self.w_int8.float() * self.w_scales.unsqueeze(1)
        residual = (weight.float() - dequant_weight).float()

        u, s, vh = torch.linalg.svd(residual, full_matrices=False)
        eff_rank = min(self.rank, s.numel())

        lora_up = u[:, :eff_rank] * s[:eff_rank].unsqueeze(0)
        lora_down = vh[:eff_rank, :].t().contiguous()

        if eff_rank < self.rank:
            pad_up = torch.zeros(
                self.out_features,
                self.rank - eff_rank,
                dtype=lora_up.dtype,
                device=lora_up.device,
            )
            pad_down = torch.zeros(
                self.in_features,
                self.rank - eff_rank,
                dtype=lora_down.dtype,
                device=lora_down.device,
            )
            lora_up = torch.cat([lora_up, pad_up], dim=1)
            lora_down = torch.cat([lora_down, pad_down], dim=1)

        self.lora_up = lora_up.to(dtype=torch.float16)
        self.lora_down = lora_down.to(dtype=torch.float16)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.in_features:
            raise ValueError(f"Expected input last dim {self.in_features}, got {x.shape[-1]}")

        orig_shape = x.shape
        x2d = x.reshape(-1, self.in_features).contiguous()

        out = self._main_branch(x2d)

        # Low-rank residual branch in FP16.
        lora_act = torch.matmul(x2d.to(dtype=torch.float16), self.lora_down)
        lora_out = torch.matmul(lora_act, self.lora_up.t())
        out = out + lora_out.float()

        if self.bias is not None:
            out = out + self.bias.float().unsqueeze(0)

        out = out.to(dtype=x.dtype)
        return out.reshape(*orig_shape[:-1], self.out_features)


def fp16_linear_reference(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
    out = torch.matmul(x, weight.t())
    if bias is not None:
        out = out + bias
    return out
