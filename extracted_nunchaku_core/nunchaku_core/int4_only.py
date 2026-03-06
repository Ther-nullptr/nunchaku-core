from __future__ import annotations

import torch

from .modules import Int4LinearCore


class Int4OnlyLinearCore(Int4LinearCore):
    """Dedicated 4-bit-only operator entrypoint (no 16-bit low-rank branch)."""

    @classmethod
    def from_fp16_weight(cls, weight: torch.Tensor, bias: torch.Tensor | None = None) -> "Int4OnlyLinearCore":
        base = super().from_fp16_weight(weight, bias)
        mod = cls(in_features=base.in_features, out_features=base.out_features, bias=base.bias)
        mod.qweight_packed = base.qweight_packed
        mod.w_scales = base.w_scales
        mod.w_int8 = base.w_int8
        return mod
