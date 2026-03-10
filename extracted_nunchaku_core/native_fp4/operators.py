from __future__ import annotations

import torch

from .layout import build_backward_scales_from_forward_quant, repack_fp4_weight_for_backward, unpack_fp4_weight_scales


def _load_fp4_backend_ops():
    try:
        from nunchaku_core import _fp4_native_cuda as module
    except Exception as exc:
        raise RuntimeError(
            "Failed to import nunchaku_core._fp4_native_cuda. "
            "Please build extracted_nunchaku_core with `python setup.py build_ext --inplace` first."
        ) from exc
    return module


_OPS = _load_fp4_backend_ops()


def ceil_divide(x: int, y: int) -> int:
    return (x + y - 1) // y


def pad_tensor(
    tensor: torch.Tensor,
    divisor: int | tuple[int, ...],
    dim: int | tuple[int, ...],
    fill_value: float | int = 0,
) -> torch.Tensor:
    if isinstance(divisor, int):
        divisor = (divisor,)
    if isinstance(dim, int):
        dim = (dim,)

    if len(divisor) != len(dim):
        raise ValueError("divisor and dim should have the same length")

    shape = list(tensor.shape)
    for d, div in zip(dim, divisor, strict=True):
        shape[d] = ceil_divide(shape[d], div) * div

    out = torch.full(shape, fill_value, dtype=tensor.dtype, device=tensor.device)
    out[tuple(slice(0, extent) for extent in tensor.shape)] = tensor
    return out


def pack_lowrank_weight(weight: torch.Tensor, down: bool) -> torch.Tensor:
    """Pack low-rank weights exactly following nunchaku_converter.pack_lowrank_weight."""

    if weight.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError(f"Unsupported weight dtype: {weight.dtype}")

    lane_n, lane_k = 1, 2
    n_pack_size, k_pack_size = 2, 2
    num_n_lanes, num_k_lanes = 8, 4

    frag_n = n_pack_size * num_n_lanes * lane_n
    frag_k = k_pack_size * num_k_lanes * lane_k

    weight = pad_tensor(weight, divisor=(frag_n, frag_k), dim=(0, 1))

    if down:
        r, c = weight.shape
        r_frags, c_frags = r // frag_n, c // frag_k
        weight = weight.view(r_frags, frag_n, c_frags, frag_k).permute(2, 0, 1, 3)
    else:
        c, r = weight.shape
        c_frags, r_frags = c // frag_n, r // frag_k
        weight = weight.view(c_frags, frag_n, r_frags, frag_k).permute(0, 2, 1, 3)

    weight = weight.reshape(c_frags, r_frags, n_pack_size, num_n_lanes, k_pack_size, num_k_lanes, lane_k)
    weight = weight.permute(0, 1, 3, 5, 2, 4, 6).contiguous()

    return weight.view(c, r)


def quantize_fp4_act_with_lora(
    x: torch.Tensor,
    lora_down_packed: torch.Tensor,
    smooth: torch.Tensor,
    pad_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if x.dim() != 2:
        raise ValueError(f"Expected x to be 2D, got shape={tuple(x.shape)}")

    batch_size, channels = x.shape
    rank = lora_down_packed.shape[1]
    batch_size_pad = ceil_divide(batch_size, pad_size) * pad_size

    qout = torch.empty(batch_size_pad, channels // 2, dtype=torch.uint8, device=x.device)
    oscales = torch.empty(channels // 16, batch_size_pad, dtype=torch.float8_e4m3fn, device=x.device)
    lora_act_out = torch.empty(batch_size_pad, rank, dtype=torch.float32, device=x.device)

    _OPS.quantize_w4a4_act_fuse_lora(x, qout, oscales, lora_down_packed, lora_act_out, smooth, False, True)
    return qout, oscales, lora_act_out


class NunchakuFP4GemmOp(torch.nn.Module):
    """Native nunchaku FP4 GEMM operator (main branch only)."""

    def __init__(self, weight: torch.Tensor, bias: torch.Tensor | None = None, dummy_rank: int = 16):
        super().__init__()
        if not weight.is_cuda:
            raise ValueError("weight must be on CUDA")
        if weight.dtype not in (torch.float16, torch.bfloat16):
            raise ValueError("weight dtype must be float16 or bfloat16")
        if weight.dim() != 2:
            raise ValueError("weight should have shape [out_features, in_features]")
        if dummy_rank <= 0 or dummy_rank % 16 != 0:
            raise ValueError("dummy_rank must be positive and divisible by 16")

        self.out_features, self.in_features = weight.shape
        self.k_pad = ceil_divide(self.in_features, 128) * 128
        self.n_pad = ceil_divide(self.out_features, 128) * 128
        self.compute_dtype = weight.dtype

        weight_pad = pad_tensor(weight, divisor=(128, 128), dim=(0, 1))

        smooth = torch.ones(self.k_pad, dtype=weight.dtype, device=weight.device)
        self.register_buffer("smooth", smooth, persistent=False)

        dummy_down_dense = torch.zeros(dummy_rank, self.k_pad, dtype=weight.dtype, device=weight.device)
        dummy_down_packed = pack_lowrank_weight(dummy_down_dense, down=True)
        self.register_buffer("dummy_down_packed", dummy_down_packed.contiguous(), persistent=False)

        qweight, wscales, _ = quantize_fp4_act_with_lora(
            weight_pad,
            lora_down_packed=self.dummy_down_packed,
            smooth=self.smooth,
            pad_size=128,
        )
        self.register_buffer("qweight", qweight.contiguous(), persistent=True)
        self.register_buffer("wscales", wscales.contiguous(), persistent=True)

        if bias is None:
            self.register_buffer("bias_pad", None, persistent=True)
        else:
            bias_pad = pad_tensor(bias.to(weight.dtype), divisor=128, dim=0)
            self.register_buffer("bias_pad", bias_pad.contiguous(), persistent=True)

    def quantize_input(self, x2d: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        qact, ascales, _ = quantize_fp4_act_with_lora(
            x2d,
            lora_down_packed=self.dummy_down_packed,
            smooth=self.smooth,
            pad_size=256,
        )
        return qact, ascales

    def forward_prequantized(self, qact: torch.Tensor, ascales: torch.Tensor) -> torch.Tensor:
        out = torch.empty(qact.shape[0], self.n_pad, dtype=self.compute_dtype, device=qact.device)

        _OPS.gemm_w4a4(
            qact,
            self.qweight,
            out,
            None,
            ascales,
            self.wscales,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            self.bias_pad,
            None,
            None,
            None,
            False,
            [],
            False,
            True,
            1.0,
            None,
            None,
            None,
            None,
            0,
        )
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.in_features:
            raise ValueError(f"Expected input last dim = {self.in_features}, got {x.shape[-1]}")

        orig_shape = x.shape
        x2d_src = x.reshape(-1, self.in_features)
        x2d = x2d_src
        if self.k_pad != self.in_features:
            x2d = pad_tensor(x2d, divisor=self.k_pad, dim=1)

        qact, ascales = self.quantize_input(x2d)
        out = self.forward_prequantized(qact, ascales)

        out = out[: x2d_src.shape[0], : self.out_features]
        return out.reshape(*orig_shape[:-1], self.out_features)


class NunchakuFP4LowRankOp(NunchakuFP4GemmOp):
    """Native nunchaku FP4 + FP16 low-rank hybrid operator."""

    def __init__(self, weight: torch.Tensor, bias: torch.Tensor | None = None, rank: int = 32):
        rank = max(16, ceil_divide(rank, 16) * 16)
        super().__init__(weight=weight, bias=bias, dummy_rank=rank)
        self.rank = rank

        # Build low-rank branch by truncated SVD on padded FP16 weight.
        weight_pad = pad_tensor(weight, divisor=(128, 128), dim=(0, 1))
        u, s, vh = torch.linalg.svd(weight_pad.float(), full_matrices=False)
        eff_rank = min(rank, s.numel())

        lora_up_dense = (u[:, :eff_rank] * s[:eff_rank].unsqueeze(0)).to(self.compute_dtype)
        lora_down_dense = vh[:eff_rank, :].to(self.compute_dtype)

        if eff_rank < rank:
            up_pad = torch.zeros(self.n_pad, rank - eff_rank, dtype=self.compute_dtype, device=weight.device)
            down_pad = torch.zeros(rank - eff_rank, self.k_pad, dtype=self.compute_dtype, device=weight.device)
            lora_up_dense = torch.cat([lora_up_dense, up_pad], dim=1)
            lora_down_dense = torch.cat([lora_down_dense, down_pad], dim=0)

        lora_down_packed = pack_lowrank_weight(lora_down_dense, down=True)
        lora_up_packed = pack_lowrank_weight(lora_up_dense, down=False)

        # Keep dense low-rank factors for validation/debug (not saved in checkpoints).
        self.register_buffer("lora_down_dense", lora_down_dense.contiguous(), persistent=False)
        self.register_buffer("lora_up_dense", lora_up_dense.contiguous(), persistent=False)
        self.register_buffer("lora_down_packed", lora_down_packed.contiguous(), persistent=True)
        self.register_buffer("lora_up_packed", lora_up_packed.contiguous(), persistent=True)
        self._lora_scales = [1.0] * ceil_divide(rank, 16)

    def quantize_input_with_lora(self, x2d: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        qact, ascales, lora_act = quantize_fp4_act_with_lora(
            x2d,
            lora_down_packed=self.lora_down_packed,
            smooth=self.smooth,
            pad_size=256,
        )
        return qact, ascales, lora_act

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.in_features:
            raise ValueError(f"Expected input last dim = {self.in_features}, got {x.shape[-1]}")

        orig_shape = x.shape
        x2d_src = x.reshape(-1, self.in_features)
        x2d = x2d_src
        if self.k_pad != self.in_features:
            x2d = pad_tensor(x2d, divisor=self.k_pad, dim=1)

        qact, ascales, lora_act = self.quantize_input_with_lora(x2d)
        out = torch.empty(qact.shape[0], self.n_pad, dtype=self.compute_dtype, device=qact.device)

        _OPS.gemm_w4a4(
            qact,
            self.qweight,
            out,
            None,
            ascales,
            self.wscales,
            None,
            None,
            lora_act,
            self.lora_up_packed,
            None,
            None,
            None,
            None,
            None,
            self.bias_pad,
            None,
            None,
            None,
            False,
            self._lora_scales,
            False,
            True,
            1.0,
            None,
            None,
            None,
            None,
            0,
        )

        out = out[: x2d_src.shape[0], : self.out_features]
        return out.reshape(*orig_shape[:-1], self.out_features)


class NunchakuFP4LowRankUnfusedOp(NunchakuFP4LowRankOp):
    """FP4 main branch + standalone BF16/FP16 low-rank branch (no kernel fusion)."""

    def __init__(
        self,
        weight: torch.Tensor,
        bias: torch.Tensor | None = None,
        rank: int = 32,
        lowrank_dtype: torch.dtype = torch.bfloat16,
    ):
        if lowrank_dtype not in (torch.float16, torch.bfloat16):
            raise ValueError("lowrank_dtype must be float16 or bfloat16")
        super().__init__(weight=weight, bias=bias, rank=rank)
        self.lowrank_dtype = lowrank_dtype
        self.register_buffer(
            "lora_up_dense_lr",
            self.lora_up_dense.to(lowrank_dtype).contiguous(),
            persistent=False,
        )
        self.register_buffer(
            "lora_down_dense_lr",
            self.lora_down_dense.to(lowrank_dtype).contiguous(),
            persistent=False,
        )

    def lowrank_only_padded(self, x2d: torch.Tensor) -> torch.Tensor:
        x_lr = x2d.to(self.lowrank_dtype)
        lora_act = torch.matmul(x_lr, self.lora_down_dense_lr.t())
        lora_out = torch.matmul(lora_act, self.lora_up_dense_lr.t())
        return lora_out.to(self.compute_dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.in_features:
            raise ValueError(f"Expected input last dim = {self.in_features}, got {x.shape[-1]}")

        orig_shape = x.shape
        x2d_src = x.reshape(-1, self.in_features)
        x2d = x2d_src
        if self.k_pad != self.in_features:
            x2d = pad_tensor(x2d, divisor=self.k_pad, dim=1)

        qact, ascales = self.quantize_input(x2d)
        out_main = self.forward_prequantized(qact, ascales)
        out = out_main + self.lowrank_only_padded(x2d)

        out = out[: x2d_src.shape[0], : self.out_features]
        return out.reshape(*orig_shape[:-1], self.out_features)


class NunchakuFP4BackwardDXOp(NunchakuFP4GemmOp):
    """Backward dX operator using transient FP4 repack for W^T."""

    def __init__(self, weight: torch.Tensor, dummy_rank: int = 16):
        super().__init__(weight=weight, bias=None, dummy_rank=dummy_rank)

        wscales_fwd_logical = unpack_fp4_weight_scales(self.wscales, self.n_pad, self.k_pad).to(torch.float16).contiguous()
        self.register_buffer("wscales_fwd_logical", wscales_fwd_logical, persistent=False)

        logical_scales_bwd, packed_scales_bwd = build_backward_scales_from_forward_quant(
            qweight=self.qweight,
            packed_wscales=self.wscales,
            out_features=self.n_pad,
            in_features=self.k_pad,
        )
        self.register_buffer("wscales_bwd_logical", logical_scales_bwd.to(torch.float16).contiguous(), persistent=False)
        self.register_buffer("wscales_bwd_packed", packed_scales_bwd.contiguous(), persistent=True)

        smooth_bwd = torch.ones(self.n_pad, dtype=self.compute_dtype, device=weight.device)
        self.register_buffer("smooth_bwd", smooth_bwd, persistent=False)

        dummy_down_bwd = torch.zeros(dummy_rank, self.n_pad, dtype=self.compute_dtype, device=weight.device)
        dummy_down_bwd_packed = pack_lowrank_weight(dummy_down_bwd, down=True)
        self.register_buffer("dummy_down_bwd_packed", dummy_down_bwd_packed.contiguous(), persistent=False)

    def quantize_grad(self, dy2d: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        qact, ascales, _ = quantize_fp4_act_with_lora(
            dy2d,
            lora_down_packed=self.dummy_down_bwd_packed,
            smooth=self.smooth_bwd,
            pad_size=256,
        )
        return qact, ascales

    def repack_qweight_for_backward(self) -> torch.Tensor:
        if hasattr(_OPS, "fp4_repack_backward"):
            return _OPS.fp4_repack_backward(
                self.qweight,
                self.wscales_fwd_logical,
                self.wscales_bwd_logical,
            )
        return repack_fp4_weight_for_backward(
            qweight=self.qweight,
            packed_wscales=self.wscales,
            out_features=self.n_pad,
            in_features=self.k_pad,
            logical_scales_t=self.wscales_bwd_logical,
        )

    def backward_prequantized(
        self,
        qdy: torch.Tensor,
        ascales: torch.Tensor,
        qweight_bwd: torch.Tensor,
        lora_act: torch.Tensor | None = None,
        lora_up: torch.Tensor | None = None,
        lora_scales: list[float] | None = None,
    ) -> torch.Tensor:
        out = torch.empty(qdy.shape[0], self.k_pad, dtype=self.compute_dtype, device=qdy.device)
        _OPS.gemm_w4a4(
            qdy,
            qweight_bwd,
            out,
            None,
            ascales,
            self.wscales_bwd_packed,
            None,
            None,
            lora_act,
            lora_up,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            False,
            [] if lora_scales is None else lora_scales,
            False,
            True,
            1.0,
            None,
            None,
            None,
            None,
            0,
        )
        return out

    def forward(self, dy: torch.Tensor) -> torch.Tensor:
        if dy.shape[-1] != self.out_features:
            raise ValueError(f"Expected grad last dim = {self.out_features}, got {dy.shape[-1]}")

        orig_shape = dy.shape
        dy2d = dy.reshape(-1, self.out_features)
        if self.n_pad != self.out_features:
            dy2d = pad_tensor(dy2d, divisor=self.n_pad, dim=1)

        qdy, ascales = self.quantize_grad(dy2d)
        qweight_bwd = self.repack_qweight_for_backward()
        out = self.backward_prequantized(qdy, ascales, qweight_bwd)
        out = out[: dy2d.shape[0], : self.in_features]
        return out.reshape(*orig_shape[:-1], self.in_features)


class NunchakuFP4LowRankBackwardDXOp(NunchakuFP4BackwardDXOp):
    """Backward dX operator with fused FP4 main branch and FP16/BF16 low-rank branch."""

    def __init__(
        self,
        weight: torch.Tensor,
        rank: int = 32,
        lowrank_dtype: torch.dtype = torch.bfloat16,
    ):
        if lowrank_dtype not in (torch.float16, torch.bfloat16):
            raise ValueError("lowrank_dtype must be float16 or bfloat16")

        rank = max(16, ceil_divide(rank, 16) * 16)
        super().__init__(weight=weight, dummy_rank=rank)
        self.rank = rank
        self.lowrank_dtype = lowrank_dtype

        weight_pad = pad_tensor(weight, divisor=(128, 128), dim=(0, 1))
        u, s, vh = torch.linalg.svd(weight_pad.float(), full_matrices=False)
        eff_rank = min(rank, s.numel())

        lora_up_dense = (u[:, :eff_rank] * s[:eff_rank].unsqueeze(0)).to(self.compute_dtype)
        lora_down_dense = vh[:eff_rank, :].to(self.compute_dtype)
        if eff_rank < rank:
            up_pad = torch.zeros(self.n_pad, rank - eff_rank, dtype=self.compute_dtype, device=weight.device)
            down_pad = torch.zeros(rank - eff_rank, self.k_pad, dtype=self.compute_dtype, device=weight.device)
            lora_up_dense = torch.cat([lora_up_dense, up_pad], dim=1)
            lora_down_dense = torch.cat([lora_down_dense, down_pad], dim=0)

        self.register_buffer("lora_down_dense", lora_down_dense.contiguous(), persistent=False)
        self.register_buffer("lora_up_dense", lora_up_dense.contiguous(), persistent=False)
        self.register_buffer("lora_down_dense_lr", lora_down_dense.to(lowrank_dtype).contiguous(), persistent=False)
        self.register_buffer("lora_up_dense_lr", lora_up_dense.to(lowrank_dtype).contiguous(), persistent=False)

        lora_down_bwd_dense = lora_up_dense.t().contiguous()
        lora_up_bwd_dense = lora_down_dense.t().contiguous()
        self.register_buffer("lora_down_bwd_dense", lora_down_bwd_dense.contiguous(), persistent=False)
        self.register_buffer("lora_up_bwd_dense", lora_up_bwd_dense.contiguous(), persistent=False)

        lora_down_bwd_packed = pack_lowrank_weight(lora_down_bwd_dense, down=True)
        lora_up_bwd_packed = pack_lowrank_weight(lora_up_bwd_dense, down=False)
        self.register_buffer("lora_down_bwd_packed", lora_down_bwd_packed.contiguous(), persistent=True)
        self.register_buffer("lora_up_bwd_packed", lora_up_bwd_packed.contiguous(), persistent=True)
        self._lora_scales = [1.0] * ceil_divide(rank, 16)

    def quantize_grad_with_lora(self, dy2d: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return quantize_fp4_act_with_lora(
            dy2d,
            lora_down_packed=self.lora_down_bwd_packed,
            smooth=self.smooth_bwd,
            pad_size=256,
        )

    def lowrank_only_padded(self, dy2d: torch.Tensor) -> torch.Tensor:
        dy_lr = dy2d.to(self.lowrank_dtype)
        lora_mid = torch.matmul(dy_lr, self.lora_up_dense_lr)
        dX_lora = torch.matmul(lora_mid, self.lora_down_dense_lr)
        return dX_lora.to(self.compute_dtype)

    def forward_unfused(self, dy: torch.Tensor) -> torch.Tensor:
        if dy.shape[-1] != self.out_features:
            raise ValueError(f"Expected grad last dim = {self.out_features}, got {dy.shape[-1]}")

        orig_shape = dy.shape
        dy2d_src = dy.reshape(-1, self.out_features)
        dy2d = dy2d_src
        if self.n_pad != self.out_features:
            dy2d = pad_tensor(dy2d, divisor=self.n_pad, dim=1)

        qdy, ascales = self.quantize_grad(dy2d)
        dy2d_pad = dy2d if qdy.shape[0] == dy2d.shape[0] else pad_tensor(dy2d, divisor=qdy.shape[0], dim=0)
        qweight_bwd = self.repack_qweight_for_backward()
        out_main = self.backward_prequantized(qdy, ascales, qweight_bwd)
        out = out_main + self.lowrank_only_padded(dy2d_pad)
        out = out[: dy2d_src.shape[0], : self.in_features]
        return out.reshape(*orig_shape[:-1], self.in_features)

    def forward(self, dy: torch.Tensor) -> torch.Tensor:
        if dy.shape[-1] != self.out_features:
            raise ValueError(f"Expected grad last dim = {self.out_features}, got {dy.shape[-1]}")

        orig_shape = dy.shape
        dy2d_src = dy.reshape(-1, self.out_features)
        dy2d = dy2d_src
        if self.n_pad != self.out_features:
            dy2d = pad_tensor(dy2d, divisor=self.n_pad, dim=1)

        qdy, ascales, lora_act = self.quantize_grad_with_lora(dy2d)
        qweight_bwd = self.repack_qweight_for_backward()
        out = self.backward_prequantized(
            qdy,
            ascales,
            qweight_bwd,
            lora_act=lora_act,
            lora_up=self.lora_up_bwd_packed,
            lora_scales=self._lora_scales,
        )
        out = out[: dy2d_src.shape[0], : self.in_features]
        return out.reshape(*orig_shape[:-1], self.in_features)

    def backward_full(self, x: torch.Tensor, dy: torch.Tensor, fused_dx: bool = True) -> dict[str, torch.Tensor]:
        if x.shape[-1] != self.in_features:
            raise ValueError(f"Expected x last dim = {self.in_features}, got {x.shape[-1]}")
        if dy.shape[-1] != self.out_features:
            raise ValueError(f"Expected dy last dim = {self.out_features}, got {dy.shape[-1]}")

        x2d_src = x.reshape(-1, self.in_features)
        dy2d_src = dy.reshape(-1, self.out_features)
        x2d = x2d_src if self.k_pad == self.in_features else pad_tensor(x2d_src, divisor=self.k_pad, dim=1)
        dy2d = dy2d_src if self.n_pad == self.out_features else pad_tensor(dy2d_src, divisor=self.n_pad, dim=1)

        dX = self.forward(dy) if fused_dx else self.forward_unfused(dy)

        x_lr = x2d.to(self.lowrank_dtype)
        dy_lr = dy2d.to(self.lowrank_dtype)
        lora_act = torch.matmul(x_lr, self.lora_down_dense_lr.t())
        dy_up = torch.matmul(dy_lr, self.lora_up_dense_lr)

        d_up = torch.matmul(dy_lr.t(), lora_act)
        d_down = torch.matmul(dy_up.t(), x_lr)

        return {
            "dx": dX,
            "lora_up_grad": d_up[: self.out_features, : self.rank].to(self.compute_dtype),
            "lora_down_grad": d_down[: self.rank, : self.in_features].to(self.compute_dtype),
        }
