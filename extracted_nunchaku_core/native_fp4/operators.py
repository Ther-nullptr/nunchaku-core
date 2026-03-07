from __future__ import annotations

import importlib.util
from pathlib import Path

import torch


def _load_nunchaku_c_ops():
    repo_root = Path(__file__).resolve().parents[2]
    candidates = sorted((repo_root / "nunchaku").glob("_C*.so"))
    if not candidates:
        raise RuntimeError(f"Cannot find nunchaku extension under {repo_root / 'nunchaku'}")

    so_path = candidates[0]
    # Extension exports PyInit__C.
    spec = importlib.util.spec_from_file_location("_C", so_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load extension spec: {so_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.ops


_OPS = _load_nunchaku_c_ops()


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

        weight_pad = pad_tensor(weight, divisor=(128, 128), dim=(0, 1))
        self.register_buffer("weight_pad", weight_pad.contiguous(), persistent=False)

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
        out = torch.empty(qact.shape[0], self.n_pad, dtype=self.weight_pad.dtype, device=qact.device)

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
        u, s, vh = torch.linalg.svd(self.weight_pad.float(), full_matrices=False)
        eff_rank = min(rank, s.numel())

        lora_up_dense = (u[:, :eff_rank] * s[:eff_rank].unsqueeze(0)).to(self.weight_pad.dtype)
        lora_down_dense = vh[:eff_rank, :].to(self.weight_pad.dtype)

        if eff_rank < rank:
            up_pad = torch.zeros(self.n_pad, rank - eff_rank, dtype=self.weight_pad.dtype, device=self.weight_pad.device)
            down_pad = torch.zeros(rank - eff_rank, self.k_pad, dtype=self.weight_pad.dtype, device=self.weight_pad.device)
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
        out = torch.empty(qact.shape[0], self.n_pad, dtype=self.weight_pad.dtype, device=qact.device)

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
