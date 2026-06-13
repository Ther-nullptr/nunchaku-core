from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict
from typing import Any

import torch
import torch.nn.functional as F

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from native_fp4 import FP4LoRAConfig, iter_fp4_lora_modules, prepare_fp4_lora_finetuning  # noqa: E402


MODES = ("accuracy", "balanced", "throughput", "memory_saving")
TARGET_MODULES = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
EXCLUDE_MODULES = ("lm_head",)


class TinyTransformerBlock(torch.nn.Module):
    def __init__(self, hidden: int, dtype: torch.dtype):
        super().__init__()
        self.q_proj = torch.nn.Linear(hidden, hidden, bias=False, dtype=dtype)
        self.k_proj = torch.nn.Linear(hidden, hidden, bias=False, dtype=dtype)
        self.v_proj = torch.nn.Linear(hidden, hidden, bias=False, dtype=dtype)
        self.o_proj = torch.nn.Linear(hidden, hidden, bias=False, dtype=dtype)
        self.gate_proj = torch.nn.Linear(hidden, hidden, bias=False, dtype=dtype)
        self.up_proj = torch.nn.Linear(hidden, hidden, bias=False, dtype=dtype)
        self.down_proj = torch.nn.Linear(hidden, hidden, bias=True, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn = self.o_proj(F.silu(self.q_proj(x) + self.k_proj(x) + self.v_proj(x)))
        mlp = self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
        return x + attn + mlp


class TinyTransformer(torch.nn.Module):
    def __init__(self, hidden: int, layers: int, dtype: torch.dtype):
        super().__init__()
        self.layers = torch.nn.ModuleList([TinyTransformerBlock(hidden, dtype) for _ in range(layers)])
        self.lm_head = torch.nn.Linear(hidden, hidden, bias=False, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return self.lm_head(x)


class DenseLoRALinear(torch.nn.Module):
    def __init__(
        self,
        linear: torch.nn.Linear,
        *,
        rank: int,
        lora_alpha: float | None,
        lowrank_dtype: torch.dtype,
        train_bias: bool,
    ):
        super().__init__()
        self.out_features, self.in_features = linear.weight.shape
        self.rank = max(16, ((int(rank) + 15) // 16) * 16)
        self.requested_rank = int(rank)
        self.lora_alpha = float(self.rank if lora_alpha is None else lora_alpha)
        self.scaling = self.lora_alpha / float(self.rank)
        self.lowrank_dtype = lowrank_dtype
        self.register_buffer("weight", linear.weight.detach().contiguous(), persistent=True)
        if linear.bias is None:
            self.register_parameter("bias", None)
        elif train_bias:
            self.bias = torch.nn.Parameter(linear.bias.detach().contiguous())
        else:
            self.register_buffer("bias", linear.bias.detach().contiguous(), persistent=True)
        self.lora_down = torch.nn.Parameter(
            torch.empty(self.rank, self.in_features, device=linear.weight.device, dtype=lowrank_dtype)
        )
        self.lora_up = torch.nn.Parameter(
            torch.empty(self.out_features, self.rank, device=linear.weight.device, dtype=lowrank_dtype)
        )
        self.reset_lora_parameters()

    def reset_lora_parameters(self) -> None:
        torch.nn.init.kaiming_uniform_(self.lora_down, a=5**0.5)
        torch.nn.init.zeros_(self.lora_up)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.linear(x, self.weight, self.bias)
        x2d = x.reshape(-1, self.in_features).to(self.lowrank_dtype)
        lora_act = torch.matmul(x2d, self.lora_down.t())
        lora_out = torch.matmul(lora_act, self.lora_up.t()).mul(self.scaling)
        return y + lora_out.to(y.dtype).reshape(*x.shape[:-1], self.out_features)


def dtype_from_name(name: str) -> torch.dtype:
    if name == "fp16":
        return torch.float16
    if name == "bf16":
        return torch.bfloat16
    raise ValueError(f"Unsupported dtype: {name}")


def name_matches(full_name: str, child_name: str, patterns: tuple[str, ...]) -> bool:
    return any(full_name == pattern or child_name == pattern or full_name.endswith(f".{pattern}") for pattern in patterns)


def convert_linear_to_dense_lora(
    module: torch.nn.Module,
    *,
    rank: int,
    lora_alpha: float | None,
    lowrank_dtype: torch.dtype,
    train_bias: bool,
    target_modules: tuple[str, ...] = TARGET_MODULES,
    exclude_modules: tuple[str, ...] = EXCLUDE_MODULES,
) -> list[str]:
    replaced: list[str] = []

    def visit(parent: torch.nn.Module, prefix: str) -> None:
        for child_name, child in list(parent.named_children()):
            full_name = f"{prefix}.{child_name}" if prefix else child_name
            if isinstance(child, torch.nn.Linear):
                is_target = name_matches(full_name, child_name, target_modules)
                is_excluded = name_matches(full_name, child_name, exclude_modules)
                if is_target and not is_excluded:
                    setattr(
                        parent,
                        child_name,
                        DenseLoRALinear(
                            child,
                            rank=rank,
                            lora_alpha=lora_alpha,
                            lowrank_dtype=lowrank_dtype,
                            train_bias=train_bias,
                        ),
                    )
                    replaced.append(full_name)
                    continue
            visit(child, full_name)

    visit(module, "")
    return replaced


def jsonable_config(cfg: FP4LoRAConfig) -> dict[str, Any]:
    data = asdict(cfg)
    data["lowrank_dtype"] = str(cfg.lowrank_dtype).replace("torch.", "")
    return data


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
    values: list[float] = []
    for _ in range(iters):
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        values.append(start.elapsed_time(end))
    return float(sum(values) / len(values))


def measure_peak_delta(fn) -> tuple[int, int, int]:
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    baseline = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    fn()
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated()
    return int(peak - baseline), int(baseline), int(peak)


def zero_grads(model: torch.nn.Module, x: torch.Tensor, optimizer: torch.optim.Optimizer | None = None) -> None:
    if optimizer is not None:
        optimizer.zero_grad(set_to_none=True)
    else:
        model.zero_grad(set_to_none=True)
    x.grad = None


def train_step(
    model: torch.nn.Module,
    x: torch.Tensor,
    target: torch.Tensor,
    optimizer: torch.optim.Optimizer,
) -> tuple[torch.Tensor, torch.Tensor]:
    zero_grads(model, x, optimizer)
    y = model(x)
    loss = F.mse_loss(y.float(), target.float())
    loss.backward()
    optimizer.step()
    return y, loss


def build_dense_state(args: argparse.Namespace, dtype: torch.dtype) -> dict[str, torch.Tensor]:
    torch.manual_seed(args.seed)
    dense = TinyTransformer(args.hidden, args.layers, dtype=dtype).cuda()
    return {key: value.detach().clone() for key, value in dense.state_dict().items()}


def make_inputs(args: argparse.Namespace, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(args.seed + 1)
    x = torch.randn(args.batch, args.hidden, device="cuda", dtype=dtype, requires_grad=True)
    target = torch.randn(args.batch, args.hidden, device="cuda", dtype=dtype)
    return x, target


def mode_backend_records(args: argparse.Namespace) -> list[tuple[str, str, str]]:
    records: list[tuple[str, str, str]] = []
    for mode in args.modes:
        if mode == "memory_saving":
            for backend in args.memory_saving_backends:
                records.append((f"{mode}_{backend}", mode, backend))
        else:
            records.append((mode, mode, args.fp4_activation_cache_d_lora_down_backend))
    return records


def run_record(
    *,
    args: argparse.Namespace,
    record_name: str,
    mode: str,
    backend: str,
    dense_state: dict[str, torch.Tensor],
    dense_y_ref: torch.Tensor,
    dtype: torch.dtype,
    lowrank_dtype: torch.dtype,
) -> dict[str, Any]:
    torch.manual_seed(args.seed)
    model = TinyTransformer(args.hidden, args.layers, dtype=dtype).cuda()
    model.load_state_dict(dense_state)
    model.train()

    result = prepare_fp4_lora_finetuning(
        model,
        mode=mode,
        rank=args.rank,
        dtype=dtype,
        lowrank_dtype=lowrank_dtype,
        use_frozen_residual=not args.no_frozen_residual,
        frozen_residual_rank=args.frozen_residual_rank,
        train_bias=args.train_bias,
        cache_lora_act=not args.no_cache_lora_act,
        activation_checkpoint=args.activation_checkpoint,
        fp4_activation_cache_d_lora_down_backend=backend,
        lr=args.lr,
        lora_weight_decay=args.lora_weight_decay,
        bias_weight_decay=args.bias_weight_decay,
    )
    optimizer = torch.optim.AdamW(result.optimizer_param_groups, lr=args.lr, eps=args.adam_eps)
    hook = result.register_cache_refresh_hook(optimizer) if result.config.cache_fused_lora_dx else None
    x, target = make_inputs(args, dtype)

    with torch.no_grad():
        initial_y = result.model(x.detach())
    initial_error = tensor_error(initial_y, dense_y_ref)

    def fn() -> None:
        train_step(result.model, x, target, optimizer)

    latency_ms = time_cuda(fn, warmup=args.warmup, iters=args.iters)
    peak_delta, peak_baseline, peak = measure_peak_delta(fn)
    y, loss = train_step(result.model, x, target, optimizer)
    torch.cuda.synchronize()

    fp4_modules = dict(iter_fp4_lora_modules(result.model))
    all_module_backends_match = all(
        child.fp4_activation_cache_d_lora_down_backend == backend for child in fp4_modules.values()
    )
    grads_finite = all(
        param.grad is not None and bool(torch.isfinite(param.grad).all())
        for group in result.optimizer_param_groups
        for param in group["params"]
    )
    x_grad_finite = bool(x.grad is not None and torch.isfinite(x.grad).all())
    cache_hook_count = None if hook is None else hook.last_refresh_count
    if hook is not None:
        hook.remove()

    expected_replaced = args.layers * 7
    checks = {
        "replaced_count_matches": len(result.replaced_modules) == expected_replaced,
        "trainable_param_count_positive": result.trainable_param_count > 0,
        "loss_finite": bool(torch.isfinite(loss)),
        "output_finite": bool(torch.isfinite(y).all()),
        "x_grad_finite": x_grad_finite,
        "trainable_grads_finite": grads_finite,
        "module_backends_match": all_module_backends_match,
        "latency_positive": latency_ms > 0.0,
        "peak_delta_nonnegative": peak_delta >= 0,
    }

    return {
        "record": record_name,
        "mode": mode,
        "backend": backend,
        "config": jsonable_config(result.config),
        "replaced_count": len(result.replaced_modules),
        "trainable_param_count": result.trainable_param_count,
        "refreshed_cache_count": result.refreshed_cache_count,
        "cache_hook_refresh_count": cache_hook_count,
        "latency_ms": {
            "train_step_with_optimizer": latency_ms,
        },
        "throughput": {
            "steps_per_second": 1000.0 / latency_ms,
            "samples_per_second": args.batch * 1000.0 / latency_ms,
        },
        "peak_memory_bytes": {
            "train_step_delta": peak_delta,
            "baseline": peak_baseline,
            "peak": peak,
        },
        "initial_forward_vs_dense": initial_error,
        "final_loss": float(loss.detach().item()),
        "checks": checks,
        "all_passed": bool(all(checks.values())),
    }


def run_dense_lora_baseline(
    *,
    args: argparse.Namespace,
    dense_state: dict[str, torch.Tensor],
    dense_y_ref: torch.Tensor,
    dtype: torch.dtype,
    lowrank_dtype: torch.dtype,
) -> dict[str, Any]:
    torch.manual_seed(args.seed)
    model = TinyTransformer(args.hidden, args.layers, dtype=dtype).cuda()
    model.load_state_dict(dense_state)
    model.train()
    for param in model.parameters():
        param.requires_grad_(False)
    replaced = convert_linear_to_dense_lora(
        model,
        rank=args.rank,
        lora_alpha=None,
        lowrank_dtype=lowrank_dtype,
        train_bias=args.train_bias,
    )
    named_params = [(name, param) for name, param in model.named_parameters() if param.requires_grad]
    lora_params = [param for name, param in named_params if not name.endswith("bias")]
    bias_params = [param for name, param in named_params if name.endswith("bias")]
    param_groups: list[dict[str, Any]] = []
    if lora_params:
        param_groups.append({"params": lora_params, "weight_decay": args.lora_weight_decay})
    if bias_params:
        param_groups.append({"params": bias_params, "weight_decay": args.bias_weight_decay})
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, eps=args.adam_eps)
    x, target = make_inputs(args, dtype)

    with torch.no_grad():
        initial_y = model(x.detach())
    initial_error = tensor_error(initial_y, dense_y_ref)

    def fn() -> None:
        train_step(model, x, target, optimizer)

    latency_ms = time_cuda(fn, warmup=args.warmup, iters=args.iters)
    peak_delta, peak_baseline, peak = measure_peak_delta(fn)
    y, loss = train_step(model, x, target, optimizer)
    torch.cuda.synchronize()

    trainable_param_count = int(sum(param.numel() for _, param in named_params))
    trainable_names = {name for name, param in model.named_parameters() if param.requires_grad}
    grads_finite = all(param.grad is not None and bool(torch.isfinite(param.grad).all()) for _, param in named_params)
    checks = {
        "replaced_count_matches": len(replaced) == args.layers * 7,
        "trainable_param_count_positive": trainable_param_count > 0,
        "only_lora_or_trainable_bias_trainable": all(
            name.endswith("lora_down")
            or name.endswith("lora_up")
            or (args.train_bias and name.endswith("bias") and "lm_head" not in name)
            for name in trainable_names
        ),
        "loss_finite": bool(torch.isfinite(loss)),
        "output_finite": bool(torch.isfinite(y).all()),
        "x_grad_finite": bool(x.grad is not None and torch.isfinite(x.grad).all()),
        "trainable_grads_finite": grads_finite,
        "latency_positive": latency_ms > 0.0,
        "peak_delta_nonnegative": peak_delta >= 0,
    }
    return {
        "record": "dense_lora",
        "replaced_count": len(replaced),
        "trainable_param_count": trainable_param_count,
        "trainable_names": sorted(trainable_names),
        "latency_ms": {
            "train_step_with_optimizer": latency_ms,
        },
        "throughput": {
            "steps_per_second": 1000.0 / latency_ms,
            "samples_per_second": args.batch * 1000.0 / latency_ms,
        },
        "peak_memory_bytes": {
            "train_step_delta": peak_delta,
            "baseline": peak_baseline,
            "peak": peak,
        },
        "initial_forward_vs_dense": initial_error,
        "final_loss": float(loss.detach().item()),
        "checks": checks,
        "all_passed": bool(all(checks.values())),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Benchmark model-level FP4 LoRA prepare() policy presets.")
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--rank", type=int, default=32)
    p.add_argument("--frozen-residual-rank", type=int, default=None)
    p.add_argument("--dtype", choices=["fp16", "bf16"], default="bf16")
    p.add_argument("--lowrank-dtype", choices=["fp16", "bf16"], default="bf16")
    p.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    p.add_argument(
        "--memory-saving-backends",
        nargs="+",
        choices=["fused", "dequant_gemm"],
        default=["fused", "dequant_gemm"],
    )
    p.add_argument("--fp4-activation-cache-d-lora-down-backend", choices=["fused", "dequant_gemm"], default="fused")
    p.add_argument("--no-frozen-residual", action="store_true")
    p.add_argument("--train-bias", action="store_true")
    p.add_argument("--no-cache-lora-act", action="store_true")
    p.add_argument("--activation-checkpoint", action="store_true")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--adam-eps", type=float, default=1e-4)
    p.add_argument("--lora-weight-decay", type=float, default=0.0)
    p.add_argument("--bias-weight-decay", type=float, default=0.0)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--iters", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--results-dir", type=str, default="results")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    dtype = dtype_from_name(args.dtype)
    lowrank_dtype = dtype_from_name(args.lowrank_dtype)
    dense_state = build_dense_state(args, dtype)
    x_ref, _ = make_inputs(args, dtype)
    dense_ref = TinyTransformer(args.hidden, args.layers, dtype=dtype).cuda()
    dense_ref.load_state_dict(dense_state)
    dense_ref.eval()
    with torch.no_grad():
        dense_y_ref = dense_ref(x_ref.detach())
    del dense_ref, x_ref
    torch.cuda.empty_cache()

    dense_lora = run_dense_lora_baseline(
        args=args,
        dense_state=dense_state,
        dense_y_ref=dense_y_ref,
        dtype=dtype,
        lowrank_dtype=lowrank_dtype,
    )
    records = [
        run_record(
            args=args,
            record_name=record_name,
            mode=mode,
            backend=backend,
            dense_state=dense_state,
            dense_y_ref=dense_y_ref,
            dtype=dtype,
            lowrank_dtype=lowrank_dtype,
        )
        for record_name, mode, backend in mode_backend_records(args)
    ]

    baseline_name = "balanced" if any(record["record"] == "balanced" for record in records) else records[0]["record"]
    baseline_latency = next(
        record["latency_ms"]["train_step_with_optimizer"] for record in records if record["record"] == baseline_name
    )
    for record in records:
        record["relative_to_baseline"] = {
            "baseline_record": baseline_name,
            "train_step_speedup": baseline_latency / record["latency_ms"]["train_step_with_optimizer"],
        }
        dense_latency = dense_lora["latency_ms"]["train_step_with_optimizer"]
        dense_peak_delta = dense_lora["peak_memory_bytes"]["train_step_delta"]
        record["relative_to_dense_lora"] = {
            "train_step_speedup": dense_latency / record["latency_ms"]["train_step_with_optimizer"],
            "peak_delta_ratio": (
                None
                if dense_peak_delta <= 0
                else record["peak_memory_bytes"]["train_step_delta"] / dense_peak_delta
            ),
        }

    payload = {
        "experiment": "fp4_lora_prepare_policy_benchmark",
        "shape": {
            "batch": args.batch,
            "hidden": args.hidden,
            "layers": args.layers,
            "rank": args.rank,
            "frozen_residual_rank": args.frozen_residual_rank,
            "dtype": args.dtype,
            "lowrank_dtype": args.lowrank_dtype,
            "modes": args.modes,
            "memory_saving_backends": args.memory_saving_backends,
            "no_frozen_residual": args.no_frozen_residual,
            "train_bias": args.train_bias,
            "cache_lora_act": not args.no_cache_lora_act,
            "activation_checkpoint": args.activation_checkpoint,
            "warmup": args.warmup,
            "iters": args.iters,
        },
        "dense_lora_baseline": dense_lora,
        "records": {record["record"]: record for record in records},
        "all_passed": bool(dense_lora["all_passed"] and all(record["all_passed"] for record in records)),
    }

    os.makedirs(args.results_dir, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(args.results_dir, f"fp4_lora_prepare_policies_{stamp}.json")
    latest_path = os.path.join(args.results_dir, "latest_fp4_lora_prepare_policies.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    with open(latest_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(json.dumps(payload, indent=2))
    print(f"Wrote {out_path}")
    print(f"Wrote {latest_path}")
    if not payload["all_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
