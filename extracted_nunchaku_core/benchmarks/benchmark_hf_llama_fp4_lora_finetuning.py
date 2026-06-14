from __future__ import annotations

import argparse
import gc
import json
import os
import re
import sys
import time
from dataclasses import asdict
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from native_fp4 import (  # noqa: E402
    DEFAULT_FP4_LORA_TARGET_MODULES,
    FP4LoRAConfig,
    iter_fp4_lora_modules,
    prepare_fp4_lora_finetuning,
)


DEFAULT_MODEL_ID = "NousResearch/Llama-2-7b-hf"
DEFAULT_MODEL_DIR = "/home/wyj24/projects/nunchaku/extracted_nunchaku_core/models/Llama-2-7b-hf"
DEFAULT_EXCLUDE_MODULES = ("lm_head",)
VALID_VARIANTS = (
    "dense_lora",
    "fp4_accuracy",
    "fp4_balanced",
    "fp4_throughput",
    "fp4_memory_saving",
    "fp4_memory_saving_dequant",
)


class DenseLoRALinear(nn.Module):
    """Frozen dense Linear plus trainable LoRA branch used as the FP16/BF16 baseline."""

    def __init__(
        self,
        linear: nn.Linear,
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
            self.bias = nn.Parameter(linear.bias.detach().contiguous())
        else:
            self.register_buffer("bias", linear.bias.detach().contiguous(), persistent=True)

        self.lora_down = nn.Parameter(
            torch.empty(self.rank, self.in_features, device=linear.weight.device, dtype=lowrank_dtype)
        )
        self.lora_up = nn.Parameter(
            torch.empty(self.out_features, self.rank, device=linear.weight.device, dtype=lowrank_dtype)
        )
        self.reset_lora_parameters()

    def reset_lora_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.lora_down, a=5**0.5)
        nn.init.zeros_(self.lora_up)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.linear(x, self.weight, self.bias)
        x2d = x.reshape(-1, self.in_features).to(self.lowrank_dtype)
        lora_act = torch.matmul(x2d, self.lora_down.t())
        lora_out = torch.matmul(lora_act, self.lora_up.t()).mul(self.scaling)
        return y + lora_out.to(y.dtype).reshape(*x.shape[:-1], self.out_features)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark HF causal-LM train steps with dense LoRA vs native FP4 LoRA modules."
    )
    parser.add_argument("--model-id", type=str, default=DEFAULT_MODEL_ID)
    parser.add_argument("--model-dir", type=str, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--hf-endpoint", type=str, default="https://hf-mirror.com")
    parser.add_argument("--dtype", choices=["fp16", "bf16"], default="bf16")
    parser.add_argument("--lowrank-dtype", choices=["fp16", "bf16"], default="bf16")
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--lora-alpha", type=float, default=None)
    parser.add_argument("--variants", nargs="+", choices=VALID_VARIANTS, default=["dense_lora", "fp4_balanced"])
    parser.add_argument("--target-modules", nargs="+", default=list(DEFAULT_FP4_LORA_TARGET_MODULES))
    parser.add_argument("--exclude-modules", nargs="+", default=list(DEFAULT_EXCLUDE_MODULES))
    parser.add_argument("--linear-prefix", type=str, default="model.layers.")
    parser.add_argument("--include-lm-head", action="store_true")
    parser.add_argument("--replace-layer-start", type=int, default=None)
    parser.add_argument("--replace-layer-end", type=int, default=None)
    parser.add_argument("--replace-name-substrings", nargs="*", default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--dataset-name", type=str, default="wikitext")
    parser.add_argument("--dataset-config", type=str, default="wikitext-2-raw-v1")
    parser.add_argument("--dataset-split", type=str, default="test")
    parser.add_argument("--dataset-max-docs", type=int, default=0)
    parser.add_argument("--dataset-offset-tokens", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--adam-eps", type=float, default=1e-4)
    parser.add_argument("--lora-weight-decay", type=float, default=0.0)
    parser.add_argument("--bias-weight-decay", type=float, default=0.0)
    parser.add_argument("--train-bias", action="store_true")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--prime-steps", type=int, default=1)
    parser.add_argument("--no-frozen-residual", action="store_true")
    parser.add_argument("--frozen-residual-rank", type=int, default=None)
    parser.add_argument("--residual-svd-method", choices=["full_svd", "svd_lowrank"], default=None)
    parser.add_argument("--residual-svd-lowrank-oversample", type=int, default=8)
    parser.add_argument("--residual-svd-lowrank-niter", type=int, default=2)
    parser.add_argument("--no-cache-lora-act", action="store_true")
    parser.add_argument("--activation-checkpoint", action="store_true")
    parser.add_argument("--model-gradient-checkpointing", action="store_true")
    parser.add_argument("--backward-weight-policy", choices=["repack", "cache"], default="repack")
    parser.add_argument("--reuse-fused-dy-up-for-d-lora-down", action="store_true")
    parser.add_argument("--overlap-lora-grad-min-rows", type=int, default=4096)
    parser.add_argument("--fp4-activation-cache-d-lora-down-backend", choices=["fused", "dequant_gemm"], default="fused")
    parser.add_argument("--no-zero-lora-up-fast-path", action="store_true")
    parser.add_argument("--attn-implementation", type=str, default=None)
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def dtype_from_name(name: str) -> torch.dtype:
    if name == "fp16":
        return torch.float16
    if name == "bf16":
        return torch.bfloat16
    raise ValueError(f"Unsupported dtype: {name}")


def set_hf_mirror(endpoint: str) -> None:
    if endpoint:
        os.environ["HF_ENDPOINT"] = endpoint
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"


def ensure_model_downloaded(model_id: str, model_dir: str) -> str:
    from huggingface_hub import snapshot_download

    config_path = os.path.join(model_dir, "config.json")
    if os.path.exists(config_path):
        index_path = os.path.join(model_dir, "model.safetensors.index.json")
        if not os.path.exists(index_path):
            return model_dir
        with open(index_path, "r", encoding="utf-8") as f:
            index = json.load(f)
        shard_names = sorted(set(index.get("weight_map", {}).values()))
        if shard_names and all(os.path.exists(os.path.join(model_dir, name)) for name in shard_names):
            return model_dir

    return snapshot_download(
        repo_id=model_id,
        local_dir=model_dir,
        local_dir_use_symlinks=False,
        allow_patterns=["*.json", "*.model", "*.txt", "*.md", "*.pdf", "*.safetensors"],
        ignore_patterns=["*.bin", "*.h5", "*.ot"],
    )


def load_tokenizer(model_dir: str):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_model(model_dir: str, dtype: torch.dtype, attn_implementation: str | None):
    from transformers import AutoModelForCausalLM

    gc.collect()
    torch.cuda.empty_cache()
    kwargs: dict[str, Any] = {
        "dtype": dtype,
        "low_cpu_mem_usage": True,
        "device_map": {"": "cuda:0"},
    }
    if attn_implementation:
        kwargs["attn_implementation"] = attn_implementation
    try:
        model = AutoModelForCausalLM.from_pretrained(model_dir, **kwargs)
    except TypeError:
        kwargs["torch_dtype"] = kwargs.pop("dtype")
        model = AutoModelForCausalLM.from_pretrained(model_dir, **kwargs)
    model.train()
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    return model


def cleanup_model(model: nn.Module | None) -> None:
    if model is not None:
        del model
    gc.collect()
    torch.cuda.empty_cache()


def load_wikitext_token_stream(
    tokenizer,
    dataset_name: str,
    dataset_config: str,
    dataset_split: str,
    dataset_max_docs: int,
) -> torch.Tensor:
    from datasets import load_dataset

    dataset = load_dataset(dataset_name, dataset_config, split=dataset_split)
    if dataset_max_docs > 0:
        dataset = dataset.select(range(min(dataset_max_docs, len(dataset))))
    texts = [text for text in dataset["text"] if text and text.strip()]
    corpus = "\n\n".join(texts)
    token_ids = tokenizer(corpus, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
    if tokenizer.bos_token_id is not None:
        token_ids = torch.cat([torch.tensor([tokenizer.bos_token_id], dtype=torch.long), token_ids], dim=0)
    return token_ids.contiguous()


def build_batch_from_stream(
    token_stream: torch.Tensor,
    *,
    seq_len: int,
    batch_size: int,
    offset_tokens: int,
) -> dict[str, torch.Tensor]:
    needed = offset_tokens + batch_size * seq_len
    if needed > token_stream.numel():
        raise ValueError(f"Token stream too short: need {needed}, have {token_stream.numel()}")
    rows = []
    for row in range(batch_size):
        start = offset_tokens + row * seq_len
        rows.append(token_stream[start : start + seq_len])
    input_ids = torch.stack(rows, dim=0).to("cuda")
    return {
        "input_ids": input_ids,
        "attention_mask": torch.ones_like(input_ids),
        "labels": input_ids.clone(),
    }


def module_name_matches(full_name: str, child_name: str, patterns: tuple[str, ...]) -> bool:
    return any(full_name == pattern or child_name == pattern or full_name.endswith(f".{pattern}") for pattern in patterns)


def extract_layer_index(full_name: str, linear_prefix: str) -> int | None:
    if not full_name.startswith(linear_prefix):
        return None
    match = re.match(rf"^{re.escape(linear_prefix)}(\d+)\.", full_name)
    if match is None:
        return None
    return int(match.group(1))


def layer_is_selected(
    full_name: str,
    linear_prefix: str,
    layer_start: int | None,
    layer_end: int | None,
) -> bool:
    layer_idx = extract_layer_index(full_name, linear_prefix)
    if layer_idx is None:
        return True
    if layer_start is not None and layer_idx < layer_start:
        return False
    if layer_end is not None and layer_idx >= layer_end:
        return False
    return True


def name_is_selected(full_name: str, name_substrings: list[str] | None) -> bool:
    if not name_substrings:
        return True
    return any(substring in full_name for substring in name_substrings)


def effective_exclude_modules(args: argparse.Namespace) -> tuple[str, ...]:
    excludes = tuple(args.exclude_modules or ())
    if args.include_lm_head:
        excludes = tuple(name for name in excludes if name != "lm_head")
    return excludes


def select_linear_names(model: nn.Module, args: argparse.Namespace) -> list[str]:
    targets = tuple(args.target_modules or ())
    excludes = effective_exclude_modules(args)
    selected: list[str] = []
    for full_name, child in model.named_modules():
        if not isinstance(child, nn.Linear):
            continue
        child_name = full_name.rsplit(".", 1)[-1]
        is_target = module_name_matches(full_name, child_name, targets)
        if args.include_lm_head and full_name == "lm_head":
            is_target = True
        is_excluded = module_name_matches(full_name, child_name, excludes)
        if (
            is_target
            and not is_excluded
            and layer_is_selected(full_name, args.linear_prefix, args.replace_layer_start, args.replace_layer_end)
            and name_is_selected(full_name, args.replace_name_substrings)
        ):
            selected.append(full_name)
    return selected


def parent_module(root: nn.Module, full_name: str) -> tuple[nn.Module, str]:
    parts = full_name.split(".")
    parent = root
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]


def replace_linear_with_dense_lora(
    model: nn.Module,
    selected_names: list[str],
    *,
    rank: int,
    lora_alpha: float | None,
    lowrank_dtype: torch.dtype,
    train_bias: bool,
) -> list[str]:
    replaced: list[str] = []
    for full_name in selected_names:
        parent, child_name = parent_module(model, full_name)
        child = getattr(parent, child_name)
        if not isinstance(child, nn.Linear):
            raise TypeError(f"Selected module is no longer nn.Linear: {full_name}")
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
    return replaced


def dense_lora_parameter_groups(
    model: nn.Module,
    *,
    train_bias: bool,
    lora_weight_decay: float,
    bias_weight_decay: float,
    lr: float,
) -> tuple[list[dict[str, Any]], list[str], int]:
    lora_params: list[nn.Parameter] = []
    bias_params: list[nn.Parameter] = []
    trainable_names: list[str] = []
    for name, param in model.named_parameters():
        if name.endswith(".lora_down") or name.endswith(".lora_up"):
            param.requires_grad_(True)
            lora_params.append(param)
            trainable_names.append(name)
        elif train_bias and name.endswith(".bias") and param.requires_grad:
            bias_params.append(param)
            trainable_names.append(name)
        else:
            param.requires_grad_(False)

    groups: list[dict[str, Any]] = []
    if lora_params:
        groups.append({"params": lora_params, "weight_decay": float(lora_weight_decay), "lr": float(lr)})
    if bias_params:
        groups.append({"params": bias_params, "weight_decay": float(bias_weight_decay), "lr": float(lr)})
    trainable_param_count = int(sum(param.numel() for group in groups for param in group["params"]))
    return groups, trainable_names, trainable_param_count


def jsonable_config(cfg: FP4LoRAConfig) -> dict[str, Any]:
    data = asdict(cfg)
    data["lowrank_dtype"] = str(cfg.lowrank_dtype).replace("torch.", "")
    return data


def tensor_error(a: torch.Tensor, b: torch.Tensor) -> dict[str, float]:
    da = a.float()
    db = b.float()
    diff = da - db
    abs_diff = diff.abs()
    return {
        "max_abs": float(abs_diff.max().item()),
        "mae": float(abs_diff.mean().item()),
        "rel_l2": float(diff.norm().item() / (db.norm().item() + 1e-12)),
    }


def zero_grads(model: nn.Module, optimizer: torch.optim.Optimizer) -> None:
    optimizer.zero_grad(set_to_none=True)
    model.zero_grad(set_to_none=True)


def forward_loss(model: nn.Module, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    outputs = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        labels=batch["labels"],
        use_cache=False,
    )
    return outputs.loss, outputs.logits


def train_step(model: nn.Module, batch: dict[str, torch.Tensor], optimizer: torch.optim.Optimizer) -> torch.Tensor:
    zero_grads(model, optimizer)
    loss, _ = forward_loss(model, batch)
    loss.backward()
    optimizer.step()
    return loss.detach()


def time_cuda(fn, *, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    samples: list[float] = []
    for _ in range(iters):
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(end))
    return float(sum(samples) / len(samples))


def measure_peak_delta(fn) -> tuple[int, int, int]:
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    baseline = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    fn()
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated()
    return int(peak - baseline), int(baseline), int(peak)


def variant_to_mode_backend(variant: str, default_backend: str) -> tuple[str, str]:
    if variant == "fp4_accuracy":
        return "accuracy", default_backend
    if variant == "fp4_balanced":
        return "balanced", default_backend
    if variant == "fp4_throughput":
        return "throughput", default_backend
    if variant == "fp4_memory_saving":
        return "memory_saving", default_backend
    if variant == "fp4_memory_saving_dequant":
        return "memory_saving", "dequant_gemm"
    raise ValueError(f"Unsupported FP4 variant: {variant}")


def run_prime_steps(
    model: nn.Module,
    batch: dict[str, torch.Tensor],
    optimizer: torch.optim.Optimizer,
    prime_steps: int,
) -> list[float]:
    losses: list[float] = []
    for _ in range(prime_steps):
        loss = train_step(model, batch, optimizer)
        losses.append(float(loss.item()))
    torch.cuda.synchronize()
    return losses


def run_dense_lora_variant(
    args: argparse.Namespace,
    *,
    model_dir: str,
    dtype: torch.dtype,
    lowrank_dtype: torch.dtype,
    batch: dict[str, torch.Tensor],
) -> tuple[dict[str, Any], torch.Tensor]:
    torch.manual_seed(args.seed)
    model = load_model(model_dir, dtype=dtype, attn_implementation=args.attn_implementation)
    if args.model_gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    for param in model.parameters():
        param.requires_grad_(False)

    selected_names = select_linear_names(model, args)
    if not selected_names:
        cleanup_model(model)
        raise RuntimeError("No nn.Linear modules selected for dense_lora baseline")
    replaced = replace_linear_with_dense_lora(
        model,
        selected_names,
        rank=args.rank,
        lora_alpha=args.lora_alpha,
        lowrank_dtype=lowrank_dtype,
        train_bias=args.train_bias,
    )
    param_groups, trainable_names, trainable_param_count = dense_lora_parameter_groups(
        model,
        train_bias=args.train_bias,
        lora_weight_decay=args.lora_weight_decay,
        bias_weight_decay=args.bias_weight_decay,
        lr=args.lr,
    )
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, eps=args.adam_eps)

    with torch.no_grad():
        initial_loss, initial_logits = forward_loss(model, batch)
    initial_logits_ref = initial_logits.detach().float().cpu()
    prime_losses = run_prime_steps(model, batch, optimizer, args.prime_steps)

    def fn() -> None:
        train_step(model, batch, optimizer)

    latency_ms = time_cuda(fn, warmup=args.warmup, iters=args.iters)
    peak_delta, peak_baseline, peak = measure_peak_delta(fn)
    final_loss = train_step(model, batch, optimizer)
    grads_finite = all(
        param.grad is not None and bool(torch.isfinite(param.grad).all())
        for group in param_groups
        for param in group["params"]
    )

    record = {
        "variant": "dense_lora",
        "selected_modules": selected_names,
        "replaced_modules": replaced,
        "replaced_count": len(replaced),
        "trainable_names": trainable_names,
        "trainable_param_count": trainable_param_count,
        "initial_loss": float(initial_loss.detach().item()),
        "prime_losses": prime_losses,
        "final_loss": float(final_loss.item()),
        "latency_ms": {
            "train_step_with_optimizer": latency_ms,
        },
        "throughput": {
            "tokens_per_second": args.batch_size * args.seq_len * 1000.0 / latency_ms,
            "steps_per_second": 1000.0 / latency_ms,
        },
        "peak_memory_bytes": {
            "train_step_delta": peak_delta,
            "baseline": peak_baseline,
            "peak": peak,
        },
        "checks": {
            "selected_modules_nonempty": bool(selected_names),
            "replaced_count_matches_selection": len(replaced) == len(selected_names),
            "trainable_param_count_positive": trainable_param_count > 0,
            "initial_loss_finite": bool(torch.isfinite(initial_loss)),
            "final_loss_finite": bool(torch.isfinite(final_loss)),
            "trainable_grads_finite": grads_finite,
            "latency_positive": latency_ms > 0.0,
            "peak_delta_nonnegative": peak_delta >= 0,
        },
    }
    record["all_passed"] = bool(all(record["checks"].values()))
    cleanup_model(model)
    return record, initial_logits_ref


def run_fp4_variant(
    args: argparse.Namespace,
    *,
    variant: str,
    model_dir: str,
    dtype: torch.dtype,
    lowrank_dtype: torch.dtype,
    batch: dict[str, torch.Tensor],
    dense_initial_logits: torch.Tensor | None,
    dense_latency_ms: float | None,
    dense_peak_delta: int | None,
) -> dict[str, Any]:
    torch.manual_seed(args.seed)
    model = load_model(model_dir, dtype=dtype, attn_implementation=args.attn_implementation)
    if args.model_gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    selected_names = select_linear_names(model, args)
    if not selected_names:
        cleanup_model(model)
        raise RuntimeError(f"No nn.Linear modules selected for {variant}")

    mode, backend = variant_to_mode_backend(variant, args.fp4_activation_cache_d_lora_down_backend)
    convert_start = time.perf_counter()
    result = prepare_fp4_lora_finetuning(
        model,
        mode=mode,
        rank=args.rank,
        lora_alpha=args.lora_alpha,
        dtype=dtype,
        lowrank_dtype=lowrank_dtype,
        use_frozen_residual=not args.no_frozen_residual,
        frozen_residual_rank=args.frozen_residual_rank,
        residual_svd_method=args.residual_svd_method,
        residual_svd_lowrank_oversample=args.residual_svd_lowrank_oversample,
        residual_svd_lowrank_niter=args.residual_svd_lowrank_niter,
        train_bias=args.train_bias,
        cache_lora_act=not args.no_cache_lora_act,
        activation_checkpoint=args.activation_checkpoint,
        backward_weight_policy=args.backward_weight_policy,
        reuse_fused_dy_up_for_d_lora_down=args.reuse_fused_dy_up_for_d_lora_down,
        overlap_lora_grad_min_rows=args.overlap_lora_grad_min_rows,
        fp4_activation_cache_d_lora_down_backend=backend,
        zero_lora_up_fast_path=not args.no_zero_lora_up_fast_path,
        target_modules=tuple(selected_names),
        exclude_modules=effective_exclude_modules(args),
        lr=args.lr,
        lora_weight_decay=args.lora_weight_decay,
        bias_weight_decay=args.bias_weight_decay,
    )
    conversion_seconds = time.perf_counter() - convert_start
    optimizer = torch.optim.AdamW(result.optimizer_param_groups, lr=args.lr, eps=args.adam_eps)
    hook = (
        result.register_cache_refresh_hook(optimizer)
        if result.config.fuse_lowrank_forward or result.config.cache_fused_lora_dx
        else None
    )

    with torch.no_grad():
        initial_loss, initial_logits = forward_loss(result.model, batch)
    initial_error_vs_dense = None
    if dense_initial_logits is not None:
        initial_error_vs_dense = tensor_error(initial_logits.detach().float().cpu(), dense_initial_logits)

    prime_losses = run_prime_steps(result.model, batch, optimizer, args.prime_steps)

    def fn() -> None:
        train_step(result.model, batch, optimizer)

    latency_ms = time_cuda(fn, warmup=args.warmup, iters=args.iters)
    peak_delta, peak_baseline, peak = measure_peak_delta(fn)
    final_loss = train_step(result.model, batch, optimizer)

    fp4_modules = dict(iter_fp4_lora_modules(result.model))
    grads_finite = all(
        param.grad is not None and bool(torch.isfinite(param.grad).all())
        for group in result.optimizer_param_groups
        for param in group["params"]
    )
    all_module_backends_match = all(child.fp4_activation_cache_d_lora_down_backend == backend for child in fp4_modules.values())
    all_module_backward_weight_policies_match = all(
        child.backward_weight_policy == args.backward_weight_policy for child in fp4_modules.values()
    )
    cache_hook_count = None if hook is None else hook.last_refresh_count
    cache_hook_forward_count = None if hook is None else hook.last_fused_lora_forward_refresh_count
    cache_hook_dx_count = None if hook is None else hook.last_fused_lora_dx_refresh_count
    if hook is not None:
        hook.remove()

    record: dict[str, Any] = {
        "variant": variant,
        "mode": mode,
        "fp4_activation_cache_d_lora_down_backend": backend,
        "config": jsonable_config(result.config),
        "selected_modules": selected_names,
        "replaced_modules": result.replaced_modules,
        "replaced_count": len(result.replaced_modules),
        "trainable_names": result.trainable_names,
        "trainable_param_count": result.trainable_param_count,
        "conversion_seconds": conversion_seconds,
        "refreshed_forward_cache_count": result.refreshed_forward_cache_count,
        "refreshed_cache_count": result.refreshed_cache_count,
        "refreshed_backward_weight_count": result.refreshed_backward_weight_count,
        "cache_summary": asdict(result.cache_summary),
        "cache_hook_refresh_count": cache_hook_count,
        "cache_hook_forward_refresh_count": cache_hook_forward_count,
        "cache_hook_dx_refresh_count": cache_hook_dx_count,
        "initial_loss": float(initial_loss.detach().item()),
        "initial_logits_vs_dense_lora": initial_error_vs_dense,
        "prime_losses": prime_losses,
        "final_loss": float(final_loss.item()),
        "latency_ms": {
            "train_step_with_optimizer": latency_ms,
        },
        "throughput": {
            "tokens_per_second": args.batch_size * args.seq_len * 1000.0 / latency_ms,
            "steps_per_second": 1000.0 / latency_ms,
        },
        "peak_memory_bytes": {
            "train_step_delta": peak_delta,
            "baseline": peak_baseline,
            "peak": peak,
        },
        "relative_to_dense_lora": None,
        "checks": {
            "selected_modules_nonempty": bool(selected_names),
            "replaced_count_matches_selection": len(result.replaced_modules) == len(selected_names),
            "trainable_param_count_positive": result.trainable_param_count > 0,
            "initial_loss_finite": bool(torch.isfinite(initial_loss)),
            "final_loss_finite": bool(torch.isfinite(final_loss)),
            "trainable_grads_finite": grads_finite,
            "module_backends_match": all_module_backends_match,
            "module_backward_weight_policies_match": all_module_backward_weight_policies_match,
            "latency_positive": latency_ms > 0.0,
            "peak_delta_nonnegative": peak_delta >= 0,
        },
    }
    if dense_latency_ms is not None:
        record["relative_to_dense_lora"] = {
            "train_step_speedup": dense_latency_ms / latency_ms,
            "peak_delta_ratio": None if not dense_peak_delta else peak_delta / dense_peak_delta,
        }
    record["all_passed"] = bool(all(record["checks"].values()))
    cleanup_model(result.model)
    return record


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.iters <= 0:
        raise ValueError("--iters must be positive")
    if args.warmup < 0 or args.prime_steps < 0:
        raise ValueError("--warmup and --prime-steps must be non-negative")

    torch.manual_seed(args.seed)
    set_hf_mirror(args.hf_endpoint)
    os.makedirs(args.results_dir, exist_ok=True)

    model_dir = ensure_model_downloaded(args.model_id, args.model_dir)
    dtype = dtype_from_name(args.dtype)
    lowrank_dtype = dtype_from_name(args.lowrank_dtype)

    tokenizer = load_tokenizer(model_dir)
    token_stream = load_wikitext_token_stream(
        tokenizer=tokenizer,
        dataset_name=args.dataset_name,
        dataset_config=args.dataset_config,
        dataset_split=args.dataset_split,
        dataset_max_docs=args.dataset_max_docs,
    )
    batch = build_batch_from_stream(
        token_stream,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        offset_tokens=args.dataset_offset_tokens,
    )

    variants = list(dict.fromkeys(args.variants))
    results: dict[str, Any] = {
        "experiment": "hf_causal_lm_fp4_lora_finetuning",
        "model_id": args.model_id,
        "model_dir": model_dir,
        "dtype": args.dtype,
        "lowrank_dtype": args.lowrank_dtype,
        "rank": args.rank,
        "lora_alpha": args.lora_alpha,
        "variants_requested": variants,
        "selection": {
            "target_modules": args.target_modules,
            "exclude_modules": list(effective_exclude_modules(args)),
            "linear_prefix": args.linear_prefix,
            "include_lm_head": args.include_lm_head,
            "replace_layer_start": args.replace_layer_start,
            "replace_layer_end": args.replace_layer_end,
            "replace_name_substrings": args.replace_name_substrings,
        },
        "train": {
            "batch_size": args.batch_size,
            "seq_len": args.seq_len,
            "lr": args.lr,
            "adam_eps": args.adam_eps,
            "lora_weight_decay": args.lora_weight_decay,
            "bias_weight_decay": args.bias_weight_decay,
            "train_bias": args.train_bias,
            "warmup": args.warmup,
            "iters": args.iters,
            "prime_steps": args.prime_steps,
            "model_gradient_checkpointing": args.model_gradient_checkpointing,
        },
        "dataset": {
            "name": args.dataset_name,
            "config": args.dataset_config,
            "split": args.dataset_split,
            "max_docs": args.dataset_max_docs,
            "offset_tokens": args.dataset_offset_tokens,
            "token_stream_tokens": int(token_stream.numel()),
        },
        "fp4_options": {
            "use_frozen_residual": not args.no_frozen_residual,
            "frozen_residual_rank": args.frozen_residual_rank,
            "residual_svd_method": args.residual_svd_method,
            "residual_svd_lowrank_oversample": args.residual_svd_lowrank_oversample,
            "residual_svd_lowrank_niter": args.residual_svd_lowrank_niter,
            "cache_lora_act": not args.no_cache_lora_act,
            "activation_checkpoint": args.activation_checkpoint,
            "backward_weight_policy": args.backward_weight_policy,
            "reuse_fused_dy_up_for_d_lora_down": args.reuse_fused_dy_up_for_d_lora_down,
            "overlap_lora_grad_min_rows": args.overlap_lora_grad_min_rows,
            "fp4_activation_cache_d_lora_down_backend": args.fp4_activation_cache_d_lora_down_backend,
            "zero_lora_up_fast_path": not args.no_zero_lora_up_fast_path,
        },
        "records": {},
        "all_passed": False,
    }

    dense_initial_logits: torch.Tensor | None = None
    dense_latency_ms: float | None = None
    dense_peak_delta: int | None = None
    if "dense_lora" in variants:
        dense_record, dense_initial_logits = run_dense_lora_variant(
            args,
            model_dir=model_dir,
            dtype=dtype,
            lowrank_dtype=lowrank_dtype,
            batch=batch,
        )
        results["records"]["dense_lora"] = dense_record
        dense_latency_ms = dense_record["latency_ms"]["train_step_with_optimizer"]
        dense_peak_delta = dense_record["peak_memory_bytes"]["train_step_delta"]

    for variant in variants:
        if variant == "dense_lora":
            continue
        record = run_fp4_variant(
            args,
            variant=variant,
            model_dir=model_dir,
            dtype=dtype,
            lowrank_dtype=lowrank_dtype,
            batch=batch,
            dense_initial_logits=dense_initial_logits,
            dense_latency_ms=dense_latency_ms,
            dense_peak_delta=dense_peak_delta,
        )
        results["records"][variant] = record

    results["all_passed"] = bool(all(record["all_passed"] for record in results["records"].values()))

    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(args.results_dir, f"hf_llama_fp4_lora_finetuning_{stamp}.json")
    latest_path = os.path.join(args.results_dir, "latest_hf_llama_fp4_lora_finetuning.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    with open(latest_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(json.dumps(results, indent=2))
    print(f"Saved benchmark to: {out_path}")
    print(f"Saved latest to: {latest_path}")
    if not results["all_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
