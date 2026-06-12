from __future__ import annotations

import argparse
import json
import os
import sys

import torch

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from native_fp4 import (  # noqa: E402
    FP4LoRAConfig,
    NunchakuFP4LoRALinear,
    clear_fused_lora_dx_caches,
    convert_linear_to_fp4_lora,
    fp4_lora_state_dict,
    freeze_non_fp4_lora_parameters,
    iter_fp4_lora_modules,
    load_fp4_lora_state_dict,
    refresh_fused_lora_dx_caches,
)


class TinyBlock(torch.nn.Module):
    def __init__(self, hidden: int, dtype: torch.dtype):
        super().__init__()
        self.q_proj = torch.nn.Linear(hidden, hidden, bias=False, dtype=dtype)
        self.k_proj = torch.nn.Linear(hidden, hidden, bias=False, dtype=dtype)
        self.down_proj = torch.nn.Linear(hidden, hidden, bias=True, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(torch.nn.functional.silu(self.q_proj(x)) + self.k_proj(x))


class TinyModel(torch.nn.Module):
    def __init__(self, hidden: int, dtype: torch.dtype):
        super().__init__()
        self.layers = torch.nn.ModuleList([TinyBlock(hidden, dtype), TinyBlock(hidden, dtype)])
        self.lm_head = torch.nn.Linear(hidden, hidden, bias=False, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return self.lm_head(x)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--rank", type=int, default=32)
    p.add_argument("--dtype", choices=["fp16", "bf16"], default="bf16")
    p.add_argument("--lowrank-dtype", choices=["fp16", "bf16"], default="bf16")
    p.add_argument("--fuse-lora-dx", action="store_true")
    p.add_argument("--cache-fused-lora-dx", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--results-dir", type=str, default="results")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.manual_seed(args.seed)

    dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16
    lowrank_dtype = torch.float16 if args.lowrank_dtype == "fp16" else torch.bfloat16
    model = TinyModel(args.hidden, dtype).cuda()

    cfg = FP4LoRAConfig(
        rank=args.rank,
        lowrank_dtype=lowrank_dtype,
        init="gaussian",
        train_bias=False,
        cache_lora_act=True,
        fuse_lora_dx=args.fuse_lora_dx,
        cache_fused_lora_dx=args.cache_fused_lora_dx,
    )
    model, replaced = convert_linear_to_fp4_lora(
        model,
        cfg,
        target_modules=("q_proj", "down_proj"),
        exclude_modules=("lm_head",),
    )
    trainable = freeze_non_fp4_lora_parameters(model)
    refreshed = refresh_fused_lora_dx_caches(model)
    cleared = clear_fused_lora_dx_caches(model)
    refreshed_after_clear = refresh_fused_lora_dx_caches(model)

    x = torch.randn(args.batch, args.hidden, device="cuda", dtype=dtype, requires_grad=True)
    y = model(x)
    loss = y.float().square().mean()
    loss.backward()

    fp4_modules = dict(iter_fp4_lora_modules(model))
    trainable_named = {name for name, param in model.named_parameters() if param.requires_grad}
    grad_named = {name for name, param in model.named_parameters() if param.grad is not None}
    expected_replaced = {
        "layers.0.q_proj",
        "layers.0.down_proj",
        "layers.1.q_proj",
        "layers.1.down_proj",
    }
    expected_adapter_keys = {f"{name}.{param}" for name in expected_replaced for param in ("lora_down", "lora_up")}

    adapter_state = fp4_lora_state_dict(model)
    model2 = TinyModel(args.hidden, dtype).cuda()
    model2, replaced2 = convert_linear_to_fp4_lora(
        model2,
        cfg,
        target_modules=("q_proj", "down_proj"),
        exclude_modules=("lm_head",),
    )
    pre_load_refreshed = refresh_fused_lora_dx_caches(model2)
    missing, unexpected = load_fp4_lora_state_dict(model2, adapter_state, strict=True)
    loaded_state = fp4_lora_state_dict(model2)
    loaded_matches = all(torch.equal(adapter_state[key], loaded_state[key]) for key in expected_adapter_keys)
    caches_cleared_after_load = all(
        child._cached_lora_down_bwd_packed is None and child._cached_lora_up_bwd_packed is None
        for _, child in iter_fp4_lora_modules(model2)
    )
    strict_mismatch_raises = False
    try:
        bad_state = dict(adapter_state)
        bad_state.pop(next(iter(expected_adapter_keys)))
        load_fp4_lora_state_dict(model2, bad_state, strict=True)
    except ValueError:
        strict_mismatch_raises = True

    checks = {
        "replaced_expected_modules": set(replaced) == expected_replaced,
        "second_model_replaced_expected_modules": set(replaced2) == expected_replaced,
        "lm_head_not_replaced": not isinstance(model.lm_head, NunchakuFP4LoRALinear),
        "all_replaced_are_fp4_lora": all(isinstance(fp4_modules[name], NunchakuFP4LoRALinear) for name in expected_replaced),
        "only_lora_trainable": trainable_named == set(trainable),
        "trainable_grads_present": set(trainable).issubset(grad_named),
        "x_grad_finite": bool(x.grad is not None and torch.isfinite(x.grad).all()),
        "output_finite": bool(torch.isfinite(y).all()),
        "cache_count_matches": refreshed == (len(replaced) if args.fuse_lora_dx and args.cache_fused_lora_dx else 0),
        "clear_count_matches": cleared == len(replaced),
        "refresh_after_clear_matches": refreshed_after_clear
        == (len(replaced) if args.fuse_lora_dx and args.cache_fused_lora_dx else 0),
        "adapter_state_keys_match": set(adapter_state) == expected_adapter_keys,
        "adapter_state_load_missing_empty": missing == [],
        "adapter_state_load_unexpected_empty": unexpected == [],
        "adapter_state_loaded_matches": loaded_matches,
        "adapter_load_clears_cache": caches_cleared_after_load,
        "strict_mismatch_raises": strict_mismatch_raises,
    }
    payload = {
        "shape": {
            "batch": args.batch,
            "hidden": args.hidden,
            "rank": args.rank,
            "dtype": args.dtype,
            "lowrank_dtype": args.lowrank_dtype,
            "fuse_lora_dx": args.fuse_lora_dx,
            "cache_fused_lora_dx": args.cache_fused_lora_dx,
        },
        "replaced": replaced,
        "trainable": trainable,
        "grad_named": sorted(grad_named),
        "cache_counts": {
            "refreshed": refreshed,
            "cleared": cleared,
            "refreshed_after_clear": refreshed_after_clear,
            "pre_load_refreshed": pre_load_refreshed,
        },
        "adapter_state_keys": sorted(adapter_state),
        "checks": checks,
        "all_passed": bool(all(checks.values())),
    }

    os.makedirs(args.results_dir, exist_ok=True)
    out_path = os.path.join(args.results_dir, "latest_native_fp4_lora_modeling_validation.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(json.dumps(payload, indent=2))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
