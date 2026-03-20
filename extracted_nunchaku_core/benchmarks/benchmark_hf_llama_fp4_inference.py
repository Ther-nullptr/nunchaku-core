from __future__ import annotations

import argparse
import gc
import json
import os
import re
import sys
import time
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from native_fp4 import NunchakuFP4GemmOp, NunchakuFP4LowRankOp  # noqa: E402


@dataclass
class LogitsReference:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    logits: torch.Tensor


@dataclass
class GenerationReference:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    generated_ids: torch.Tensor


def time_cuda(fn, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    samples = []
    for _ in range(iters):
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(end))
    return float(sum(samples) / len(samples))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", type=str, default="NousResearch/Llama-2-7b-hf")
    parser.add_argument(
        "--model-dir",
        type=str,
        default="/home/wyj24/projects/nunchaku/extracted_nunchaku_core/models/Llama-2-7b-hf",
    )
    parser.add_argument("--dtype", choices=["fp16", "bf16"], default="fp16")
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--variants", type=str, nargs="+", default=["fp16", "fp4", "fp4_hybrid"])
    parser.add_argument("--lowrank-mode", choices=["full_svd", "svd_lowrank"], default="svd_lowrank")
    parser.add_argument("--svd-lowrank-oversample", type=int, default=8)
    parser.add_argument("--svd-lowrank-niter", type=int, default=2)
    parser.add_argument("--linear-prefix", type=str, default="model.layers.")
    parser.add_argument("--include-lm-head", action="store_true")
    parser.add_argument("--replace-layer-start", type=int, default=None)
    parser.add_argument("--replace-layer-end", type=int, default=None)
    parser.add_argument("--replace-name-substrings", type=str, nargs="*", default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--prefill-lengths", type=int, nargs="+", default=[128, 512, 1024])
    parser.add_argument("--decode-prompt-length", type=int, default=512)
    parser.add_argument("--decode-steps", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--dataset-name", type=str, default="wikitext")
    parser.add_argument("--dataset-config", type=str, default="wikitext-2-raw-v1")
    parser.add_argument("--dataset-split", type=str, default="test")
    parser.add_argument("--accuracy-seq-len", type=int, default=512)
    parser.add_argument("--accuracy-num-seqs", type=int, default=32)
    parser.add_argument("--logits-num-seqs", type=int, default=4)
    parser.add_argument("--generation-prompt-length", type=int, default=128)
    parser.add_argument("--generation-steps", type=int, default=32)
    parser.add_argument("--generation-num-prompts", type=int, default=4)
    parser.add_argument("--dataset-max-docs", type=int, default=0)
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--hf-endpoint", type=str, default="https://hf-mirror.com")
    return parser.parse_args()


def get_dtype(name: str) -> torch.dtype:
    return torch.float16 if name == "fp16" else torch.bfloat16


def set_hf_mirror(endpoint: str) -> None:
    os.environ["HF_ENDPOINT"] = endpoint
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"


def ensure_model_downloaded(model_id: str, model_dir: str) -> str:
    from huggingface_hub import snapshot_download

    config_path = os.path.join(model_dir, "config.json")
    index_path = os.path.join(model_dir, "model.safetensors.index.json")
    if os.path.exists(config_path) and os.path.exists(index_path):
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


def should_replace_module(full_name: str, linear_prefix: str, include_lm_head: bool) -> bool:
    return full_name.startswith(linear_prefix) or (include_lm_head and full_name == "lm_head")


def extract_layer_index(full_name: str, linear_prefix: str) -> int | None:
    if not full_name.startswith(linear_prefix):
        return None
    match = re.match(rf"^{re.escape(linear_prefix)}(\d+)\.", full_name)
    if match is None:
        return None
    return int(match.group(1))


def layer_is_selected(full_name: str, linear_prefix: str, layer_start: int | None, layer_end: int | None) -> bool:
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


def replace_linear_modules(
    module: nn.Module,
    variant: str,
    dtype: torch.dtype,
    rank: int,
    linear_prefix: str,
    include_lm_head: bool,
    layer_start: int | None,
    layer_end: int | None,
    name_substrings: list[str] | None,
    factor_mode: str,
    svd_lowrank_oversample: int,
    svd_lowrank_niter: int,
    prefix: str = "",
) -> tuple[int, int]:
    replaced_modules = 0
    replaced_params = 0

    for child_name, child in list(module.named_children()):
        full_name = f"{prefix}.{child_name}" if prefix else child_name
        if (
            isinstance(child, nn.Linear)
            and should_replace_module(full_name, linear_prefix, include_lm_head)
            and layer_is_selected(full_name, linear_prefix, layer_start, layer_end)
            and name_is_selected(full_name, name_substrings)
        ):
            weight = child.weight.detach().to(device="cuda", dtype=dtype).contiguous()
            bias = None
            if child.bias is not None:
                bias = child.bias.detach().to(device="cuda", dtype=dtype).contiguous()

            if variant == "fp4":
                replacement = NunchakuFP4GemmOp(weight=weight, bias=bias)
            elif variant == "fp4_hybrid":
                replacement = NunchakuFP4LowRankOp(
                    weight=weight,
                    bias=bias,
                    rank=rank,
                    factor_mode=factor_mode,
                    svd_lowrank_oversample=svd_lowrank_oversample,
                    svd_lowrank_niter=svd_lowrank_niter,
                )
            else:
                raise ValueError(f"Unsupported variant: {variant}")

            setattr(module, child_name, replacement)
            replaced_modules += 1
            replaced_params += child.weight.numel() + (0 if child.bias is None else child.bias.numel())
            del child
            torch.cuda.empty_cache()
        else:
            child_modules, child_params = replace_linear_modules(
                module=child,
                variant=variant,
                dtype=dtype,
                rank=rank,
                linear_prefix=linear_prefix,
                include_lm_head=include_lm_head,
                layer_start=layer_start,
                layer_end=layer_end,
                name_substrings=name_substrings,
                factor_mode=factor_mode,
                svd_lowrank_oversample=svd_lowrank_oversample,
                svd_lowrank_niter=svd_lowrank_niter,
                prefix=full_name,
            )
            replaced_modules += child_modules
            replaced_params += child_params

    return replaced_modules, replaced_params


def load_tokenizer(model_dir: str):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_model(model_dir: str, dtype: torch.dtype):
    from transformers import AutoModelForCausalLM

    gc.collect()
    torch.cuda.empty_cache()

    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        dtype=dtype,
        low_cpu_mem_usage=True,
        device_map={"": "cuda:0"},
    )
    model.eval()
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
    seq_len: int,
    batch_size: int,
    offset_tokens: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    needed = offset_tokens + batch_size * seq_len
    if needed > token_stream.numel():
        raise ValueError(f"Token stream too short: need {needed}, have {token_stream.numel()}")

    batch = []
    for i in range(batch_size):
        start = offset_tokens + i * seq_len
        batch.append(token_stream[start : start + seq_len])
    input_ids = torch.stack(batch, dim=0).to("cuda")
    attention_mask = torch.ones_like(input_ids)
    return input_ids, attention_mask


def benchmark_prefill(model, input_ids: torch.Tensor, attention_mask: torch.Tensor, warmup: int, iters: int) -> dict[str, float]:
    def fn() -> torch.Tensor:
        return model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True).logits

    ms = time_cuda(fn, warmup=warmup, iters=iters)
    tokens = int(input_ids.numel())
    return {
        "batch_size": int(input_ids.shape[0]),
        "seq_len": int(input_ids.shape[1]),
        "ms": ms,
        "tokens_per_s": tokens * 1000.0 / ms,
    }


def run_prefill_plus_decode(model, input_ids: torch.Tensor, attention_mask: torch.Tensor, decode_steps: int) -> torch.Tensor:
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True)
    next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    past_key_values = outputs.past_key_values

    for _ in range(decode_steps):
        outputs = model(input_ids=next_token, past_key_values=past_key_values, use_cache=True)
        next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        past_key_values = outputs.past_key_values
    return next_token


def benchmark_decode(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    decode_steps: int,
    warmup: int,
    iters: int,
) -> dict[str, float]:
    def prefill_fn() -> torch.Tensor:
        return model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True).logits

    def full_fn() -> torch.Tensor:
        return run_prefill_plus_decode(model, input_ids=input_ids, attention_mask=attention_mask, decode_steps=decode_steps)

    prefill_ms = time_cuda(prefill_fn, warmup=warmup, iters=iters)
    full_ms = time_cuda(full_fn, warmup=warmup, iters=iters)
    decode_only_ms = max(full_ms - prefill_ms, 0.0)
    decode_ms_per_token = decode_only_ms / max(decode_steps, 1)
    tokens_per_s = int(input_ids.shape[0]) * 1000.0 / max(decode_ms_per_token, 1e-6)

    return {
        "batch_size": int(input_ids.shape[0]),
        "prompt_len": int(input_ids.shape[1]),
        "decode_steps": decode_steps,
        "prefill_ms": prefill_ms,
        "prefill_plus_decode_ms": full_ms,
        "decode_only_ms": decode_only_ms,
        "decode_ms_per_token": decode_ms_per_token,
        "decode_tokens_per_s": tokens_per_s,
    }


def greedy_generate(model, input_ids: torch.Tensor, attention_mask: torch.Tensor, steps: int) -> torch.Tensor:
    generated = []
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True)
    next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    past_key_values = outputs.past_key_values

    for _ in range(steps):
        generated.append(next_token.cpu())
        outputs = model(input_ids=next_token, past_key_values=past_key_values, use_cache=True)
        next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        past_key_values = outputs.past_key_values

    if not generated:
        return torch.empty(input_ids.shape[0], 0, dtype=torch.long)
    return torch.cat(generated, dim=1)


def compute_perplexity_sum(model, token_stream: torch.Tensor, seq_len: int, num_seqs: int, offset_tokens: int) -> tuple[float, int]:
    total_nll_sum = 0.0
    total_tokens = 0

    for i in range(num_seqs):
        start = offset_tokens + i * seq_len
        input_ids = token_stream[start : start + seq_len].unsqueeze(0).to("cuda")
        attention_mask = torch.ones_like(input_ids)

        with torch.inference_mode():
            logits = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).logits.float()

        shift_logits = logits[:, :-1, :]
        shift_labels = input_ids[:, 1:]
        nll_sum = F.cross_entropy(
            shift_logits.reshape(-1, shift_logits.shape[-1]),
            shift_labels.reshape(-1),
            reduction="sum",
        ).item()

        total_nll_sum += nll_sum
        total_tokens += int(shift_labels.numel())

    return total_nll_sum, total_tokens


def build_logits_references(model, token_stream: torch.Tensor, seq_len: int, num_seqs: int, offset_tokens: int) -> list[LogitsReference]:
    refs: list[LogitsReference] = []
    for i in range(num_seqs):
        start = offset_tokens + i * seq_len
        input_ids = token_stream[start : start + seq_len].unsqueeze(0)
        attention_mask = torch.ones_like(input_ids)
        with torch.inference_mode():
            logits = model(
                input_ids=input_ids.to("cuda"),
                attention_mask=attention_mask.to("cuda"),
                use_cache=False,
            ).logits.float().cpu()
        refs.append(LogitsReference(input_ids=input_ids.cpu(), attention_mask=attention_mask.cpu(), logits=logits))
    return refs


def build_generation_references(
    model,
    token_stream: torch.Tensor,
    prompt_len: int,
    num_prompts: int,
    generate_steps: int,
    offset_tokens: int,
) -> list[GenerationReference]:
    refs: list[GenerationReference] = []
    for i in range(num_prompts):
        start = offset_tokens + i * prompt_len
        input_ids = token_stream[start : start + prompt_len].unsqueeze(0)
        attention_mask = torch.ones_like(input_ids)
        with torch.inference_mode():
            generated_ids = greedy_generate(
                model,
                input_ids=input_ids.to("cuda"),
                attention_mask=attention_mask.to("cuda"),
                steps=generate_steps,
            )
        refs.append(
            GenerationReference(
                input_ids=input_ids.cpu(),
                attention_mask=attention_mask.cpu(),
                generated_ids=generated_ids,
            )
        )
    return refs


def evaluate_variant_accuracy(
    model,
    token_stream: torch.Tensor,
    ppl_seq_len: int,
    ppl_num_seqs: int,
    ppl_offset_tokens: int,
    baseline_ppl_nll_sum: float,
    baseline_ppl_token_count: int,
    logits_refs: list[LogitsReference],
    generation_refs: list[GenerationReference],
) -> dict[str, float]:
    variant_ppl_nll_sum, variant_ppl_token_count = compute_perplexity_sum(
        model=model,
        token_stream=token_stream,
        seq_len=ppl_seq_len,
        num_seqs=ppl_num_seqs,
        offset_tokens=ppl_offset_tokens,
    )

    logits_abs_sum = 0.0
    logits_count = 0
    logits_ref_norm_sq = 0.0
    logits_diff_norm_sq = 0.0
    top1_match = 0
    top1_total = 0

    for ref in logits_refs:
        input_ids = ref.input_ids.to("cuda")
        attention_mask = ref.attention_mask.to("cuda")
        with torch.inference_mode():
            variant_logits = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).logits.float()

        base_logits = ref.logits.to(variant_logits.device)
        diff = variant_logits - base_logits
        logits_abs_sum += diff.abs().sum().item()
        logits_count += diff.numel()
        logits_ref_norm_sq += base_logits.square().sum().item()
        logits_diff_norm_sq += diff.square().sum().item()

        base_top1 = base_logits.argmax(dim=-1)
        variant_top1 = variant_logits.argmax(dim=-1)
        top1_match += int((base_top1 == variant_top1).sum().item())
        top1_total += int(base_top1.numel())

    gen_token_match = 0
    gen_token_total = 0
    for ref in generation_refs:
        input_ids = ref.input_ids.to("cuda")
        attention_mask = ref.attention_mask.to("cuda")
        with torch.inference_mode():
            variant_generated = greedy_generate(
                model,
                input_ids=input_ids,
                attention_mask=attention_mask,
                steps=int(ref.generated_ids.shape[1]),
            )
        gen_token_match += int((variant_generated == ref.generated_ids).sum().item())
        gen_token_total += int(ref.generated_ids.numel())

    baseline_nll = baseline_ppl_nll_sum / max(baseline_ppl_token_count, 1)
    variant_nll = variant_ppl_nll_sum / max(variant_ppl_token_count, 1)

    return {
        "baseline_nll": baseline_nll,
        "variant_nll": variant_nll,
        "baseline_perplexity": float(torch.exp(torch.tensor(baseline_nll)).item()),
        "variant_perplexity": float(torch.exp(torch.tensor(variant_nll)).item()),
        "perplexity_ratio": float(torch.exp(torch.tensor(variant_nll - baseline_nll)).item()),
        "logits_mae": logits_abs_sum / max(logits_count, 1),
        "logits_rel_l2": (logits_diff_norm_sq ** 0.5) / max(logits_ref_norm_sq ** 0.5, 1e-12),
        "top1_agreement": top1_match / max(top1_total, 1),
        "generation_token_agreement": gen_token_match / max(gen_token_total, 1),
    }


def run_variant(
    model_dir: str,
    variant: str,
    dtype: torch.dtype,
    rank: int,
    linear_prefix: str,
    include_lm_head: bool,
    layer_start: int | None,
    layer_end: int | None,
    name_substrings: list[str] | None,
    factor_mode: str,
    svd_lowrank_oversample: int,
    svd_lowrank_niter: int,
    prefill_inputs: list[tuple[torch.Tensor, torch.Tensor]],
    decode_inputs: tuple[torch.Tensor, torch.Tensor],
    decode_steps: int,
    warmup: int,
    iters: int,
    token_stream: torch.Tensor,
    accuracy_seq_len: int,
    accuracy_num_seqs: int,
    accuracy_offset_tokens: int,
    baseline_ppl_nll_sum: float,
    baseline_ppl_token_count: int,
    logits_refs: list[LogitsReference] | None,
    generation_refs: list[GenerationReference] | None,
    preloaded_model: nn.Module | None = None,
) -> dict[str, object]:
    torch.cuda.empty_cache()
    model = preloaded_model if preloaded_model is not None else load_model(model_dir=model_dir, dtype=dtype)
    loaded_here = preloaded_model is None
    payload: dict[str, object] = {"variant": variant}

    if variant != "fp16":
        convert_start = time.perf_counter()
        replaced_modules, replaced_params = replace_linear_modules(
            module=model,
            variant=variant,
            dtype=dtype,
            rank=rank,
            linear_prefix=linear_prefix,
            include_lm_head=include_lm_head,
            layer_start=layer_start,
            layer_end=layer_end,
            name_substrings=name_substrings,
            factor_mode=factor_mode,
            svd_lowrank_oversample=svd_lowrank_oversample,
            svd_lowrank_niter=svd_lowrank_niter,
        )
        payload["conversion_seconds"] = time.perf_counter() - convert_start
        payload["replaced_modules"] = replaced_modules
        payload["replaced_params"] = replaced_params
    else:
        payload["conversion_seconds"] = 0.0
        payload["replaced_modules"] = 0
        payload["replaced_params"] = 0

    payload["prefill"] = [
        benchmark_prefill(model, input_ids=input_ids, attention_mask=attention_mask, warmup=warmup, iters=iters)
        for input_ids, attention_mask in prefill_inputs
    ]
    payload["decode"] = benchmark_decode(
        model,
        input_ids=decode_inputs[0],
        attention_mask=decode_inputs[1],
        decode_steps=decode_steps,
        warmup=warmup,
        iters=iters,
    )

    if logits_refs is None or generation_refs is None:
        payload["accuracy"] = None
    else:
        payload["accuracy"] = evaluate_variant_accuracy(
            model=model,
            token_stream=token_stream,
            ppl_seq_len=accuracy_seq_len,
            ppl_num_seqs=accuracy_num_seqs,
            ppl_offset_tokens=accuracy_offset_tokens,
            baseline_ppl_nll_sum=baseline_ppl_nll_sum,
            baseline_ppl_token_count=baseline_ppl_token_count,
            logits_refs=logits_refs,
            generation_refs=generation_refs,
        )

    if loaded_here:
        cleanup_model(model)
    return payload


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    set_hf_mirror(args.hf_endpoint)
    os.makedirs(args.results_dir, exist_ok=True)
    model_dir = ensure_model_downloaded(args.model_id, args.model_dir)
    dtype = get_dtype(args.dtype)

    tokenizer = load_tokenizer(model_dir)
    token_stream = load_wikitext_token_stream(
        tokenizer=tokenizer,
        dataset_name=args.dataset_name,
        dataset_config=args.dataset_config,
        dataset_split=args.dataset_split,
        dataset_max_docs=args.dataset_max_docs,
    )

    max_needed = max(args.prefill_lengths + [args.decode_prompt_length, args.generation_prompt_length, args.accuracy_seq_len])
    if token_stream.numel() < max_needed * max(args.batch_size, args.accuracy_num_seqs, args.generation_num_prompts, 2):
        raise RuntimeError(f"Token stream too short for requested settings: only {token_stream.numel()} tokens")

    prefill_inputs = []
    cursor = 0
    for seq_len in args.prefill_lengths:
        prefill_inputs.append(build_batch_from_stream(token_stream, seq_len=seq_len, batch_size=args.batch_size, offset_tokens=cursor))
        cursor += seq_len * args.batch_size

    decode_inputs = build_batch_from_stream(
        token_stream,
        seq_len=args.decode_prompt_length,
        batch_size=args.batch_size,
        offset_tokens=cursor,
    )
    cursor += args.decode_prompt_length * args.batch_size

    accuracy_offset_tokens = cursor
    logits_offset_tokens = accuracy_offset_tokens + args.accuracy_num_seqs * args.accuracy_seq_len
    generation_offset_tokens = logits_offset_tokens + args.logits_num_seqs * args.accuracy_seq_len

    fp16_model = load_model(model_dir=model_dir, dtype=dtype)
    baseline_ppl_nll_sum, baseline_ppl_token_count = compute_perplexity_sum(
        model=fp16_model,
        token_stream=token_stream,
        seq_len=args.accuracy_seq_len,
        num_seqs=args.accuracy_num_seqs,
        offset_tokens=accuracy_offset_tokens,
    )
    logits_refs = build_logits_references(
        model=fp16_model,
        token_stream=token_stream,
        seq_len=args.accuracy_seq_len,
        num_seqs=args.logits_num_seqs,
        offset_tokens=logits_offset_tokens,
    )
    generation_refs = build_generation_references(
        model=fp16_model,
        token_stream=token_stream,
        prompt_len=args.generation_prompt_length,
        num_prompts=args.generation_num_prompts,
        generate_steps=args.generation_steps,
        offset_tokens=generation_offset_tokens,
    )
    results = {
        "model_id": args.model_id,
        "model_dir": model_dir,
        "dtype": args.dtype,
        "rank": args.rank,
        "variants_requested": args.variants,
        "lowrank_mode": args.lowrank_mode,
        "linear_prefix": args.linear_prefix,
        "include_lm_head": args.include_lm_head,
        "replace_layer_start": args.replace_layer_start,
        "replace_layer_end": args.replace_layer_end,
        "replace_name_substrings": args.replace_name_substrings,
        "batch_size": args.batch_size,
        "prefill_lengths": args.prefill_lengths,
        "decode_prompt_length": args.decode_prompt_length,
        "decode_steps": args.decode_steps,
        "dataset_name": args.dataset_name,
        "dataset_config": args.dataset_config,
        "dataset_split": args.dataset_split,
        "accuracy_seq_len": args.accuracy_seq_len,
        "accuracy_num_seqs": args.accuracy_num_seqs,
        "logits_num_seqs": args.logits_num_seqs,
        "generation_prompt_length": args.generation_prompt_length,
        "generation_steps": args.generation_steps,
        "generation_num_prompts": args.generation_num_prompts,
        "token_stream_tokens": int(token_stream.numel()),
        "variants": {},
    }

    valid_variants = {"fp16", "fp4", "fp4_hybrid"}
    if any(variant not in valid_variants for variant in args.variants):
        raise ValueError(f"variants must be chosen from {sorted(valid_variants)}")

    for variant in args.variants:
        results["variants"][variant] = run_variant(
            model_dir=model_dir,
            variant=variant,
            dtype=dtype,
            rank=args.rank,
            linear_prefix=args.linear_prefix,
            include_lm_head=args.include_lm_head,
            layer_start=args.replace_layer_start,
            layer_end=args.replace_layer_end,
            name_substrings=args.replace_name_substrings,
            factor_mode=args.lowrank_mode,
            svd_lowrank_oversample=args.svd_lowrank_oversample,
            svd_lowrank_niter=args.svd_lowrank_niter,
            prefill_inputs=prefill_inputs,
            decode_inputs=decode_inputs,
            decode_steps=args.decode_steps,
            warmup=args.warmup,
            iters=args.iters,
            token_stream=token_stream,
            accuracy_seq_len=args.accuracy_seq_len,
            accuracy_num_seqs=args.accuracy_num_seqs,
            accuracy_offset_tokens=accuracy_offset_tokens,
            baseline_ppl_nll_sum=baseline_ppl_nll_sum,
            baseline_ppl_token_count=baseline_ppl_token_count,
            logits_refs=logits_refs if variant != "fp16" else None,
            generation_refs=generation_refs if variant != "fp16" else None,
            preloaded_model=fp16_model if variant == "fp16" else None,
        )
        if variant == "fp16":
            cleanup_model(fp16_model)
            fp16_model = None

    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(args.results_dir, f"llama_fp4_inference_{stamp}.json")
    latest_path = os.path.join(args.results_dir, "latest_llama_fp4_inference.json")

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    with open(latest_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(json.dumps(results, indent=2))
    print(f"Saved benchmark to: {out_path}")


if __name__ == "__main__":
    main()
