# Nunchaku Core Optimization Summary

## 1. Scope

This document summarizes the optimization work completed in `extracted_nunchaku_core`, with emphasis on:

- standalone extraction of the Nunchaku FP4 backend
- forward FP4 GEMM and FP4 + 16-bit low-rank hybrid operators
- fusion and ablation experiments for the low-rank branch
- FP4 backward operator design and full-backward optimization

The implementation is targeted at **NVIDIA GeForce RTX 5090** and uses the native Blackwell FP4 MMA path.

## 2. Environment

- GPU: `NVIDIA GeForce RTX 5090`
- Driver: `590.44.01`
- PyTorch: `2.9.1+cu128`
- CUDA runtime: `12.8`
- Conda env: `triton`

## 3. Delivered Components

The extracted standalone library lives under:

- [extracted_nunchaku_core](/home/wyj24/projects/nunchaku/extracted_nunchaku_core)

Key implementation pieces:

- standalone vendored CUDA backend:
  - [fp4_backend](/home/wyj24/projects/nunchaku/extracted_nunchaku_core/fp4_backend)
- native extension entry:
  - [fp4_native_ops.cpp](/home/wyj24/projects/nunchaku/extracted_nunchaku_core/csrc/fp4_native_ops.cpp)
- Python wrappers:
  - [operators.py](/home/wyj24/projects/nunchaku/extracted_nunchaku_core/native_fp4/operators.py)
- forward benchmark:
  - [benchmark_native_fp4.py](/home/wyj24/projects/nunchaku/extracted_nunchaku_core/benchmarks/benchmark_native_fp4.py)
- backward benchmark:
  - [benchmark_native_fp4_backward.py](/home/wyj24/projects/nunchaku/extracted_nunchaku_core/benchmarks/benchmark_native_fp4_backward.py)
- backward validation:
  - [validate_native_fp4_backward.py](/home/wyj24/projects/nunchaku/extracted_nunchaku_core/benchmarks/validate_native_fp4_backward.py)

## 4. Main Optimizations

### 4.1 Standalone FP4 backend extraction

Goal:

- remove dependency on the original repository runtime layout
- make the backend directly buildable and usable from `extracted_nunchaku_core`

What was done:

- vendored the CUDA backend into `fp4_backend`
- added standalone extension build and pybind plumbing
- adapted the build for RTX 5090 / Blackwell FP4 path

Result:

- the extracted package is now self-contained and can be built in-place under `extracted_nunchaku_core`

### 4.2 Forward FP4 GEMM extraction

Goal:

- keep only the pure FP4 main branch
- provide a clean PyTorch wrapper and benchmark against PyTorch FP16 GEMM

Result from [latest_native_fp4.json](/home/wyj24/projects/nunchaku/extracted_nunchaku_core/results/latest_native_fp4.json):

- `fp16_ms = 0.6466 ms`
- `fp4_gemm_ms = 0.1391 ms`
- `speedup vs fp16 = 4.648x`

### 4.3 Forward FP4 + 16-bit low-rank hybrid operator

Goal:

- extract the paper’s core hybrid idea:
  - FP4 main branch
  - 16-bit low-rank residual branch

Result from [latest_native_fp4.json](/home/wyj24/projects/nunchaku/extracted_nunchaku_core/results/latest_native_fp4.json):

- `fp4_hybrid_ms = 0.1831 ms`
- `speedup vs fp16 = 3.532x`

Interpretation:

- the low-rank branch adds compute, so hybrid is slower than pure FP4 GEMM
- but it remains substantially faster than FP16 baseline

### 4.4 Fusion ablation: fused vs unfused FP4 + BF16 low-rank branch

Goal:

- measure the cost of a naive unfused low-rank branch
- compare it with the fused path used by Nunchaku

Result from [latest_fp4_bf16_fusion_ablation.json](/home/wyj24/projects/nunchaku/extracted_nunchaku_core/results/latest_fp4_bf16_fusion_ablation.json):

- `fp4_only_ms = 0.1379 ms`
- `fp4_bf16_fused_ms = 0.1810 ms`
- `fp4_bf16_unfused_ms = 0.2939 ms`
- `fused speedup vs fp16 = 3.553x`
- `unfused speedup vs fp16 = 2.188x`
- `unfused / fused = 1.624x`

Interpretation:

- naive unfused low-rank execution is about **62.4%** slower than the fused implementation
- fused and unfused numerical accuracy is effectively the same:
  - `fused_mae_vs_fp16 = 73.0114`
  - `unfused_mae_vs_fp16 = 73.0114`
  - `unfused_vs_fused_mae = 0.0325`

### 4.5 Backward dX: transient FP4 repack instead of persistent duplicated weights

Goal:

- support `dX = dY @ W^T` on the native FP4 layout
- avoid storing a second permanent `qweight_bwd`

Design:

- do not pre-store a second FP4 weight tensor
- keep backward logical scales
- generate `qweight_bwd` transiently during backward

Why:

- FP4 Blackwell layout cannot be reused for `W^T` by a simple view transpose
- pre-storing both forward and backward packed weights would roughly double compressed weight storage

Result:

- permanent memory does not include a second FP4 weight copy
- backward remains compatible with native FP4 GEMM

### 4.6 CUDA repack kernel

Goal:

- replace the Python reference repack with a CUDA implementation

Key file:

- [fp4_repack_cuda.cu](/home/wyj24/projects/nunchaku/extracted_nunchaku_core/csrc/fp4_repack_cuda.cu)

Result from [latest_native_fp4_backward.json](/home/wyj24/projects/nunchaku/extracted_nunchaku_core/results/latest_native_fp4_backward.json):

- `repack_python_ms = 1.4378 ms`
- `repack_ms = 0.04136 ms`
- `repack_speedup_vs_python = 34.76x`

Interpretation:

- the Python repack path was too expensive to use in a realistic backward pipeline
- CUDA repack reduces repack cost to about `22.1%` of pure FP4 `dX`

### 4.7 Fused backward dX with FP4 main branch + 16-bit low-rank branch

Goal:

- implement fused backward `dX = dY @ (W^T + B^T A^T)`

Result from [latest_native_fp4_backward.json](/home/wyj24/projects/nunchaku/extracted_nunchaku_core/results/latest_native_fp4_backward.json):

- `fp16_dx_ms = 0.6078 ms`
- `fp4_dx_ms = 0.1874 ms`
- `fp4_dx_hybrid_unfused_ms = 0.2630 ms`
- `fp4_dx_hybrid_fused_ms = 0.2280 ms`

Speedup:

- pure FP4 backward `dX` vs FP16: `3.244x`
- hybrid unfused `dX` vs FP16: `2.312x`
- hybrid fused `dX` vs FP16: `2.666x`

Interpretation:

- fusion recovers a meaningful part of the low-rank overhead
- fused hybrid `dX` is about `13.3%` faster than unfused hybrid `dX`

### 4.8 Full backward optimization path

Implemented variants:

1. `fp4_full_backward_unfused`
2. `fp4_full_backward_fused`
3. `fp4_full_backward_shared_recompute`
4. `fp4_full_backward_shared_cached`
5. `fp4_full_backward_shared_packed`
6. `fp4_full_backward_shared_dual`
7. `fp4_full_backward_shared_packed_overlap`

Their measured latency from [latest_native_fp4_backward.json](/home/wyj24/projects/nunchaku/extracted_nunchaku_core/results/latest_native_fp4_backward.json):

| Variant | Latency (ms) | Speedup vs FP16 |
| --- | ---: | ---: |
| FP16 full backward | 0.7520 | 1.000x |
| FP4 full backward unfused | 0.3492 | 2.154x |
| FP4 full backward fused | 0.3117 | 2.413x |
| shared recompute | 0.3071 | 2.449x |
| shared cached | 0.2917 | 2.578x |
| shared packed | 0.2770 | 2.715x |
| shared dual | 0.2956 | 2.544x |
| shared packed overlap | 0.2673 | 2.813x |

Meaning of each optimization:

- `shared_recompute`:
  - reuse the fused `dX` path
  - still recompute low-rank helpers
- `shared_cached`:
  - reuse forward cache `x @ A`
  - removes one low-rank recomputation
- `shared_packed`:
  - reuse the packed `dY @ A` internal output
  - decode it for `dB`
  - avoids recomputing dense `dY @ A`
- `shared_dual`:
  - directly emit both packed and dense low-rank activations from quantize
  - removes explicit decode
- `shared_packed_overlap`:
  - keeps `shared_packed` numerics
  - overlaps repack, `d_up`, `d_down`, and fused `dX` across CUDA streams

Net gains:

- fused full backward vs unfused full backward:
  - `0.3492 -> 0.3117 ms`
  - about **10.7%** faster
- shared cached vs fused:
  - `0.3117 -> 0.2917 ms`
  - about **6.4%** faster
- shared packed vs fused:
  - `0.3117 -> 0.2770 ms`
  - about **11.1%** faster
- shared packed overlap vs shared packed:
  - `0.2770 -> 0.2673 ms`
  - about **3.5%** faster
- shared packed overlap vs fused:
  - `0.3117 -> 0.2673 ms`
  - about **14.2%** faster

### 4.9 Dual-output quantize ablation

Goal:

- test whether writing dense `dY @ A` directly during quantization is better than the packed-output-plus-decode route

Result:

- `shared_dual_ms = 0.2956 ms`
- `shared_packed_ms = 0.2770 ms`

Interpretation:

- direct dual-output quantize is **slower** than packed + decode in the current implementation
- the extra dense global reduction/write cost inside quantize is larger than the decode kernel cost

This is an important negative result:

- removing a kernel boundary is not sufficient by itself
- the writeback pattern matters more than the nominal kernel count

## 5. Numerical Validation

Validation result:

- [latest_native_fp4_backward_validation.json](/home/wyj24/projects/nunchaku/extracted_nunchaku_core/results/latest_native_fp4_backward_validation.json)

Current status:

- `all_passed = true`

Important checks:

- `qweight_bwd_cuda_matches_reference = true`
- `full_shared_packed_overlap_dx_matches_fused_rel_l2_lt_5e-4 = true`
- `full_shared_packed_overlap_up_rel_l2_lt_1e-5 = true`
- `full_shared_packed_overlap_down_rel_l2_lt_5e-4 = true`

Representative errors:

- packed low-rank decode vs dense reference:
  - `rel_l2 = 3.252e-4`
- best full-backward path `shared_packed_overlap`:
  - `dx vs fused rel_l2 = 4.236e-6`
  - `lora_down_grad vs reference rel_l2 = 4.221e-4`

Conclusion:

- all optimized backward variants remain numerically aligned with the fused reference path under the chosen thresholds

## 6. Key Design Decisions

### 6.1 What was intentionally avoided

- no permanent second FP4 packed weight tensor for backward
- no dependency on the original repository runtime `.so`
- no reliance on FlashInfer for the native FP4 GEMM path

### 6.2 What turned out to work best

The current best full-backward design is:

1. transient CUDA repack for `W^T`
2. fused FP4 main branch for `dX`
3. forward low-rank cache reuse for `dA`
4. packed low-rank activation reuse plus decode for `dB`
5. multi-stream overlap of repack, `dX`, `dA`, and `dB`

## 7. Commit Timeline

Major optimization commits:

- `eb7c8da` `extract: vendor standalone fp4 cuda backend under core`
- `140dbfd` `benchmark: refresh standalone fp4 speed and correctness reports`
- `379f7ae` `ablation: add unfused fp4+bf16 path and fusion benchmark`
- `be37b48` `benchmark: record fp4-bf16 fusion ablation results`
- `92c5e7b` `backward: add fp4 transient repack dx operators and benchmarks`
- `0e7fb0f` `backward: add cuda fp4 repack kernel and refresh benchmarks`
- `fdef60a` `backward: add cached full fp4 backward path and benchmarks`
- `3fba0bc` `backward: decode packed lowrank activations for dy reuse`
- `cc1f93a` `backward: add dual-output quantize ablation for dy reuse`
- `25f16b2` `backward: overlap repack with packed lowrank branches`

## 8. Current Best Results

Forward:

- pure FP4 GEMM: **4.648x** vs FP16
- FP4 + 16-bit low-rank hybrid: **3.532x** vs FP16

Backward:

- pure FP4 `dX`: **3.244x** vs FP16
- fused hybrid `dX`: **2.666x** vs FP16
- best full backward (`shared_packed_overlap`): **2.813x** vs FP16

## 9. Remaining Work

The main remaining optimization target is:

- fuse `d_up = dY^T @ (x @ A)` into the same `dY` consumption pass

At the moment:

- `shared_packed_overlap` already overlaps multiple branches
- but `d_up` is still issued as a separate GEMM

So the current implementation is a strong scheduling-level fusion, but not yet the final single-pass large backward fusion.
