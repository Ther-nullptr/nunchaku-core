# Extracted Nunchaku Core

This directory isolates the key SVDQuant idea from Nunchaku (paper: https://arxiv.org/abs/2411.05007):

- 4-bit main branch (INT4 quantized GEMM path)
- 16-bit low-rank residual branch (FP16 LoRA-style correction)

## Layout

- `csrc/`: CUDA extension for INT4 quantization + pack/unpack.
- `nunchaku_core/`: PyTorch wrappers.
- `benchmarks/`: speed benchmark vs FP16 baseline.
- `results/`: benchmark outputs.

Optimization summary:

- `OPTIMIZATION_SUMMARY.md`: forward extraction, fusion ablations, backward optimizations, and measured gains on RTX 5090.

## Build

```bash
conda activate triton
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
pip install -e .
```

## Benchmark

```bash
python benchmarks/benchmark_speed.py --device cuda --dtype fp16
```

The benchmark reports:

- `svdq_speedup_vs_fp16`: INT4 main + FP16 low-rank branch
- `int4_only_speedup_vs_fp16`: INT4 main only

For the dedicated 4-bit-only extraction path (without low-rank branch):

```bash
python benchmarks/benchmark_int4_only.py --dtype fp16
```

## Native FP4 Operators

This directory now contains a fully self-contained native FP4 backend:

- Backend CUDA/C++ source: `fp4_backend/src/`
- PyBind entry: `csrc/fp4_native_ops.cpp`
- Built extension: `nunchaku_core._fp4_native_cuda`

The FP4 MMA execution path is inside:

- `fp4_backend/src/kernels/zgemm/gemm_w4a4.cuh` (`mma_fp4` + `mma.sync...mxf4nvf4`)

Python wrappers:

- `native_fp4.NunchakuFP4GemmOp`: pure FP4 GEMM operator.
- `native_fp4.NunchakuFP4LowRankOp`: FP4 main branch + FP16 low-rank branch.

Build native extensions in-place:

```bash
conda activate triton
python setup.py build_ext --inplace
```

Benchmark against PyTorch FP16:

```bash
python benchmarks/benchmark_nunchaku_native_fp4.py --dtype fp16
```

FP4+BF16 fusion ablation (fused vs non-fused low-rank branch):

```bash
python benchmarks/benchmark_fp4_bf16_fusion_ablation.py --dtype fp16 --lowrank-dtype bf16
```
