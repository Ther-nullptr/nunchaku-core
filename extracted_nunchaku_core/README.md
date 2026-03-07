# Extracted Nunchaku Core

This directory isolates the key SVDQuant idea from Nunchaku (paper: https://arxiv.org/abs/2411.05007):

- 4-bit main branch (INT4 quantized GEMM path)
- 16-bit low-rank residual branch (FP16 LoRA-style correction)

## Layout

- `csrc/`: CUDA extension for INT4 quantization + pack/unpack.
- `nunchaku_core/`: PyTorch wrappers.
- `benchmarks/`: speed benchmark vs FP16 baseline.
- `results/`: benchmark outputs.

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

This directory also includes native nunchaku FP4 wrappers:

- `native_fp4.NunchakuFP4GemmOp`: pure FP4 GEMM operator.
- `native_fp4.NunchakuFP4LowRankOp`: FP4 main branch + FP16 low-rank branch.

Benchmark against PyTorch FP16:

```bash
python benchmarks/benchmark_nunchaku_native_fp4.py --dtype fp16
```
