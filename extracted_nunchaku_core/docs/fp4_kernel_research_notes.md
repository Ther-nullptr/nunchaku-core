# FP4 kernel research notes

本文件记录本轮从 KernelWiki / FlashInfer contest workflow / personal-vault 映射到当前 FP4 LoRA 微调内核的可执行结论。

## 已阅读资料

- `/home/wyj24/projects/personal-vault/wiki/projects/fp4-low-bit-finetuning/README.md`
- `/home/wyj24/projects/personal-vault/wiki/projects/fp4-low-bit-finetuning/research-process.md`
- `/home/wyj24/projects/personal-vault/wiki/projects/fp4-low-bit-finetuning/reference/svdquant-absorbing-outliers-by-low-rank-components-for-4-bit-diffusion-models.md`
- `/home/wyj24/projects/personal-vault/wiki/projects/fp4-low-bit-finetuning/reference/qlora-efficient-finetuning-of-quantized-llms.md`
- `/home/wyj24/projects/mlsys2026-flashinfer-contest/prompts/README.md`
- `/home/wyj24/projects/kernel-design-agents/docs/agent-flow.md`
- `/home/wyj24/projects/KernelWiki/wiki/kernels/nvfp4-gemm.md`
- `/home/wyj24/projects/KernelWiki/wiki/hardware/nvfp4.md`
- `/home/wyj24/projects/KernelWiki/sources/prs/sglang/PR-10101.md`
- `/home/wyj24/projects/KernelWiki/sources/prs/flashinfer/PR-2660.md`
- `/home/wyj24/projects/KernelWiki/sources/prs/cutlass/PR-2995.md`

## 映射到当前工程

- personal-vault 的核心假设是 SVDQuant 双分支扩展到可微调：冻结 FP4 backbone + BF16/FP16 trainable LoRA。当前 `NunchakuFP4LoRALinear` 已覆盖 autograd、optimizer 参数组、LoRA-only checkpoint、outlier overrides 和 activation-cache 显存模式。
- KernelWiki 的 `kernel-nvfp4-gemm` 说明标准大 GEMM 的正确方向是 Blackwell tensor core / TMEM / TMA / block-scale pipeline；当前 forward/dX backbone 已经复用 Nunchaku 原生 FP4 GEMM 路径。
- `fp4_activation_cache_lora_down_grad` 不是标准大 GEMM，而是 `rank x K` 的 skinny reduction：`dA = (dY @ B).T @ dequant(qact)`。因此本轮不直接迁移完整 tcgen05 GEMM，而是按 Kernel Design Agents 的流程做小候选、验证、benchmark、保留/拒绝。

## 本轮候选记录

目标：优化 `fp4_activation_cache_lora_down_grad` 的 rank32/rank64/rank128/rank256/rank512 常见 LoRA 形状，降低 `fp4_activation_cache_d_lora_down=True` 显存模式的 backward 开销。

Rank32，RTX 5090，BF16，`benchmark_fp4_lora_activation_cache_policy.py --warmup 5 --iters 10`：

| candidate | 2048 fused dA ms | 4096 fused dA ms | status | reason |
| --- | ---: | ---: | --- | --- |
| baseline `kVec=2,rVec=16` | 0.1294 | 0.3913 | replaced | stable but CTA count is higher on large K |
| `kVec=1,rVec=32` | 0.2083 | 0.6562 | rejected | avoids one rank split but doubles dy/rank loads per two columns |
| `kVec=2,rVec=32` | n/a | n/a | rejected | ptxas reports static shared memory `0x10000` > `0xc000` |
| `kVec=3,rVec=16` | 0.1296 | 0.3478 | promoted | keeps rank tiling, reduces CTA count, stays within 48KB static smem |

Stabler 30-iter promoted result:

| shape | fused dA ms | fused vs dequant+GEMM | dA fused vs dequant rel_l2 |
| --- | ---: | ---: | ---: |
| 2048^2, rank32 | 0.1281 | 0.528x | 2.86e-3 |
| 4096^2, rank32 | 0.3460 | 0.585x | 7.84e-5 |

结论：`kVec=3,rVec=16` 对 2048 基本持平，对 4096 相比旧 `kVec=2,rVec=16` 约 `1.13x`。这仍慢于 `dequant -> GEMM`，但能减少 dense `x_hat` 物化，是显存压力模式的局部改进。

Rank64，RTX 5090，BF16：

| candidate | 2048 fused dA ms | 4096 fused dA ms | status | reason |
| --- | ---: | ---: | --- | --- |
| baseline `kVec=2,rVec=16,threads=256` | 0.4176 | 1.5043 | replaced | repeats activation decode across four rank tiles |
| `kVec=2,rVec=32,threads=128` | 0.3943 | 1.3416 | rejected | reduces rank tiles but keeps the same column tile count |
| `kVec=3,rVec=32,threads=128` | 0.3206 | 0.9876 | promoted | reduces both rank and column tile counts while staying within 48KB static smem |

Correctness gate:

```bash
conda run -n triton python benchmarks/validate_native_fp4_lora_training.py \
  --m 257 \
  --in-features 512 \
  --out-features 768 \
  --rank 64 \
  --dtype bf16 \
  --lowrank-dtype bf16 \
  --fuse-lora-dx \
  --cache-fused-lora-dx \
  --fp4-activation-cache-d-lora-down
```

该验证 `all_passed=true`，`lora_down_grad_vs_manual rel_l2=0`，FP4-cache `dA` 相对 exact saved-x `dA` 的 rel_l2 约 `9.68e-2`，符合该显存模式的近似边界。

Rank128，RTX 5090，BF16：

| candidate | 2048 fused dA ms | 4096 fused dA ms | status | reason |
| --- | ---: | ---: | --- | --- |
| baseline `kVec=2,rVec=16,threads=256` | 0.7334 | 3.3392 | replaced | repeats activation decode across eight rank tiles |
| `kVec=3,rVec=32,threads=128` | 0.5633 | 1.8640 | promoted | reuses the rank64 tile, halves rank tiles, and reduces column tile count |

Correctness gate:

```bash
conda run -n triton python benchmarks/validate_native_fp4_lora_training.py \
  --m 257 \
  --in-features 512 \
  --out-features 768 \
  --rank 128 \
  --dtype bf16 \
  --lowrank-dtype bf16 \
  --fuse-lora-dx \
  --cache-fused-lora-dx \
  --fp4-activation-cache-d-lora-down
```

该验证 `all_passed=true`，`lora_down_grad_vs_manual rel_l2=0`，FP4-cache `dA` 相对 exact saved-x `dA` 的 rel_l2 约 `9.77e-2`。

Rank256，RTX 5090，BF16：

| candidate | 2048 fused dA ms | 4096 fused dA ms | status | reason |
| --- | ---: | ---: | --- | --- |
| baseline `kVec=2,rVec=16,threads=256` | 1.4103 | 5.7223 | replaced | repeats activation decode across sixteen rank tiles |
| `kVec=3,rVec=64,threads=64` | 1.2138 | 4.0133 | rejected | fewer rank tiles, but higher per-thread accumulator pressure and lower row parallelism |
| `kVec=3,rVec=32,threads=128` | 1.0643 | 3.6840 | promoted | reuses the rank64/rank128 tile and is faster than the wider rank tile |

Correctness gate:

```bash
conda run -n triton python benchmarks/validate_native_fp4_lora_training.py \
  --m 257 \
  --in-features 512 \
  --out-features 768 \
  --rank 256 \
  --dtype bf16 \
  --lowrank-dtype bf16 \
  --fuse-lora-dx \
  --cache-fused-lora-dx \
  --fp4-activation-cache-d-lora-down
```

该验证 `all_passed=true`，`lora_down_grad_vs_manual rel_l2=0`，FP4-cache `dA` 相对 exact saved-x `dA` 的 rel_l2 约 `9.75e-2`。

Rank512，RTX 5090，BF16：

| candidate | 2048 fused dA ms | 4096 fused dA ms | status | reason |
| --- | ---: | ---: | --- | --- |
| baseline `kVec=2,rVec=16,threads=256` | 3.1901 | 11.5392 | replaced | repeats activation decode across thirty-two rank tiles |
| `kVec=3,rVec=32,threads=128` | 2.0542 | 7.2254 | promoted | same proven tile remains faster while staying under 48KB static smem |

Correctness gate:

```bash
conda run -n triton python benchmarks/validate_native_fp4_lora_training.py \
  --m 257 \
  --in-features 512 \
  --out-features 768 \
  --rank 512 \
  --dtype bf16 \
  --lowrank-dtype bf16 \
  --fuse-lora-dx \
  --cache-fused-lora-dx \
  --fp4-activation-cache-d-lora-down
```

该验证 `all_passed=true`，`lora_down_grad_vs_manual rel_l2=0`，FP4-cache `dA` 相对 exact saved-x `dA` 的 rel_l2 约 `9.69e-2`。

## Backend policy follow-up

由于 rank32-rank512 的 fused `dA` fast path 仍慢于 `dequant + GEMM`，本轮把训练接口改成显式 backend 策略，而不是继续只追加 rank specialization：

- `fp4_activation_cache_d_lora_down_backend="fused"`：默认路径，直接从 `qact + ascales` 计算 `dA`，不在 backward 物化 dense `x_hat`，用于峰值显存优先。
- `fp4_activation_cache_d_lora_down_backend="dequant_gemm"`：先用 CUDA dequant fast path 生成 dense `x_hat`，再用 torch GEMM 算 `dA`，用于显存允许时的速度消融。

5090 BF16 复测，`m=in=out=4096, rank=64, warmup=5, iters=15`：

| path | latency ms | note |
| --- | ---: | --- |
| saved BF16 `x` dA | 0.0550 | exact gradient, highest saved activation memory |
| FP4 cache `dequant_gemm` | 0.2142 | fastest FP4-cache backend, transient dense `x_hat` |
| FP4 cache fused | 0.9916 | lower transient memory, current CUDA reduction still slow |

Correctness gates:

- `validate_native_fp4_lora_training.py --rank 64 --fp4-activation-cache-d-lora-down --fp4-activation-cache-d-lora-down-backend fused` 通过。
- `validate_native_fp4_lora_training.py --rank 64 --fp4-activation-cache-d-lora-down --fp4-activation-cache-d-lora-down-backend dequant_gemm` 通过。
- `validate_fp4_lora_training_policies.py --modes memory_saving --steps 2 --fp4-activation-cache-d-lora-down-backend dequant_gemm` 通过。

## 下一步

- 若继续追求性能，需要把 `dA` 改写成更接近 tensor-core GEMM 的形式，例如分块 dequant 到 shared/register 后做 rank tile MMA，或者显式分离 `qact` dequant staging 与 rank GEMM reduction。
- 对 rank512 以外的更高 LoRA rank 仍需继续做 tile sweep；当前 rank > 512 保持旧 `kVec=2,rVec=16`，避免无证据推广。
- `dA` 精度瓶颈来自 FP4 activation cache 本身，kernel tile 只能优化速度，不能消除约 `1e-1` 的 exact saved-x `dA` 误差。
