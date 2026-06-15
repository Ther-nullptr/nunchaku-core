# FP4 kernel research notes

本文件记录本轮从 KernelWiki / FlashInfer contest workflow / personal-vault 映射到当前 FP4 LoRA 微调内核的可执行结论。

## 已阅读资料

- `/home/wyj24/projects/personal-vault/wiki/projects/fp4-low-bit-finetuning/README.md`
- `/home/wyj24/projects/personal-vault/wiki/projects/fp4-low-bit-finetuning/research-process.md`
- `/home/wyj24/projects/personal-vault/wiki/projects/fp4-low-bit-finetuning/reference/svdquant-absorbing-outliers-by-low-rank-components-for-4-bit-diffusion-models.md`
- `/home/wyj24/projects/personal-vault/wiki/projects/fp4-low-bit-finetuning/reference/qlora-efficient-finetuning-of-quantized-llms.md`
- `/home/wyj24/projects/mlsys2026-flashinfer-contest/prompts/README.md`
- `/home/wyj24/projects/kernel-design-agents/docs/agent-flow.md`
- `/home/wyj24/projects/mlsys2026-flashinfer-contest-solution/README.md`
- `/home/wyj24/projects/mlsys2026-flashinfer-contest-solution/verify.py`
- `/home/wyj24/projects/mlsys2026-flashinfer-contest-solution/submissions/moe-fp8/solution/python/main.py`
- `/home/wyj24/projects/mlsys2026-flashinfer-contest-solution/submissions/gdn-prefill/solution/python/main.py`
- `/home/wyj24/projects/KernelWiki/wiki/kernels/nvfp4-gemm.md`
- `/home/wyj24/projects/KernelWiki/wiki/hardware/nvfp4.md`
- `/home/wyj24/projects/KernelWiki/wiki/techniques/kernel-fusion.md`
- `/home/wyj24/projects/KernelWiki/wiki/techniques/epilogue-fusion.md`
- `/home/wyj24/projects/KernelWiki/wiki/patterns/memory-bound.md`
- `/home/wyj24/projects/KernelWiki/sources/prs/sglang/PR-10101.md`
- `/home/wyj24/projects/KernelWiki/sources/prs/flashinfer/PR-2660.md`
- `/home/wyj24/projects/KernelWiki/sources/prs/cutlass/PR-2995.md`

## 映射到当前工程

- personal-vault 的核心假设是 SVDQuant 双分支扩展到可微调：冻结 FP4 backbone + BF16/FP16 trainable LoRA。当前 `NunchakuFP4LoRALinear` 已覆盖 autograd、optimizer 参数组、LoRA-only checkpoint、outlier overrides 和 activation-cache 显存模式。
- KernelWiki 的 `kernel-nvfp4-gemm` 说明标准大 GEMM 的正确方向是 Blackwell tensor core / TMEM / TMA / block-scale pipeline；当前 forward/dX backbone 已经复用 Nunchaku 原生 FP4 GEMM 路径。
- `fp4_activation_cache_lora_down_grad` 不是标准大 GEMM，而是 `rank x K` 的 skinny reduction：`dA = (dY @ B).T @ dequant(qact)`。因此本轮不直接迁移完整 tcgen05 GEMM，而是按 Kernel Design Agents 的流程做小候选、验证、benchmark、保留/拒绝。
- `mlsys2026-flashinfer-contest-solution` 的可迁移工程模式不是直接复制某个 kernel，而是固定实验闭环：提交入口自包含、按 shape 分派、JIT/cache 只做一次、verification harness 固定 warmup/iters/容差并汇总 speedup。当前工程新增 `scripts/run_tmux_experiment.py`，把长实验统一成 tmux session + log + metadata，避免 benchmark 证据散落在临时 shell 中。
- `moe-fp8` 的四路径分派和 `gdn-prefill` 的 short/long path dispatcher 对当前 FP4 LoRA 的启示是：不要把所有形状强行交给同一个 kernel。当前接口已经采用 `overlap_lora_grad_min_rows`、`fp4_activation_cache_min_rows`、`fp4_activation_cache_d_lora_down_backend`、target policy 和 sensitivity/outlier overrides 做 shape/policy 分派；本轮进一步加入 `fp4_lora_sequence_finetune_config` / HF `fp4_auto_seq_policy`，用 flattened rows 在短序列 exact preset 与长序列 memory-saving preset 之间自动分派。后续新增 kernel 候选也应先接入这些 gate，再进入默认 preset。

## 本轮候选记录

目标：优化 `fp4_activation_cache_lora_down_grad` 的 rank32/rank64/rank128/rank256/rank512 常见 LoRA 形状，降低 `fp4_activation_cache_d_lora_down=True` 显存模式的 backward 开销。

Rank32，RTX 5090，BF16，`benchmark_fp4_lora_activation_cache_policy.py --warmup 5 --iters 10`：

| candidate | 2048 fused dA ms | 4096 fused dA ms | status | reason |
| --- | ---: | ---: | --- | --- |
| baseline `kVec=2,rVec=16` | 0.1294 | 0.3913 | replaced | stable but CTA count is higher on large K |
| `kVec=1,rVec=32` | 0.2083 | 0.6562 | rejected | avoids one rank split but doubles dy/rank loads per two columns |
| `kVec=2,rVec=32,threads=128` | n/a | 0.3757 | rejected | current compiler accepts the 32KB smem variant, but it increases dy traffic and is slower than split-rank tiling |
| `kVec=3,rVec=16` | 0.1296 | 0.3478 | replaced | keeps rank tiling, reduces CTA count, stays within 48KB static smem |
| `kVec=3,rVec=32,threads=128` | n/a | 0.3586 | rejected | one rank tile but higher accumulator pressure and lower row parallelism |
| `kVec=4,rVec=16,threads=128` | 0.1126 | 0.3060 | promoted | reduces column CTA count while preserving narrow rank tiling |

Stabler 30-iter promoted result:

| shape | fused dA ms | fused vs dequant+GEMM | dA fused vs dequant rel_l2 |
| --- | ---: | ---: | ---: |
| 2048^2, rank32 | 0.1126 | 0.604x | 2.86e-3 |
| 4096^2, rank32 | 0.3060 | 0.659x | 7.84e-5 |

结论：`kVec=4,rVec=16,threads=128` 对 2048 和 4096 都优于旧 rank32 fast path；4096 相比旧 `kVec=2,rVec=16` 约 `1.28x`，相比上一版 `kVec=3,rVec=16` 约 `1.13x`。这仍慢于 `dequant -> GEMM`，但能减少 dense `x_hat` 物化，是显存压力模式的局部改进。

2026-06-15 复测尝试把 rank32 覆盖到单个 rank tile：`kVec=2,rVec=32` 为 `0.3757ms`，`kVec=3,rVec=32` 为 `0.3586ms`，均慢于 production `kVec=4,rVec=16` 的 `0.3095ms` 同量级结果。结论不变：rank32 下减少 FP4 activation 重读不抵消更高 dy 读流量、寄存器压力和 occupancy 损失，因此 production kernel 保持 split-rank tile。

Rank64，RTX 5090，BF16：

| candidate | 2048 fused dA ms | 4096 fused dA ms | status | reason |
| --- | ---: | ---: | --- | --- |
| baseline `kVec=2,rVec=16,threads=256` | 0.4176 | 1.5043 | replaced | repeats activation decode across four rank tiles |
| `kVec=2,rVec=32,threads=128` | 0.3943 | 1.3416 | rejected | reduces rank tiles but keeps the same column tile count |
| `kVec=3,rVec=32,threads=128` | 0.3206 | 0.9876 | promoted | reduces both rank and column tile counts while staying within 48KB static smem |
| `kVec=4,rVec=32,threads=64` | 0.3267 | 0.9804 | rejected | marginal 4096 gain is noise-level and 2048 regresses due to lower row parallelism |

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

## dY 读取融合 follow-up

为评估 `dX/dA/dB` 整体融合的上限，`benchmark_native_fp4_lora_training_breakdown.py` 新增了 overlap 子图计时、correctness 和 `dy_read_model`：

- `cached_pack_sequential_exact`：FP4 `dX` quantize/fused `dy_up`、exact `dB=dY.T@lora_act`、exact `dy_up=dY@B` 三条 consumer，按 3 次读 `dY` 建模。
- `reuse_fused_dy_up`：复用 fused `dX` 路径产出的 `dy_up`，exact `dB` 仍需读 `dY`，按 2 次读 `dY` 建模。
- `hypothetical_single_dy_read_fusion`：理想大融合 kernel，在一次读 `dY` 时同时喂给 FP4 `dX`、`dB` 和 `dy_up/dA`。

RTX 5090 BF16，`m=in=out=4096, rank=32, warmup=5, iters=10`：

| item | value |
| --- | ---: |
| `dy_bytes` | 33.55 MB |
| exact current estimated `dY` reads | 100.66 MB |
| reuse estimated `dY` reads | 67.11 MB |
| hypothetical fused read | 33.55 MB |
| `fused_dx_cached_pack` | 0.2158 ms |
| `fused_backward_overlap_exact` | 0.2595 ms |
| `fused_backward_overlap_reuse` | 0.2908 ms |
| `dense_lora_grad_pair_with_dy_up` | 0.0853 ms |
| exact overlap `dX` rel_l2 vs cached dX | 1.88e-6 |
| exact overlap `dA/dB` rel_l2 vs sequential | 0 |
| reuse overlap `dA` rel_l2 vs sequential | 5.75e-4 |

解释：

- 当前 exact overlap 主要通过和 FP4 repack/dX 重叠获得端到端收益；低秩梯度子图单独 overlap 并不一定更快。
- reuse 路径确实把 `dY` consumer 从 3 次降到 2 次，但它复用的是 quantize/fused low-rank 输出，`dA` 会变成近似梯度。
- 真正的一次读 `dY` 大融合 kernel 仍有理论空间：相对 exact current 可把 `dY` 读流量降到约 `1/3`，相对 reuse 降到约 `1/2`。但这个 kernel 必须同时处理 FP4 quantize/GEMM 输入、`dB` 规约和 `dy_up/dA` 链，不能只用多 stream overlap 替代。

随后对 overlap helper 的 Python-side stream/event 创建做消融。实现上保留 `NUNCHAKU_FP4_LORA_CACHE_OVERLAP_RESOURCES=1` 内部开关，但默认关闭，因为 4096 主形状没有收益：

| shape | cached wall ms | uncached wall ms | cached speedup | correctness |
| --- | ---: | ---: | ---: | --- |
| 1024 forced-overlap | 0.4439 | 0.4482 | 1.010x | all_passed |
| 4096 default-gate | 0.9071 | 0.8837 | 0.974x | all_passed |

结论：复用 Python `torch.cuda.Stream/Event` 对象不是当前瓶颈，且会在 4096 形状引入轻微负收益；默认仍保持 helper 内临时创建资源。下一步真正值得做的是一次读 `dY` 的融合 kernel 或 rank-specialized low-rank gradient kernel，而不是继续堆多 stream 调度。

本轮继续按 Kernel Design Agents 的“候选实现-验证-拒绝”流程尝试了 rank-small dense `dB=dY.T@lora_act` scalar-reduction CUDA 原型，接口为 `native_fp4.lora_up_grad`，只供 benchmark 调用：

| dtype/rank | torch `dB` ms | CUDA scalar `dB` ms | CUDA speedup | rel_l2 vs torch |
| --- | ---: | ---: | ---: | ---: |
| BF16/rank32 | 0.0295 | 0.4878 | 0.060x | 1.60e-4 |
| FP16/rank32 | 0.0226 | 0.4931 | 0.046x | 3.29e-4 |

结论：普通 block-level scalar reduction 无法替代 cuBLAS/Tensor Core GEMM；即使 rank 很小，也会因为重复读取 `lora_act` 和没有 Tensor Core 路径而大幅落后。后续低秩梯度优化应转向 CUTLASS grouped/small-N GEMM、或把 `dB` 规约融合进 FP4 dX quantize 读取 `dY` 的路径。

## Zero-init LoRA-up fast path

本轮尝试继续扩大 rank32 fused `dA` 的 rank tile，结果不如已推广的 `kVec=4,rVec=16,threads=128`：

| candidate | 2048 fused dA ms | 4096 fused dA ms | status | reason |
| --- | ---: | ---: | --- | --- |
| current `kVec=4,rVec=16,threads=128` | ~0.1126 | ~0.3060 | kept | 已知稳定最优 |
| `kVec=2,rVec=32,threads=128` | 0.1345 | 0.3724 | rejected | full-rank tile 降低重复 decode，但列 tile 变窄，整体更慢 |
| `kVec=4,rVec=32,threads=64` | 0.1574 | 0.3951 | rejected | 行并行度不足，occupancy/latency 退化 |

因此本轮没有改 fused `dA` CUDA tile，转而优化 zero-init task LoRA 的首步训练路径。标准 LoRA 初始化 `lora_up=0` 时：

- forward 的 LoRA out 为 0；
- LoRA dX 为 0；
- `dA = (dY @ B).T @ x` 为 0；
- 只有 `dB = dY.T @ (x @ A.T)` 需要计算。

已实现的 `zero_lora_up_fast_path` 用 parameter version 判断是否仍处于初始化零 `lora_up` 状态，不在热路径做全张量检查；optimizer step 或 adapter load 后自动失效。

RTX 5090 BF16，`m=in=out=4096, rank=32, warmup=5, iters=20`：

| config | baseline train step ms | zero-up fast train step ms | speedup | correctness |
| --- | ---: | ---: | ---: | --- |
| fused dX cached-pack | 0.9058 | 0.7439 | 1.218x | all_passed |
| throughput fused forward + fused dX | 0.8768 | 0.7994 | 1.097x | all_passed |
| fused dX cached-pack + zero-up overlap | 0.8371 | 0.7254 | 1.154x | all_passed |
| throughput fused forward + fused dX + zero-up overlap | 0.7713 | 0.7260 | 1.062x | all_passed |

throughput disabled-baseline 使用 native fused forward 生成近似 `lora_act`，`d_lora_up_baseline_vs_exact` 约 `7.1e-4`；fast path 使用 dense `x@A.T`，`d_lora_up_fast_vs_exact=0`。初始 packed LoRA forward/dX cache 均不生成；optimizer post-step hook 在 `lora_up` 更新后恢复正常 cache refresh。

`overlap_lora_grad=True` 下新增了 zero-up 专用 backward overlap：FP4 main `dX`、`dB=dY.T@(x@A.T)` 和可选 frozen residual dX 分别放到 CUDA streams 上执行，不依赖 LoRA packed dX cache，也不新增 resident memory。单看 zero-up fast path 自身，fused dX cached-pack 从 `0.7439ms` 降到 `0.7254ms`（`1.025x`），throughput fused-forward+fused-dX 从 `0.7994ms` 降到 `0.7260ms`（`1.101x`）。带 frozen residual 的短测结果为：fused dX cached-pack `1.0189 -> 0.9127ms`（`1.116x`），throughput fused-forward+fused-dX `1.0146 -> 0.9908ms`（`1.024x`）。
