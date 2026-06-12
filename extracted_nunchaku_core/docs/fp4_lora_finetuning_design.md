# FP4 + BF16 LoRA 微调接口设计

## 目标

把当前已经独立出来的 Nunchaku FP4 推理/反向算子推进到可接入微调的接口层：

```text
y = FP4_GEMM(x, W0) + scaling * (x @ A.T) @ B.T + bias
```

- `W0` 是冻结的 FP4 backbone 权重。
- `A/B` 是可训练的 BF16/FP16 LoRA 权重。
- forward 的 FP4 主分支复用现有 `gemm_w4a4` CUDA kernel。
- backward 的 `dX_main = dY @ W0` 复用现有 transient repack + FP4 GEMM 路径。
- `dA/dB` 先用 PyTorch BF16/FP16 matmul，作为 P0 可训练接口；后续再把低秩梯度规约融合进 CUDA。

## 本轮调研结论

### Nunchaku / SVDQuant 路径

- 原始核心创新是 `4-bit backbone + 16-bit low-rank branch`。
- 当前 extracted core 已经具备：
  - `NunchakuFP4GemmOp`：纯 FP4 forward。
  - `NunchakuFP4LowRankOp`：FP4 + low-rank fused forward。
  - `NunchakuFP4BackwardDXOp`：纯 FP4 backward dX。
  - `NunchakuFP4LowRankBackwardDXOp`：混合 backward dX 和多种 full backward 消融。
- 现有 full backward 更像固定 residual low-rank 分支的算子验证；微调需要显式暴露 trainable LoRA 参数。

### KernelWiki / contest solution

- NVFP4/FP4 的高性能路径依赖 Blackwell 原生 FP4 MMA，以及每 16 个元素一组的细粒度 scale。
- DeepGEMM / `tcgen05.mma` 资料更偏 SM100/B200 数据中心 Blackwell；本机 RTX 5090 路径此前可编译的是 Nunchaku 手写 `mma.sync ... mxf4nvf4` 路径。
- contest workflow 强调每个候选实现都要留下 task contract、正确性验证、benchmark 证据和保留/回滚决策。

### personal-vault 里的 FP4 finetuning 需求

- 目标不是训练 FP4 backbone，而是冻结量化 backbone，只训练 LoRA。
- 关键公式：

```text
forward:  y  = x @ Q4(W0).T + x @ A.T @ B.T
backward: dX = dY @ Q4(W0)   + dY @ B @ A
          dB = dY.T @ (x @ A.T)
          dA = (dY @ B).T @ x
```

- 主要工程风险：
  - backward 不能常驻保存第二份 transposed FP4 packed weight，否则压缩权重内存接近翻倍。
  - 如果完全不缓存 forward LoRA activation，`dB` 需要重算 `x @ A.T`。
  - outlier 激活仍会限制 FP4 主分支精度，需要训练时的 scale/smooth 策略继续单独研究。

## 已落地接口

新增：

- `native_fp4.training.NunchakuFP4LoRALinear`
- `benchmarks/validate_native_fp4_lora_training.py`

模块语义：

```python
from native_fp4 import NunchakuFP4LoRALinear

op = NunchakuFP4LoRALinear(
    weight=linear.weight,
    bias=linear.bias,
    rank=32,
    lowrank_dtype=torch.bfloat16,
    init="zero",
)
y = op(x)
```

参数：

- `weight`：CUDA 上的 FP16/BF16 dense 权重，构造时量化为 frozen FP4 backbone。
- `bias`：可选。默认作为 frozen buffer；`train_bias=True` 时作为可训练参数。
- `rank`：LoRA rank，会向上补齐到 16 的倍数，便于后续复用 Nunchaku 低秩 packed layout。
- `lora_alpha`：默认等于补齐后的 rank，因此默认 `scaling=1`。
- `lowrank_dtype`：`torch.bfloat16` 或 `torch.float16`。
- `init`：
  - `zero`：标准 LoRA 零效果初始化，`A` Kaiming，`B` 为 0。
  - `gaussian`：`A/B` 都用小方差正态，用于 correctness/压力测试。
  - `residual_svd`：用 `W0 - dequant(Q4(W0))` 的低秩 SVD 初始化 LoRA，贴近 SVDQuant residual branch。
- `cache_lora_act`：是否保存 forward 的 `x @ A.T`，避免 backward 计算 `dB` 时重算。

## 当前 backward 边界

P0 backward 的计算拆分如下：

```text
dX_main = NunchakuFP4BackwardDXOp(dY)
dy_up   = dY @ B
dX      = dX_main + scaling * dy_up @ A
dB      = scaling * dY.T @ lora_act
dA      = scaling * dy_up.T @ x
```

其中：

- `dX_main` 调 CUDA FP4 kernel。
- `dX_lora/dA/dB` 暂时调 PyTorch matmul。
- 当前没有训练 FP4 backbone，也不会产生 backbone `dW`。
- 当前默认不预存 backward packed weight；沿用 transient CUDA repack，符合“不让常驻内存乘 2”的约束。
- `NunchakuFP4LoRALinear` 会让 forward/backward op 共享同一份 resident `qweight/wscales`，只额外保存 backward scale metadata。

## 后续优化路线

P1：把 `dy_up = dY @ B` 和 `dX_main` 的 quantize/repack 调度重叠。

P2：复用 `quantize_grad_with_lora_dual` 思路，让一次 `dY` 读取同时产出 FP4 quantized `dY` 和 dense/packed `dy_up`。

P3：把 `dA/dB` 的低秩 GEMM 改成小 rank 专用 CUDA kernel，减少 PyTorch kernel launch 和中间张量开销。

P4：加入 activation cache policy：

- `save_bf16`：保存 BF16 `x` 和 `lora_act`，速度优先。
- `recompute_lora_act`：只保存 `x`，少存一个 `[M, rank]`。
- `checkpoint`：进一步重算上游 activation，面向长序列微调。

P5：加入 outlier-aware FP4 训练策略：

- activation smooth / online scale 统计。
- 对敏感 projection 保留 BF16。
- residual LoRA 初始化和 task LoRA 初始化分离。

## 验证命令

```bash
cd /home/wyj24/projects/nunchaku/extracted_nunchaku_core
conda run -n triton python benchmarks/validate_native_fp4_lora_training.py \
  --m 257 \
  --in-features 3072 \
  --out-features 3584 \
  --rank 32 \
  --dtype bf16 \
  --lowrank-dtype bf16
```

验证项：

如果要验证 backward 重算 `x @ A.T` 的路径，在命令末尾追加 `--no-cache-lora-act`。

- forward wrapper 是否等价于手写 `FP4 main + LoRA dense branch`。
- `dX` 是否等价于 `FP4 backward dX + LoRA dense dX`。
- `dA/dB/bias` 是否等价于手写 BF16/FP16 matmul 梯度。
