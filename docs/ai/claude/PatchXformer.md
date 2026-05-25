# PatchXformer 模型说明文档

## 概述

PatchXformer 是一个专为工业软测量（Soft Sensing）任务设计的双轴注意力 Transformer 模型。它融合了 PatchTST 的时间 Patch 注意力机制与 iTransformer 的变量注意力机制，并引入了三个软测量专用组件：**变量重要性门控**、**滞后感知位置编码**和**非对称交叉变量注意力**。

### 与现有模型的对比

| 特性 | PatchTST | iTransformer | TimeXer | **PatchXformer** |
|------|----------|--------------|---------|-----------------|
| Patch 策略 | 所有变量 | 无 | 仅目标变量 | 所有变量 |
| 时间注意力 | 逐变量 | 无 | 逐变量（目标） | 逐变量 |
| 变量注意力 | 无 | 全序列 | 全局 Token 交叉注意力 | 逐 Patch 位置 |
| 软测量支持 | 否 | 否 | 部分 | 完整（门控、滞后、非对称） |

### 核心设计决策

1. **所有变量统一 Patch 嵌入** — 不同于 TimeXer 仅对目标变量做 Patch，PatchXformer 对所有变量统一做 Patch 嵌入，为每个输入信号提供丰富的时间表示。
2. **逐 Patch 位置的变量注意力** — 在每个 Patch 位置进行跨变量交互，提供细粒度的时间分辨率。
3. **双轴块：时间 → 变量** — 每个块先捕获变量内的时间模式，再捕获跨变量依赖关系。
4. **变量重要性门控** — 可学习的 Sigmoid 门控机制，自动抑制无关的过程变量。
5. **滞后感知位置编码** — 为跨越测量滞后边界的 Patch 添加可学习嵌入。
6. **非对称注意力** — 当目标变量存在于输入中时，过程变量作为 K/V，目标变量作为 Q。

---

## 架构

```
Input [B, L, N]
    │
    ▼
RevIN Normalization
    │
    ▼
Patch Embedding (per variable) → [B, N, P, H]
    │
    ▼
Lag-Aware Position Encoding
    │
    ▼
┌─────────────────────────────────────┐
│  Dual-Axis Block × num_layers       │
│  ┌─────────────────────────────────┐│
│  │ Temporal Self-Attention          ││
│  │ (along patches, per variable)    ││
│  └─────────────────────────────────┘│
│              │                       │
│              ▼                       │
│  ┌─────────────────────────────────┐│
│  │ Variable Importance Gate         ││
│  │ (optional)                       ││
│  └─────────────────────────────────┘│
│              │                       │
│              ▼                       │
│  ┌─────────────────────────────────┐│
│  │ Variate Attention                ││
│  │ (across variables, per patch)    ││
│  │ - Asymmetric: target=Q, proc=KV ││
│  │ - Self-attn: all variables      ││
│  └─────────────────────────────────┘│
└─────────────────────────────────────┘
    │
    ▼
Final LayerNorm → [B, N, P, H]
    │
    ▼
Extract Target Variable → [B, 1, P, H]
    │
    ▼
Flatten → [B, P*H]
    │
    ▼
Dropout + Linear → [B, output_len]
    │
    ▼
RevIN Denormalization
    │
    ▼
Output [B, output_len, 1]
```

---

## 模块组成

### 1. PatchXformerConfig

配置数据类，继承自 `BasicTSModelConfig`。

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `input_len` | int | 必填 | 输入序列长度 |
| `output_len` | int | 必填 | 预测步长 |
| `num_features` | int | 必填 | 输入变量数 |
| `patch_len` | int | 16 | Patch 长度 |
| `patch_stride` | int | 8 | Patch 步长 |
| `padding` | bool | True | 是否在 Patch 前填充 |
| `hidden_size` | int | 256 | 隐藏维度 |
| `n_heads` | int | 4 | 注意力头数 |
| `intermediate_size` | int | 512 | FFN 中间维度 |
| `hidden_act` | str | "gelu" | 激活函数 |
| `num_layers` | int | 2 | 双轴块层数 |
| `dropout` | float | 0.1 | Dropout 率 |
| `use_revin` | bool | True | 是否使用 RevIN |
| `output_attentions` | bool | False | 是否输出注意力权重 |
| `measurement_lag` | int | 0 | 测量滞后步数 |
| `use_variable_gate` | bool | True | 是否使用变量重要性门控 |
| `use_asymmetric_attn` | bool | True | 是否使用非对称注意力 |
| `target_var_index` | int | -1 | 目标变量索引（-1 表示最后一个） |

### 2. VariableImportanceGate

变量重要性门控模块。

**计算流程：**
1. 跨 Patch 池化：`[B, N, P, H]` → mean → `[B, N, H]`
2. 线性投影：`[B, N, H]` → `[B, N, 1]`
3. Sigmoid 激活 → `[B, N, 1, 1]`（广播形状）
4. 与输入逐元素相乘

**作用：** 自动学习每个过程变量对目标预测的重要性权重，抑制噪声变量的贡献。

### 3. DualAxisBlock

双轴注意力块，包含：

- **时间注意力子层**：MultiHeadAttention + MLPLayer + 残差连接 + LayerNorm
- **变量注意力子层**：MultiHeadAttention + MLPLayer + 残差连接 + LayerNorm
- **可选变量门控**：在时间注意力和变量注意力之间

**数据流：**
```
[B, N, P, H] → reshape [B*N, P, H] → Temporal Self-Attn → reshape [B, N, P, H]
    → Variable Gate (optional)
    → reshape [B*P, N, H] → Variate Attn → reshape [B, N, P, H]
```

**非对称注意力模式：**
- 当 `use_asymmetric=True` 且 `N > 1`：目标变量作为 Query，过程变量作为 Key/Value
- 过程变量的表示在变量注意力后保持不变，仅目标变量被更新
- 当 `use_asymmetric=False` 或 `N == 1`：标准自注意力

### 4. PatchXformerBackbone

骨干网络，负责特征提取。

**组件：**
- `patch_embedding`：PatchEmbedding（来自 basicts.modules.embed）
- `lag_embedding`：nn.Parameter `[1, 1, 1, H]`（当 measurement_lag > 0）
- `dual_axis_blocks`：nn.ModuleList of DualAxisBlock
- `final_norm`：nn.LayerNorm

**num_patches 计算：**
```python
num_patches = (input_len - patch_len) // patch_stride + 1
if padding:
    num_patches += 1
# 例：input_len=96, patch_len=16, patch_stride=8, padding=True → num_patches=12
```

**滞后边界计算：**
```python
lag_boundary_time = input_len - measurement_lag
# 对于 patch_start < lag_boundary_time <= patch_end 的 Patch，添加 lag_embedding
```

### 5. PatchXformerForForecasting

预测任务包装器。

**组件：**
- `backbone`：PatchXformerBackbone
- `revin`：RevIN（可选）
- `head_dropout`：nn.Dropout
- `flatten`：nn.Flatten
- `forecasting_head`：nn.Linear(num_patches * hidden_size → output_len)

**输入/输出：**
- 输入：`[B, input_len, num_features]`
- 输出：`[B, output_len, 1]`（或 dict 当 output_attentions=True）

---

## 张量形状变化

| 阶段 | 形状 | 说明 |
|------|------|------|
| 输入 | `[B, L, N]` | 原始多变量时间序列 |
| RevIN 后 | `[B, L, N]` | 归一化输入 |
| Patch 嵌入后 | `[B, N, P, H]` | 逐变量 Patch 嵌入 |
| 滞后编码后 | `[B, N, P, H]` | 添加滞后位置信息 |
| 时间注意力（内部） | `[B*N, P, H]` | 逐变量注意力 |
| 门控权重 | `[B, N, 1, 1]` | 逐变量重要性 |
| 变量注意力（内部） | `[B*P, N, H]` | 逐 Patch 跨变量注意力 |
| 目标提取 | `[B, 1, P, H]` | 仅目标变量 |
| 展平 | `[B, P*H]` | 线性头输入 |
| 输出 | `[B, output_len, 1]` | 最终预测 |

---

## 使用方式

### 基本用法

```python
from basicts.models.PatchXformer import PatchXformerForForecasting, PatchXformerConfig

config = PatchXformerConfig(
    input_len=96,
    output_len=12,
    num_features=37,
    patch_len=16,
    patch_stride=8,
    hidden_size=256,
    n_heads=4,
    intermediate_size=512,
    num_layers=2,
    dropout=0.1,
    measurement_lag=6,
    use_variable_gate=True,
    use_asymmetric_attn=True,
)

model = PatchXformerForForecasting(config)
```

### 与 BasicTS 框架集成

```python
from basicts import BasicTSLauncher
from basicts.configs import BasicTSSoftSensorConfig
from basicts.models.PatchXformer import PatchXformerForForecasting, PatchXformerConfig

model_config = PatchXformerConfig(
    input_len=96,
    output_len=12,
    num_features=37,
    measurement_lag=6,
    use_variable_gate=True,
    use_asymmetric_attn=True,
)

BasicTSLauncher.launch_training(BasicTSSoftSensorConfig(
    model=PatchXformerForForecasting,
    model_config=model_config,
    dataset_name="EthyDistillation",
    target_vars=20,
    exclude_target_from_input=True,
    measurement_lag=6,
    input_len=96,
    output_len=12,
    # ... 其他训练配置
))
```

### Exclude-Target 模式

当 `exclude_target_from_input=True` 时：
- 输入仅包含过程变量（num_features = 过程变量数）
- 即使 `use_asymmetric_attn=True`，变量注意力也会自动回退为标准自注意力
- 模型作为纯软测量器运行，不使用目标变量的自回归信息

### 获取注意力权重

```python
config = PatchXformerConfig(..., output_attentions=True)
model = PatchXformerForForecasting(config)

output = model(inputs)
# output["prediction"]: [B, output_len, 1]
# output["attn_weights"]: List[Tuple[temporal_attn, variate_attn]] per layer
```

---

## 错误处理

| 场景 | 处理方式 |
|------|----------|
| `num_features < 2` 且 `use_asymmetric_attn=True` | 抛出 ValueError |
| `patch_len > input_len` | 抛出 ValueError |
| `measurement_lag >= input_len` | 抛出 ValueError |
| `hidden_size % n_heads != 0` | 抛出 ValueError |
| `inputs.shape[1] != input_len` | 抛出 ValueError |

---

## 文件结构

```
src/basicts/models/PatchXformer/
├── __init__.py                          # 导出 PatchXformerForForecasting, Backbone, Config
├── arch/
│   ├── __init__.py                      # 导出架构模块
│   ├── layers.py                        # VariableImportanceGate, DualAxisBlock
│   └── patchxformer_arch.py             # PatchXformerBackbone, PatchXformerForForecasting
└── config/
    ├── __init__.py
    └── patchxformer_config.py           # PatchXformerConfig 数据类

experiments/EthyDistillation_benchmark/
└── patchxformer_excl_target.py          # EthyDistillation 基准实验配置
```

---

## 依赖模块

PatchXformer 复用了 BasicTS 框架中的以下现有模块：

- `basicts.modules.embed.PatchEmbedding` — Patch 嵌入（含 ReplicationPad1d 填充和位置编码）
- `basicts.modules.transformer.MultiHeadAttention` — 多头注意力
- `basicts.modules.mlps.MLPLayer` — 前馈网络
- `basicts.modules.norm.RevIN` — 可逆实例归一化

---

## 推荐超参数（EthyDistillation 数据集）

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| input_len | 96 | 输入窗口长度 |
| output_len | 12 | 预测步长（6 估计 + 6 预测） |
| patch_len | 16 | Patch 长度 |
| patch_stride | 8 | 50% 重叠 |
| hidden_size | 256 | 隐藏维度 |
| n_heads | 4 | 注意力头数 |
| num_layers | 2 | 双轴块层数 |
| measurement_lag | 6 | 测量滞后 |
| batch_size | 64 | 批大小 |
| learning_rate | 1e-3 | 初始学习率 |
| weight_decay | 1e-4 | 权重衰减 |

---

## 生成信息

- **生成工具**: Kiro AI (Claude)
- **生成日期**: 2026-05-25
- **基于版本**: PatchXformer v1.0 (初始实现)
