# Requirements Document

## Introduction

当前 BasicTS 框架中的软测量（Soft Sensor）任务主要通过 `BasicTSSoftSensorDataset` 和 `BasicTSSoftSensorTaskFlow` 实现。现有设计按变量维度切分输入（过程变量）和目标（质量变量），适用于 encoder-only 模型（如 DLinear、LSTM）在 `exclude_target_from_input=False` 模式下的使用。

然而，对于 Informer 等 encoder-decoder 架构模型，其天然具备两个输入入口（encoder input 和 decoder input），可以通过更合理的时间窗口划分和 mask 策略来实现软测量，而无需修改模型本身的网络结构。

### Encoder-Decoder 在软测量中的语义

Encoder-decoder 架构的两个输入入口在软测量任务中具有明确的功能分工：

- **Encoder（历史信息引导）**: 接收质量变量最后一次已知测量值之前的完整历史数据（所有变量真值可用）。Encoder 通过 self-attention 学习历史时间模式，并将历史上下文编码为隐状态，供 decoder 通过 cross-attention 查询。其核心作用是提供长期历史趋势和周期性信息的指导。

- **Decoder（变量相关性学习与续写预测）**: 接收从质量变量未知时刻开始的数据窗口。在滞后区间内，过程变量的真值可用而质量变量被 mask，decoder 需要通过以下机制推断质量变量：
  1. **Cross-attention**: 关注 encoder 编码的历史上下文，获取历史趋势引导
  2. **Self-attention**: 在 decoder 输入的过程变量之间建立变量相关性，学习"过程变量 → 质量变量"的映射关系
  3. **Causal mask**: 保证自回归续写的因果性，逐步生成质量变量的估计/预测值

这种设计使得 encoder-decoder 模型在软测量中同时具备：
- 历史信息的长程依赖建模（encoder cross-attention）
- 过程变量与质量变量之间的实时相关性学习（decoder self-attention）
- 基于已知信息的自回归续写能力（causal generation）

### 与 Encoder-Only 模型的对比

Encoder-only 模型（如 PatchTST、iTransformer、TimeXer）采用 N 输入通道 → N 输出通道的设计，无法在输入中不包含目标变量的情况下生成目标变量的预测。而 encoder-decoder 模型通过 decoder 的 projection 层（`nn.Linear(hidden_size, num_features)`）可以从隐状态空间映射到任意变量，结合 mask 策略即可实现不依赖目标变量未来真值的软测量。

## Glossary

### 时间窗口定义

- `t_begin`: 单次输入时间序列数据的起始时间步
- `t_end`: 单次输入时间序列数据的结束时间步（左闭右开），`t_end - t_begin = input_len`
- `t0`: 第一个质量变量未知的时间步，`t0 = t_end - measurement_lag`
- `measurement_lag`: 质量变量的测量滞后步数
- `pred_len`: 超出输入窗口的短期预测步数（用户自定义，默认为 0）
- `lookback`: decoder 回望窗口长度，用于引导 decoder 输出

### 窗口划分（左闭右开区间）

```
输入数据:  [t_begin, t_end)                         长度 = input_len
t0 = t_end - measurement_lag                       第一个质量变量未知的时间步

Encoder:  [t_begin, t0)                             长度 = input_len - measurement_lag
Decoder:  [t0 - lookback, t_end + pred_len)         长度 = measurement_lag + pred_len + lookback

Decoder 内部:
  [t0 - lookback, t0)        → lookback 区间，所有变量保留真值
  [t0, t_end)                → 滞后区间，过程变量保留真值，质量变量置零
  [t_end, t_end + pred_len)  → 预测区间，过程变量保留真值（如可获取）或置零，质量变量置零

预测目标: [t0, t_end + pred_len) 的质量变量          长度 = measurement_lag + pred_len
  [t0, t_end)                → 估计区间（已发生但未测量）
  [t_end, t_end + pred_len)  → 预测区间（未来值，当 pred_len > 0 时存在）
```

- **Encoder 输入**: `[t_begin, t0)`，长度 = `input_len - measurement_lag`
  - 所有变量保留真值（质量变量在 t0 之前已完成测量）
  - 提供完整的历史上下文信息

- **Decoder 输入**: `[t0 - lookback, t_end + pred_len)`，长度 = `measurement_lag + pred_len + lookback`
  - Lookback 区间 `[t0 - lookback, t0)`: 所有变量保留真值（已测量完成，作为引导）
  - 滞后区间 `[t0, t_end)`: 过程变量保留真值，质量变量置零（已发生但未测量）
  - 预测区间 `[t_end, t_end + pred_len)`: 所有变量置零（过程变量和质量变量均为未来未知值）

- **模型 output_len 配置**: 应设为 `measurement_lag + pred_len + lookback`（decoder 完整输出长度）

### lookback 参数

- `lookback=0`: decoder 输入仅包含滞后+预测区间，长度 = measurement_lag + pred_len
- `lookback>0`: decoder 向前回望 lookback 步已知数据作为引导

### pred_len 参数

- `pred_len=0`: 纯软测量估计，输出仅覆盖滞后区间
- `pred_len>0`: 软测量 + 短期预测，利用 transformer 的续写能力

## Requirements

### REQ-1: Encoder-Decoder 软测量数据集

**描述**: 新建或扩展数据集类，支持为 encoder-decoder 模型生成按时间维度划分的 encoder/decoder 输入对。

**验收标准**:
- [ ] 数据集返回的 item 包含 `inputs`（encoder 输入）和 `targets`（decoder 输入）两个张量
- [ ] encoder 输入 shape: `[input_len - measurement_lag, num_features]`
- [ ] decoder 输入（targets）shape: `[measurement_lag + pred_len + lookback, num_features]`
- [ ] decoder 输入中，lookback 区间所有变量保留真值
- [ ] decoder 输入中，滞后区间的质量变量置零，过程变量保留真值
- [ ] decoder 输入中，预测区间的所有变量置零（过程变量和质量变量均为未来未知值）
- [ ] 支持 `lookback` 参数配置，默认值为 0
- [ ] 支持 `pred_len` 参数配置，默认值为 0
- [ ] `lookback=0` 且 `pred_len=0` 时退化为纯滞后区间估计

### REQ-2: Encoder-Decoder 维度一致性

**描述**: encoder 和 decoder 的输入变量维度必须一致，均为全部 `num_features` 通道。

**验收标准**:
- [ ] encoder 输入和 decoder 输入的最后一个维度（变量维度）相同，均为 num_features
- [ ] 不对变量维度做任何切分或排除操作
- [ ] 模型配置中的 `num_features` 对应数据集的全部变量数

### REQ-3: 输出处理与预测扩展

**描述**: 模型输出为 `[batch, measurement_lag + pred_len + lookback, num_features]`，需要从中提取质量变量的有效预测区间。

**验收标准**:
- [ ] postprocess 从模型输出中跳过前 lookback 步，提取后 `measurement_lag + pred_len` 步
- [ ] 从提取结果中仅保留质量变量通道（`target_vars`）
- [ ] 输出目标长度 = `measurement_lag + pred_len`，其中：
  - `[0, measurement_lag)` 为估计区间（已发生未测量的质量变量）
  - `[measurement_lag, measurement_lag + pred_len)` 为预测区间（未来的质量变量）
- [ ] 当 `pred_len == 0` 时为纯软测量估计
- [ ] 当 `pred_len > 0` 时为软测量 + 短期预测
- [ ] 支持 inverse transform 还原到原始尺度
- [ ] 与现有 metric 计算逻辑兼容（支持 eval_horizons 按时间步评估）

### REQ-3.1: Taskflow 适配

**描述**: 提供适配 encoder-decoder 软测量的 taskflow，处理归一化、mask 和后处理逻辑。

**验收标准**:
- [ ] preprocess 阶段对 encoder 和 decoder 输入分别进行归一化
- [ ] preprocess 阶段对 decoder 输入中滞后+预测区间的质量变量置零
- [ ] postprocess 阶段按上述 REQ-3 逻辑提取有效预测
- [ ] 与现有 `BasicTSSoftSensorTaskFlow` 的 metric 计算逻辑兼容

### REQ-4: 配置支持

**描述**: 在 `BasicTSSoftSensorConfig` 或新配置类中支持 encoder-decoder 软测量的参数。

**验收标准**:
- [ ] 支持 `lookback` 参数配置（默认 0）
- [ ] 支持 `pred_len` 参数配置（默认 0）
- [ ] 配置中的 `exclude_target_from_input` 对 encoder-decoder 模型不适用（始终使用全部变量）
- [ ] 配置验证：`lookback >= 0`，`lookback <= input_len - measurement_lag`
- [ ] 配置验证：`pred_len >= 0`
- [ ] 模型的 `output_len` 自动计算为 `measurement_lag + pred_len + lookback`

### REQ-5: Informer 实验配置

**描述**: 为 EthyDistillation 和 Debutanizer benchmark 提供 Informer 软测量实验配置。

**验收标准**:
- [ ] 提供 EthyDistillation 数据集上的 Informer 实验配置
- [ ] 提供 Debutanizer 数据集上的 Informer 实验配置
- [ ] 实验配置中明确标注 encoder-decoder 软测量的设计说明
- [ ] 配置可正常启动训练（无维度错误）

### REQ-6: 与现有框架兼容

**描述**: 新增功能不影响现有 encoder-only 模型的软测量流程。

**验收标准**:
- [ ] 现有 DLinear、LSTM、NLinear 等模型的实验配置无需修改即可正常运行
- [ ] 现有 `BasicTSSoftSensorDataset` 和 `BasicTSSoftSensorTaskFlow` 保持不变
- [ ] 新增的 dataset/taskflow 作为独立模块，不影响已有逻辑

## 约束与假设

1. **模型不修改**: Informer 模型本身的网络结构不做修改，仅通过数据预处理实现软测量适配
2. **维度一致**: encoder 和 decoder 的 `num_features` 必须相同，等于数据集的全部变量数
3. **信息不泄露**: decoder 输入中滞后区间的质量变量必须置零，预测区间的所有变量必须置零，防止模型在训练时直接看到答案
4. **通用性**: 设计应适用于所有 encoder-decoder 架构模型（Informer、Autoformer 等），不仅限于 Informer
5. **解码策略无关**: 框架设计不限制模型内部的解码策略。模型的 `forward` 接口统一为接收 encoder/decoder 输入、返回完整预测序列 `[batch, measurement_lag + pred_len + lookback, num_features]`。无论模型内部采用一次性生成（如 Informer 的生成式解码器）还是逐步自回归生成（如 GPT 式 token-by-token），均由模型 `forward` 内部封装实现，框架层（dataset、taskflow、runner）无需感知或修改
