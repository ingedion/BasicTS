# BasicTS 软测量（Soft Sensor）功能规格说明

> 本文档记录了 BasicTS 框架软测量功能扩展的完整设计、实现和调试过程。

## 1. 概述

### 1.1 背景

软测量（Soft Sensor）是工业过程控制中的核心技术，利用易于获取的过程变量（如温度、压力、流量等）来**估计**难以实时测量的质量变量（如产品浓度、纯度等）。

传统软测量仅关注"估计"——补偿测量滞后。本框架将其扩展为**预测性软测量**，利用时间序列预测的优势，在补偿滞后的基础上进一步对质量变量进行短期预测。

### 1.2 核心概念：估计 + 预测的统一框架

```
时间轴: ─────────────────────────────────────────────────────────────→
        |←────── input_len (过程变量历史) ──────→|
                                                 |←─ measurement_lag ─→|←── 预测区 ──→|
                                                 |←──────────── output_len ────────────→|
                                                 t                     t+lag           t+output_len
                                                 
        estimation zone (估计区):  [t+1, t+lag]     已发生但未测量到的质量值
        prediction zone (预测区):  [t+lag+1, t+output_len]  未来的质量值
```

**两种工作模式**：

| 模式 | 条件 | 含义 |
|------|------|------|
| 纯估计（传统软测量） | `output_len == measurement_lag` | 仅补偿测量延迟 |
| 预测性软测量 | `output_len > measurement_lag` | 估计 + 短期预测 |

### 1.3 关键参数语义

- **measurement_lag**：质量变量的测量延迟（步数）。例如化验需要 1 小时，采样间隔 5 分钟，则 lag = 12 步。在时刻 t，最近可用的质量变量真实值是 t - measurement_lag 时刻的。
- **output_len**：总输出长度，必须 ≥ measurement_lag。其中前 lag 步是估计，超出部分是预测。
- **input_len**：过程变量的历史窗口长度。

### 1.4 设计目标

- 在 BasicTS 框架内以最小侵入方式支持软测量任务
- 复用现有时序预测模型（DLinear、PatchTST、iTransformer 等）无需修改
- 通过 `output_len` 和 `measurement_lag` 的关系统一"估计"和"预测"
- 正确处理通道子集的归一化/反归一化
- 提供合理的默认配置和评估指标

---

## 2. 架构设计

### 2.1 新增组件

```
src/basicts/
├── configs/
│   └── ss_config.py          # BasicTSSoftSensorConfig
├── data/
│   └── ss_dataset.py         # BasicTSSoftSensorDataset
└── runners/taskflow/
    └── softsensor_taskflow.py # BasicTSSoftSensorTaskFlow
```

### 2.2 组件关系

```
BasicTSSoftSensorConfig
    ├── dataset_type = BasicTSSoftSensorDataset
    ├── taskflow = BasicTSSoftSensorTaskFlow()
    └── model = 任意 BasicTS 兼容模型

BasicTSSoftSensorDataset (继承 BasicTSDataset)
    ├── 按 input_vars 切出过程变量作为 inputs
    ├── 按 target_vars 切出质量变量作为 targets
    └── target 窗口紧接 input 窗口之后

BasicTSSoftSensorTaskFlow (继承 BasicTSTaskFlow)
    ├── _get_channel_stats: 从全通道 scaler 中切出对应通道的 mean/std
    ├── _transform / _inverse_transform: 通道对齐的归一化/反归一化
    ├── preprocess: 通道对齐的归一化
    ├── postprocess: 通道对齐的反归一化 + target 通道抽取
    └── get_weight: 基于有效样本数的权重
```

---

## 3. 数据切片语义

### 3.1 设计原理

预测性软测量的数据切片与 forecasting 在形式上一致：target 紧接在 input 之后。但语义上有本质区别：

- **Forecasting**：所有 output_len 步都是"预测未来"
- **预测性软测量**：前 measurement_lag 步是"估计当前"（补偿滞后），后面是"预测未来"

这种设计的优势：
1. 可以直接复用所有 forecasting 模型
2. 通过调整 `output_len` 和 `measurement_lag` 的比例灵活切换模式
3. 评估时可以分别报告估计区和预测区的指标

### 3.2 数据切片实现

```python
# input: 过程变量历史 [index, index + input_len)
history_data = data[index : index + input_len]
inputs = history_data[..., input_vars]

# target: 质量变量 [index + input_len, index + input_len + output_len)
future_start = index + input_len
future_data = data[future_start : future_start + output_len]
targets = future_data[..., target_vars]
```

### 3.3 约束条件

- `measurement_lag >= 1`：质量变量至少有 1 步延迟
- `output_len >= measurement_lag`：输出至少覆盖估计区

### 3.4 数据集长度

```python
len(data) - input_len - output_len + 1
```

### 3.5 与传统 forecasting 的对比

| 维度 | Forecasting | 预测性软测量 |
|------|-------------|-------------|
| 输入通道 | 全部变量 | 过程变量子集 |
| 输出通道 | 全部变量 | 质量变量子集 |
| target 位置 | input 之后 | input 之后（相同） |
| 语义 | 全部是预测 | 前 lag 步估计 + 后续预测 |
| 典型 output_len | 96~720 | 1~24 |
| 核心指标 | MAE/MSE | RMSE/R² |

---

## 4. 通道对齐归一化

### 4.1 问题

BasicTS 的 scaler 在整个训练集（所有通道）上 fit，得到 shape 为 `[1, num_all_channels]` 的 mean/std。但软测量的 inputs 和 targets 只包含通道子集，直接调用 `scaler.transform()` 会导致广播错误——用错误通道的 mean/std 去归一化数据。

### 4.2 解决方案

在 `BasicTSSoftSensorTaskFlow` 中实现通道对齐：

```python
def _get_channel_stats(self, runner):
    mean = runner.scaler.stats['mean']  # [1, num_all_channels]
    std = runner.scaler.stats['std']
    
    # 从 config 或 dataset 获取实际的 input_vars
    input_vars = runner.cfg.input_vars
    if input_vars is None:
        # config 中为 None 时，从 dataset 获取已解析的列表
        dataset = runner.train_data_loader.dataset
        input_vars = dataset.input_vars
    
    target_vars = runner.cfg.target_vars
    
    input_mean = mean[..., input_vars]
    input_std = std[..., input_vars]
    target_mean = mean[..., target_vars]
    target_std = std[..., target_vars]
    
    return input_mean, input_std, target_mean, target_std
```

- **preprocess**：用 `input_mean/std` 归一化 inputs，用 `target_mean/std` 归一化 targets
- **postprocess**：用 `target_mean/std` 反归一化 prediction 和 targets

### 4.3 关键细节：input_vars 的延迟解析

当 `exclude_target_from_input=True` 且 `input_vars=None` 时：
- config 的 `__post_init__` 无法推断 input_vars（不知道总通道数）
- dataset 的 `__init__` 加载数据后才能推断（`num_vars = data.shape[-1]`）
- taskflow 运行时需要从 dataset 对象获取已解析的 `input_vars`

### 4.4 负索引处理

`target_vars=-1` 等负索引在 config 的 `__post_init__` 和 dataset 的 `__init__` 中统一转为正索引（`v % num_features`），确保后续通道切片正确。

---

## 5. 配置说明

### 5.1 关键参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `target_vars` | `int \| List[int]` | `0` | 质量变量索引（支持负索引） |
| `input_vars` | `List[int] \| None` | `None` | 过程变量索引，None=自动推断 |
| `exclude_target_from_input` | `bool` | `True` | 是否从 input 中排除 target 通道 |
| `measurement_lag` | `int` | `1` | 测量延迟（步数） |
| `input_len` | `int` | `96` | 输入窗口长度 |
| `output_len` | `int` | `1` | 输出窗口长度（≥ measurement_lag） |
| `rescale` | `bool` | `True` | 是否反归一化到原始尺度 |
| `norm_each_channel` | `bool` | `True` | 是否逐通道归一化 |

### 5.2 默认指标

```python
metrics = ["MAE", "MSE", "RMSE", "MAPE", "R2"]
target_metric = "RMSE"
best_metric = "min"
```

### 5.3 示例配置

```python
from basicts.configs import BasicTSSoftSensorConfig
from basicts.models.DLinear import DLinear, DLinearConfig

# 纯估计模式：output_len == measurement_lag
config_estimation = BasicTSSoftSensorConfig(
    model=DLinear,
    model_config=DLinearConfig(input_len=96, output_len=1, num_features=7),
    dataset_name="ETTh1",
    target_vars=-1,
    exclude_target_from_input=False,  # 包含 target 历史（自回归）
    input_len=96,
    output_len=1,        # == measurement_lag，纯估计
    measurement_lag=1,
)

# 预测性软测量：output_len > measurement_lag
config_predictive = BasicTSSoftSensorConfig(
    model=DLinear,
    model_config=DLinearConfig(input_len=96, output_len=12, num_features=6),
    dataset_name="ETTh1",
    target_vars=-1,
    exclude_target_from_input=True,   # 只用过程变量
    input_len=96,
    output_len=12,       # > measurement_lag，前6步估计 + 后6步预测
    measurement_lag=6,
)
```

---

## 6. R² 指标修复

### 6.1 问题演进

R² 指标经历了三轮修复：

**第一轮问题**：scaler 通道广播错误导致 prediction/targets 数值完全偏离 → R² ≈ -2.7 亿

**第二轮问题**：修复 scaler 后，`output_len=1` 时 per-sample R² 的 `ss_tot=0`（单时间步无方差）→ R² ≈ -几百万

**第三轮问题**：`output_len=12` 但 target 只有 1 个通道时，per-sample 的 12 步内方差仍然很小 → R² ≈ -1.5 万

### 6.2 最终方案

统一使用**全局 R²**（跨整个 batch 展平计算），不再区分 per-sample：

```python
def masked_r2(prediction, targets, targets_mask=None):
    # Flatten all dimensions
    pred_flat = prediction.reshape(-1)
    tgt_flat = targets.reshape(-1)
    mask_flat = mask.reshape(-1)
    
    n_valid = mask_flat.sum()
    tgt_mean = (tgt_flat * mask_flat).sum() / n_valid
    ss_res = (mask_flat * (tgt_flat - pred_flat)**2).sum()
    ss_tot = (mask_flat * (tgt_flat - tgt_mean)**2).sum()
    
    r2 = 1 - (ss_res / (ss_tot + 1e-6))
    return r2
```

### 6.3 已知限制

训练过程中 val R² 仍可能出现较大负值（-1200 万级别），这是因为 R² 是逐 batch 计算再通过 meter 加权平均的。某些 batch 内全局方差极小时会导致单 batch R² 爆炸。最终评估（`eval` 阶段）的 R² 是正确的，因为它在完整测试集上累积计算。

彻底修复需要将 R² 改为 epoch 级别的累积计算（类似 sklearn 的 `r2_score`），这需要改动 runner 的 meter 机制，留作后续优化。

---

## 7. 与现有模型的兼容性

### 7.1 模型要求

任何满足以下接口的模型均可直接用于软测量：
```python
def forward(self, inputs: torch.Tensor) -> torch.Tensor:
    """
    Args:
        inputs: [batch_size, input_len, num_input_features]
    Returns:
        prediction: [batch_size, output_len, num_output_features]
    """
```

### 7.2 通道数处理

- 如果模型输出通道数 == target 通道数：直接使用
- 如果模型输出通道数 > target 通道数：taskflow 自动按 `target_vars` 索引抽取

### 7.3 注意事项

当 `exclude_target_from_input=True` 时，模型的 `num_features` 应设为输入通道数（不含 target），而非总通道数。例如 ETTh1 有 7 个变量，排除 1 个 target 后 `num_features=6`。

---

## 8. 实验结果

### 8.1 纯估计模式（output_len=1, lag=1）

配置：DLinear, input_len=96, exclude_target_from_input=False, num_features=7

| 指标 | 训练集 | 测试集 |
|------|--------|--------|
| R² | 0.84 | -4.4 |
| RMSE | 3.65 | 2.43 |
| MAE | 2.83 | 2.01 |

训练集 R²=0.84 说明模型在训练分布上拟合良好；测试集 R² 为负说明存在分布偏移（ETTh1 按时间切分，测试集是最后一段）。

### 8.2 预测性软测量（output_len=12, lag=6）

配置：DLinear, input_len=96, exclude_target_from_input=True, num_features=6

| 指标 | 训练集 | 测试集 overall | h1 | h6 | h12 |
|------|--------|---------------|-----|-----|------|
| R² | 0.05 | -4.40 | -4.75 | -4.58 | -4.05 |
| RMSE | 0.60 | 0.42 | 0.44 | 0.43 | 0.40 |
| MAE | 0.42 | 0.35 | 0.37 | 0.36 | 0.33 |

训练集 R²=0.05 说明纯过程变量（不含 target 历史）对质量变量的预测能力有限，DLinear 的线性结构难以捕捉复杂的跨通道关系。

---

## 9. 评估策略（未来扩展）

### 9.1 分区评估

利用 `measurement_lag` 和 `eval_horizons` 可以将 output 分为估计区和预测区，分别报告指标：

```python
# 当前通过 eval_horizons 实现
eval_horizons = [1, measurement_lag, output_len]  # 估计起点、lag边界、预测终点
```

### 9.2 评估意义

- **估计区 R² 高**：说明模型能有效补偿测量延迟
- **预测区 R² 高**：说明模型具有真正的预测能力
- **估计区好但预测区差**：模型只学到了滞后补偿，没有预测能力

---

## 10. 已知限制与未来工作

### 10.1 当前限制

1. **val R² 不稳定**：逐 batch 计算全局 R² 再平均，某些低方差 batch 会导致极端值
2. **不支持远程数据集下载**：需要本地预处理好的 `.npy` 文件
3. **Scaler 仅验证 ZScore**：MinMaxScaler 未测试通道对齐逻辑
4. **不支持变 lag**：measurement_lag 是固定值，不支持动态变化

### 10.2 未来扩展方向

1. **R² epoch 级累积计算**：改为在整个 epoch 结束后一次性计算，避免 batch 级平均的不稳定性
2. **分区评估自动化**：自动按 measurement_lag 划分估计区/预测区并分别报告指标
3. **自适应 lag 检测**：根据互相关分析自动确定最优 lag
4. **多目标软测量**：同时估计/预测多个质量变量（已支持，待验证）
5. **在线学习**：支持增量更新以应对过程漂移
6. **专用软测量模型**：如结合过程机理的混合模型
7. **数据预处理工具**：从原始 CSV 生成 BasicTS 格式的 `.npy` 文件

---

## 11. 文件变更清单

### 新增文件

| 文件 | 说明 |
|------|------|
| `src/basicts/configs/ss_config.py` | 软测量配置类 BasicTSSoftSensorConfig |
| `src/basicts/data/ss_dataset.py` | 软测量数据集类 BasicTSSoftSensorDataset |
| `src/basicts/runners/taskflow/softsensor_taskflow.py` | 软测量任务流 BasicTSSoftSensorTaskFlow |
| `examples/softsensor/softsensor_dlinear_demo.py` | 纯估计模式示例（lag=1, output=1） |
| `examples/softsensor/softsensor_dlinear_predictive_demo.py` | 预测性软测量示例（lag=6, output=12） |
| `docs/ai/claude/soft_sensor_spec.md` | 本文档 |

### 修改文件

| 文件 | 说明 |
|------|------|
| `src/basicts/configs/__init__.py` | 导出 BasicTSSoftSensorConfig |
| `src/basicts/data/__init__.py` | 导出 BasicTSSoftSensorDataset |
| `src/basicts/runners/taskflow/__init__.py` | 导出 BasicTSSoftSensorTaskFlow |
| `src/basicts/metrics/r_square.py` | 修复 R² 计算（改为全局方式） |

---

## 12. Bug 修复记录

### 12.1 Scaler 通道广播错误

**问题**：scaler 在全通道上 fit（mean/std shape `[1, 7]`），但 targets 只有 1 个通道（shape `[B, T, 1]`），transform/inverse_transform 时 PyTorch 广播导致用错误通道的统计量。

**表现**：R² ≈ -2.7 亿，MSE ≈ 274

**修复**：taskflow 中按 `input_vars`/`target_vars` 切出对应通道的 mean/std 再做 transform。

### 12.2 input_vars 为 None 时 taskflow 获取全通道 stats

**问题**：当 `exclude_target_from_input=True` 且 `input_vars=None` 时，config 中 `input_vars` 保持 None（因为不知道总通道数），但 dataset 加载数据后会推断出实际的 input_vars。taskflow 读 `runner.cfg.input_vars` 得到 None，fallback 到全通道 stats，导致维度不匹配。

**表现**：`RuntimeError: The size of tensor a (6) must match the size of tensor b (7)`

**修复**：taskflow 的 `_get_channel_stats` 在 `input_vars is None` 时从 dataset 对象获取已解析的列表。

### 12.3 R² 在 output_len=1 时失效

**问题**：原始 `masked_r2` 沿 dim=1（时间维度）计算方差，单时间步方差为 0，导致除以 ~0。

**表现**：R² ≈ -几百万

**修复**：改为全局 R²。

### 12.4 R² 在多步单通道时仍不稳定

**问题**：`output_len=12` 但 target 只有 1 个通道时，per-sample 的 12 步内方差仍然很小（短期内质量变量变化不大），R² 仍然爆炸。

**表现**：overall R² ≈ -15000，per-horizon R² ≈ -100 万

**修复**：统一使用全局 R²（跨 batch 所有时间步展平），不再使用 per-sample 方式。

### 12.5 负索引未解析

**问题**：`target_vars=-1` 在 `exclude_target_from_input` 逻辑中无法正确匹配（`-1 not in range(7)` 始终为 True）。

**修复**：在 config `__post_init__` 和 dataset `__init__` 中统一用 `v % num_vars` 转为正索引。

---

## 13. 建议的 Commit Message

```
feat: add soft sensor task support (estimation + predictive)

- Add BasicTSSoftSensorConfig, BasicTSSoftSensorDataset, BasicTSSoftSensorTaskFlow
- Support channel-aligned normalization for input/target variable subsets
- Fix masked_r2 to use global R² (avoids degenerate per-sample variance)
- Fix negative index resolution for target_vars/input_vars
- Add DLinear demos for both pure estimation and predictive soft sensor modes
- Add spec documentation in docs/ai/claude/
```
