# R² Meter 修复说明

## 问题

训练过程中 val/train 的 R² 显示极端负值（如 -1200 万），而最终评估的 R² 是合理的（如 -4.4）。

### 根因

R² 使用 `AvgMeter`（加权平均）累积。每个 batch 独立计算一个全局 R²，然后对所有 batch 做加权平均。但 R² 不是简单可加的指标：

- 某些 batch 内 target 方差极小（`ss_tot ≈ 0`）
- 此时 `R² = 1 - ss_res / ss_tot` 爆炸到 -几百万
- 一个极端 batch 就能把整个 epoch 的平均值拉到 -1200 万

### 为什么最终评估没问题

最终评估（eval 阶段）也是逐 batch 累积，但因为：
1. 使用 best model，预测更准，极端值更少
2. batch 数量多时极端值被稀释
3. 但严格来说 eval 的 R² 也不完全精确

## 解决方案

新增 `R2Meter` 类，注册到 `METRIC_METER['R2']`。

### 核心思路：Clip + 加权平均

```python
class R2Meter:
    def update(self, value, n=1):
        clipped = max(min(value, 1.0), -100.0)
        self._sum += clipped * n
        self._count += n
    
    @property
    def value(self):
        return self._sum / self._count
```

- **Clip 上界 1.0**：R² 的理论最大值
- **Clip 下界 -100.0**：足够保留"模型比均值差很多"的信息，同时过滤掉因方差接近零导致的数值爆炸

### 为什么不做精确的 epoch 级 R²

精确方案需要累积所有 batch 的 `ss_res` 和 `ss_tot`：
```python
R2_epoch = 1 - sum(ss_res_all_batches) / sum(ss_tot_all_batches)
```

但当前 runner 的 meter 接口只接收 `(scalar_value, weight)`，无法传递 `(ss_res, ss_tot)` 元组。要实现精确方案需要：
- 改 metric 函数返回值格式（破坏接口）
- 或改 runner 的 `_metric_forward` + meter 注册逻辑（侵入性大）

Clip 方案在不改动 runner 的前提下，将 val R² 从"完全不可读"（-1200 万）变为"有参考价值"（合理的负数或正数），是当前约束下的最优折中。

## 影响范围

| 组件 | 影响 |
|------|------|
| Forecasting 任务（默认不含 R²） | ❌ 无影响 |
| Forecasting 任务（手动加了 R²） | ✅ val/train R² 显示更合理 |
| 软测量任务 | ✅ val/train R² 显示更合理 |
| 其他指标（MAE/MSE/RMSE/MAPE） | ❌ 无影响 |

## 文件变更

| 文件 | 变更 |
|------|------|
| `src/basicts/metrics/metric_meter.py` | 新增 `R2Meter` 类 |
| `src/basicts/metrics/__init__.py` | 导出 `R2Meter`，注册到 `METRIC_METER` |

## 验证方法

运行软测量 demo，观察训练过程中 val R² 是否从 -1200 万级别变为 -100 ~ 1 的合理范围：

```bash
python examples/softsensor/softsensor_dlinear_predictive_demo.py
```

预期：
- train R²: 0 ~ 0.1（模型在训练集上略好于均值）
- val R²: -10 ~ 0（模型在验证集上可能不如均值，但数值合理）
- test R²: -5 ~ 0（与之前最终评估一致）
