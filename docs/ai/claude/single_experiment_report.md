# 单次实验报告生成器 (SingleExperimentReporter) 技术文档

> 创建日期: 2026-05-26
> 状态: 已完成（全部核心功能实现）

---

## 概述

`basicts.analysis.single` 是 BasicTS 框架的**单次实验深度分析模块**，用于针对单个实验检查点生成完整的训练报告。与现有的 `ExperimentAnalyzer`（多模型横向对比）互补，本模块聚焦于单个实验的纵向分析：

- 训练/验证/测试损失曲线（含 Early Stopping 标注）
- 各评估指标随 epoch 变化的趋势曲线
- 测试集最终指标汇总（含软测量区间分析）
- 预测值与真实值的对比可视化

## 模块结构

```
src/basicts/analysis/single/
├── __init__.py              ← 模块入口，导出 SingleExperimentReporter
├── __main__.py              ← CLI 入口 (python -m basicts.analysis.single)
├── models.py                ← 数据模型 (ExperimentConfig, TrainingCurveData)
├── checkpoint_validator.py  ← 检查点目录验证与配置解析
├── loss_curve_plotter.py    ← 损失曲线绘制（TensorBoard / 日志回退）
├── metric_curve_plotter.py  ← 指标曲线绘制
├── metric_summarizer.py     ← 测试指标汇总与表格生成
├── prediction_plotter.py    ← 预测对比曲线绘制
├── report_assembler.py      ← Markdown 报告组装
└── reporter.py              ← 主协调器，组合所有组件
```

## 使用方式

### Python API

```python
from basicts.analysis import SingleExperimentReporter

# 一键生成完整报告
reporter = SingleExperimentReporter("checkpoints/Debutanizer_benchmark/DLinear")
report_path = reporter.generate_report()
# 输出: Report generated: D:\...\report\report.md

# 分步调用（适合 Jupyter Notebook 交互使用）
reporter = SingleExperimentReporter("checkpoints/Debutanizer_benchmark/DLinear")

# 绘制损失曲线
fig = reporter.plot_loss_curves()       # 返回 matplotlib Figure

# 绘制指标曲线
fig = reporter.plot_metric_curves()     # 返回 matplotlib Figure

# 绘制预测对比图
fig = reporter.plot_predictions()       # 返回 matplotlib Figure

# 获取指标字典
metrics = reporter.summarize_metrics()
# {'overall': {'MAE': 0.1879, ...}, 'horizon_1': {...}, ...}
```

### CLI

```bash
# 基本用法（自动选择最新的哈希子目录）
python -m basicts.analysis.single --ckpt_dir checkpoints/Debutanizer_benchmark/DLinear

# 指定具体哈希目录
python -m basicts.analysis.single --ckpt_dir checkpoints/Debutanizer_benchmark/DLinear/b370b4e80fd5fff8e6b963acde661bd5

# 自定义输出目录和预测曲线样本数
python -m basicts.analysis.single --ckpt_dir checkpoints/Debutanizer_benchmark/DLinear \
    --output_dir my_reports/dlinear_report \
    --num_samples 500
```

### CLI 参数

| 参数 | 必选 | 默认值 | 说明 |
|------|------|--------|------|
| `--ckpt_dir` | ✅ | — | 实验检查点目录路径（哈希目录或模型目录） |
| `--output_dir` | ❌ | `{ckpt_dir}/report/` | 报告输出目录 |
| `--num_samples` | ❌ | 200 | 预测曲线展示的时间步数量 [1, 10000] |

### 退出码

| 退出码 | 含义 |
|--------|------|
| 0 | 成功 |
| 1 | `--ckpt_dir` 路径不存在或不是目录 |
| 2 | `--num_samples` 超出有效范围 [1, 10000] |

## 输入要求

### 检查点目录结构

```
{ckpt_dir}/                         ← 可以是哈希目录或模型目录
├── cfg.json                        ← 实验配置（必需）
├── test_metrics.json               ← 测试指标（必需）
├── tensorboard/                    ← TensorBoard 事件文件（可选，优先数据源）
│   └── events.out.tfevents.*
├── training_log_*.log              ← 训练日志（可选，TensorBoard 不可用时回退）
└── test_results/                   ← 预测结果（可选，需 save_results=True）
    ├── prediction.npy
    └── targets.npy
```

**必需文件**:
- `cfg.json` — 实验配置，包含模型名称、数据集、输入/输出长度等
- `test_metrics.json` — 测试集评估指标

**可选文件**（缺失时对应报告章节标注"数据不可用"）:
- `tensorboard/` 或 `training_log_*.log` — 用于绘制损失/指标曲线
- `test_results/prediction.npy` + `targets.npy` — 用于绘制预测对比图

### 模型目录自动解析

当提供模型目录（包含多个哈希子目录）时，自动选择**修改时间最晚**的子目录：

```
checkpoints/Debutanizer_benchmark/DLinear/
├── 1b5489b328dffb1a96fea74ea3892e15/    ← 较早的实验
└── b370b4e80fd5fff8e6b963acde661bd5/    ← 最新的实验 ← 自动选择
```

## 输出结构

```
{output_dir}/
├── report.md                ← 完整 Markdown 报告
└── fig/
    ├── loss_curves.png      ← 损失曲线图
    ├── metric_curves.png    ← 指标曲线图
    └── prediction_curves.png ← 预测对比图
```

## 报告内容

生成的 `report.md` 包含以下章节：

### 1. 实验配置摘要

以表格形式展示：生成时间、模型名称、数据集名称、输入长度、输出长度、测量滞后。

### 2. 损失曲线 (Loss Curves)

- 同一张图中绘制 Train Loss、Val Loss、Test Loss
- 不同颜色 + 图例区分
- X 轴 "Epoch"（整数刻度从 1 开始），Y 轴 "Loss"
- 若检测到 Early Stopping，以垂直虚线标注停止位置
- 数据源优先级：TensorBoard 事件文件 > 训练日志文件

### 3. 指标曲线 (Metric Curves)

- 为 cfg.json 中配置的每个 metric 生成独立子图
- 每个子图展示 train 和 val 曲线
- 跳过无数据的指标

### 4. 最终指标表格 (Final Metrics)

- Overall 指标表格（MAE、MSE、RMSE、R2，保留 4 位小数）
- Per-Horizon R2 表格
- 软测量任务额外展示：估计区间/预测区间平均 R2 及差值 Δ

### 5. 预测对比图 (Prediction Plots)

- 为每个 eval_horizon 绘制独立子图
- 真实值（"真实值"）与预测值（"预测值"）双曲线
- 子图标题标注 "Horizon {n} (R2={value})"
- 展示样本数 = min(num_samples, 可用数据长度)

## 核心组件

### CheckpointValidator

验证检查点目录完整性，解析 cfg.json 为 `ExperimentConfig` 数据类。

**从 cfg.json 提取的字段**:
- `model.name` → model_name
- `dataset_name` → dataset_name
- `model_config.input_len` → input_len
- `model_config.output_len` → output_len
- `measurement_lag` → measurement_lag（默认 0）
- `metrics` → metrics 列表
- `eval_horizons` → eval_horizons 列表
- `target_vars` → target_vars 列表
- `measurement_lag >= 1` → is_soft_sensor

### LossCurvePlotter

**数据源优先级**:
1. TensorBoard 事件文件（`tensorboard.backend.event_processing.event_accumulator`）
2. 训练日志文件（正则解析 `training_log_*.log`）

**Early Stopping 检测**: 从日志中匹配 "Early stopping at epoch N" 模式。

### MetricSummarizer

**软测量区间分析**（当 measurement_lag ≥ 1 时）:
- 估计区间 (Estimation Zone): horizon ≤ measurement_lag 的 R2 均值
- 预测区间 (Prediction Zone): horizon > measurement_lag 的 R2 均值
- Δ = 估计区间 R2 − 预测区间 R2

### PredictionPlotter

**数据加载策略**:
1. 标准 `np.load(allow_pickle=True)`
2. 回退到 `np.memmap` 读取（适配 BasicTS 的 memmap 保存方式）
3. 通过文件大小和 output_len 推断 shape

## 优雅降级策略

| 缺失数据 | 行为 |
|----------|------|
| TensorBoard 事件文件 | 回退到训练日志解析 |
| TensorBoard + 训练日志均缺失 | 跳过损失/指标曲线，报告中注明 |
| test_results/ 目录缺失 | 跳过预测曲线，报告中注明需设置 save_results=True |
| 部分 loss 类型缺失 | 仅绘制可用曲线 |
| 部分 metric 数据缺失 | 跳过对应子图 |
| matplotlib 未安装 | 跳过所有绘图，仅生成文本报告 |

**关键原则**: 必需文件（cfg.json、test_metrics.json）缺失时立即报错；可选数据缺失时优雅降级，不中断报告生成。

## 与 ExperimentAnalyzer 的关系

| 特性 | ExperimentAnalyzer | SingleExperimentReporter |
|------|-------------------|--------------------------|
| 分析范围 | 多模型横向对比 | 单次实验纵向分析 |
| 输入 | 实验目录（含多个模型） | 单个检查点目录 |
| 核心图表 | 柱状图、热力图、模型对比 | 损失曲线、指标趋势、预测对比 |
| 适用场景 | 基准测试结果汇总 | 单次训练质量评估 |
| 导入方式 | `from basicts.analysis import ExperimentAnalyzer` | `from basicts.analysis import SingleExperimentReporter` |

两者共存于 `basicts.analysis` 模块中，互不干扰。

## 依赖

- **必需**: numpy（已在 BasicTS 环境中）
- **可选**: matplotlib（图表生成）、tensorboard（TensorBoard 事件读取）

```bash
pip install matplotlib tensorboard
```

未安装 matplotlib 时仅生成纯文本报告；未安装 tensorboard 时自动回退到日志解析。

## 示例输出

以 Debutanizer 数据集上的 DLinear 实验为例：

```
python -m basicts.analysis.single --ckpt_dir checkpoints/Debutanizer_benchmark/DLinear
```

生成报告结构：
```
checkpoints/Debutanizer_benchmark/DLinear/b370b4e80fd5fff8e6b963acde661bd5/report/
├── report.md
└── fig/
    ├── loss_curves.png
    ├── metric_curves.png
    └── prediction_curves.png
```

报告头部示例：

| Parameter | Value |
|-----------|-------|
| Generation Time | 2026-05-25 22:25:57 |
| Model Name | DLinear |
| Dataset Name | Debutanizer |
| Input Length | 32 |
| Output Length | 6 |
| Measurement Lag | 3 |

## 已知限制

1. **CJK 字体**: 预测曲线图例使用中文标签（"真实值"、"预测值"），在未安装 CJK 字体的系统上可能显示为方块。可通过安装 SimHei 或 Noto Sans CJK 字体解决。
2. **大规模数据**: 当 test_results 中的 npy 文件较大时，加载可能需要较多内存。通过 `--num_samples` 参数控制绘图范围不影响加载开销。
3. **TensorBoard 格式兼容性**: 仅支持标准 TensorBoard scalar 事件格式。自定义 summary writer 可能不兼容。
