# 实现计划：单次实验报告

## 概述

为 BasicTS 框架实现 `SingleExperimentReporter` 模块，提供针对单次软测量实验的结构化训练报告生成能力。实现遵循现有 `basicts.analysis` 模块的架构风格，采用组合模式，由独立的分析组件协同工作，通过中央 Reporter 类统一协调。

## 任务列表

- [ ] 1. 搭建模块结构与数据模型
  - [ ] 1.1 在 `src/basicts/analysis/` 下创建 `single` 子包，定义数据类
    - 创建 `src/basicts/analysis/single/__init__.py`，导出 `SingleExperimentReporter`
    - 创建 `src/basicts/analysis/single/models.py`，包含 `ExperimentConfig` 和 `TrainingCurveData` 数据类（按设计文档规格）
    - _对应需求: 1.4, 8.1_

  - [ ] 1.2 更新 `src/basicts/analysis/__init__.py`，导出 `SingleExperimentReporter`
    - 添加 `from .single import SingleExperimentReporter`
    - 更新 `__all__` 列表
    - _对应需求: 8.1_

- [ ] 2. 实现 CheckpointValidator（检查点验证器）
  - [ ] 2.1 创建 `src/basicts/analysis/single/checkpoint_validator.py`
    - 实现 `resolve_checkpoint_dir(path)` 静态方法：若路径含 `cfg.json` 则直接返回；否则查找修改时间最晚的子目录
    - 实现 `validate()` 方法：检查 `cfg.json` 和 `test_metrics.json` 是否存在，解析配置为 `ExperimentConfig`，缺失时抛出 `FileNotFoundError`（含文件名和期望路径），JSON 无效时抛出 `ValueError`
    - 提取字段：`model_name`（来自 `model.name`）、`dataset_name`、`input_len`（来自 `model_config.input_len`）、`output_len`（来自 `model_config.output_len`）、`measurement_lag`（默认 0）、`metrics`、`eval_horizons`、`target_vars`，根据 `measurement_lag >= 1` 判断 `is_soft_sensor`
    - _对应需求: 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7_

  - [ ]* 2.2 编写 CheckpointValidator 的属性测试
    - **属性 1：必需文件验证** — 对任意目录路径，验证器准确报告缺失的必需文件集合
    - **属性 2：配置提取保持源值** — 对任意有效 cfg.json，提取的字段与源 JSON 完全匹配
    - **属性 3：最新目录选择** — 对含多个子目录的模型目录，返回修改时间最晚的子目录
    - _验证需求: 1.1, 1.2, 1.3, 1.4, 1.6_

- [ ] 3. 实现 LossCurvePlotter（损失曲线绘制器）
  - [ ] 3.1 创建 `src/basicts/analysis/single/loss_curve_plotter.py`
    - 实现 `extract_loss_data()`：优先从 `tensorboard/` 子目录的 TensorBoard 事件文件读取（使用 `event_accumulator`）；不可用时回退到解析最新的 `training_log_*.log` 文件
    - 实现训练日志解析器：用正则提取每个 epoch 的 train/loss、val/loss、test/loss；检测 "Early stopping at epoch N" 模式
    - 实现 `plot()`：单图绘制损失曲线，不同颜色区分，图例标签 "Train Loss"、"Val Loss"、"Test Loss"；X 轴 "Epoch"（整数刻度从 1 开始），Y 轴 "Loss"；若检测到 Early Stopping 则添加垂直虚线标注；仅绘制可用曲线；保存为 PNG（≥150 DPI）；返回 Figure 或 None
    - _对应需求: 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8_

  - [ ]* 3.2 编写 LossCurvePlotter 的属性测试
    - **属性 4：训练日志损失解析往返** — 对任意有效训练日志内容，解析的损失值与日志中的值数值匹配
    - **属性 5：部分损失曲线处理** — 对任意可用损失类型子集，图中线条数量等于可用类型数
    - _验证需求: 2.2, 2.7_

- [ ] 4. 实现 MetricCurvePlotter（指标曲线绘制器）
  - [ ] 4.1 创建 `src/basicts/analysis/single/metric_curve_plotter.py`
    - 实现 `extract_metric_data()`：从 TensorBoard 或训练日志中提取 `cfg.json` 配置的所有指标的逐 epoch 数据
    - 实现 `plot()`：每个有数据的指标生成一个子图，展示 train 和 val 曲线（不同颜色）；跳过无数据的指标；返回 Figure 或 None
    - _对应需求: 3.1, 3.2, 3.3, 3.4_

  - [ ]* 4.2 编写 MetricCurvePlotter 的属性测试
    - **属性 6：指标子图数量匹配配置** — 对任意有数据的配置指标列表，图中子图数量与之完全匹配
    - _验证需求: 3.1, 3.2_

- [ ] 5. 检查点 - 确保所有测试通过
  - 运行所有已有测试，确认通过。如有问题询问用户。

- [ ] 6. 实现 MetricSummarizer（指标汇总器）
  - [ ] 6.1 创建 `src/basicts/analysis/single/metric_summarizer.py`
    - 实现 `summarize()`：读取 `test_metrics.json`，返回包含 "overall" 和 "horizon_N" 键的字典
    - 实现 `to_markdown_table()`：生成 Markdown 表格，overall 指标（MAE、MSE、RMSE、R2 保留 4 位小数）和逐 horizon R2 表格
    - 实现 `compute_zone_averages()`：对软测量任务（measurement_lag >= 1），计算估计区间（horizon ≤ measurement_lag）和预测区间（horizon > measurement_lag）的平均 R2 及差值 Δ
    - 优雅处理 `test_metrics.json` 缺失/无效的情况，返回占位文本
    - _对应需求: 4.1, 4.2, 4.3, 4.4, 4.5_

  - [ ]* 6.2 编写 MetricSummarizer 的属性测试
    - **属性 7：指标表格值格式正确** — 对任意含 "overall" 部分的有效 test_metrics.json，表格中每个值保留 4 位小数
    - **属性 8：区间 R2 计算正确性** — 对任意 horizon R2 值集合和 measurement_lag >= 1，区间平均值等于对应子集的算术平均
    - **属性 14：summarize_metrics 返回完整数据** — 对任意有效 test_metrics.json，返回字典包含所有键和匹配的值
    - _验证需求: 4.1, 4.2, 4.3, 4.4, 8.6_

- [ ] 7. 实现 PredictionPlotter（预测曲线绘制器）
  - [ ] 7.1 创建 `src/basicts/analysis/single/prediction_plotter.py`
    - 实现 `plot()`：从 `test_results/` 加载 `prediction.npy` 和 `targets.npy`；仅提取 `target_vars` 指定的维度；每个 `eval_horizons` 条目生成一个子图；绘制真实值和预测值曲线（不同颜色，图例 "真实值"、"预测值"）；子图标题标注 "Horizon {n} (R2={value})"；展示样本数为 `min(num_samples, 可用长度)`；返回 Figure 或 None
    - 优雅处理 `test_results/` 目录缺失的情况，提示用户设置 `save_results=True`
    - _对应需求: 5.1, 5.2, 5.3, 5.4, 5.5, 5.6_

  - [ ]* 7.2 编写 PredictionPlotter 的属性测试
    - **属性 9：目标变量维度提取** — 对任意预测数组和 target_vars 配置，仅绘制指定维度
    - **属性 10：预测子图数量匹配 eval_horizons** — 对任意 eval_horizons 列表，图中子图数量完全匹配
    - **属性 11：样本数量限制** — 对任意 num_samples 和数据长度，展示步数等于 min(num_samples, 数据长度)
    - _验证需求: 5.1, 5.2, 5.4, 5.6_

- [ ] 8. 实现 ReportAssembler 和 SingleExperimentReporter
  - [ ] 8.1 创建 `src/basicts/analysis/single/report_assembler.py`（报告组装器）
    - 实现 `assemble()`：创建 `report/` 和 `report/fig/` 目录；按顺序组装 Markdown 报告各节：实验配置摘要（生成时间、模型名称、数据集名称、input_len、output_len、measurement_lag）、损失曲线、指标曲线、最终指标表格、预测对比图；图片使用相对路径 `fig/<filename>.png`；标注跳过的节；保存为 `report.md`；返回绝对路径
    - _对应需求: 6.1, 6.2, 6.3, 6.4, 6.5, 6.7_

  - [ ] 8.2 创建 `src/basicts/analysis/single/reporter.py`（主入口类）
    - 实现 `__init__()`：调用 `CheckpointValidator` 验证并解析配置；存储 `output_dir` 和 `num_samples`
    - 实现 `generate_report()`：协调所有组件（LossCurvePlotter、MetricCurvePlotter、MetricSummarizer、PredictionPlotter、ReportAssembler）；保存图片到 `report/fig/`；控制台输出报告路径；返回报告路径字符串
    - 实现 `plot_loss_curves()`、`plot_metric_curves()`、`plot_predictions()`：委托给对应绘制器，返回 Figure 对象
    - 实现 `summarize_metrics()`：委托给 MetricSummarizer，返回字典
    - _对应需求: 8.1, 8.2, 8.3, 8.4, 8.5, 8.6, 6.6_

  - [ ]* 8.3 编写 ReportAssembler 的属性测试
    - **属性 12：报告内容完整性** — 对任意实验配置，报告包含所有头部字段，图片引用使用相对路径，跳过的节包含不可用提示
    - _验证需求: 6.4, 6.5, 6.7_

- [ ] 9. 实现 CLI 命令行入口
  - [ ] 9.1 创建 `src/basicts/analysis/single/__main__.py`
    - 使用 `argparse`：必选参数 `--ckpt_dir`，可选参数 `--output_dir`（默认 `{ckpt_dir}/report/`），可选参数 `--num_samples`（默认 200，范围 1-10000）
    - 验证 `--ckpt_dir` 存在且为目录（否则退出码 1）
    - 验证 `--num_samples` 范围（超出则退出码 2）
    - `--help` 显示所有参数说明及默认值
    - 成功时实例化 `SingleExperimentReporter` 并调用 `generate_report()`
    - _对应需求: 7.1, 7.2, 7.3, 7.4, 7.5, 7.6_

  - [ ]* 9.2 编写 CLI num_samples 验证的属性测试
    - **属性 13：num_samples 范围验证** — 对任意整数值，CLI 当且仅当值在 [1, 10000] 范围内时接受；超出范围导致非零退出码
    - _验证需求: 7.3, 7.5_

- [ ] 10. 最终检查点 - 确保所有测试通过
  - 运行全部测试，确认通过。如有问题询问用户。

## 备注

- 标记 `*` 的任务为可选项，可跳过以加速 MVP 交付
- 每个任务标注了对应的需求编号，便于追溯
- 检查点任务确保增量验证
- 属性测试验证设计文档中的通用正确性属性
- 实现遵循现有 `basicts.analysis` 模块架构（组合模式）
- 所有测试文件放置在 `tests/basicts_test/` 下，遵循现有测试结构
- TensorBoard 事件读取依赖 `tensorboard` 包，实现需优雅处理其缺失情况

## 任务依赖图

```json
{
  "waves": [
    { "id": 0, "tasks": ["1.1"] },
    { "id": 1, "tasks": ["1.2", "2.1"] },
    { "id": 2, "tasks": ["2.2", "3.1", "4.1", "6.1"] },
    { "id": 3, "tasks": ["3.2", "4.2", "6.2", "7.1"] },
    { "id": 4, "tasks": ["7.2", "8.1"] },
    { "id": 5, "tasks": ["8.2", "8.3"] },
    { "id": 6, "tasks": ["9.1"] },
    { "id": 7, "tasks": ["9.2"] }
  ]
}
```
