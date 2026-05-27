# Requirements Document

## Introduction

本功能为 BasicTS 框架提供针对**单独一次软测量实验**的训练报告生成能力。与现有的多模型对比分析模块（`ExperimentAnalyzer`）不同，本功能聚焦于单个实验的完整训练过程分析，包括训练/验证/测试损失曲线、最终评估指标汇总、以及预测曲线与真实值的对比可视化。

用户完成一次实验训练后，可通过本功能快速生成一份结构化的训练报告，用于评估模型训练质量和预测效果。

## Glossary

- **Report_Generator**: 单独实验分析报告生成器，负责解析单次实验产出物并生成完整训练报告
- **Loss_Curve_Plotter**: 损失曲线绘制器，负责从 TensorBoard 事件文件中提取并绘制训练集、验证集和测试集的损失曲线
- **Metric_Summarizer**: 指标汇总器，负责读取 test_metrics.json 并生成结构化的指标摘要
- **Prediction_Plotter**: 预测曲线绘制器，负责加载 prediction.npy 和 targets.npy 并绘制预测对比图
- **Experiment_Checkpoint**: 单次实验的检查点目录，包含 cfg.json、test_metrics.json、tensorboard/、test_results/ 等文件
- **Training_Log**: 训练日志文件（training_log_*.log），记录每个 epoch 的训练、验证和测试指标
- **TensorBoard_Event**: TensorBoard 事件文件（events.out.tfevents.*），记录训练过程中的标量指标

## Requirements

### Requirement 1: 实验检查点目录定位与验证

**User Story:** 作为研究人员，我希望报告生成器能自动定位并验证实验检查点目录的完整性，以便确保报告生成所需的数据文件齐全。

#### Acceptance Criteria

1. WHEN 用户提供一个实验检查点路径时，THE Report_Generator SHALL 验证该路径下存在 cfg.json 文件
2. WHEN 用户提供一个实验检查点路径时，THE Report_Generator SHALL 验证该路径下存在 test_metrics.json 文件
3. IF cfg.json 或 test_metrics.json 不存在，THEN THE Report_Generator SHALL 返回错误信息，包含缺失的文件名称和期望的完整路径
4. WHEN cfg.json 存在且可解析为有效 JSON 时，THE Report_Generator SHALL 从中提取以下参数：模型名称（model.name）、数据集名称（dataset_name）、输入长度（model_config.input_len）、输出长度（model_config.output_len）、测量滞后（measurement_lag）
5. IF cfg.json 存在但无法解析为有效 JSON，THEN THE Report_Generator SHALL 返回错误信息，指出该文件格式无效及其完整路径
6. WHEN 用户提供模型目录路径（包含一个或多个哈希命名子目录）而非具体哈希目录时，THE Report_Generator SHALL 选择该目录下修改时间最晚的哈希子目录作为实验检查点
7. IF 用户提供的模型目录路径下不存在任何子目录，THEN THE Report_Generator SHALL 返回错误信息，指出该模型目录下未找到任何实验检查点

### Requirement 2: 训练损失曲线绘制

**User Story:** 作为研究人员，我希望能看到训练集、验证集和测试集的损失函数随 epoch 变化的曲线，以便评估模型的收敛情况和过拟合风险。

#### Acceptance Criteria

1. WHEN tensorboard/ 子目录中存在包含标量数据的 TensorBoard 事件文件时，THE Loss_Curve_Plotter SHALL 从事件文件中提取 train/loss、val/loss 和 test/loss 标量数据
2. WHEN tensorboard/ 子目录中不存在 TensorBoard 事件文件但存在 training_log_*.log 文件时，THE Loss_Curve_Plotter SHALL 从时间戳最新的日志文件中解析每个 epoch 的 train/loss、val/loss 和 test/loss 数值
3. THE Loss_Curve_Plotter SHALL 在同一张图中绘制训练集、验证集和测试集的损失曲线，使用不同颜色和图例区分，图例标签分别为 "Train Loss"、"Val Loss" 和 "Test Loss"
4. THE Loss_Curve_Plotter SHALL 在图表中标注 X 轴为 "Epoch"、Y 轴为 "Loss"，X 轴刻度从 1 开始且仅显示整数值
5. IF TensorBoard 事件文件和训练日志文件均不存在，THEN THE Loss_Curve_Plotter SHALL 跳过损失曲线绘制并在报告中注明数据不可用
6. WHEN 训练日志或 TensorBoard 事件中包含 "Early stopping at epoch N" 记录时，THE Loss_Curve_Plotter SHALL 在图中以垂直虚线标注实际停止的 epoch 位置并附带文字标签说明该位置为 EarlyStopping 触发点
7. IF 三条损失曲线中仅部分数据可用（例如缺少 test/loss），THEN THE Loss_Curve_Plotter SHALL 仅绘制可用的损失曲线并在图例中只显示已绘制的曲线
8. THE Loss_Curve_Plotter SHALL 将生成的损失曲线图保存为 PNG 格式文件，分辨率不低于 150 DPI

### Requirement 3: 训练指标曲线绘制

**User Story:** 作为研究人员，我希望能看到各评估指标（如 MAE、RMSE、R2）随 epoch 变化的曲线，以便全面了解模型训练过程中的性能变化趋势。

#### Acceptance Criteria

1. WHEN TensorBoard 事件文件或训练日志文件存在时，THE Loss_Curve_Plotter SHALL 提取 cfg.json 中配置的所有 metrics 指标的训练过程数据
2. THE Loss_Curve_Plotter SHALL 为每个指标生成独立的子图，展示训练集和验证集上该指标随 epoch 的变化
3. THE Loss_Curve_Plotter SHALL 在每个子图中使用不同颜色区分训练集（train）和验证集（val）曲线
4. IF 某个指标的数据不可用，THEN THE Loss_Curve_Plotter SHALL 跳过该指标的子图并在报告中注明

### Requirement 4: 最终评估指标汇总

**User Story:** 作为研究人员，我希望能在报告中看到模型在测试集上的最终评估指标，包括总体指标和各预测步长的指标，以便快速了解模型的最终性能。

#### Acceptance Criteria

1. IF test_metrics.json 存在且包含 "overall" 部分，THEN THE Metric_Summarizer SHALL 读取并以 Markdown 表格展示 overall 部分的 MAE、MSE、RMSE、R2 四项指标，数值保留 4 位小数
2. IF test_metrics.json 包含以 "horizon_" 为前缀的键，THEN THE Metric_Summarizer SHALL 以 Markdown 表格展示各预测步长的指标值，表格列包含模型名称、配置标识以及每个 horizon 对应的 R2 值
3. THE Metric_Summarizer SHALL 以 Markdown 表格格式呈现所有指标数据，表格包含表头行和分隔行，每行对应一个实验结果
4. IF 实验为软测量任务且 measurement_lag ≥ 1，THEN THE Metric_Summarizer SHALL 分别计算估计区间（horizon ≤ measurement_lag）和预测区间（horizon > measurement_lag）的平均 R2，并在表格中展示两个区间的平均 R2 及其差值（Δ = 估计区间 R2 − 预测区间 R2）
5. IF test_metrics.json 不存在或无法解析，THEN THE Metric_Summarizer SHALL 在报告对应位置输出提示信息表明指标数据不可用，且不中断报告生成流程

### Requirement 5: 预测曲线对比绘制

**User Story:** 作为研究人员，我希望能看到模型预测值与真实值的对比曲线，以便直观评估模型的预测精度和趋势跟踪能力。

#### Acceptance Criteria

1. WHEN test_results 目录下存在 prediction.npy 和 targets.npy 时，THE Prediction_Plotter SHALL 加载预测结果和真实值数据，并仅提取 cfg.json 中 target_vars 指定的特征维度进行绘制
2. THE Prediction_Plotter SHALL 针对 cfg.json 中配置的每个 eval_horizons 条目绘制独立的子图，所有子图排列在同一张图中并保存为单个图像文件
3. THE Prediction_Plotter SHALL 在每个子图中同时绘制真实值曲线和预测值曲线，使用两种视觉可区分的颜色并附带图例标注"真实值"和"预测值"
4. THE Prediction_Plotter SHALL 从 test_metrics.json 中读取对应 horizon 的 R2 指标值，并将其标注在对应子图的标题中，格式为 "Horizon {n} (R2={value})"
5. IF test_results 目录不存在或 prediction.npy/targets.npy 文件缺失，THEN THE Prediction_Plotter SHALL 跳过预测曲线绘制并在报告中注明需要在训练配置中设置 save_results=True
6. THE Prediction_Plotter SHALL 支持通过参数控制展示的时间步数量，默认展示 200 个时间步；IF 可用样本数少于指定展示数量，THEN THE Prediction_Plotter SHALL 展示全部可用样本

### Requirement 6: 报告生成与输出

**User Story:** 作为研究人员，我希望所有分析结果能整合为一份结构化的 Markdown 报告，以便存档和分享。

#### Acceptance Criteria

1. THE Report_Generator SHALL 生成一份 Markdown 格式的报告文件（文件名为 `report.md`），按以下顺序包含各节：实验配置摘要、损失曲线图、指标曲线图、最终指标表格、预测对比图
2. THE Report_Generator SHALL 将报告文件保存到实验检查点哈希目录下的 `report/` 子目录中；IF 该子目录不存在，THEN THE Report_Generator SHALL 自动创建该目录
3. THE Report_Generator SHALL 将所有生成的图片文件以 PNG 格式保存到 `report/fig/` 子目录中；IF 该子目录不存在，THEN THE Report_Generator SHALL 自动创建该目录
4. THE Report_Generator SHALL 在报告头部包含以下实验基本信息：生成时间（精确到秒）、模型名称、数据集名称、输入长度、输出长度、测量滞后（measurement_lag）
5. THE Report_Generator SHALL 在报告中使用相对路径 `fig/<filename>.png` 引用图片文件
6. WHEN 报告生成完成时，THE Report_Generator SHALL 在控制台输出报告文件的绝对路径
7. IF 某个分析步骤因数据缺失而被跳过，THEN THE Report_Generator SHALL 在报告对应节中注明该部分数据不可用

### Requirement 7: 命令行接口

**User Story:** 作为研究人员，我希望能通过命令行快速生成单独实验的分析报告，以便集成到自动化工作流中。

#### Acceptance Criteria

1. THE Report_Generator SHALL 提供命令行入口，支持通过 `python -m basicts.analysis.single --ckpt_dir <path>` 调用，其中 `--ckpt_dir` 为必选参数
2. THE Report_Generator SHALL 支持 `--output_dir` 参数指定报告输出目录，默认为检查点哈希目录下的 `report/`
3. THE Report_Generator SHALL 支持 `--num_samples` 参数控制预测曲线展示的样本数量，默认值为 200，接受的值范围为 1 到 10000 的正整数
4. IF 用户提供的 `--ckpt_dir` 路径不存在或不是一个目录，THEN THE Report_Generator SHALL 输出错误信息指明该路径不存在或非目录，并以非零退出码退出
5. IF 用户提供的 `--num_samples` 值不在 1 到 10000 范围内，THEN THE Report_Generator SHALL 输出错误信息指明有效范围，并以非零退出码退出
6. WHEN 用户使用 `--help` 参数调用时，THE Report_Generator SHALL 显示所有支持参数的说明文本，包括各参数的默认值和用途描述

### Requirement 8: 编程接口

**User Story:** 作为研究人员，我希望能通过 Python API 调用报告生成功能，以便在 Jupyter Notebook 或自定义脚本中灵活使用。

#### Acceptance Criteria

1. THE Report_Generator SHALL 提供 `SingleExperimentReporter` 类，可通过 `from basicts.analysis import SingleExperimentReporter` 导入
2. WHEN 用户实例化 `SingleExperimentReporter(ckpt_dir)` 时，THE Report_Generator SHALL 自动完成检查点验证和配置解析
3. IF 实例化时提供的 ckpt_dir 路径无效或缺少必需文件（cfg.json、test_metrics.json），THEN THE Report_Generator SHALL 抛出异常，包含错误信息指明缺失的文件名称和期望路径
4. THE Report_Generator SHALL 提供 `generate_report()` 方法一键生成完整报告，并返回生成的报告文件路径（字符串类型）
5. THE Report_Generator SHALL 提供独立方法 `plot_loss_curves()`、`plot_metric_curves()`、`plot_predictions()`，每个方法返回 matplotlib Figure 对象以支持 Jupyter Notebook 中的交互式显示
6. THE Report_Generator SHALL 提供 `summarize_metrics()` 方法，返回包含测试集评估指标的字典（dict），包含 overall 指标和各 horizon 指标
