# BasicTS 分析模块 (basicts.analysis) 技术文档

> 创建日期: 2026-05-18
> 状态: Step 1 已完成（基础图表 + 报告生成）

---

## 概述

`basicts.analysis` 是 BasicTS 框架的实验后处理与可视化模块，用于多模型基准实验的结果汇总、图表生成和报告输出。该模块作为框架核心组件集成，支持所有任务类型（软测量、预测、分类等）。

## 模块结构

```
src/basicts/analysis/
├── __init__.py       ← 模块入口，导出 ExperimentAnalyzer
├── __main__.py       ← CLI 入口 (python -m basicts.analysis)
├── analyzer.py       ← 统一接口，组合 collector/visualizer/reporter
├── collector.py      ← 扫描 checkpoints，解析 cfg.json + test_metrics.json
├── visualizer.py     ← 图表生成（柱状图、热力图、分区对比、预测曲线）
├── reporter.py       ← markdown 报告生成
└── attention.py      ← (预留) 注意力权重可视化接口
```

## 使用方式

### Python API

```python
from basicts.analysis import ExperimentAnalyzer

# 一键分析
analyzer = ExperimentAnalyzer("checkpoints/Debutanizer_benchmark")
analyzer.run_all()

# 或分步调用
analyzer.collect()
analyzer.plot_overall_comparison()
analyzer.plot_horizon_comparison()
analyzer.plot_heatmap()
analyzer.plot_zone_comparison()       # 软测量专属
analyzer.plot_predictions(top_k=3)    # 需要 save_results=True
analyzer.generate_report()
```

### CLI

```bash
python -m basicts.analysis --experiment_dir checkpoints/Debutanizer_benchmark
python -m basicts.analysis -e checkpoints/Debutanizer_benchmark -o custom_output_dir
python -m basicts.analysis -e checkpoints/Debutanizer_benchmark -m R2 RMSE MAE
```

## 依赖

- **必需**: numpy, pandas (已在 BasicTS 环境中)
- **可选**: matplotlib, seaborn (图表生成)

```bash
pip install matplotlib seaborn
```

未安装 matplotlib 时模块仍可正常工作（collector + reporter），仅图表生成功能不可用。

## 核心组件

### 1. ResultCollector (collector.py)

**功能**: 扫描实验目录，解析 cfg.json 和 test_metrics.json，组织为结构化数据。

**目录结构要求**:
```
experiment_dir/
├── ModelA/
│   ├── hash1/
│   │   ├── cfg.json
│   │   ├── test_metrics.json
│   │   └── test_results/        (可选, save_results=True)
│   │       ├── inputs.npy
│   │       ├── prediction.npy
│   │       └── targets.npy
│   └── hash2/
│       └── ...
└── ModelB/
    └── ...
```

**关键设计**:
- 通过 cfg.json 中的 `model.name` 自动识别模型类型
- 内置短名映射: `iTransformerForForecasting` → `iTransformer`, `PatchTSTForForecasting` → `PatchTST`
- 通过 `taskflow.name` 自动识别任务类型（软测量/预测/分类）
- 处理 cfg.json 中的 NaN 值（BasicTS 特有）
- 结果按 Overall R² 降序排列

### 2. ExperimentVisualizer (visualizer.py)

**功能**: 生成多种对比图表。

| 方法 | 输出 | 说明 |
|------|------|------|
| `plot_overall_comparison()` | overall_comparison.png | 模型×配置的指标柱状图 |
| `plot_horizon_comparison()` | horizon_comparison.png | 各 horizon 分组柱状图 |
| `plot_heatmap()` | heatmap.png | 模型×horizon 热力图 |
| `plot_zone_comparison()` | zone_comparison.png | 估计区/预测区对比（软测量） |
| `plot_predictions()` | predictions.png | 模型×horizon 网格预测曲线 |

**预测曲线加载逻辑**:
1. 先尝试 `np.load(allow_pickle=True)` — 标准 npy 格式
2. 失败后回退到 `np.memmap` 读取 — 适配 BasicTS 的 memmap 保存方式
3. 通过文件大小和 output_len 推断 shape

**多步预测曲线设计**:
- 布局: top-k 模型（行）× eval_horizons（列）的网格图
- 自动从 cfg.json 读取 eval_horizons 确定绘制哪些步长
- 每个子图标注对应 horizon 的 R² 值
- 以 Debutanizer (top_k=3, eval_horizons=[1,3,6]) 为例: 3×3 = 9 个子图
- 不需要重新训练，prediction.npy 已包含所有步长的完整输出

### 3. ExperimentReporter (reporter.py)

**功能**: 生成 markdown 格式的实验报告。

**报告内容**:
- 实验配置信息（自动从 cfg.json 提取）
- 结果汇总表格
- 最佳模型识别
- Horizon 分析表格
- 估计区/预测区对比表格（软测量）
- 图表引用（相对路径）

### 4. AttentionVisualizer (attention.py) — 预留接口

**状态**: 接口已定义，实现待后续开发。

**计划功能**:
- 单样本注意力热力图
- 跨变量注意力模式（iTransformer）
- 时序注意力模式（PatchTST）
- 层间注意力对比

**使用前提**: 模型配置中设置 `output_attentions=True`

## 实验配置要点

### 启用预测结果保存

在实验配置中添加 `save_results=True`:

```python
BasicTSLauncher.launch_training(BasicTSSoftSensorConfig(
    ...
    save_results=True,    # 保存 prediction.npy / targets.npy
    ...
))
```

**存储开销估算**（以 Debutanizer 为例）:
- 测试集 480 样本, output_len=6, float32
- 单实验: ~452 KB (inputs + prediction + targets)
- 8 组实验: ~3.6 MB

**注意**: `save_results` 是 training-independent key，不影响 md5 hash，因此添加后重跑会写入同一目录。

### Checkpoint 路径控制

通过 `ckpt_save_dir` 参数按实验名分类:

```python
BasicTSSoftSensorConfig(
    ...
    ckpt_save_dir="checkpoints/Debutanizer_benchmark/TimeXer",
    ...
)
```

路径结构: `{ckpt_save_dir}/{md5_hash}/`

## 已验证的实验

### Debutanizer Benchmark

- 4 模型 (DLinear, PatchTST, iTransformer, TimeXer) × 2 配置 (excl/incl target)
- 预测性软测量: lag=3, output=6, eval_horizons=[1, 3, 6]
- 分析输出: `checkpoints/Debutanizer_benchmark/analysis/`

### ETTh1 TSFsoftsensor_test

- 4 模型 × 2 配置
- 预测性软测量: lag=6, output=12, eval_horizons=[1, 6, 12]
- 分析报告: `docs/ai/GLM_5/soft_sensor_benchmark_report.md`

## 后续扩展方向

1. **注意力可视化** (attention.py): 实现 Transformer 注意力权重热力图
2. **训练曲线对比**: 从 tensorboard 日志中提取 loss/metric 曲线并叠加对比
3. **统计显著性检验**: 多次运行的均值±标准差，t-test 对比
4. **交互式报告**: 生成 HTML 报告（plotly 交互图表）
5. **自动调参建议**: 基于结果分析给出超参数调整建议
