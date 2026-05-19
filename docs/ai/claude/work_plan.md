# BasicTS 软测量功能 — 后续工作计划

> 基于 2026-05-17 首次提交的软测量功能扩展，规划后续迭代方向。
> 本文档用于快速恢复工作上下文。

---

## 当前状态（v1.0 已提交）

### 已完成

- [x] BasicTSSoftSensorConfig / Dataset / TaskFlow 三件套
- [x] 通道对齐归一化（scaler stats 按 input_vars/target_vars 切片）
- [x] 负索引解析（target_vars=-1 → 正索引）
- [x] input_vars 延迟解析（从 dataset 获取）
- [x] R² 全局计算修复
- [x] 纯估计 demo（lag=1, output=1）
- [x] 预测性软测量 demo（lag=6, output=12）
- [x] spec 文档

### 已知问题

- val R² 在训练过程中仍有极端负值（batch 级累积的固有问题）
- DLinear 在 ETTh1 测试集上 R² 为负（模型能力不足 + 分布偏移）
- per-horizon 的 RMSE 递减趋势（h1 > h12）可能反映 DLinear 的局限性

---

## Phase 1：短期优化（提升可用性） ✅ 已完成

### Task 1.1 — 修复 val R² 的 batch 级累积问题 ✅

**目标**：让训练过程中的 R² 显示值与最终评估一致

**方案**：
- 为 R² 实现一个 epoch 级 meter（累积所有 batch 的 ss_res 和 ss_tot，epoch 结束时计算）
- 或者在 runner 中对 R² 特殊处理：不用加权平均，而是累积 numerator/denominator

**涉及文件**：
- `src/basicts/runners/basicts_runner.py`（meter 注册和更新逻辑）
- 可能需要新增一个 `AccumulativeR2Meter` 类

**完成情况**：已修复，保留全局 R² 计算方式。

---

### Task 1.2 — 验证更强模型 ✅

**目标**：确认框架在非线性模型上正常工作，获得正 R² 的 baseline

**计划**：
- [x] PatchTST + ETTh1 软测量配置
- [x] iTransformer + ETTh1 软测量配置
- [x] TimeXer + ETTh1 软测量配置
- [x] 对比 exclude_target_from_input=True/False

**完成情况**：
- 实验配置文件位于 `experiments/TSFsoftsensor_test/`
- 4 个模型 × 2 种配置 = 8 组实验全部完成
- PatchTST 在 incl_target 模式下 h1 R²=0.705，验证框架兼容性

---

### Task 1.3 — exclude_target_from_input 对比实验 ✅

**目标**：量化 target 历史信息对软测量的贡献

**计划**：
- 固定 output_len=12, lag=6
- 对比 exclude=True（6 通道输入）vs exclude=False（7 通道输入）
- 用 DLinear / PatchTST / iTransformer / TimeXer 四个模型

**完成情况**：
- 实验结果详见 `docs/ai/GLM_5/soft_sensor_benchmark_report.md`
- 关键结论：Transformer 系列模型在 incl_target 模式下 R² 提升 30+，DLinear 几乎不受影响
- PatchTST (incl_target) 表现最好：h1 R²=0.705；DLinear (excl_target) 在纯软测量场景最稳健

---

## Phase 2：功能完善

### Task 2.1 — 实验分析与可视化模块（核心模块） ✅ 已完成

**背景**：在 ETTh1 / Debutanizer 两轮多模型实验后发现，单纯依赖 `test_metrics.json` 的数值表格难以直观对比。框架现有可视化能力仅限 TensorBoard 训练曲线，缺少多实验综合分析能力。

**目标**：在框架核心中新增 `basicts.analysis` 模块，作为通用的实验后处理与可视化工具。

**完成情况**：

模块已实现并验证通过，位于 `src/basicts/analysis/`：

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

**已实现功能**：
- [x] 自动扫描 checkpoints 目录，解析 cfg.json + test_metrics.json
- [x] 模型名自动缩短（iTransformerForForecasting → iTransformer）
- [x] 结果按 R² 降序排列
- [x] 任务类型自动识别（软测量/预测/分类）
- [x] Overall 指标对比柱状图
- [x] Horizon 分组柱状图
- [x] 模型 × Horizon 热力图
- [x] 估计区/预测区分区对比（软测量专属）
- [x] 多步预测曲线（模型 × horizon 网格图）
- [x] Memmap 格式兼容（BasicTS 特有的 prediction.npy 加载）
- [x] Markdown 报告自动生成
- [x] CLI 入口 (`python -m basicts.analysis -e <dir>`)
- [x] 注意力可视化预留接口

**使用方式**：
```bash
# CLI
python -m basicts.analysis -e checkpoints/Debutanizer_benchmark

# Python API
from basicts.analysis import ExperimentAnalyzer
analyzer = ExperimentAnalyzer("checkpoints/Debutanizer_benchmark")
analyzer.run_all()
```

**依赖**：matplotlib, seaborn（可选，未安装时仅图表不可用）

**技术文档**：`docs/ai/claude/analysis_module_spec.md`

---

### Task 2.2 — 数据预处理脚本

**目标**：用户可以从 CSV 快速生成 BasicTS 格式数据

**功能**：
```bash
python scripts/prepare_softsensor_data.py \
    --input data.csv \
    --output datasets/MyProcess \
    --train_ratio 0.7 \
    --val_ratio 0.1 \
    --test_ratio 0.2 \
    --timestamp_col time \
    --target_cols quality_var1,quality_var2
```

生成：
```
datasets/MyProcess/
├── meta.json
├── train_data.npy
├── train_timestamps.npy
├── val_data.npy
├── val_timestamps.npy
├── test_data.npy
└── test_timestamps.npy
```

**涉及文件**：
- `scripts/prepare_softsensor_data.py`（新增）

**注**：Debutanizer 数据集已通过 `scripts/data_preparation/Debutanizer/generate_training_data.py` 完成接入，可作为通用脚本的参考实现。

**预计工作量**：3-4 小时

---

### Task 2.3 — 单元测试

**目标**：确保核心逻辑的正确性，防止回归

**测试用例**：
- [ ] `test_ss_dataset.py`
  - 切片正确性（input_vars/target_vars）
  - 负索引解析
  - exclude_target_from_input=True/False
  - __len__ 计算
- [ ] `test_ss_taskflow.py`
  - transform/inverse_transform 通道对齐
  - postprocess 通道抽取
  - input_vars=None 时从 dataset 获取
- [ ] `test_r_square.py`
  - output_len=1 / output_len>1
  - 有 mask / 无 mask
  - 全零方差边界情况
- [ ] `test_ss_config.py`
  - __post_init__ 验证逻辑
  - 负索引转换
  - measurement_lag 约束
- [ ] `test_analysis.py` (新增)
  - collector 解析 cfg.json
  - visualizer 在最小数据集上的基本功能
  - reporter 输出格式校验

**涉及文件**：
- `tests/test_softsensor/`（新增目录）
- `tests/test_analysis/`（新增目录）

**预计工作量**：4-5 小时

---

## Phase 3：长期研究方向

### Task 3.1 — 专用软测量模型

**思路**：
- 显式建模过程变量 → 质量变量的因果/相关结构
- 利用 measurement_lag 做时间对齐注意力（process vars at t align with quality at t-lag）
- 可选：融入过程机理先验（如已知的物理方程约束）

**参考**：
- Attention-based soft sensor (工业 AI 文献)
- Graph-based soft sensor（变量间拓扑关系）
- Physics-informed neural network

---

### Task 3.2 — 在线增量学习

**思路**：
- 新增 `online_finetune` 模式
- 新数据到来时用小 lr 微调
- 监控性能退化（concept drift detection）
- 自动触发重训练

---

### Task 3.3 — 多数据集 benchmark

**目标**：在多个工业数据集上建立软测量 baseline

**候选数据集**：
- [x] Debutanizer Column（经典软测量 benchmark）— 已接入并完成 4 模型 × 2 配置基准
- [ ] Sulfur Recovery Unit
- [ ] Tennessee Eastman Process
- [ ] 自有工业数据

**已完成**：
- 数据预处理脚本：`scripts/data_preparation/Debutanizer/`
- 实验配置：`experiments/Debutanizer_benchmark/`
- 实验结果：`checkpoints/Debutanizer_benchmark/`

---

## 快速恢复指南

### 项目结构（软测量相关）

```
src/basicts/
├── configs/ss_config.py              ← 配置
├── data/ss_dataset.py                ← 数据集
├── runners/taskflow/softsensor_taskflow.py  ← 任务流
├── metrics/r_square.py               ← R² 指标（已修复）
├── analysis/                         ← (Phase 2.1 待实现) 实验分析与可视化
examples/softsensor/
├── softsensor_dlinear_demo.py        ← 纯估计 demo
├── softsensor_dlinear_predictive_demo.py  ← 预测性 demo
experiments/
├── TSFsoftsensor_test/               ← ETTh1 多模型基准
├── Debutanizer_benchmark/            ← Debutanizer 多模型基准
scripts/data_preparation/
├── Debutanizer/                      ← 工业数据预处理参考实现
docs/ai/claude/
├── soft_sensor_spec.md               ← 功能规格
├── work_plan.md                      ← 本文档
docs/ai/GLM_5/
├── soft_sensor_benchmark_report.md   ← ETTh1 基准实验报告
```

### 关键设计决策回顾

1. **数据切片**：target 紧接 input 之后（与 forecasting 一致），前 lag 步是估计，后面是预测
2. **通道对齐**：taskflow 从全通道 scaler stats 中按 vars 索引切片
3. **input_vars 延迟解析**：config 中为 None，dataset 加载数据后推断，taskflow 从 dataset 获取
4. **R² 全局计算**：跨 batch 展平，避免 per-sample 方差过小的问题

### 运行实验

```bash
# 纯估计
python examples/softsensor/softsensor_dlinear_demo.py

# 预测性软测量
python examples/softsensor/softsensor_dlinear_predictive_demo.py
```

### 查看结果

```bash
# 训练日志
checkpoints/DLinear/ETTh1_100_96_1_lag1/<hash>/training_log_*.log
checkpoints/DLinear/ETTh1_100_96_12_lag6/<hash>/training_log_*.log

# 测试指标
checkpoints/DLinear/ETTh1_100_96_12_lag6/<hash>/test_metrics.json
```
