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

### Task 2.2 — 数据预处理脚本 ✅ 已完成（参考实现方式）

**目标**：用户可以从原始数据快速生成 BasicTS 格式数据

**完成情况**：

经评估，决定不开发通用预处理模块，保持每个数据集独立脚本的方式（与 BasicTS 框架设计一致）。

已完成的参考实现：
- [x] `scripts/data_preparation/Debutanizer/generate_training_data.py` — txt 无时间戳场景
- [x] `scripts/data_preparation/EthyDistillation/generate_training_data.py` — xlsx 有时间戳场景

输出格式统一为 npy + meta.json，新数据集只需复制现有脚本修改即可。

**技术文档**：`docs/ai/claude/ethy_distillation_spec.md`

---

### Task 2.3 — 单元测试 ✅ 已完成

**状态**：已为 Phase 3 新增模块编写单元测试。

**测试文件**：
- `tests/basicts_test/test_models_pls_svr_lstm.py` — 23 个模型级单元测试（PLS/SVR/LSTM 的实例化、forward、fit、序列化、梯度流等）
- `tests/basicts_test/test_non_gradient_runner.py` — 3 个集成测试（NonGradientRunner + PLS/SVR 完整 pipeline）

**运行方式**：
```bash
python -m pytest tests/basicts_test/ -v
```

---

## Phase 3：传统模型 Baseline 与研究方向

### Task 3.0 — 传统软测量 Baseline 模型 ✅ 已完成

**背景**：Phase 1-2 的实验中 Transformer 系列模型在工业数据集上 R² 均为负值，无法确认是模型问题还是数据/配置问题。需要先用传统模型建立 baseline，确认数据集的可预测性，再评估 Transformer 的增益。

**目标**：在 BasicTS 框架内实现 4 个传统 baseline 模型，构成能力递进的对比体系：

| 顺序 | 模型 | 定位 | 回答的问题 | 输入方式 |
|------|------|------|-----------|----------|
| 1 | **NLinear**（单变量） | 自回归参考 | 目标变量自身历史能预测多少？ | 仅目标历史序列 (单通道) |
| 2 | **PLS** (偏最小二乘) | 线性多变量 | 过程变量与质量变量的线性相关性有多强？ | 多变量当前时刻 (excl_target) |
| 3 | **SVR** (支持向量回归) | 非线性静态 | 非线性建模带来多少增益？ | 多变量当前时刻 (excl_target) |
| 4 | **LSTM** | 非线性时序 | 加入时序依赖还能提升多少？ | 多变量历史序列 |

**实验设计**：

四个模型构成一条清晰的能力递进链：
- NLinear（单变量）= **自回归参考**（纯靠目标自身历史，不用过程变量）
- PLS = **线性下界**（纯线性、无时序依赖，多变量）
- SVR = **非线性静态**（非线性、无时序依赖，多变量）
- LSTM = **经典非线性时序**（多变量 + 时序 + 非线性）

**诊断逻辑**：
- PLS R² > 0 → 数据集本身可预测，过程变量有信息量
- SVR > PLS → 非线性建模有增益
- LSTM > SVR → 时序依赖有增益
- Transformer < LSTM → 问题在数据/配置层面，不是框架问题

**架构设计**：

非梯度优化模型（PLS、SVR）不强行适配 BasicTS 的梯度训练 pipeline，而是：
1. 新增 `NonGradientRunner`（继承 `BasicTSRunner`），重写 `train()` 为一次性 fit
2. 模型仍为 `nn.Module`，内部包装 sklearn 模型
3. 评估流程（`_eval_loop`、metrics、结果保存、per-horizon 评估）完全复用
4. 实验配置风格与现有模型一致，统一管理

```
src/basicts/runners/
├── basicts_runner.py          ← 现有，不改动
├── non_gradient_runner.py     ← 新增，处理非梯度模型的训练/评估
```

**实现计划**（按开发顺序）：

#### 3.0.1 — NonGradientRunner 基础设施 + PLS baseline ✅

- 新增 `src/basicts/runners/non_gradient_runner.py`
  - 重写 `train()`：加载全部训练数据 → 调用 `model.fit(X, y)` → 直接进入 test
  - 重写 `_init_train()`：跳过 optimizer/lr_scheduler 创建
  - 复用 `_eval_loop()` / `_init_test()` / metrics 计算 / 结果保存
  - checkpoint 保存：用 pickle 保存 sklearn 模型
- 新增 `src/basicts/models/PLS/`
  - `nn.Module` 包装 sklearn `PLSRegression`
  - `fit(X, y)`：展平 [B, T, C] → [B, T*C]，调用 sklearn fit
  - `forward(inputs)`：展平 → predict → reshape 为 [B, output_len, 1]

#### 3.0.2 — SVR baseline ✅

- 新增 `src/basicts/models/SVR/`
- `nn.Module` 包装 sklearn `SVR`（RBF 核）
- 输入方式与 PLS 一致（展平多变量窗口）
- 多步输出策略：MultiOutputRegressor（n_jobs=1 避免 Windows 多进程冲突）
- 复用 NonGradientRunner

#### 3.0.3 — LSTM baseline ✅

- 新增 `src/basicts/models/LSTM/`
- 标准 LSTM encoder → Linear head 结构
- 支持多变量输入、多步输出、双向模式
- 走正常梯度优化 pipeline（BasicTSRunner）

#### 3.0.4 — 整体实验配置 ✅

- Debutanizer benchmark 实验配置已完成：
  - `nlinear_univariate.py` — NLinear 单变量自回归
  - `pls.py` — PLS（NonGradientRunner）
  - `svr.py` — SVR（NonGradientRunner）
  - `lstm_excl_target.py` / `lstm_incl_target.py` — LSTM
- `run.sh` 已更新（已有模型注释掉，只跑新 baseline）
- Debutanizer 实验已全部跑通，analysis 报告已生成

**实验结果（Debutanizer，Overall R²）**：

| 模型 | 配置 | R² |
|------|------|-----|
| NLinear | incl_target (单变量) | **0.8739** |
| TimeXer | incl_target | 0.3709 |
| PatchTST | incl_target | 0.1437 |
| PLS | excl_target | **0.0980** |
| iTransformer | incl_target | -0.3169 |
| LSTM | incl_target | -0.3640 |
| SVR | excl_target | -0.5077 |
| LSTM | excl_target | -0.5834 |

**关键发现**：
- PLS 在 h1 R²=0.44，确认数据集可预测，过程变量有信息量
- NLinear 单变量自回归 R²=0.87，说明目标变量自身历史有极强预测力
- SVR h1 R²=0.11，非线性增益有限（可能因为展平窗口维度过高）
- LSTM incl_target 的 h1 R²=0.51，但 overall 为负（远期预测差）

**验证数据集**：
- 先在 Debutanizer 上验证（小数据集，快速迭代）
- 确认 R² > 0 后迁移到 EthyDistillation

**预期结果**：
- 建立可信的 baseline R² 水平
- 明确 Transformer 系列需要达到的最低标准
- 为后续模型调优提供方向（是否需要更长 input_len、不同 lag 等）

---

### Task 3.3 — 多数据集 benchmark

**目标**：在多个工业数据集上建立软测量 baseline

**候选数据集**：
- [x] Debutanizer Column（经典软测量 benchmark）— 已接入并完成 4 模型 × 2 配置基准
- [x] Ethylene Distillation Column（乙烯精馏塔）— 已接入并完成基准实验
- [ ] Sulfur Recovery Unit
- [ ] Tennessee Eastman Process
- [ ] 自有工业数据

**已完成**：
- 数据预处理脚本：`scripts/data_preparation/Debutanizer/` + `scripts/data_preparation/EthyDistillation/`
- 实验配置：`experiments/Debutanizer_benchmark/` + `experiments/EthyDistillation_benchmark/`
- 实验结果：`checkpoints/Debutanizer_benchmark/` + `checkpoints/EthyDistillation_benchmark/`

---

## 快速恢复指南

### 当前进度

- **Phase 1**: ✅ 全部完成
- **Phase 2**: ✅ Task 2.1 + 2.2 + 2.3 全部完成
- **Phase 3**: ✅ Task 3.0（传统 Baseline）已完成，Debutanizer 实验已跑通

### 下一步行动

1. 在 EthyDistillation 上跑传统 baseline 实验
2. 综合分析 Debutanizer + EthyDistillation 结果，确定 Transformer 调优方向
3. 根据实验结论决定是否需要调整 input_len / lag / 模型超参

### 项目结构（软测量相关）

```
src/basicts/
├── configs/ss_config.py              ← 软测量配置
├── data/ss_dataset.py                ← 软测量数据集
├── runners/taskflow/softsensor_taskflow.py  ← 软测量任务流
├── runners/non_gradient_runner.py    ← (待实现) 非梯度模型 runner
├── metrics/r_square.py               ← R² 指标
├── analysis/                         ← 实验分析与可视化模块
│   ├── analyzer.py                   ← 统一接口
│   ├── collector.py                  ← 结果收集
│   ├── visualizer.py                 ← 图表生成
│   ├── reporter.py                   ← 报告生成
│   └── attention.py                  ← (预留) 注意力可视化
├── models/
│   ├── DLinear/                      ← 已有
│   ├── NLinear/                      ← 已有（Task 3.0.1 单变量配置复用）
│   ├── PatchTST/                     ← 已有
│   ├── iTransformer/                 ← 已有
│   ├── TimeXer/                      ← 已有
│   ├── LSTM/                         ← (待实现) Task 3.0.5
│   ├── PLS/                          ← (待实现) Task 3.0.3
│   └── SVR/                          ← (待实现) Task 3.0.4
experiments/
├── TSFsoftsensor_test/               ← ETTh1 多模型基准
├── Debutanizer_benchmark/            ← Debutanizer 多模型基准
├── EthyDistillation_benchmark/       ← 乙烯精馏塔多模型基准
scripts/data_preparation/
├── Debutanizer/                      ← 脱丁烷塔数据预处理
├── EthyDistillation/                 ← 乙烯精馏塔数据预处理
docs/ai/claude/
├── soft_sensor_spec.md               ← 功能规格
├── work_plan.md                      ← 本文档
├── analysis_module_spec.md           ← 分析模块技术文档
├── ethy_distillation_spec.md         ← 乙烯精馏塔数据集文档
docs/ai/GLM_5/
├── soft_sensor_benchmark_report.md   ← ETTh1 基准实验报告
```

### 关键设计决策回顾

1. **数据切片**：target 紧接 input 之后（与 forecasting 一致），前 lag 步是估计，后面是预测
2. **通道对齐**：taskflow 从全通道 scaler stats 中按 vars 索引切片
3. **input_vars 延迟解析**：config 中为 None，dataset 加载数据后推断，taskflow 从 dataset 获取
4. **R² 全局计算**：跨 batch 展平，避免 per-sample 方差过小的问题
5. **checkpoint 路径控制**：通过 `ckpt_save_dir` 按实验名分类
6. **数据预处理策略**：每个数据集独立脚本，输出格式统一（npy + meta.json）
7. **非梯度模型集成策略**：不强行适配梯度 pipeline，通过 NonGradientRunner 子类实现一次性 fit + 复用评估流程，模型仍为 nn.Module 以统一管理

### 运行实验

```bash
# 数据预处理
python scripts/data_preparation/Debutanizer/generate_training_data.py
python scripts/data_preparation/EthyDistillation/generate_training_data.py

# 跑实验
bash experiments/Debutanizer_benchmark/run.sh
bash experiments/EthyDistillation_benchmark/run.sh

# 分析结果
python -m basicts.analysis -e checkpoints/Debutanizer_benchmark
python -m basicts.analysis -e checkpoints/EthyDistillation_benchmark
```

### 已有实验结果概览

| 数据集 | 最佳模型 | 最佳 R² | 配置 |
|--------|----------|---------|------|
| ETTh1 | TimeXer | 0.424 | incl_target, lag=6, out=12 |
| Debutanizer | NLinear | 0.874 | incl_target (单变量), lag=3, out=6 |
| Debutanizer | PLS | 0.098 | excl_target, lag=3, out=6 |
| EthyDistillation | TimeXer | -2.44 | incl_target, lag=6, out=12 |

**结论**：
- Debutanizer 数据集可预测性已确认（PLS h1 R²=0.44，NLinear overall R²=0.87）
- 目标变量自身历史是最强预测信号（NLinear 单变量远超所有多变量模型）
- EthyDistillation 仍需传统 baseline 验证
