# 乙烯精馏塔数据集接入与基准实验

> 创建日期: 2026-05-20
> 状态: 数据集接入完成，基准实验已跑通

---

## 数据集概况

| 属性 | 值 |
|------|------|
| 数据集名称 | EthyDistillation |
| 来源 | 工业乙烯精馏塔过程数据 |
| 原始文件 | `datasets/raw_data/ethy/ethy.xlsx` |
| 样本数 | 10080 |
| 变量数 | 38 |
| 时间范围 | 2016-10-13 13:00 ~ 2016-10-20 12:59 |
| 采样频率 | 1 分钟 |
| 缺失值 | 无 |
| 目标变量 | 塔顶乙烷浓度 (index=20) |

## 变量列表

```
进料压力、干燥器入口温度、进料量、塔顶压力、塔顶温度、
进料乙烯浓度、进料乙烷浓度、塔压差、9#温度、142#温度、
147#温度、162#温度、168#温度、塔釜采出温度、中沸器出口物料温度、
中沸器液位、中沸器流量、到DA410量、裂解气入口温度、裂解气出口温度、
【塔顶乙烷浓度】(目标)、塔顶甲烷浓度、回流量、采出量、不凝气量、
回流罐液位、冷凝器1液位、冷凝器2液位、冷凝器3液位、冷凝后温度、
灵敏板乙烯浓度、丙烯冷剂流量、丙烯入口温度、凝液罐液位、
再沸器出口物料温度、塔釜液位、塔釜采出量、塔釜乙烯损失
```

## 数据预处理

**脚本**: `scripts/data_preparation/EthyDistillation/generate_training_data.py`

**处理流程**:
1. 读取 xlsx 文件，解析 datetime 索引
2. 提取 38 个数值变量
3. 生成时间特征: time_of_day + day_of_week (2维)
4. 按 70%/10%/20% 划分 train/val/test
5. 保存为 npy 格式 + meta.json

**输出**:
```
datasets/EthyDistillation/
├── meta.json              (含变量名、目标索引等元信息)
├── train_data.npy         (7056, 38) float32
├── val_data.npy           (1008, 38) float32
├── test_data.npy          (2016, 38) float32
├── train_timestamps.npy   (7056, 2)
├── val_timestamps.npy     (1008, 2)
└── test_timestamps.npy    (2016, 2)
```

## 基准实验配置

**目录**: `experiments/EthyDistillation_benchmark/`

| 参数 | 值 |
|------|------|
| input_len | 96 |
| output_len | 12 |
| measurement_lag | 6 |
| target_vars | 20 (塔顶乙烷浓度) |
| eval_horizons | [1, 6, 12] |
| batch_size | 64 |
| num_epochs | 100 |
| hidden_size | 256 |
| save_results | True |

**实验矩阵**: 4 模型 × 2 配置 = 8 组

## 基准实验结果

### 结果汇总 (按 R² 降序)

| 模型 | 配置 | R2 | RMSE | h1 R2 | h6 R2 | h12 R2 |
|------|------|-----|------|-------|-------|--------|
| TimeXer | incl_target | -2.44 | 6.82 | -1.36 | -2.69 | -3.56 |
| PatchTST | incl_target | -6.18 | 9.37 | -4.37 | -6.41 | -6.40 |
| iTransformer | incl_target | -12.91 | 15.82 | -12.32 | -14.08 | -12.86 |
| DLinear | incl_target | -79.14 | 57.37 | -82.95 | -79.89 | -77.48 |
| DLinear | excl_target | -88.64 | 74.54 | -91.30 | -89.08 | -87.84 |
| TimeXer | excl_target | -93.68 | 110.31 | -94.32 | -93.48 | -93.21 |
| iTransformer | excl_target | -97.70 | 118.96 | -98.08 | -97.68 | -97.64 |
| PatchTST | excl_target | -98.01 | 116.57 | -97.86 | -98.06 | -98.12 |

### 关键发现

1. **所有模型 R² 均为负值**: 当前配置下预测能力不足，需要后续调优
2. **incl_target 显著优于 excl_target**: 目标自相关性仍是主要信号源
3. **模型排名**: TimeXer > PatchTST > iTransformer > DLinear (incl_target 模式)
4. **数据集验证通过**: 框架能正常加载和训练，pipeline 无报错

### 后续调优方向

- 增大 input_len (192/288)，覆盖更长历史窗口
- 尝试不同 measurement_lag (3/6/12/24)
- 检查目标变量数据分布，处理可能的异常值
- 调整学习率和训练策略
- 探索变量筛选（利用注意力权重分析）

## 关于通用数据预处理的决策

经评估，决定**不开发通用预处理模块**，保持每个数据集独立脚本的方式：

- 工业数据格式差异大，通用模块参数过多反而不清晰
- 与 BasicTS 框架现有设计一致（每个数据集独立 generate_training_data.py）
- 输出格式统一（npy + meta.json）已是最好的"通用性"
- 新数据集只需复制现有脚本修改即可
