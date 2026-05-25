# 需求文档

## 简介

本文档定义了一个名为 **PatchXformer** 的新型 Transformer 模型的需求，该模型专为 BasicTS 框架中的软测量任务设计。模型结合了双轴注意力机制（来自 PatchTST 的时间维度 patch 注意力和来自 iTransformer 的变量维度注意力），以及软测量特有的机制，包括非对称变量角色、滞后感知位置编码和变量重要性门控。

与 TimeXer 仅对目标变量进行 patch 处理并使用单个全局 token 进行跨变量交互不同，PatchXformer 对所有变量进行 patch 处理，并在每个 patch 位置执行变量注意力，实现更丰富的跨变量信息交换。

## 术语表

- **PatchXformer**: 用于软测量任务的新型 Transformer 模型，结合 Patch 级双轴注意力与软测量特有机制。
- **Backbone（骨干网络）**: 核心神经网络架构（PatchXformerBackbone），执行特征提取，不包含任务特定的输出头。
- **ForForecasting（预测包装器）**: 任务特定的包装器（PatchXformerForForecasting），在 Backbone 之上添加预测头。
- **Temporal_Attention（时间注意力）**: 沿 patch（时间）维度在每个变量内独立应用的多头自注意力，捕获时间模式。
- **Variate_Attention（变量注意力）**: 在每个 patch 位置跨变量应用的多头注意力，捕获变量间依赖关系。
- **Patch_Embedding（Patch嵌入）**: 将每个变量的时间序列分割为固定长度的 patch 并投影到隐藏维度的模块。
- **Dual_Axis_Block（双轴块）**: 包含一个 Temporal_Attention 层和一个 Variate_Attention 层的编码器块，带残差连接和层归一化。
- **Variable_Importance_Gate（变量重要性门控）**: 可学习的门控机制，产生每个变量的重要性权重以调节变量在 Variate_Attention 中的贡献。
- **Lag_Aware_Position_Encoding（滞后感知位置编码）**: 融入 measurement_lag 信息的位置编码，区分估计区域和预测区域的 patch。
- **Process_Variables（过程变量）**: 用于预测质量变量的输入传感器测量值（如温度、压力、流量）。
- **Target_Variable（目标变量）**: 被预测的质量变量（如乙烷浓度）。
- **Measurement_Lag（测量滞后）**: 过程变量可用性与目标变量测量之间的时间步延迟。
- **RevIN（可逆实例归一化）**: 用于输入归一化和输出反归一化。
- **Config_Dataclass（配置数据类）**: 继承自 BasicTSModelConfig 的 Python 数据类（PatchXformerConfig），包含所有模型超参数。
- **BasicTSSoftSensorConfig**: BasicTS 中软测量任务的实验配置类。
- **Asymmetric_Cross_Variate_Attention（非对称跨变量注意力）**: Variate_Attention 的变体，过程变量作为 key/value，目标变量作为 query，实现从过程变量到目标变量的定向信息流。

## 需求

### 需求1：模型配置

**用户故事：** 作为研究人员，我希望有一个暴露所有超参数的 PatchXformer 配置数据类，以便轻松配置实验。

#### 验收标准

1. Config_Dataclass 应继承自 BasicTSModelConfig，定义为名为 PatchXformerConfig 的 Python 数据类。
2. Config_Dataclass 应包含 input_len、output_len、num_features、patch_len、patch_stride、hidden_size、n_heads、intermediate_size、hidden_act、num_layers、dropout、use_revin 和 output_attentions 字段，默认值与现有模型约定一致。
3. Config_Dataclass 应包含软测量特有字段：measurement_lag（默认0）、use_variable_gate（默认True）和 use_asymmetric_attn（默认True）。
4. 当 measurement_lag 设为0时，PatchXformer 应平等对待所有位置，不进行滞后感知编码调整。
5. Config_Dataclass 应包含 padding 字段（默认True），控制是否在 patching 前对输入序列进行填充。

### 需求2：所有变量的Patch嵌入

**用户故事：** 作为研究人员，我希望所有输入变量被独立地进行 patch 处理和嵌入，以便为双轴注意力保留时间和变量两个维度。

#### 验收标准

1. Backbone 应对每个变量独立应用 Patch_Embedding，产生形状为 [batch_size, num_variables, num_patches, hidden_size] 的张量。
2. 当 Config_Dataclass 中启用 padding 时，Patch_Embedding 应在 patching 前使用复制填充对输入序列进行填充，确保完全覆盖输入长度。
3. Patch_Embedding 应使用 basicts.modules.embed 中现有的 PatchEmbedding 模块，对每个变量应用位置编码。
4. Backbone 应根据 input_len、patch_len、patch_stride 和 padding 配置计算 num_patches，与 PatchTST 约定一致。

### 需求3：时间注意力（变量内）

**用户故事：** 作为研究人员，我希望在每个变量的 patch 序列内应用时间自注意力，以便模型在跨变量交互之前独立捕获每个变量的时间动态。

#### 验收标准

1. Temporal_Attention 应沿 patch 维度对每个变量独立应用多头自注意力，操作重塑为 [batch_size * num_variables, num_patches, hidden_size] 的张量。
2. Temporal_Attention 应使用 basicts.modules.transformer 中现有的 MultiHeadAttention 模块。
3. Temporal_Attention 应包含残差连接和后层归一化，遵循与 PatchTST 编码器层相同的模式。
4. Temporal_Attention 应在注意力子层之后包含前馈网络（MLPLayer），带残差连接和层归一化。

### 需求4：变量注意力（跨变量）

**用户故事：** 作为研究人员，我希望在每个 patch 位置进行跨变量注意力，以便模型以细粒度的时间分辨率捕获变量间依赖关系，而不是通过单个全局 token。

#### 验收标准

1. Variate_Attention 应在每个 patch 位置跨变量维度应用多头注意力，操作重塑为 [batch_size * num_patches, num_variables, hidden_size] 的张量。
2. 当 Config_Dataclass 中 use_asymmetric_attn 为 True 时，Variate_Attention 应使用过程变量作为 key 和 value，目标变量作为 query 进行交叉注意力。
3. 当 Config_Dataclass 中 use_asymmetric_attn 为 False 时，Variate_Attention 应对所有变量平等地应用标准自注意力。
4. Variate_Attention 应使用 basicts.modules.transformer 中现有的 MultiHeadAttention 模块。
5. Variate_Attention 应包含残差连接和后层归一化。
6. Variate_Attention 应在注意力子层之后包含前馈网络（MLPLayer），带残差连接和层归一化。

### 需求5：双轴块组合

**用户故事：** 作为研究人员，我希望时间注意力和变量注意力组合成可重复的块，以便模型能够逐步细化时间和跨变量表示。

#### 验收标准

1. Dual_Axis_Block 应由一个 Temporal_Attention 子层后接一个 Variate_Attention 子层按顺序组成。
2. Backbone 应按顺序堆叠 num_layers 个 Dual_Axis_Block。
3. Backbone 应在最后一个 Dual_Axis_Block 之后应用最终层归一化。

### 需求6：变量重要性门控

**用户故事：** 作为研究人员，我希望有一个可学习的门控机制来加权变量贡献，以便模型能够自动识别哪些过程变量对预测目标最有信息量。

#### 验收标准

1. 当 use_variable_gate 为 True 时，Variable_Importance_Gate 应从嵌入表示中计算每个变量的重要性分数。
2. Variable_Importance_Gate 应使用线性投影后接 sigmoid 激活产生形状为 [batch_size, num_variables, 1, 1] 的权重向量。
3. Variable_Importance_Gate 应在每个 Dual_Axis_Block 中的 Variate_Attention 之前，将重要性权重与变量表示逐元素相乘。
4. 当 use_variable_gate 为 False 时，Backbone 应跳过门控机制，直接将表示传递给 Variate_Attention。

### 需求7：滞后感知位置编码

**用户故事：** 作为研究人员，我希望模型将测量滞后信息编码到 patch 位置中，以便模型区分估计区域和预测区域的 patch。

#### 验收标准

1. 当 measurement_lag 大于0时，Lag_Aware_Position_Encoding 应向相对于预测范围落在测量滞后窗口内的 patch 位置添加可学习的滞后嵌入。
2. Lag_Aware_Position_Encoding 应使用形状为 [1, 1, 1, hidden_size] 的可学习嵌入向量，添加到时间覆盖范围与滞后边界重叠的 patch。
3. 当 measurement_lag 为0时，Backbone 应使用 Patch_Embedding 的标准位置编码，不进行滞后感知调整。

### 需求8：预测头

**用户故事：** 作为研究人员，我希望有一个将编码表示映射到预测的预测头，以便模型产生与 BasicTS 软测量评估管道兼容的输出。

#### 验收标准

1. ForForecasting 应从 Backbone 输出中提取目标变量表示，并在 patch 和隐藏维度上展平。
2. ForForecasting 应应用从 (num_patches * hidden_size) 到 output_len 的线性投影以产生最终预测。
3. ForForecasting 应产生形状为 [batch_size, output_len, num_features] 的输出，其中 num_features 等于目标变量数量（通常为1）。
4. 当 use_revin 为 True 时，ForForecasting 应在 Backbone 之前对输入应用 RevIN 归一化，在预测头之后对预测应用 RevIN 反归一化。

### 需求9：框架集成

**用户故事：** 作为研究人员，我希望 PatchXformer 与 BasicTS 无缝集成，以便我可以使用与现有模型相同的模式通过 BasicTSSoftSensorConfig 运行实验。

#### 验收标准

1. PatchXformer 模块应位于 src/basicts/models/PatchXformer/，遵循目录结构：__init__.py、arch/__init__.py、arch/patchxformer_arch.py、arch/layers.py、config/patchxformer_config.py。
2. ForForecasting 的 forward 方法应接受形状为 [batch_size, input_len, num_features] 的输入，以及可选的形状为 [batch_size, input_len, num_timestamps] 的 inputs_timestamps。
3. 当 output_attentions 为 True 时，ForForecasting 应返回包含 "prediction" 和 "attn_weights" 键的字典，其中包含时间和变量注意力权重。
4. 当 output_attentions 为 False 时，ForForecasting 应仅返回预测张量。
5. PatchXformer 的 __init__.py 应导出 PatchXformerForForecasting、PatchXformerBackbone 和 PatchXformerConfig。

### 需求10：与排除目标模式的兼容性

**用户故事：** 作为研究人员，我希望 PatchXformer 在目标变量从输入中排除时（exclude_target_from_input=True）能正确工作，以便模型作为纯软测量器运行，不使用自回归目标信息。

#### 验收标准

1. 当 BasicTSSoftSensorConfig 中 exclude_target_from_input 为 True 时，ForForecasting 应接受仅包含过程变量的输入（num_features 等于不含目标的过程变量数量）。
2. 当 exclude_target_from_input 为 True 时，Variate_Attention 应将所有输入变量视为过程变量，use_asymmetric_attn 应应用标准自注意力，因为输入中没有目标变量。
3. 当 exclude_target_from_input 为 False 时，Variate_Attention 应通过目标变量的位置（数据集预处理后的最后一个变量）来识别目标变量以进行非对称注意力。
4. Config_Dataclass 应包含 target_var_index 字段（默认-1，表示最后一个变量），用于在 use_asymmetric_attn 为 True 且 exclude_target_from_input 为 False 时识别目标变量位置。

### 需求11：Dropout和正则化

**用户故事：** 作为研究人员，我希望在整个模型中一致地应用 dropout，以便在通常较小的软测量数据集中控制过拟合。

#### 验收标准

1. Backbone 应在 Patch_Embedding 之后、Temporal_Attention 内、Variate_Attention 内和前馈网络内使用 Config_Dataclass 中的 dropout 率应用 dropout。
2. ForForecasting 应在预测头的最终线性投影之前应用 dropout。
3. dropout 率应通过 Config_Dataclass 中的单个 dropout 字段进行配置，统一应用于所有组件。
