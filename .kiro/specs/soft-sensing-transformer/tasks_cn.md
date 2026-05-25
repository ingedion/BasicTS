# 实现计划：PatchXformer

## 概述

实现 PatchXformer 模型——一种用于软测量任务的双轴注意力 Transformer——遵循现有 BasicTS 模型模式（PatchTST、iTransformer、TimeXer）。实现按照配置→层→骨干网络→预测包装器→导出→实验配置→测试的顺序进行，确保每一步都建立在前一步之上并完全集成。

## 任务

- [ ] 1. 搭建项目结构和配置
  - [ ] 1.1 创建 PatchXformerConfig 数据类
    - 创建 `src/basicts/models/PatchXformer/config/patchxformer_config.py`
    - 定义 `PatchXformerConfig(BasicTSModelConfig)`，包含所有字段：input_len, output_len, num_features, patch_len (16), patch_stride (8), padding (True), hidden_size (256), n_heads (4), intermediate_size (512), hidden_act ("gelu"), num_layers (2), dropout (0.1), use_revin (True), output_attentions (False), measurement_lag (0), use_variable_gate (True), use_asymmetric_attn (True), target_var_index (-1)
    - 遵循与 PatchTSTConfig 和 iTransformerConfig 相同的 `dataclass` + `field(default=..., metadata={...})` 模式
    - _需求：1.1, 1.2, 1.3, 1.4, 1.5_

- [ ] 2. 实现核心层
  - [ ] 2.1 实现 VariableImportanceGate
    - 创建 `src/basicts/models/PatchXformer/arch/layers.py`
    - 实现 `VariableImportanceGate(nn.Module)`，包含 `__init__(self, hidden_size, num_patches)` 和 `forward(self, hidden_states)` 方法
    - 跨 patch 池化（对 dim=2 取均值），线性投影到 [B, N, 1]，sigmoid 激活，重塑为 [B, N, 1, 1]，与输入逐元素相乘
    - _需求：6.1, 6.2, 6.3_

  - [ ] 2.2 实现 DualAxisBlock
    - 在 `src/basicts/models/PatchXformer/arch/layers.py` 中实现 `DualAxisBlock(nn.Module)`
    - 包含 temporal_attn (MultiHeadAttention)、temporal_ffn (MLPLayer)、temporal_norm1/norm2 (LayerNorm)
    - 包含 variate_attn (MultiHeadAttention)、variate_ffn (MLPLayer)、variate_norm1/norm2 (LayerNorm)
    - 包含可选的 VariableImportanceGate
    - 前向传播：重塑 [B,N,P,H]→[B*N,P,H] 用于时间注意力，重塑回，应用门控，重塑 [B,N,P,H]→[B*P,N,H] 用于变量注意力（非对称或自注意力），重塑回
    - 使用残差连接和后层归一化模式
    - _需求：3.1, 3.2, 3.3, 3.4, 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 5.1, 6.3, 6.4_

  - [ ]* 2.3 编写变量重要性门控边界的属性测试
    - **属性 5：变量重要性门控边界**
    - 验证门控产生范围在 [0, 1] 内、形状为 [B, N, 1, 1] 的权重，且输出等于逐元素乘积
    - **验证：需求 6.1, 6.2, 6.3**

  - [ ]* 2.4 编写门控旁路的属性测试
    - **属性 6：门控禁用时的旁路**
    - 验证当 use_variable_gate=False 时，时间注意力输出不变地传递到变量注意力
    - **验证：需求 6.4**

- [ ] 3. 实现 PatchXformerBackbone
  - [ ] 3.1 实现 PatchXformerBackbone
    - 创建 `src/basicts/models/PatchXformer/arch/patchxformer_arch.py`
    - 实现 `PatchXformerBackbone(nn.Module)`，包含 PatchEmbedding（来自 basicts.modules.embed）、lag_embedding (nn.Parameter)、DualAxisBlock 的 ModuleList、最终 LayerNorm
    - 从配置计算 num_patches（与 PatchTST 相同的公式）
    - 逐变量应用 patch 嵌入：转置输入→patch→重塑为 [B, N, P, H]
    - 当 measurement_lag > 0 时，为覆盖滞后边界的 patch 应用滞后嵌入
    - 堆叠双轴块，应用最终层归一化
    - 返回 [B, N, P, H] 和可选的注意力权重
    - _需求：2.1, 2.2, 2.3, 2.4, 5.2, 5.3, 7.1, 7.2, 7.3, 11.1_

  - [ ]* 3.2 编写 Patch 嵌入形状的属性测试
    - **属性 2：Patch 嵌入形状**
    - 验证骨干网络 patch 嵌入产生 [B, N, num_patches, hidden_size]，且 num_patches 公式正确
    - **验证：需求 2.1, 2.4**

  - [ ]* 3.3 编写时间注意力变量独立性的属性测试
    - **属性 3：时间注意力变量独立性**
    - 修改变量 i 的值，验证变量 j≠i 的时间注意力输出不变
    - **验证：需求 3.1**

  - [ ]* 3.4 编写滞后编码激活的属性测试
    - **属性 7：滞后编码激活**
    - 验证滞后嵌入仅添加到正确的 patch；当 measurement_lag=0 时，不应用滞后嵌入
    - **验证：需求 1.4, 7.1, 7.3**

- [ ] 4. 检查点 - 确保所有测试通过
  - 确保所有测试通过，如有问题请询问用户。

- [ ] 5. 实现 PatchXformerForForecasting
  - [ ] 5.1 实现 PatchXformerForForecasting
    - 在 `src/basicts/models/PatchXformer/arch/patchxformer_arch.py` 中实现 `PatchXformerForForecasting(nn.Module)`
    - 包含 backbone、可选 RevIN、head_dropout、flatten、forecasting_head (nn.Linear)
    - 前向传播：RevIN 归一化→骨干网络→提取目标变量→展平→dropout→线性层→重塑为 [B, output_len, 1]→RevIN 反归一化
    - 处理 output_attentions：返回包含 "prediction" 和 "attn_weights" 的字典或仅返回预测张量
    - 接受输入 [B, L, N] 和可选的 inputs_timestamps [B, L, T]
    - 添加输入验证（对无效形状/配置抛出 ValueError）
    - _需求：8.1, 8.2, 8.3, 8.4, 9.2, 9.3, 9.4, 10.1, 10.2, 10.3, 11.2, 11.3_

  - [ ]* 5.2 编写输出形状不变性的属性测试
    - **属性 1：输出形状不变性**
    - 生成随机 [B, L, N] 输入，验证输出形状为 [B, output_len, 1]，无论批大小和 num_features
    - **验证：需求 8.3, 9.2**

  - [ ]* 5.3 编写 RevIN 往返的属性测试
    - **属性 8：RevIN 往返**
    - 验证 RevIN 归一化后接反归一化≈恒等，在浮点容差范围内
    - **验证：需求 8.4**

  - [ ]* 5.4 编写变量自注意力等变性的属性测试
    - **属性 4：变量自注意力等变性**
    - 当 use_asymmetric_attn=False 时，验证变量注意力的排列等变性
    - **验证：需求 4.3, 10.2**

  - [ ]* 5.5 编写排除目标模式正确性的属性测试
    - **属性 9：排除目标模式正确性**
    - 验证 use_asymmetric_attn=True 但仅有过程变量时回退到自注意力并产生有效输出
    - **验证：需求 10.1, 10.2**

  - [ ]* 5.6 编写非对称注意力方向性的属性测试
    - **属性 10：非对称注意力方向性**
    - 验证目标依赖于过程变量（交叉注意力），而过程变量不被变量注意力改变
    - **验证：需求 4.2, 10.3**

- [ ] 6. 设置模块导出和框架集成
  - [ ] 6.1 创建模块 __init__.py 文件
    - 创建 `src/basicts/models/PatchXformer/__init__.py`，导出 PatchXformerForForecasting、PatchXformerBackbone、PatchXformerConfig
    - 创建 `src/basicts/models/PatchXformer/arch/__init__.py`，从 patchxformer_arch 导出
    - 创建 `src/basicts/models/PatchXformer/config/__init__.py`（空或最小内容）
    - 遵循与 PatchTST 和 iTransformer __init__.py 文件相同的模式
    - _需求：9.1, 9.5_

  - [ ] 6.2 在模型包中注册 PatchXformer
    - 更新 `src/basicts/models/__init__.py` 以包含 PatchXformer 导入
    - 确保 `from basicts.models.PatchXformer import ...` 可正常工作
    - _需求：9.1_

- [ ] 7. 创建实验配置
  - [ ] 7.1 创建 EthyDistillation 基准实验配置
    - 创建 `experiments/EthyDistillation_benchmark/patchxformer_excl_target.py`
    - 遵循与 `itransformer_excl_target.py` 和 `timexer_excl_target.py` 相同的模式
    - 配置：input_len=96, output_len=12, num_features=37, patch_len=16, patch_stride=8, hidden_size=256, n_heads=4, num_layers=2, measurement_lag=6, use_variable_gate=True, use_asymmetric_attn=True（由于 exclude_target=True 将回退到自注意力）
    - 使用 BasicTSSoftSensorConfig，设置 target_vars=20, exclude_target_from_input=True, measurement_lag=6
    - _需求：9.1, 10.1, 10.2_

- [ ] 8. 编写单元测试
  - [ ]* 8.1 编写 PatchXformer 单元测试
    - 创建 `tests/basicts_test/test_patchxformer.py`
    - 测试 config 数据类字段存在性和默认值
    - 测试模块实例化和子模块结构
    - 测试使用具体输入的前向传播（37 个特征，input_len=96，output_len=12）
    - 测试 output_attentions=True 返回字典，output_attentions=False 返回张量
    - 测试错误情况（无效配置抛出 ValueError）
    - 测试 exclude_target_from_input 模式
    - 遵循与 test_models_pls_svr_lstm.py 相同的导入模式（基于文件的导入）
    - _需求：1.1—1.5, 8.1—8.4, 9.1—9.5, 10.1—10.4, 11.1—11.3_

- [ ] 9. 最终检查点 - 确保所有测试通过
  - 确保所有测试通过，如有问题请询问用户。

## 备注

- 标记 `*` 的任务为可选，可跳过以加快 MVP 进度
- 每个任务引用特定需求以实现可追溯性
- 检查点确保增量验证
- 属性测试验证设计文档中的通用正确性属性
- 单元测试验证特定示例和边界情况
- 实现使用 basicts.modules 中的现有模块（PatchEmbedding、MultiHeadAttention、MLPLayer、RevIN）
- 所有基于属性的测试应使用 `hypothesis` 库配合 `torch` 策略扩展

## 任务依赖图

```json
{
  "waves": [
    { "id": 0, "tasks": ["1.1"] },
    { "id": 1, "tasks": ["2.1", "2.2"] },
    { "id": 2, "tasks": ["2.3", "2.4", "3.1"] },
    { "id": 3, "tasks": ["3.2", "3.3", "3.4", "5.1"] },
    { "id": 4, "tasks": ["5.2", "5.3", "5.4", "5.5", "5.6", "6.1"] },
    { "id": 5, "tasks": ["6.2", "7.1"] },
    { "id": 6, "tasks": ["8.1"] }
  ]
}
```
