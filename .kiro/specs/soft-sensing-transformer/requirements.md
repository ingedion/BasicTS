# Requirements Document

## Introduction

This document specifies the requirements for a new Transformer-based model called **PatchXformer** designed for soft sensing tasks within the BasicTS framework. The model combines dual-axis attention (temporal patches from PatchTST and variate attention from iTransformer) with soft-sensor-specific mechanisms including asymmetric variable roles, lag-aware positional encoding, and variable importance gating. Unlike TimeXer which only patches the target variable and uses a single global token for cross-variate interaction, PatchXformer patches all variables and performs variate attention at every patch position, enabling richer cross-variable information exchange.

## Glossary

- **PatchXformer**: The new Transformer model for soft sensing tasks combining dual-axis attention with soft-sensor-specific mechanisms.
- **Backbone**: The core neural network architecture (PatchXformerBackbone) that performs feature extraction without task-specific heads.
- **ForForecasting**: The task-specific wrapper (PatchXformerForForecasting) that adds a forecasting head on top of the Backbone.
- **Temporal_Attention**: Multi-head self-attention applied along the patch (time) dimension within each variable independently, capturing temporal patterns.
- **Variate_Attention**: Multi-head attention applied across variables at each patch position, capturing inter-variable dependencies.
- **Patch_Embedding**: The module that segments each variable's time series into fixed-length patches and projects them into hidden dimension.
- **Dual_Axis_Block**: A single encoder block containing one Temporal_Attention layer followed by one Variate_Attention layer with residual connections and layer normalization.
- **Variable_Importance_Gate**: A learnable gating mechanism that produces per-variable importance weights to modulate variable contributions in Variate_Attention.
- **Lag_Aware_Position_Encoding**: Positional encoding that incorporates measurement_lag information to distinguish estimation-zone patches from prediction-zone patches.
- **Process_Variables**: The input sensor measurements (e.g., temperature, pressure, flow rates) used to predict quality variables.
- **Target_Variable**: The quality variable being predicted (e.g., ethane concentration).
- **Measurement_Lag**: The number of time steps delay between process variable availability and target variable measurement.
- **RevIN**: Reversible Instance Normalization used for input normalization and output denormalization.
- **Config_Dataclass**: A Python dataclass (PatchXformerConfig) inheriting from BasicTSModelConfig that holds all model hyperparameters.
- **BasicTSSoftSensorConfig**: The experiment configuration class for soft sensor tasks in BasicTS.
- **Asymmetric_Cross_Variate_Attention**: A variant of Variate_Attention where Process_Variables serve as key/value and Target_Variable serves as query, enabling directed information flow from process to target.

## Requirements

### Requirement 1: Model Configuration

**User Story:** As a researcher, I want a configuration dataclass for PatchXformer that exposes all hyperparameters, so that I can easily configure experiments.

#### Acceptance Criteria

1. THE Config_Dataclass SHALL inherit from BasicTSModelConfig and be defined as a Python dataclass named PatchXformerConfig.
2. THE Config_Dataclass SHALL include fields for input_len, output_len, num_features, patch_len, patch_stride, hidden_size, n_heads, intermediate_size, hidden_act, num_layers, dropout, use_revin, and output_attentions with sensible defaults matching existing model conventions.
3. THE Config_Dataclass SHALL include soft-sensor-specific fields: measurement_lag (default 0), use_variable_gate (default True), and use_asymmetric_attn (default True).
4. WHEN measurement_lag is set to 0, THE PatchXformer SHALL treat all positions equally without lag-aware encoding adjustments.
5. THE Config_Dataclass SHALL include a padding field (default True) controlling whether input sequences are padded before patching.

### Requirement 2: Patch Embedding for All Variables

**User Story:** As a researcher, I want all input variables to be independently patched and embedded, so that both temporal and variate dimensions are preserved for dual-axis attention.

#### Acceptance Criteria

1. THE Backbone SHALL apply Patch_Embedding independently to each variable, producing a tensor of shape [batch_size, num_variables, num_patches, hidden_size].
2. WHEN padding is enabled in Config_Dataclass, THE Patch_Embedding SHALL pad the input sequence using replication padding before patching to ensure complete coverage of the input length.
3. THE Patch_Embedding SHALL use the existing PatchEmbedding module from basicts.modules.embed with positional encoding applied per variable.
4. THE Backbone SHALL compute num_patches based on input_len, patch_len, patch_stride, and padding configuration consistent with PatchTST conventions.

### Requirement 3: Temporal Attention (Within-Variable)

**User Story:** As a researcher, I want temporal self-attention applied within each variable's patch sequence, so that the model captures temporal dynamics independently per variable before cross-variable interaction.

#### Acceptance Criteria

1. THE Temporal_Attention SHALL apply multi-head self-attention along the patch dimension for each variable independently, operating on tensors reshaped to [batch_size * num_variables, num_patches, hidden_size].
2. THE Temporal_Attention SHALL use the existing MultiHeadAttention module from basicts.modules.transformer.
3. THE Temporal_Attention SHALL include a residual connection and post-layer-normalization following the same pattern as PatchTST encoder layers.
4. THE Temporal_Attention SHALL include a feed-forward network (MLPLayer) after the attention sublayer with residual connection and layer normalization.

### Requirement 4: Variate Attention (Cross-Variable)

**User Story:** As a researcher, I want cross-variable attention at each patch position, so that the model captures inter-variable dependencies at fine temporal granularity rather than through a single global token.

#### Acceptance Criteria

1. THE Variate_Attention SHALL apply multi-head attention across the variable dimension at each patch position, operating on tensors reshaped to [batch_size * num_patches, num_variables, hidden_size].
2. WHEN use_asymmetric_attn is True in Config_Dataclass, THE Variate_Attention SHALL use Process_Variables as key and value while using Target_Variable as query in a cross-attention mechanism.
3. WHEN use_asymmetric_attn is False in Config_Dataclass, THE Variate_Attention SHALL apply standard self-attention across all variables equally.
4. THE Variate_Attention SHALL use the existing MultiHeadAttention module from basicts.modules.transformer.
5. THE Variate_Attention SHALL include a residual connection and post-layer-normalization.
6. THE Variate_Attention SHALL include a feed-forward network (MLPLayer) after the attention sublayer with residual connection and layer normalization.

### Requirement 5: Dual-Axis Block Composition

**User Story:** As a researcher, I want temporal and variate attention composed into repeatable blocks, so that the model can progressively refine both temporal and cross-variable representations.

#### Acceptance Criteria

1. THE Dual_Axis_Block SHALL consist of one Temporal_Attention sublayer followed by one Variate_Attention sublayer in sequence.
2. THE Backbone SHALL stack num_layers Dual_Axis_Blocks sequentially.
3. THE Backbone SHALL apply a final layer normalization after the last Dual_Axis_Block.

### Requirement 6: Variable Importance Gating

**User Story:** As a researcher, I want a learnable gating mechanism that weights variable contributions, so that the model can automatically identify which process variables are most informative for predicting the target.

#### Acceptance Criteria

1. WHEN use_variable_gate is True, THE Variable_Importance_Gate SHALL compute per-variable importance scores from the embedded representations.
2. THE Variable_Importance_Gate SHALL produce a weight vector of shape [batch_size, num_variables, 1, 1] using a linear projection followed by sigmoid activation.
3. THE Variable_Importance_Gate SHALL multiply the importance weights element-wise with variable representations before Variate_Attention in each Dual_Axis_Block.
4. WHEN use_variable_gate is False, THE Backbone SHALL skip the gating mechanism and pass representations directly to Variate_Attention.

### Requirement 7: Lag-Aware Positional Encoding

**User Story:** As a researcher, I want the model to encode measurement lag information into patch positions, so that the model distinguishes between patches in the estimation zone and prediction zone.

#### Acceptance Criteria

1. WHEN measurement_lag is greater than 0, THE Lag_Aware_Position_Encoding SHALL add a learnable lag embedding to patch positions that fall within the measurement lag window relative to the prediction horizon.
2. THE Lag_Aware_Position_Encoding SHALL use a learnable embedding vector of shape [1, 1, 1, hidden_size] added to patches whose temporal coverage overlaps with the lag boundary.
3. WHEN measurement_lag is 0, THE Backbone SHALL use standard positional encoding from Patch_Embedding without lag-aware adjustments.

### Requirement 8: Forecasting Head

**User Story:** As a researcher, I want a forecasting head that maps encoded representations to predictions, so that the model produces output compatible with the BasicTS soft sensor evaluation pipeline.

#### Acceptance Criteria

1. THE ForForecasting SHALL extract the target variable representation from the Backbone output and flatten it across patches and hidden dimensions.
2. THE ForForecasting SHALL apply a linear projection from (num_patches * hidden_size) to output_len to produce the final prediction.
3. THE ForForecasting SHALL produce output with shape [batch_size, output_len, num_features] where num_features equals the number of target variables (typically 1).
4. WHEN use_revin is True, THE ForForecasting SHALL apply RevIN normalization to inputs before the Backbone and RevIN denormalization to predictions after the forecasting head.

### Requirement 9: Framework Integration

**User Story:** As a researcher, I want PatchXformer to integrate seamlessly with BasicTS, so that I can run experiments using BasicTSSoftSensorConfig with the same patterns as existing models.

#### Acceptance Criteria

1. THE PatchXformer module SHALL be located at src/basicts/models/PatchXformer/ following the directory structure: __init__.py, arch/__init__.py, arch/patchxformer_arch.py, arch/layers.py, config/patchxformer_config.py.
2. THE ForForecasting forward method SHALL accept inputs of shape [batch_size, input_len, num_features] and optionally inputs_timestamps of shape [batch_size, input_len, num_timestamps].
3. WHEN output_attentions is True, THE ForForecasting SHALL return a dictionary with keys "prediction" and "attn_weights" containing temporal and variate attention weights.
4. WHEN output_attentions is False, THE ForForecasting SHALL return only the prediction tensor.
5. THE PatchXformer __init__.py SHALL export PatchXformerForForecasting, PatchXformerBackbone, and PatchXformerConfig.

### Requirement 10: Compatibility with Exclude-Target Mode

**User Story:** As a researcher, I want PatchXformer to work correctly when the target variable is excluded from input (exclude_target_from_input=True), so that the model operates as a pure soft sensor without autoregressive target information.

#### Acceptance Criteria

1. WHEN exclude_target_from_input is True in BasicTSSoftSensorConfig, THE ForForecasting SHALL accept inputs containing only process variables (num_features equals number of process variables without target).
2. WHEN exclude_target_from_input is True, THE Variate_Attention SHALL treat all input variables as process variables and use_asymmetric_attn SHALL apply standard self-attention since no target variable is present in the input.
3. WHEN exclude_target_from_input is False, THE Variate_Attention SHALL identify the target variable by its position (last variable after dataset preprocessing) for asymmetric attention.
4. THE Config_Dataclass SHALL include a target_var_index field (default -1, meaning last variable) to identify the target variable position when use_asymmetric_attn is True and exclude_target_from_input is False.

### Requirement 11: Dropout and Regularization

**User Story:** As a researcher, I want consistent dropout applied throughout the model, so that overfitting is controlled in the typically small soft sensor datasets.

#### Acceptance Criteria

1. THE Backbone SHALL apply dropout after Patch_Embedding, within Temporal_Attention, within Variate_Attention, and within feed-forward networks using the dropout rate from Config_Dataclass.
2. THE ForForecasting SHALL apply dropout before the final linear projection in the forecasting head.
3. THE dropout rate SHALL be configurable through a single dropout field in Config_Dataclass applied uniformly across all components.
