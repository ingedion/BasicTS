# Implementation Plan: PatchXformer

## Overview

Implement the PatchXformer model —a dual-axis attention Transformer for soft sensing tasks —following the existing BasicTS model patterns (PatchTST, iTransformer, TimeXer). The implementation proceeds from configuration —layers —backbone —forecasting wrapper —exports —experiment config —tests, ensuring each step builds on the previous and integrates fully.

## Tasks

- [x] 1. Set up project structure and configuration
  - [x] 1.1 Create PatchXformerConfig dataclass
    - Create `src/basicts/models/PatchXformer/config/patchxformer_config.py`
    - Define `PatchXformerConfig(BasicTSModelConfig)` with all fields: input_len, output_len, num_features, patch_len (16), patch_stride (8), padding (True), hidden_size (256), n_heads (4), intermediate_size (512), hidden_act ("gelu"), num_layers (2), dropout (0.1), use_revin (True), output_attentions (False), measurement_lag (0), use_variable_gate (True), use_asymmetric_attn (True), target_var_index (-1)
    - Follow the same `dataclass` + `field(default=..., metadata={...})` pattern as PatchTSTConfig and iTransformerConfig
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5_

- [x] 2. Implement core layers
  - [x] 2.1 Implement VariableImportanceGate
    - Create `src/basicts/models/PatchXformer/arch/layers.py`
    - Implement `VariableImportanceGate(nn.Module)` with `__init__(self, hidden_size, num_patches)` and `forward(self, hidden_states)` method
    - Pool across patches (mean over dim=2), linear projection to [B, N, 1], sigmoid activation, reshape to [B, N, 1, 1], element-wise multiply with input
    - _Requirements: 6.1, 6.2, 6.3_

  - [x] 2.2 Implement DualAxisBlock
    - In `src/basicts/models/PatchXformer/arch/layers.py`, implement `DualAxisBlock(nn.Module)`
    - Include temporal_attn (MultiHeadAttention), temporal_ffn (MLPLayer), temporal_norm1/norm2 (LayerNorm)
    - Include variate_attn (MultiHeadAttention), variate_ffn (MLPLayer), variate_norm1/norm2 (LayerNorm)
    - Include optional VariableImportanceGate
    - Forward: reshape [B,N,P,H]→[B*N,P,H] for temporal attn, reshape back, apply gate, reshape [B,N,P,H]→[B*P,N,H] for variate attn (asymmetric or self), reshape back
    - Use residual connections and post-layer-normalization pattern
    - _Requirements: 3.1, 3.2, 3.3, 3.4, 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 5.1, 6.3, 6.4_

  - [ ]* 2.3 Write property test for Variable Importance Gate bounds
    - **Property 5: Variable Importance Gate Bounds**
    - Verify gate produces weights in [0, 1] with shape [B, N, 1, 1] and output equals element-wise product
    - **Validates: Requirements 6.1, 6.2, 6.3**

  - [ ]* 2.4 Write property test for Gate Bypass
    - **Property 6: Gate Bypass When Disabled**
    - Verify that when use_variable_gate=False, temporal attention output passes unchanged to variate attention
    - **Validates: Requirements 6.4**

- [x] 3. Implement PatchXformerBackbone
  - [x] 3.1 Implement PatchXformerBackbone
    - Create `src/basicts/models/PatchXformer/arch/patchxformer_arch.py`
    - Implement `PatchXformerBackbone(nn.Module)` with PatchEmbedding (from basicts.modules.embed), lag_embedding (nn.Parameter), ModuleList of DualAxisBlocks, final LayerNorm
    - Compute num_patches from config (same formula as PatchTST)
    - Apply patch embedding per variable: transpose input —patch —reshape to [B, N, P, H]
    - Apply lag embedding to patches overlapping lag boundary when measurement_lag > 0
    - Stack dual-axis blocks, apply final layer norm
    - Return [B, N, P, H] and optional attention weights
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 5.2, 5.3, 7.1, 7.2, 7.3, 11.1_

  - [ ]* 3.2 Write property test for Patch Embedding Shape
    - **Property 2: Patch Embedding Shape**
    - Verify backbone patch embedding produces [B, N, num_patches, hidden_size] with correct num_patches formula
    - **Validates: Requirements 2.1, 2.4**

  - [ ]* 3.3 Write property test for Temporal Attention Variable Independence
    - **Property 3: Temporal Attention Variable Independence**
    - Modify values of variable i, verify temporal attention output for variable j≠i is unchanged
    - **Validates: Requirements 3.1**

  - [ ]* 3.4 Write property test for Lag Encoding Activation
    - **Property 7: Lag Encoding Activation**
    - Verify lag embedding added only to correct patches; when measurement_lag=0, no lag embedding applied
    - **Validates: Requirements 1.4, 7.1, 7.3**

- [x] 4. Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 5. Implement PatchXformerForForecasting
  - [x] 5.1 Implement PatchXformerForForecasting
    - In `src/basicts/models/PatchXformer/arch/patchxformer_arch.py`, implement `PatchXformerForForecasting(nn.Module)`
    - Include backbone, optional RevIN, head_dropout, flatten, forecasting_head (nn.Linear)
    - Forward: RevIN norm —backbone —extract target var —flatten —dropout —linear —reshape to [B, output_len, 1] —RevIN denorm
    - Handle output_attentions: return dict with "prediction" and "attn_weights" or just prediction tensor
    - Accept inputs [B, L, N] and optional inputs_timestamps [B, L, T]
    - Add input validation (ValueError for invalid shapes/configs)
    - _Requirements: 8.1, 8.2, 8.3, 8.4, 9.2, 9.3, 9.4, 10.1, 10.2, 10.3, 11.2, 11.3_

  - [ ]* 5.2 Write property test for Output Shape Invariant
    - **Property 1: Output Shape Invariant**
    - Generate random [B, L, N] inputs, verify output shape is [B, output_len, 1] regardless of batch size and num_features
    - **Validates: Requirements 8.3, 9.2**

  - [ ]* 5.3 Write property test for RevIN Round-Trip
    - **Property 8: RevIN Round-Trip**
    - Verify RevIN norm followed by denorm —identity within floating-point tolerance
    - **Validates: Requirements 8.4**

  - [ ]* 5.4 Write property test for Variate Self-Attention Equivariance
    - **Property 4: Variate Self-Attention Equivariance**
    - When use_asymmetric_attn=False, verify permutation equivariance of variate attention
    - **Validates: Requirements 4.3, 10.2**

  - [ ]* 5.5 Write property test for Exclude-Target Mode Correctness
    - **Property 9: Exclude-Target Mode Correctness**
    - Verify model with use_asymmetric_attn=True but only process variables falls back to self-attention and produces valid output
    - **Validates: Requirements 10.1, 10.2**

  - [ ]* 5.6 Write property test for Asymmetric Attention Directionality
    - **Property 10: Asymmetric Attention Directionality**
    - Verify target depends on process vars (cross-attention) while process vars unchanged by variate attention
    - **Validates: Requirements 4.2, 10.3**

- [x] 6. Set up module exports and framework integration
  - [x] 6.1 Create module __init__.py files
    - Create `src/basicts/models/PatchXformer/__init__.py` exporting PatchXformerForForecasting, PatchXformerBackbone, PatchXformerConfig
    - Create `src/basicts/models/PatchXformer/arch/__init__.py` exporting from patchxformer_arch
    - Create `src/basicts/models/PatchXformer/config/__init__.py` (empty or minimal)
    - Follow the same pattern as PatchTST and iTransformer __init__.py files
    - _Requirements: 9.1, 9.5_

  - [x] 6.2 Register PatchXformer in the models package
    - Update `src/basicts/models/__init__.py` to include PatchXformer imports
    - Ensure `from basicts.models.PatchXformer import ...` works
    - _Requirements: 9.1_

- [x] 7. Create experiment configuration
  - [x] 7.1 Create EthyDistillation benchmark experiment config
    - Create `experiments/EthyDistillation_benchmark/patchxformer_excl_target.py`
    - Follow the same pattern as `itransformer_excl_target.py` and `timexer_excl_target.py`
    - Configure: input_len=96, output_len=12, num_features=37, patch_len=16, patch_stride=8, hidden_size=256, n_heads=4, num_layers=2, measurement_lag=6, use_variable_gate=True, use_asymmetric_attn=True (will fallback to self-attn since exclude_target=True)
    - Use BasicTSSoftSensorConfig with target_vars=20, exclude_target_from_input=True, measurement_lag=6
    - _Requirements: 9.1, 10.1, 10.2_

- [ ] 8. Write unit tests
  - [ ]* 8.1 Write unit tests for PatchXformer
    - Create `tests/basicts_test/test_patchxformer.py`
    - Test config dataclass field existence and defaults
    - Test module instantiation and submodule structure
    - Test forward pass with concrete inputs (37 features, input_len=96, output_len=12)
    - Test output_attentions=True returns dict, output_attentions=False returns tensor
    - Test error cases (invalid configs raise ValueError)
    - Test exclude_target_from_input mode
    - Follow the same import pattern as test_models_pls_svr_lstm.py (file-based imports)
    - _Requirements: 1.1—.5, 8.1—.4, 9.1—.5, 10.1—0.4, 11.1—1.3_

- [x] 9. Final checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional and can be skipped for faster MVP
- Each task references specific requirements for traceability
- Checkpoints ensure incremental validation
- Property tests validate universal correctness properties from the design document
- Unit tests validate specific examples and edge cases
- The implementation uses existing modules (PatchEmbedding, MultiHeadAttention, MLPLayer, RevIN) from basicts.modules
- All property-based tests should use `hypothesis` library with `torch` strategy extensions

## Task Dependency Graph

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
