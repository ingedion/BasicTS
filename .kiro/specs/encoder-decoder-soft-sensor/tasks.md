# Implementation Plan: Encoder-Decoder 架构模型适配软测量任务

## Overview

本实现计划将 encoder-decoder 架构模型（Informer、Autoformer 等）的软测量适配能力集成到 BasicTS 框架中。核心实现包括：新建 `EncDecSoftSensorDataset`（时间维度划分 encoder/decoder 输入对）、`EncDecSoftSensorTaskFlow`（归一化 + mask + 后处理）、`EncDecSoftSensorRunner`（推理阶段保留 decoder input）、`EncDecSoftSensorConfig`（参数验证与自动计算），以及 Informer 实验配置。所有新增模块独立于现有 encoder-only 软测量流程，不修改任何已有代码。

## Tasks

- [x] 1. Set up core interfaces and config
  - [x] 1.1 Create `EncDecSoftSensorConfig` dataclass
    - Create file `src/basicts/configs/encdec_ss_config.py`
    - Inherit from `BasicTSSoftSensorConfig`
    - Add fields: `lookback` (default 0), `pred_len` (default 0)
    - Implement `__post_init__` with validation: `lookback >= 0`, `lookback <= input_len - measurement_lag`, `pred_len >= 0`
    - Auto-compute `output_len = measurement_lag + pred_len + lookback`
    - Set default `dataset_type`, `taskflow`, and `runner` to the new enc-dec classes
    - Override `ckpt_save_dir` pattern to include lookback and pred_len
    - _Requirements: 4.1, 4.2, 4.4, 4.5, 4.6_

  - [x] 1.2 Register config in `src/basicts/configs/__init__.py`
    - Add import for `EncDecSoftSensorConfig`
    - Add to `__ALL__` list
    - _Requirements: 6.1, 6.2, 6.3_

- [x] 2. Implement EncDecSoftSensorDataset
  - [x] 2.1 Create `EncDecSoftSensorDataset` class
    - Create file `src/basicts/data/encdec_ss_dataset.py`
    - Inherit from `BasicTSDataset` (same base as `BasicTSSoftSensorDataset`)
    - Constructor params: `dataset_name`, `input_len`, `measurement_lag`, `target_vars`, `mode`, `lookback=0`, `pred_len=0`, `use_timestamps=False`, `data_file_path=None`, `memmap=False`
    - Compute `encoder_len = input_len - measurement_lag` and `decoder_len = lookback + measurement_lag + pred_len`
    - Validate: `input_len > measurement_lag`, `lookback >= 0`, `lookback <= input_len - measurement_lag`, `target_vars` indices in valid range
    - Load data from `{data_file_path}/{mode}_data.npy` (and timestamps if enabled)
    - `__getitem__`: return dict with `inputs` [encoder_len, num_features], `targets` [decoder_len, num_features] (with masking applied), and optional timestamps
    - `__len__`: `data_len - input_len - pred_len + 1`
    - Apply masking in `__getitem__`: lookback zone keeps true values, lag zone masks target_vars to 0, prediction zone all zeros
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.1, 2.2, 2.3_

  - [x] 2.2 Register dataset in `src/basicts/data/__init__.py`
    - Add import for `EncDecSoftSensorDataset`
    - Add to `__all__` list
    - _Requirements: 6.1, 6.2, 6.3_

  - [x]* 2.3 Write property tests for EncDecSoftSensorDataset
    - **Property 1: Dataset shape invariant**
    - **Property 2: Lookback zone preserves true values**
    - **Property 3: Lag zone masking correctness**
    - **Property 4: Prediction zone all zeros**
    - **Validates: Requirements 1.2, 1.3, 1.4, 1.5, 1.6, 2.1, 2.2**
    - Create file `tests/test_encdec_ss_dataset.py`
    - Use hypothesis to generate random configs (input_len, measurement_lag, lookback, pred_len, num_features)
    - Generate synthetic random data arrays for testing
    - Verify shape invariants and masking correctness across 100+ iterations

- [x] 3. Implement EncDecSoftSensorTaskFlow
  - [x] 3.1 Create `EncDecSoftSensorTaskFlow` class
    - Create file `src/basicts/runners/taskflow/encdec_ss_taskflow.py`
    - Inherit from `BasicTSSoftSensorTaskFlow`
    - Override `preprocess`: extract ground_truth from `targets[:, lookback:, target_vars]`, z-score normalize inputs and targets using full-channel scaler, mask `targets[:, lookback:lookback+measurement_lag, target_vars] = 0`, mask `targets[:, lookback+measurement_lag:, :] = 0`, store ground_truth and targets_mask
    - Override `postprocess`: extract `prediction[:, lookback:, target_vars]`, inverse transform to original scale, set `forward_return['targets'] = ground_truth`
    - Override `get_weight`: weight based on valid target mask
    - _Requirements: 3.1.1, 3.1.2, 3.1.3, 3.1.4, 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7_

  - [x] 3.2 Register taskflow in `src/basicts/runners/taskflow/__init__.py`
    - Add import for `EncDecSoftSensorTaskFlow`
    - _Requirements: 6.1, 6.2, 6.3_

  - [x]* 3.3 Write property tests for EncDecSoftSensorTaskFlow
    - **Property 5: Postprocess extraction correctness**
    - **Property 6: Normalization round-trip**
    - **Property 7: Taskflow preprocess masking after normalization**
    - **Validates: Requirements 3.1, 3.2, 3.3, 3.6, 3.1.2**
    - Create file `tests/test_encdec_ss_taskflow.py`
    - Use hypothesis to generate random batch data and model outputs
    - Verify postprocess extraction shape, normalization round-trip, and masking correctness

- [x] 4. Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 5. Implement EncDecSoftSensorRunner
  - [x] 5.1 Create `EncDecSoftSensorRunner` class
    - Create file `src/basicts/runners/encdec_ss_runner.py`
    - Inherit from `BasicTSRunner`
    - Override `_forward` method: move data to device, build kwargs from `self.forward_params`, do NOT replace `targets` with `torch.empty_like()` during non-training phases (key difference from `BasicTSRunner`), pass `step`/`epoch`/`train` if in model signature, call `model(inputs, **kwargs)`, wrap return as dict if tensor
    - _Requirements: 3.1.4 (implicit), 6.1, 6.2, 6.3_

  - [x] 5.2 Register runner in `src/basicts/runners/__init__.py`
    - Add import for `EncDecSoftSensorRunner`
    - Add to `__all__` list
    - _Requirements: 6.1, 6.2, 6.3_

- [x] 6. Create Informer experiment configurations
  - [x] 6.1 Create Debutanizer Informer experiment config
    - Create file `experiments/Debutanizer_benchmark/informer_encdec_ss.py`
    - Use `EncDecSoftSensorConfig` with `Informer` model
    - Configure: `dataset_name="Debutanizer"`, `input_len=32`, `measurement_lag=3`, `lookback=6`, `pred_len=3`, `target_vars=-1`
    - Set `InformerConfig` with `input_len=encoder_len`, `output_len=decoder_len`, `label_len=lookback`, `num_features=8`
    - Include training hyperparameters: optimizer, lr_scheduler, callbacks, metrics, eval_horizons
    - Add docstring explaining encoder-decoder soft sensor design
    - _Requirements: 5.1, 5.3, 5.4_

  - [x] 6.2 Create EthyDistillation Informer experiment config
    - Create file `experiments/EthyDistillation_benchmark/informer_encdec_ss.py`
    - Use `EncDecSoftSensorConfig` with `Informer` model
    - Configure for EthyDistillation dataset parameters (num_features, target_vars, measurement_lag)
    - Set appropriate `lookback` and `pred_len` values
    - Include training hyperparameters matching the benchmark pattern
    - Add docstring explaining encoder-decoder soft sensor design
    - _Requirements: 5.2, 5.3, 5.4_

- [x] 7. Config validation and integration tests
  - [x]* 7.1 Write property tests for EncDecSoftSensorConfig
    - **Property 8: Config validation rejects invalid parameters**
    - **Property 9: Output length auto-computation**
    - **Validates: Requirements 4.4, 4.5, 4.6**
    - Create file `tests/test_encdec_ss_config.py`
    - Use hypothesis to generate random parameter combinations
    - Verify validation errors for invalid lookback/pred_len
    - Verify output_len auto-computation correctness

  - [x]* 7.2 Write integration tests
    - Create file `tests/test_encdec_ss_integration.py`
    - Test Informer end-to-end 1 step: config → Dataset → TaskFlow → Model → no dimension errors
    - Test `lookback=0, pred_len=0` degenerates to pure lag estimation
    - Test existing DLinear config still works (no regression)
    - Test experiment config files import without errors
    - _Requirements: 5.4, 6.1, 6.2, 6.3_

- [x] 8. Final checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional and can be skipped for faster MVP
- Each task references specific requirements for traceability
- Checkpoints ensure incremental validation
- Property tests validate universal correctness properties from the design document
- Unit tests validate specific examples and edge cases
- All new modules are independent from existing encoder-only soft sensor flow (no modifications to existing code)
- The design uses Python throughout, matching the project's language

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["1.1"] },
    { "id": 1, "tasks": ["1.2", "2.1"] },
    { "id": 2, "tasks": ["2.2", "2.3", "5.1"] },
    { "id": 3, "tasks": ["3.1", "5.2"] },
    { "id": 4, "tasks": ["3.2", "3.3"] },
    { "id": 5, "tasks": ["6.1", "6.2", "7.1"] },
    { "id": 6, "tasks": ["7.2"] }
  ]
}
```
