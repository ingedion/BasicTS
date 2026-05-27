# Implementation Plan: Single Experiment Report

## Overview

Implement the `SingleExperimentReporter` module for the BasicTS framework, providing structured training report generation for individual soft-sensing experiments. The implementation follows the existing `basicts.analysis` module architecture, using a composition pattern with independent analysis components coordinated by a central reporter class.

## Tasks

- [x] 1. Set up module structure and data models
  - [x] 1.1 Create the `single` subpackage under `src/basicts/analysis/` with `__init__.py` and define the `ExperimentConfig` and `TrainingCurveData` dataclasses in a `models.py` file
    - Create `src/basicts/analysis/single/__init__.py` exporting `SingleExperimentReporter`
    - Create `src/basicts/analysis/single/models.py` with `ExperimentConfig` and `TrainingCurveData` dataclasses as specified in the design
    - _Requirements: 1.4, 8.1_

  - [x] 1.2 Update `src/basicts/analysis/__init__.py` to export `SingleExperimentReporter` alongside existing `ExperimentAnalyzer`
    - Add `from .single import SingleExperimentReporter` to the module's `__init__.py`
    - Update `__all__` list to include `SingleExperimentReporter`
    - _Requirements: 8.1_

- [x] 2. Implement CheckpointValidator
  - [x] 2.1 Create `src/basicts/analysis/single/checkpoint_validator.py` implementing the `CheckpointValidator` class
    - Implement `resolve_checkpoint_dir(path)` static method: if path contains `cfg.json`, return it directly; otherwise find the subdirectory with the most recent modification time
    - Implement `validate()` method: check existence of `cfg.json` and `test_metrics.json`, parse `cfg.json` into `ExperimentConfig`, raise `FileNotFoundError` with file name and expected path if files are missing, raise `ValueError` if JSON is invalid
    - Extract fields: `model_name` (from `model.name`), `dataset_name`, `input_len` (from `model_config.input_len`), `output_len` (from `model_config.output_len`), `measurement_lag` (default 0), `metrics`, `eval_horizons`, `target_vars`, determine `is_soft_sensor` from `measurement_lag >= 1`
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7_

  - [ ]* 2.2 Write property tests for CheckpointValidator
    - **Property 1: Required file validation** — For any directory path, the validator reports exactly the set of missing required files
    - **Property 2: Config extraction preserves source values** — For any valid cfg.json, extracted fields match source JSON
    - **Property 3: Latest directory selection** — For any model directory with multiple subdirectories, returns the one with most recent mtime
    - **Validates: Requirements 1.1, 1.2, 1.3, 1.4, 1.6**

- [x] 3. Implement LossCurvePlotter
  - [x] 3.1 Create `src/basicts/analysis/single/loss_curve_plotter.py` implementing the `LossCurvePlotter` class
    - Implement `extract_loss_data()`: first attempt to read from TensorBoard event files in `tensorboard/` subdirectory (using `tensorboard.backend.event_processing.event_accumulator`); if unavailable, fall back to parsing the most recent `training_log_*.log` file
    - Implement training log parser: extract per-epoch train/loss, val/loss, test/loss values using regex; detect "Early stopping at epoch N" pattern
    - Implement `plot()`: create a single figure with loss curves using different colors, legend labels "Train Loss", "Val Loss", "Test Loss"; X-axis "Epoch" (integer ticks starting from 1), Y-axis "Loss"; add vertical dashed line at early stopping epoch if detected; only plot available curves; save as PNG at 150+ DPI; return Figure or None if no data
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8_

  - [ ]* 3.2 Write property tests for LossCurvePlotter
    - **Property 4: Training log loss parsing round-trip** — For any valid training log content, parsed loss values match the values in the log
    - **Property 5: Partial loss curve handling** — For any subset of available loss types, the figure contains exactly as many lines as available types
    - **Validates: Requirements 2.2, 2.7**

- [x] 4. Implement MetricCurvePlotter
  - [x] 4.1 Create `src/basicts/analysis/single/metric_curve_plotter.py` implementing the `MetricCurvePlotter` class
    - Implement `extract_metric_data()`: extract per-epoch metric values (from TensorBoard or training log) for all metrics configured in `cfg.json`
    - Implement `plot()`: generate one subplot per metric with available data, each showing train and val curves in different colors; skip metrics without data; return Figure or None
    - _Requirements: 3.1, 3.2, 3.3, 3.4_

  - [ ]* 4.2 Write property test for MetricCurvePlotter
    - **Property 6: Metric subplot count matches configuration** — For any list of configured metrics with available data, the figure contains exactly that many subplots
    - **Validates: Requirements 3.1, 3.2**

- [x] 5. Implement MetricSummarizer
  - [x] 5.1 Create `src/basicts/analysis/single/metric_summarizer.py` implementing the `MetricSummarizer` class
    - Implement `summarize()`: read `test_metrics.json`, return dict with "overall" and "horizon_N" keys
    - Implement `to_markdown_table()`: generate Markdown table for overall metrics (MAE, MSE, RMSE, R2 to 4 decimal places) and per-horizon R2 table
    - Implement `compute_zone_averages()`: for soft-sensor tasks (measurement_lag >= 1), compute estimation zone (horizon <= measurement_lag) and prediction zone (horizon > measurement_lag) average R2, plus delta
    - Handle missing/invalid `test_metrics.json` gracefully by returning placeholder text
    - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5_

  - [ ]* 5.2 Write property tests for MetricSummarizer
    - **Property 7: Metrics table contains all values formatted correctly** — For any valid test_metrics.json with "overall" section, table contains each value to 4 decimal places
    - **Property 8: Zone R2 calculation correctness** — For any set of horizon R2 values and measurement_lag >= 1, zone averages equal the arithmetic means of the respective subsets
    - **Property 14: summarize_metrics returns complete data** — For any valid test_metrics.json, returned dict contains all keys and values matching source
    - **Validates: Requirements 4.1, 4.2, 4.3, 4.4, 8.6**

- [x] 6. Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 7. Implement PredictionPlotter
  - [x] 7.1 Create `src/basicts/analysis/single/prediction_plotter.py` implementing the `PredictionPlotter` class
    - Implement `plot()`: load `prediction.npy` and `targets.npy` from `test_results/`; extract only `target_vars` dimensions; create one subplot per `eval_horizons` entry; plot ground truth and prediction curves with distinct colors and legend ("真实值", "预测值"); annotate subplot title with "Horizon {n} (R2={value})" from test_metrics; limit displayed samples to `min(num_samples, available_length)`; return Figure or None if data missing
    - Handle missing `test_results/` directory gracefully, noting user should set `save_results=True`
    - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5, 5.6_

  - [ ]* 7.2 Write property tests for PredictionPlotter
    - **Property 9: Target variable dimension extraction** — For any prediction array and target_vars config, only specified dimensions are plotted
    - **Property 10: Prediction subplot count matches eval_horizons** — For any eval_horizons list, figure contains exactly that many subplots
    - **Property 11: Sample count limiting** — For any num_samples and data length, displayed steps equal min(num_samples, data_length)
    - **Validates: Requirements 5.1, 5.2, 5.4, 5.6**

- [x] 8. Implement ReportAssembler and SingleExperimentReporter
  - [x] 8.1 Create `src/basicts/analysis/single/report_assembler.py` implementing the `ReportAssembler` class
    - Implement `assemble()`: create `report/` and `report/fig/` directories; compose Markdown report with sections in order: experiment config summary (generation time, model name, dataset name, input_len, output_len, measurement_lag), loss curves, metric curves, final metrics table, prediction plots; use relative paths `fig/<filename>.png` for images; note skipped sections; save as `report.md`; return absolute path
    - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 6.7_

  - [x] 8.2 Create `src/basicts/analysis/single/reporter.py` implementing the `SingleExperimentReporter` class
    - Implement `__init__()`: call `CheckpointValidator` to validate and parse config; store `output_dir` and `num_samples`
    - Implement `generate_report()`: orchestrate all components (LossCurvePlotter, MetricCurvePlotter, MetricSummarizer, PredictionPlotter, ReportAssembler); save figures to `report/fig/`; print report path to console; return report path string
    - Implement `plot_loss_curves()`, `plot_metric_curves()`, `plot_predictions()`: delegate to respective plotters, return Figure objects
    - Implement `summarize_metrics()`: delegate to MetricSummarizer, return dict
    - _Requirements: 8.1, 8.2, 8.3, 8.4, 8.5, 8.6, 6.6_

  - [ ]* 8.3 Write property test for ReportAssembler
    - **Property 12: Report content completeness** — For any experiment config, report contains all header fields, image references use relative paths, and skipped sections contain unavailability notice
    - **Validates: Requirements 6.4, 6.5, 6.7**

- [x] 9. Implement CLI entry point
  - [x] 9.1 Create `src/basicts/analysis/single/__main__.py` implementing the CLI interface
    - Use `argparse` with required `--ckpt_dir` parameter, optional `--output_dir` (default: `{ckpt_dir}/report/`), optional `--num_samples` (default: 200, range 1-10000)
    - Validate `--ckpt_dir` exists and is a directory (exit code 1 if not)
    - Validate `--num_samples` range (exit code 2 if invalid)
    - Provide `--help` with descriptions and defaults for all parameters
    - On success, instantiate `SingleExperimentReporter` and call `generate_report()`
    - _Requirements: 7.1, 7.2, 7.3, 7.4, 7.5, 7.6_

  - [ ]* 9.2 Write property test for CLI num_samples validation
    - **Property 13: num_samples range validation** — For any integer value, CLI accepts if and only if in [1, 10000]; values outside cause non-zero exit
    - **Validates: Requirements 7.3, 7.5**

- [x] 10. Final checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional and can be skipped for faster MVP
- Each task references specific requirements for traceability
- Checkpoints ensure incremental validation
- Property tests validate universal correctness properties from the design document
- Unit tests validate specific examples and edge cases
- The implementation follows the existing `basicts.analysis` module architecture (composition pattern)
- All test files should be placed under `tests/analysis_test/` following the existing test structure
- TensorBoard event reading requires the `tensorboard` package; the implementation should handle its absence gracefully

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["1.1"] },
    { "id": 1, "tasks": ["1.2", "2.1"] },
    { "id": 2, "tasks": ["2.2", "3.1", "4.1", "5.1"] },
    { "id": 3, "tasks": ["3.2", "4.2", "5.2", "7.1"] },
    { "id": 4, "tasks": ["7.2", "8.1"] },
    { "id": 5, "tasks": ["8.2", "8.3"] },
    { "id": 6, "tasks": ["9.1"] },
    { "id": 7, "tasks": ["9.2"] }
  ]
}
```
