"""Prediction plotter for single experiment analysis.

Loads prediction results (prediction.npy, targets.npy) from test_results/
and plots prediction vs ground truth curves for each eval horizon.
"""

import json
import os
from typing import TYPE_CHECKING, Dict, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

if TYPE_CHECKING:
    from .models import ExperimentConfig


class PredictionPlotter:
    """加载预测结果并绘制预测对比图。"""

    def __init__(self, ckpt_dir: str, config: "ExperimentConfig", num_samples: int = 200):
        """
        Args:
            ckpt_dir: 实验检查点目录路径
            config: 解析后的实验配置
            num_samples: 展示的时间步数量，默认 200
        """
        self.ckpt_dir = ckpt_dir
        self.config = config
        self.num_samples = num_samples

    def plot(self) -> Optional[Figure]:
        """绘制预测对比子图，返回 Figure 或 None。

        加载 prediction.npy 和 targets.npy，提取 target_vars 指定的维度，
        为每个 eval_horizons 条目绘制独立子图，标注 R2 值。

        Returns:
            matplotlib Figure 对象，如果数据不可用则返回 None。
        """
        # Check test_results directory
        test_results_dir = os.path.join(self.ckpt_dir, "test_results")
        if not os.path.isdir(test_results_dir):
            return None

        pred_path = os.path.join(test_results_dir, "prediction.npy")
        target_path = os.path.join(test_results_dir, "targets.npy")

        if not os.path.exists(pred_path) or not os.path.exists(target_path):
            return None

        # Load prediction and target arrays
        preds = self._load_npy(pred_path)
        targets = self._load_npy(target_path)

        if preds is None or targets is None:
            return None

        # Load test metrics for R2 annotation
        test_metrics = self._load_test_metrics()

        # Get eval_horizons (1-indexed in config, convert to 0-indexed for array access)
        eval_horizons = self.config.eval_horizons
        if not eval_horizons:
            return None

        # Get target_vars (feature dimension indices)
        target_vars = self.config.target_vars

        # Extract target_vars dimensions
        # Data shape: (samples, output_len, num_features) or (samples, output_len)
        preds = self._extract_target_vars(preds, target_vars)
        targets = self._extract_target_vars(targets, target_vars)

        if preds is None or targets is None:
            return None

        # Determine number of samples to display
        available_length = preds.shape[0]
        n_display = min(self.num_samples, available_length)

        # Create subplots - one per eval_horizon
        n_horizons = len(eval_horizons)
        ncols = min(n_horizons, 3)
        nrows = (n_horizons + ncols - 1) // ncols

        fig, axes = plt.subplots(
            nrows, ncols,
            figsize=(6 * ncols, 4 * nrows),
            squeeze=False,
        )

        for idx, horizon in enumerate(eval_horizons):
            row = idx // ncols
            col = idx % ncols
            ax = axes[row][col]

            # horizon is 1-indexed, convert to 0-indexed for array access
            h_idx = horizon - 1

            # Extract data for this horizon
            # After target_vars extraction, shape is (samples, output_len) or (samples, output_len, n_target_vars)
            pred_h = self._extract_horizon(preds, h_idx)
            target_h = self._extract_horizon(targets, h_idx)

            if pred_h is None or target_h is None:
                ax.set_title(f"Horizon {horizon} (数据不可用)")
                ax.set_visible(True)
                continue

            # Limit to num_samples
            pred_h = pred_h[:n_display]
            target_h = target_h[:n_display]

            # Plot ground truth and prediction
            ax.plot(
                range(len(target_h)),
                target_h,
                color="#333333",
                linewidth=1.2,
                label="真实值",
            )
            ax.plot(
                range(len(pred_h)),
                pred_h,
                color="#2196F3",
                linewidth=1.0,
                alpha=0.8,
                label="预测值",
            )

            # Annotate title with R2 value
            r2_value = self._get_horizon_r2(test_metrics, horizon)
            if r2_value is not None:
                ax.set_title(f"Horizon {horizon} (R2={r2_value:.4f})")
            else:
                ax.set_title(f"Horizon {horizon}")

            ax.legend(fontsize=8, loc="upper right")
            ax.grid(True, alpha=0.3)
            ax.set_xlabel("Sample Index")
            ax.set_ylabel("Value")

        # Hide unused subplots
        for idx in range(n_horizons, nrows * ncols):
            row = idx // ncols
            col = idx % ncols
            axes[row][col].set_visible(False)

        fig.suptitle(
            f"Prediction vs Ground Truth - {self.config.model_name} on {self.config.dataset_name}",
            fontsize=12,
            fontweight="bold",
        )
        fig.tight_layout()

        return fig

    def _load_npy(self, path: str) -> Optional[np.ndarray]:
        """Load a .npy file, handling both standard numpy and memmap formats.

        Args:
            path: Path to the .npy file.

        Returns:
            numpy array or None if loading fails.
        """
        try:
            data = np.load(path, allow_pickle=True)
            return data
        except Exception:
            pass

        # Fall back to memmap loading (files saved via np.memmap)
        try:
            output_len = self.config.output_len or 1
            # Try to infer shape from file size
            file_size = os.path.getsize(path)
            # Assume float32 (4 bytes per element)
            total_elements = file_size // 4

            # Try shape: (samples, output_len, num_features)
            num_features = len(self.config.target_vars) if self.config.target_vars else 1
            # First try with all features from raw config
            raw_num_features = self.config.raw_config.get("model_config", {}).get("num_features", num_features)

            if total_elements % (output_len * raw_num_features) == 0:
                n_samples = total_elements // (output_len * raw_num_features)
                data = np.memmap(
                    path, dtype=np.float32, mode="r",
                    shape=(n_samples, output_len, raw_num_features),
                )
                return np.array(data)

            # Try shape: (samples, output_len, 1) - single target variable
            if total_elements % (output_len * 1) == 0:
                n_samples = total_elements // (output_len * 1)
                data = np.memmap(
                    path, dtype=np.float32, mode="r",
                    shape=(n_samples, output_len, 1),
                )
                return np.array(data)

            # Try shape: (samples, output_len)
            if total_elements % output_len == 0:
                n_samples = total_elements // output_len
                data = np.memmap(
                    path, dtype=np.float32, mode="r",
                    shape=(n_samples, output_len),
                )
                return np.array(data)

        except Exception:
            pass

        return None

    def _extract_target_vars(self, data: np.ndarray, target_vars: list) -> Optional[np.ndarray]:
        """Extract only target_vars dimensions from the data array.

        Args:
            data: Array with shape (samples, output_len, num_features) or (samples, output_len).
            target_vars: List of feature dimension indices to extract.

        Returns:
            Array with only target_vars dimensions, or None if extraction fails.
        """
        if data is None:
            return None

        if data.ndim == 3 and target_vars:
            # Shape: (samples, output_len, num_features) -> extract target_vars
            try:
                return data[:, :, target_vars]
            except (IndexError, TypeError):
                return data
        elif data.ndim == 2:
            # Shape: (samples, output_len) - already single feature
            return data

        return data

    def _extract_horizon(self, data: np.ndarray, h_idx: int) -> Optional[np.ndarray]:
        """Extract data for a specific horizon index.

        Args:
            data: Array with shape (samples, output_len, ...) or (samples, output_len).
            h_idx: 0-indexed horizon index.

        Returns:
            1D array of values for the specified horizon, or None if index is out of bounds.
        """
        if data is None:
            return None

        try:
            if data.ndim == 3:
                # Shape: (samples, output_len, n_target_vars)
                # Take first target var for plotting
                return data[:, h_idx, 0]
            elif data.ndim == 2:
                # Shape: (samples, output_len)
                return data[:, h_idx]
            elif data.ndim == 1:
                return data
        except (IndexError, TypeError):
            return None

        return None

    def _load_test_metrics(self) -> Dict:
        """Load test_metrics.json from the checkpoint directory.

        Returns:
            Dictionary of test metrics, or empty dict if unavailable.
        """
        metrics_path = os.path.join(self.ckpt_dir, "test_metrics.json")
        if not os.path.exists(metrics_path):
            return {}

        try:
            with open(metrics_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError, OSError):
            return {}

    def _get_horizon_r2(self, test_metrics: Dict, horizon: int) -> Optional[float]:
        """Get R2 value for a specific horizon from test metrics.

        Args:
            test_metrics: Loaded test_metrics.json content.
            horizon: 1-indexed horizon number.

        Returns:
            R2 value as float, or None if not available.
        """
        horizon_key = f"horizon_{horizon}"
        horizon_data = test_metrics.get(horizon_key, {})
        r2 = horizon_data.get("R2", None)
        if r2 is not None:
            try:
                return float(r2)
            except (ValueError, TypeError):
                return None
        return None
