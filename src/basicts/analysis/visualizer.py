"""
Experiment Visualizer.

Generates charts for multi-model benchmark comparison.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .collector import ExperimentResult


def _check_matplotlib():
    """Check if matplotlib is available."""
    try:
        import matplotlib
        import matplotlib.pyplot as plt
        import seaborn as sns
        return True
    except ImportError:
        print("Warning: matplotlib/seaborn not installed. Install with:")
        print("  pip install matplotlib seaborn")
        return False


class ExperimentVisualizer:
    """
    Generates comparison charts from experiment results.

    Supports:
    - Overall metric comparison (bar chart / heatmap)
    - Horizon-wise performance comparison
    - Estimation vs prediction zone comparison (soft sensor)
    - Prediction curves (requires save_results=True)
    """

    def __init__(self, results: List[ExperimentResult], output_dir: str):
        self.results = results
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Style settings
        self._figsize_bar = (12, 6)
        self._figsize_heatmap = (10, 6)
        self._dpi = 150

    def plot_overall_comparison(
        self,
        metrics: Optional[List[str]] = None,
        filename: str = "overall_comparison.png"
    ) -> Optional[str]:
        """
        Generate bar chart comparing overall metrics across all experiments.

        Args:
            metrics: List of metric names to plot. Default: ["R2", "RMSE"]
            filename: Output filename.

        Returns:
            Path to saved figure, or None if matplotlib unavailable.
        """
        if not _check_matplotlib():
            return None

        import matplotlib.pyplot as plt
        import seaborn as sns

        if metrics is None:
            metrics = ["R2", "RMSE"]

        n_metrics = len(metrics)
        fig, axes = plt.subplots(1, n_metrics, figsize=(6 * n_metrics, 6))
        if n_metrics == 1:
            axes = [axes]

        # Prepare data
        labels = []
        for r in self.results:
            labels.append(f"{r.model_name}\n({r.config_label})")

        for ax, metric in zip(axes, metrics):
            values = []
            colors = []
            for r in self.results:
                val = r.metrics.get("overall", {}).get(metric, None)
                values.append(val if val is not None else 0)
                # Color by config
                colors.append("#2196F3" if r.exclude_target else "#FF9800")

            bars = ax.bar(range(len(values)), values, color=colors, alpha=0.8, edgecolor="black", linewidth=0.5)
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels, fontsize=8, ha="center")
            ax.set_ylabel(metric, fontsize=12)
            ax.set_title(f"Overall {metric}", fontsize=13, fontweight="bold")
            ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
            ax.grid(axis="y", alpha=0.3)

            # Add value labels on bars
            for bar, val in zip(bars, values):
                y_pos = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2, y_pos,
                       f"{val:.3f}", ha="center", va="bottom" if val >= 0 else "top",
                       fontsize=7)

        # Legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor="#2196F3", alpha=0.8, label="excl_target (pure soft sensor)"),
            Patch(facecolor="#FF9800", alpha=0.8, label="incl_target (target in input)"),
        ]
        fig.legend(handles=legend_elements, loc="upper right", fontsize=9)

        plt.tight_layout()
        save_path = self.output_dir / filename
        plt.savefig(save_path, dpi=self._dpi, bbox_inches="tight")
        plt.close()
        print(f"Saved: {save_path}")
        return str(save_path)

    def plot_horizon_comparison(
        self,
        metric: str = "R2",
        filename: str = "horizon_comparison.png"
    ) -> Optional[str]:
        """
        Generate grouped bar chart comparing performance across horizons.

        Args:
            metric: Metric to compare.
            filename: Output filename.

        Returns:
            Path to saved figure.
        """
        if not _check_matplotlib():
            return None

        import matplotlib.pyplot as plt

        # Get all horizons
        horizons = []
        for r in self.results:
            for key in r.metrics:
                if key.startswith("horizon_"):
                    h = int(key.split("_")[1])
                    if h not in horizons:
                        horizons.append(h)
        horizons = sorted(horizons)

        if not horizons:
            print("No horizon data found.")
            return None

        n_experiments = len(self.results)
        x = np.arange(len(horizons))
        width = 0.8 / n_experiments

        fig, ax = plt.subplots(figsize=self._figsize_bar)

        colors = plt.cm.Set2(np.linspace(0, 1, n_experiments))

        for i, r in enumerate(self.results):
            values = []
            for h in horizons:
                val = r.metrics.get(f"horizon_{h}", {}).get(metric, 0)
                values.append(val)

            offset = (i - n_experiments / 2 + 0.5) * width
            label = f"{r.model_name} ({r.config_label})"
            ax.bar(x + offset, values, width, label=label, color=colors[i],
                  alpha=0.85, edgecolor="black", linewidth=0.3)

        ax.set_xlabel("Horizon", fontsize=12)
        ax.set_ylabel(metric, fontsize=12)
        ax.set_title(f"{metric} by Prediction Horizon", fontsize=13, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([f"h{h}" for h in horizons], fontsize=10)
        ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
        ax.legend(fontsize=8, loc="best")
        ax.grid(axis="y", alpha=0.3)

        plt.tight_layout()
        save_path = self.output_dir / filename
        plt.savefig(save_path, dpi=self._dpi, bbox_inches="tight")
        plt.close()
        print(f"Saved: {save_path}")
        return str(save_path)

    def plot_zone_comparison(
        self,
        metric: str = "R2",
        filename: str = "zone_comparison.png"
    ) -> Optional[str]:
        """
        Compare estimation zone vs prediction zone performance (soft sensor specific).

        Automatically splits horizons by measurement_lag:
        - Estimation zone: horizon <= measurement_lag
        - Prediction zone: horizon > measurement_lag

        Args:
            metric: Metric to compare.
            filename: Output filename.

        Returns:
            Path to saved figure.
        """
        if not _check_matplotlib():
            return None

        import matplotlib.pyplot as plt

        # Filter soft sensor experiments
        ss_results = [r for r in self.results if r.is_soft_sensor and r.measurement_lag]
        if not ss_results:
            print("No soft sensor experiments with measurement_lag found.")
            return None

        fig, ax = plt.subplots(figsize=(10, 6))

        labels = []
        est_values = []
        pred_values = []

        for r in ss_results:
            lag = r.measurement_lag
            est_metrics = []
            pred_metrics = []

            for key, values in r.metrics.items():
                if not key.startswith("horizon_"):
                    continue
                h = int(key.split("_")[1])
                val = values.get(metric, None)
                if val is None:
                    continue
                if h <= lag:
                    est_metrics.append(val)
                else:
                    pred_metrics.append(val)

            est_avg = np.mean(est_metrics) if est_metrics else 0
            pred_avg = np.mean(pred_metrics) if pred_metrics else 0

            labels.append(f"{r.model_name}\n({r.config_label})")
            est_values.append(est_avg)
            pred_values.append(pred_avg)

        x = np.arange(len(labels))
        width = 0.35

        bars1 = ax.bar(x - width / 2, est_values, width, label="Estimation Zone",
                       color="#4CAF50", alpha=0.8, edgecolor="black", linewidth=0.5)
        bars2 = ax.bar(x + width / 2, pred_values, width, label="Prediction Zone",
                       color="#F44336", alpha=0.8, edgecolor="black", linewidth=0.5)

        ax.set_xlabel("Model (Configuration)", fontsize=11)
        ax.set_ylabel(metric, fontsize=12)
        ax.set_title(f"Estimation Zone vs Prediction Zone — {metric}", fontsize=13, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8, ha="center")
        ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
        ax.legend(fontsize=10)
        ax.grid(axis="y", alpha=0.3)

        # Add value labels
        for bar, val in zip(bars1, est_values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                   f"{val:.3f}", ha="center", va="bottom", fontsize=7)
        for bar, val in zip(bars2, pred_values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                   f"{val:.3f}", ha="center", va="bottom" if val >= 0 else "top", fontsize=7)

        plt.tight_layout()
        save_path = self.output_dir / filename
        plt.savefig(save_path, dpi=self._dpi, bbox_inches="tight")
        plt.close()
        print(f"Saved: {save_path}")
        return str(save_path)

    def plot_heatmap(
        self,
        metric: str = "R2",
        filename: str = "heatmap.png"
    ) -> Optional[str]:
        """
        Generate heatmap of model × horizon performance.

        Args:
            metric: Metric to visualize.
            filename: Output filename.

        Returns:
            Path to saved figure.
        """
        if not _check_matplotlib():
            return None

        import matplotlib.pyplot as plt
        import seaborn as sns

        # Build matrix: rows = experiments, cols = [overall, h1, h3, h6, ...]
        horizons = sorted(set(
            key for r in self.results for key in r.metrics if key.startswith("horizon_")
        ))
        columns = ["overall"] + horizons

        labels = []
        data = []
        for r in self.results:
            labels.append(f"{r.model_name} ({r.config_label})")
            row = []
            row.append(r.metrics.get("overall", {}).get(metric, np.nan))
            for h in horizons:
                row.append(r.metrics.get(h, {}).get(metric, np.nan))
            data.append(row)

        data = np.array(data)
        col_labels = ["Overall"] + [f"h{h.split('_')[1]}" for h in horizons]

        fig, ax = plt.subplots(figsize=self._figsize_heatmap)
        sns.heatmap(
            data, annot=True, fmt=".3f", cmap="RdYlGn",
            xticklabels=col_labels, yticklabels=labels,
            center=0, ax=ax, linewidths=0.5,
            annot_kws={"size": 9}
        )
        ax.set_title(f"{metric} Heatmap — Model × Horizon", fontsize=13, fontweight="bold")
        ax.set_xlabel("Horizon", fontsize=11)
        ax.set_ylabel("Experiment", fontsize=11)

        plt.tight_layout()
        save_path = self.output_dir / filename
        plt.savefig(save_path, dpi=self._dpi, bbox_inches="tight")
        plt.close()
        print(f"Saved: {save_path}")
        return str(save_path)

    def plot_predictions(
        self,
        top_k: int = 3,
        num_samples: int = 200,
        filename: str = "predictions.png"
    ) -> Optional[str]:
        """
        Plot prediction vs ground truth curves for top-k models at each eval horizon.

        Requires save_results=True during training (prediction.npy, targets.npy).

        Args:
            top_k: Number of best models to plot.
            num_samples: Number of time steps to show.
            filename: Output filename.

        Returns:
            Path to saved figure, or None if data unavailable.
        """
        if not _check_matplotlib():
            return None

        import matplotlib.pyplot as plt

        # Sort by overall R2 (descending)
        sorted_results = sorted(
            self.results,
            key=lambda r: r.metrics.get("overall", {}).get("R2", -999),
            reverse=True
        )[:top_k]

        # Check which have saved predictions
        plot_results = []
        for r in sorted_results:
            pred_path = os.path.join(r.experiment_dir, "test_results", "prediction.npy")
            target_path = os.path.join(r.experiment_dir, "test_results", "targets.npy")
            if os.path.exists(pred_path) and os.path.exists(target_path):
                plot_results.append(r)

        if not plot_results:
            print("No prediction results found. Set save_results=True in config to enable.")
            return None

        # Determine horizons to plot from eval_horizons in config
        # Default: use first, middle, last step of output_len
        sample_result = plot_results[0]
        output_len = sample_result.output_len or 1
        eval_horizons_cfg = sample_result.config.get("eval_horizons", None)

        if eval_horizons_cfg and output_len > 1:
            # Use configured eval_horizons (1-indexed in config)
            horizons_to_plot = [h - 1 for h in eval_horizons_cfg]  # convert to 0-indexed
            horizon_labels = [f"h{h}" for h in eval_horizons_cfg]
        elif output_len > 1:
            # Default: first, middle, last
            horizons_to_plot = [0, output_len // 2, output_len - 1]
            horizon_labels = [f"h{h+1}" for h in horizons_to_plot]
        else:
            horizons_to_plot = [0]
            horizon_labels = ["h1"]

        n_horizons = len(horizons_to_plot)
        n_models = len(plot_results)

        fig, axes = plt.subplots(
            n_models, n_horizons,
            figsize=(6 * n_horizons, 3.5 * n_models),
            squeeze=False
        )

        for row, r in enumerate(plot_results):
            pred_path = os.path.join(r.experiment_dir, "test_results", "prediction.npy")
            target_path = os.path.join(r.experiment_dir, "test_results", "targets.npy")

            try:
                preds = np.load(pred_path, allow_pickle=True)
                targets = np.load(target_path, allow_pickle=True)
            except Exception:
                try:
                    pred_size = os.path.getsize(pred_path)
                    target_size = os.path.getsize(target_path)
                    ol = r.output_len or 1
                    n_pred = pred_size // (4 * ol * 1)
                    n_target = target_size // (4 * ol * 1)
                    preds = np.memmap(pred_path, dtype=np.float32, mode='r',
                                     shape=(n_pred, ol, 1))
                    targets = np.memmap(target_path, dtype=np.float32, mode='r',
                                       shape=(n_target, ol, 1))
                except Exception as e2:
                    print(f"Warning: Failed to load predictions from {r.experiment_dir}: {e2}")
                    continue

            for col, (h_idx, h_label) in enumerate(zip(horizons_to_plot, horizon_labels)):
                ax = axes[row, col]

                # Extract specific horizon
                if preds.ndim == 3:
                    pred_h = preds[:, h_idx, 0]
                    target_h = targets[:, h_idx, 0]
                elif preds.ndim == 2:
                    pred_h = preds[:, h_idx]
                    target_h = targets[:, h_idx]
                else:
                    pred_h = preds.flatten()
                    target_h = targets.flatten()

                n = min(num_samples, len(pred_h))
                ax.plot(range(n), target_h[:n], label="Ground Truth",
                       color="#333333", linewidth=1.2)
                ax.plot(range(n), pred_h[:n], label="Prediction",
                       color="#2196F3", linewidth=1.0, alpha=0.8)

                # Title: model name + horizon
                r2_h = r.metrics.get(f"horizon_{h_idx+1}", {}).get("R2", None)
                r2_str = f"R²={r2_h:.3f}" if r2_h is not None else ""
                ax.set_title(f"{r.model_name} ({r.config_label}) @ {h_label}  {r2_str}",
                           fontsize=9)
                ax.grid(alpha=0.3)

                if row == 0:
                    ax.legend(fontsize=7, loc="upper right")
                if row == n_models - 1:
                    ax.set_xlabel("Sample Index", fontsize=9)
                if col == 0:
                    ax.set_ylabel("Value", fontsize=9)

        plt.suptitle("Prediction vs Ground Truth (Top Models × Horizons)",
                    fontsize=13, fontweight="bold", y=1.01)
        plt.tight_layout()
        save_path = self.output_dir / filename
        plt.savefig(save_path, dpi=self._dpi, bbox_inches="tight")
        plt.close()
        print(f"Saved: {save_path}")
        return str(save_path)
