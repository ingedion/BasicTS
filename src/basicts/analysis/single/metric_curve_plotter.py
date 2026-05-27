"""Metric curve plotter for single experiment analysis.

Extracts per-epoch metric values (MAE, MSE, RMSE, R2, etc.) from TensorBoard
event files or training logs, and generates subplot figures showing train/val
curves for each configured metric.
"""

import glob
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from .models import ExperimentConfig


class MetricCurvePlotter:
    """绘制各评估指标随 epoch 变化的曲线。"""

    def __init__(self, ckpt_dir: str, config: ExperimentConfig):
        """
        Args:
            ckpt_dir: 实验检查点目录路径
            config: 解析后的实验配置
        """
        self.ckpt_dir = ckpt_dir
        self.config = config

    def extract_metric_data(self) -> Dict[str, Dict[str, List[float]]]:
        """提取指标数据。外层键为指标名，内层键为 'train'/'val'。

        优先从 TensorBoard 事件文件提取，回退到训练日志解析。

        Returns:
            嵌套字典，例如:
            {
                "MAE": {"train": [0.15, 0.14, ...], "val": [0.19, 0.18, ...]},
                "RMSE": {"train": [...], "val": [...]},
            }
        """
        # Try TensorBoard first
        data = self._extract_from_tensorboard()
        if data:
            return data

        # Fall back to training log
        data = self._extract_from_training_log()
        return data

    def _extract_from_tensorboard(self) -> Dict[str, Dict[str, List[float]]]:
        """从 TensorBoard 事件文件中提取指标数据。

        Returns:
            指标数据字典，如果 TensorBoard 不可用则返回空字典。
        """
        tb_dir = Path(self.ckpt_dir) / "tensorboard"
        if not tb_dir.exists():
            return {}

        event_files = list(tb_dir.glob("events.out.tfevents.*"))
        if not event_files:
            return {}

        try:
            from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        except ImportError:
            return {}

        result: Dict[str, Dict[str, List[float]]] = {}
        metrics = self.config.metrics

        # Use the latest event file
        latest_event = max(event_files, key=lambda f: os.path.getmtime(f))

        try:
            ea = EventAccumulator(str(latest_event))
            ea.Reload()
            available_tags = ea.Tags().get("scalars", [])

            for metric_name in metrics:
                metric_data: Dict[str, List[float]] = {}

                # TensorBoard tags are typically "train/MAE", "val/MAE"
                train_tag = f"train/{metric_name}"
                val_tag = f"val/{metric_name}"

                if train_tag in available_tags:
                    events = ea.Scalars(train_tag)
                    metric_data["train"] = [e.value for e in events]

                if val_tag in available_tags:
                    events = ea.Scalars(val_tag)
                    metric_data["val"] = [e.value for e in events]

                if metric_data:
                    result[metric_name] = metric_data

        except Exception:
            return {}

        return result

    def _extract_from_training_log(self) -> Dict[str, Dict[str, List[float]]]:
        """从训练日志文件中提取指标数据。

        Returns:
            指标数据字典，如果日志不可用则返回空字典。
        """
        log_dir = Path(self.ckpt_dir)
        log_files = sorted(log_dir.glob("training_log_*.log"))

        if not log_files:
            return {}

        # Use the most recent log file (last in sorted order by timestamp name)
        latest_log = log_files[-1]

        result: Dict[str, Dict[str, List[float]]] = {}
        metrics = self.config.metrics

        # Initialize result structure
        for metric_name in metrics:
            result[metric_name] = {"train": [], "val": []}

        try:
            with open(latest_log, "r", encoding="utf-8") as f:
                for line in f:
                    # Parse train result lines
                    # Format: Result <train>: [train/time: ..., train/MAE: 0.1506, ...]
                    if "Result <train>:" in line:
                        self._parse_metric_line(line, "train", metrics, result)
                    # Parse val result lines
                    # Format: Result <val>: [val/time: ..., val/MAE: 0.1959, ...]
                    elif "Result <val>:" in line:
                        self._parse_metric_line(line, "val", metrics, result)
        except (IOError, OSError):
            return {}

        # Remove metrics that have no data
        result = {
            metric_name: data
            for metric_name, data in result.items()
            if data.get("train") or data.get("val")
        }

        # Remove empty splits within metrics
        for metric_name in result:
            result[metric_name] = {
                split: values
                for split, values in result[metric_name].items()
                if values
            }

        return result

    def _parse_metric_line(
        self,
        line: str,
        split: str,
        metrics: List[str],
        result: Dict[str, Dict[str, List[float]]],
    ) -> None:
        """从日志行中解析指标值。

        Args:
            line: 日志行文本
            split: 数据集划分 ('train' 或 'val')
            metrics: 要提取的指标名称列表
            result: 结果字典（就地修改）
        """
        for metric_name in metrics:
            # Pattern: train/MAE: 0.1506 or val/MAE: 0.1959
            pattern = rf"{split}/{metric_name}:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"
            match = re.search(pattern, line)
            if match:
                try:
                    value = float(match.group(1))
                    result[metric_name][split].append(value)
                except (ValueError, KeyError):
                    pass

    def plot(self) -> Optional[Figure]:
        """绘制指标曲线子图，返回 Figure 或 None。

        为每个有数据的指标生成一个子图，每个子图中展示 train 和 val 曲线。
        使用 2 列网格布局。

        Returns:
            matplotlib Figure 对象，如果无数据则返回 None。
        """
        metric_data = self.extract_metric_data()

        if not metric_data:
            return None

        # Determine grid layout (2 columns)
        num_metrics = len(metric_data)
        ncols = 2
        nrows = (num_metrics + ncols - 1) // ncols  # ceiling division

        fig, axes = plt.subplots(
            nrows, ncols, figsize=(6 * ncols, 4 * nrows), squeeze=False
        )

        colors = {"train": "#1f77b4", "val": "#ff7f0e"}
        labels = {"train": "Train", "val": "Val"}

        for idx, (metric_name, data) in enumerate(metric_data.items()):
            row = idx // ncols
            col = idx % ncols
            ax = axes[row][col]

            for split in ["train", "val"]:
                if split in data and data[split]:
                    values = data[split]
                    epochs = list(range(1, len(values) + 1))
                    ax.plot(
                        epochs,
                        values,
                        color=colors[split],
                        label=f"{labels[split]} {metric_name}",
                        linewidth=1.5,
                    )

            ax.set_xlabel("Epoch")
            ax.set_ylabel(metric_name)
            ax.set_title(metric_name)
            ax.legend()
            ax.grid(True, alpha=0.3)

            # Ensure X-axis shows integer ticks starting from 1
            ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

        # Hide unused subplots
        for idx in range(num_metrics, nrows * ncols):
            row = idx // ncols
            col = idx % ncols
            axes[row][col].set_visible(False)

        fig.tight_layout()
        return fig
