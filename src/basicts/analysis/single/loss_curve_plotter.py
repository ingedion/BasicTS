"""Loss curve plotter for single experiment analysis.

Extracts loss data from TensorBoard event files or training logs,
and plots train/val/test loss curves with optional early stopping annotation.
"""

import glob
import os
import re
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

if TYPE_CHECKING:
    from .models import ExperimentConfig


class LossCurvePlotter:
    """从 TensorBoard 事件或训练日志中提取并绘制损失曲线。"""

    # Loss curve colors and labels
    LOSS_STYLES = {
        "train": {"color": "#1f77b4", "label": "Train Loss"},
        "val": {"color": "#ff7f0e", "label": "Val Loss"},
        "test": {"color": "#2ca02c", "label": "Test Loss"},
    }

    def __init__(self, ckpt_dir: str, config: "ExperimentConfig"):
        """
        Args:
            ckpt_dir: 实验检查点目录路径
            config: 解析后的实验配置
        """
        self.ckpt_dir = ckpt_dir
        self.config = config
        self._early_stop_epoch: Optional[int] = None

    @property
    def early_stop_epoch(self) -> Optional[int]:
        """Early stopping 触发的 epoch（None 表示未触发）。"""
        return self._early_stop_epoch

    def plot(self) -> Optional[Figure]:
        """绘制损失曲线图，返回 Figure 或 None（数据不可用时）。"""
        loss_data = self.extract_loss_data()

        if not loss_data:
            return None

        fig, ax = plt.subplots(figsize=(10, 6))

        for key, values in loss_data.items():
            if values and key in self.LOSS_STYLES:
                style = self.LOSS_STYLES[key]
                epochs = list(range(1, len(values) + 1))
                ax.plot(epochs, values, color=style["color"], label=style["label"], linewidth=1.5)

        # Check if any curves were actually plotted
        if not ax.get_lines():
            plt.close(fig)
            return None

        # Add early stopping vertical line
        if self._early_stop_epoch is not None:
            ax.axvline(
                x=self._early_stop_epoch,
                color="red",
                linestyle="--",
                linewidth=1.0,
                alpha=0.7,
                label=f"Early Stopping (epoch {self._early_stop_epoch})",
            )

        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title(f"Loss Curves - {self.config.model_name} on {self.config.dataset_name}")
        ax.legend()

        # X-axis: integer ticks starting from 1
        lines = ax.get_lines()
        if lines:
            all_x = []
            for line in lines:
                xdata = line.get_xdata()
                if len(xdata) > 0:
                    all_x.extend(xdata)
            if all_x:
                max_epoch = int(max(all_x))
                # Show reasonable number of ticks
                if max_epoch <= 20:
                    ax.set_xticks(range(1, max_epoch + 1))
                else:
                    step = max(1, max_epoch // 10)
                    ticks = list(range(1, max_epoch + 1, step))
                    if max_epoch not in ticks:
                        ticks.append(max_epoch)
                    ax.set_xticks(ticks)

        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        return fig

    def extract_loss_data(self) -> Dict[str, List[float]]:
        """提取损失数据，键为 'train', 'val', 'test'，值为各 epoch 的 loss 列表。"""
        # Try TensorBoard first
        loss_data, early_stop = self._try_tensorboard()
        if loss_data:
            self._early_stop_epoch = early_stop
            return loss_data

        # Fall back to training log
        loss_data, early_stop = self._try_training_log()
        if loss_data:
            self._early_stop_epoch = early_stop
            return loss_data

        return {}

    def _try_tensorboard(self) -> Tuple[Dict[str, List[float]], Optional[int]]:
        """Attempt to read loss data from TensorBoard event files.

        Returns:
            Tuple of (loss_data dict, early_stop_epoch or None)
        """
        tb_dir = os.path.join(self.ckpt_dir, "tensorboard")
        if not os.path.isdir(tb_dir):
            return {}, None

        # Check for event files
        event_files = glob.glob(os.path.join(tb_dir, "events.out.tfevents.*"))
        if not event_files:
            return {}, None

        try:
            from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        except ImportError:
            # tensorboard package not available, fall back to log parsing
            return {}, None

        try:
            ea = EventAccumulator(tb_dir)
            ea.Reload()

            scalar_tags = ea.Tags().get("scalars", [])
            loss_data: Dict[str, List[float]] = {}

            # Map TensorBoard tag names to our keys
            tag_map = {
                "train/loss": "train",
                "val/loss": "val",
                "test/loss": "test",
            }

            for tag, key in tag_map.items():
                if tag in scalar_tags:
                    events = ea.Scalars(tag)
                    if events:
                        loss_data[key] = [e.value for e in events]

            # Try to detect early stopping from text summaries or log
            early_stop = self._detect_early_stop_from_log()

            if loss_data:
                return loss_data, early_stop

        except Exception:
            # If TensorBoard reading fails for any reason, fall back
            pass

        return {}, None

    def _try_training_log(self) -> Tuple[Dict[str, List[float]], Optional[int]]:
        """Attempt to read loss data from the most recent training log file.

        Returns:
            Tuple of (loss_data dict, early_stop_epoch or None)
        """
        log_file = self._find_latest_log_file()
        if log_file is None:
            return {}, None

        try:
            with open(log_file, "r", encoding="utf-8") as f:
                content = f.read()
        except (IOError, OSError):
            return {}, None

        return self._parse_training_log(content)

    def _find_latest_log_file(self) -> Optional[str]:
        """Find the most recent training_log_*.log file in the checkpoint directory."""
        log_pattern = os.path.join(self.ckpt_dir, "training_log_*.log")
        log_files = glob.glob(log_pattern)

        if not log_files:
            return None

        # Sort by filename (timestamp in name) to get the most recent
        log_files.sort()
        return log_files[-1]

    def _parse_training_log(self, content: str) -> Tuple[Dict[str, List[float]], Optional[int]]:
        """Parse training log content to extract loss values and early stopping info.

        The BasicTS log format has lines like:
            Result <train>: [train/time: 0.96 (s), train/lr: 5.00e-04, train/loss: 0.9724, ...]
            Result <val>: [val/time: 0.06 (s), val/loss: 1.3912, ...]
            Result <test>: [test/time: 0.13 (s), test/loss: 0.8379, ...]

        And early stopping:
            Early stopping at epoch 49.

        Args:
            content: Full text content of the training log file

        Returns:
            Tuple of (loss_data dict, early_stop_epoch or None)
        """
        train_losses: List[float] = []
        val_losses: List[float] = []
        test_losses: List[float] = []
        early_stop_epoch: Optional[int] = None

        # Regex patterns for loss extraction
        # Match: Result <train>: [..., train/loss: 0.9724, ...]
        train_loss_pattern = re.compile(r"Result <train>:.*?train/loss:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)")
        val_loss_pattern = re.compile(r"Result <val>:.*?val/loss:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)")
        test_loss_pattern = re.compile(r"Result <test>:.*?test/loss:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)")

        # Early stopping pattern
        early_stop_pattern = re.compile(r"Early stopping at epoch (\d+)")

        # Track which epoch we're in to avoid duplicates from "best model" evaluation
        current_epoch: Optional[int] = None
        epoch_pattern = re.compile(r"Epoch (\d+)\s*/\s*\d+")
        training_finished = False

        for line in content.split("\n"):
            # Detect training finished (after this, results are from best model evaluation)
            if "The training finished at" in line:
                training_finished = True
                continue

            # Skip results after training finished (best model evaluation)
            if training_finished:
                continue

            # Track current epoch
            epoch_match = epoch_pattern.search(line)
            if epoch_match:
                current_epoch = int(epoch_match.group(1))
                continue

            # Only extract losses when we're in a valid epoch context
            if current_epoch is not None:
                # Extract train loss (only from per-epoch results, not horizon results)
                train_match = train_loss_pattern.search(line)
                if train_match and "@ horizon" not in line:
                    train_losses.append(float(train_match.group(1)))

                # Extract val loss
                val_match = val_loss_pattern.search(line)
                if val_match and "@ horizon" not in line:
                    val_losses.append(float(val_match.group(1)))

                # Extract test loss (only overall, not per-horizon)
                test_match = test_loss_pattern.search(line)
                if test_match and "@ horizon" not in line:
                    test_losses.append(float(test_match.group(1)))

            # Detect early stopping
            early_match = early_stop_pattern.search(line)
            if early_match:
                early_stop_epoch = int(early_match.group(1))

        # Build result dict with only available data
        loss_data: Dict[str, List[float]] = {}
        if train_losses:
            loss_data["train"] = train_losses
        if val_losses:
            loss_data["val"] = val_losses
        if test_losses:
            loss_data["test"] = test_losses

        return loss_data, early_stop_epoch

    def _detect_early_stop_from_log(self) -> Optional[int]:
        """Detect early stopping epoch from training log (used as supplement for TensorBoard)."""
        log_file = self._find_latest_log_file()
        if log_file is None:
            return None

        try:
            with open(log_file, "r", encoding="utf-8") as f:
                content = f.read()
        except (IOError, OSError):
            return None

        early_stop_pattern = re.compile(r"Early stopping at epoch (\d+)")
        match = early_stop_pattern.search(content)
        if match:
            return int(match.group(1))
        return None
