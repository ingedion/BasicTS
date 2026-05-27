"""Metric summarizer for single experiment analysis.

Reads test_metrics.json and generates structured metric summaries
including overall metrics, per-horizon metrics, and zone averages
for soft-sensor tasks.
"""

import json
import os
from typing import Dict, List, Optional

from .models import ExperimentConfig


class MetricSummarizer:
    """读取 test_metrics.json 并生成结构化摘要。"""

    def __init__(self, ckpt_dir: str, config: ExperimentConfig):
        """
        Args:
            ckpt_dir: 实验检查点目录路径
            config: 实验配置
        """
        self.ckpt_dir = ckpt_dir
        self.config = config
        self._metrics_path = os.path.join(ckpt_dir, "test_metrics.json")

    def summarize(self) -> dict:
        """返回指标字典，包含 'overall' 和各 'horizon_N' 键。

        Returns:
            dict with "overall" key mapping to metric dict, and "horizon_N" keys
            mapping to per-horizon metric dicts. Returns empty dict if file is
            missing or invalid.
        """
        try:
            with open(self._metrics_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            return {}

        if not isinstance(data, dict):
            return {}

        result = {}

        # Extract overall metrics
        if "overall" in data and isinstance(data["overall"], dict):
            result["overall"] = data["overall"]

        # Extract horizon metrics
        for key, value in data.items():
            if key.startswith("horizon_") and isinstance(value, dict):
                result[key] = value

        return result

    def to_markdown_table(self) -> str:
        """生成 Markdown 格式的指标表格。

        Returns:
            Markdown string containing overall metrics table and per-horizon R2 table.
            Returns placeholder text if data is unavailable.
        """
        metrics_data = self.summarize()

        if not metrics_data:
            return "> ⚠️ 指标数据不可用：test_metrics.json 文件缺失或格式无效。\n"

        sections: List[str] = []

        # Overall metrics table
        if "overall" in metrics_data:
            overall = metrics_data["overall"]
            sections.append("### Overall Metrics\n")
            sections.append("| Metric | Value |")
            sections.append("|--------|-------|")
            for metric_name in ["MAE", "MSE", "RMSE", "R2"]:
                if metric_name in overall:
                    value = overall[metric_name]
                    sections.append(f"| {metric_name} | {value:.4f} |")
            sections.append("")

        # Per-horizon R2 table
        horizon_keys = sorted(
            [k for k in metrics_data if k.startswith("horizon_")],
            key=lambda k: int(k.split("_")[1])
        )

        if horizon_keys:
            sections.append("### Per-Horizon R2\n")
            # Build header
            horizons = [k.split("_")[1] for k in horizon_keys]
            header = "| Model | " + " | ".join(f"Horizon {h}" for h in horizons) + " |"
            separator = "|-------|" + "|".join("--------" for _ in horizons) + "|"
            sections.append(header)
            sections.append(separator)

            # Build data row
            model_name = self.config.model_name
            r2_values = []
            for key in horizon_keys:
                r2 = metrics_data[key].get("R2")
                if r2 is not None:
                    r2_values.append(f"{r2:.4f}")
                else:
                    r2_values.append("N/A")
            row = f"| {model_name} | " + " | ".join(r2_values) + " |"
            sections.append(row)
            sections.append("")

        # Zone averages (for soft-sensor tasks)
        zone_averages = self.compute_zone_averages()
        if zone_averages is not None:
            sections.append("### Zone Averages (Soft-Sensor)\n")
            sections.append("| Zone | Avg R2 |")
            sections.append("|------|--------|")
            sections.append(f"| Estimation (horizon ≤ {self.config.measurement_lag}) | {zone_averages['estimation_zone_r2']:.4f} |")
            sections.append(f"| Prediction (horizon > {self.config.measurement_lag}) | {zone_averages['prediction_zone_r2']:.4f} |")
            sections.append(f"| Δ (Estimation - Prediction) | {zone_averages['delta']:.4f} |")
            sections.append("")

        return "\n".join(sections)

    def compute_zone_averages(self) -> Optional[Dict[str, float]]:
        """计算估计区间和预测区间的平均 R2（仅软测量任务）。

        For soft-sensor tasks (measurement_lag >= 1), computes:
        - estimation zone: average R2 for horizons <= measurement_lag
        - prediction zone: average R2 for horizons > measurement_lag
        - delta: estimation zone avg R2 - prediction zone avg R2

        Returns:
            Dict with keys 'estimation_zone_r2', 'prediction_zone_r2', 'delta',
            or None if not a soft-sensor task or data is unavailable.
        """
        if not self.config.is_soft_sensor or self.config.measurement_lag < 1:
            return None

        metrics_data = self.summarize()
        if not metrics_data:
            return None

        measurement_lag = self.config.measurement_lag

        estimation_r2_values: List[float] = []
        prediction_r2_values: List[float] = []

        # Collect R2 values for each horizon
        for key, value in metrics_data.items():
            if not key.startswith("horizon_"):
                continue
            try:
                horizon = int(key.split("_")[1])
            except (IndexError, ValueError):
                continue

            r2 = value.get("R2")
            if r2 is None:
                continue

            if horizon <= measurement_lag:
                estimation_r2_values.append(r2)
            else:
                prediction_r2_values.append(r2)

        # Need at least one value in each zone to compute averages
        if not estimation_r2_values or not prediction_r2_values:
            return None

        estimation_avg = sum(estimation_r2_values) / len(estimation_r2_values)
        prediction_avg = sum(prediction_r2_values) / len(prediction_r2_values)
        delta = estimation_avg - prediction_avg

        return {
            "estimation_zone_r2": estimation_avg,
            "prediction_zone_r2": prediction_avg,
            "delta": delta,
        }
