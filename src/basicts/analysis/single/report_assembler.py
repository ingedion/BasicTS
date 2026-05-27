"""Report assembler for single experiment analysis.

Composes all analysis results into a structured Markdown report,
including experiment config summary, loss curves, metric curves,
final metrics table, and prediction plots.
"""

import os
from datetime import datetime
from typing import TYPE_CHECKING, List, Optional

if TYPE_CHECKING:
    from .models import ExperimentConfig


class ReportAssembler:
    """将各分析结果组装为 Markdown 报告。"""

    def __init__(self, config: "ExperimentConfig", output_dir: str):
        """
        Args:
            config: 实验配置
            output_dir: 报告输出目录路径
        """
        self.config = config
        self.output_dir = output_dir

    def assemble(
        self,
        loss_fig_path: Optional[str],
        metric_fig_path: Optional[str],
        prediction_fig_path: Optional[str],
        metrics_table: str,
        skipped_sections: List[str],
    ) -> str:
        """组装报告并保存，返回报告文件绝对路径。

        Creates report/ and report/fig/ directories if needed, composes
        the Markdown report with all sections, and saves as report.md.

        Args:
            loss_fig_path: Absolute path to loss curves PNG, or None if skipped
            metric_fig_path: Absolute path to metric curves PNG, or None if skipped
            prediction_fig_path: Absolute path to prediction plots PNG, or None if skipped
            metrics_table: Markdown-formatted metrics table string
            skipped_sections: List of section names that were skipped

        Returns:
            Absolute path to the generated report.md file
        """
        # Create output directories
        report_dir = self.output_dir
        fig_dir = os.path.join(report_dir, "fig")
        os.makedirs(report_dir, exist_ok=True)
        os.makedirs(fig_dir, exist_ok=True)

        # Compose report sections
        sections: List[str] = []

        # Header and experiment config summary
        sections.append(self._build_header())

        # Loss curves section
        sections.append(self._build_loss_section(loss_fig_path, skipped_sections))

        # Metric curves section
        sections.append(self._build_metric_section(metric_fig_path, skipped_sections))

        # Final metrics table section
        sections.append(self._build_metrics_table_section(metrics_table, skipped_sections))

        # Prediction plots section
        sections.append(self._build_prediction_section(prediction_fig_path, skipped_sections))

        # Assemble full report
        report_content = "\n".join(sections)

        # Save report
        report_path = os.path.join(report_dir, "report.md")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report_content)

        return os.path.abspath(report_path)

    def _build_header(self) -> str:
        """Build the report header with experiment config summary."""
        generation_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        lines = [
            f"# Single Experiment Report: {self.config.model_name} on {self.config.dataset_name}",
            "",
            "## Experiment Configuration",
            "",
            f"| Parameter | Value |",
            f"|-----------|-------|",
            f"| Generation Time | {generation_time} |",
            f"| Model Name | {self.config.model_name} |",
            f"| Dataset Name | {self.config.dataset_name} |",
            f"| Input Length | {self.config.input_len} |",
            f"| Output Length | {self.config.output_len} |",
            f"| Measurement Lag | {self.config.measurement_lag} |",
            "",
        ]
        return "\n".join(lines)

    def _build_loss_section(self, loss_fig_path: Optional[str], skipped_sections: List[str]) -> str:
        """Build the loss curves section."""
        lines = ["## Loss Curves", ""]

        if "loss_curves" in skipped_sections or loss_fig_path is None:
            lines.append("> ⚠️ 该部分数据不可用：损失曲线数据缺失。")
            lines.append("")
        else:
            filename = os.path.basename(loss_fig_path)
            lines.append(f"![Loss Curves](fig/{filename})")
            lines.append("")

        return "\n".join(lines)

    def _build_metric_section(self, metric_fig_path: Optional[str], skipped_sections: List[str]) -> str:
        """Build the metric curves section."""
        lines = ["## Metric Curves", ""]

        if "metric_curves" in skipped_sections or metric_fig_path is None:
            lines.append("> ⚠️ 该部分数据不可用：指标曲线数据缺失。")
            lines.append("")
        else:
            filename = os.path.basename(metric_fig_path)
            lines.append(f"![Metric Curves](fig/{filename})")
            lines.append("")

        return "\n".join(lines)

    def _build_metrics_table_section(self, metrics_table: str, skipped_sections: List[str]) -> str:
        """Build the final metrics table section."""
        lines = ["## Final Metrics", ""]

        if "metrics_table" in skipped_sections or not metrics_table:
            lines.append("> ⚠️ 该部分数据不可用：指标数据缺失。")
            lines.append("")
        else:
            lines.append(metrics_table)
            lines.append("")

        return "\n".join(lines)

    def _build_prediction_section(self, prediction_fig_path: Optional[str], skipped_sections: List[str]) -> str:
        """Build the prediction plots section."""
        lines = ["## Prediction Plots", ""]

        if "prediction_plots" in skipped_sections or prediction_fig_path is None:
            lines.append("> ⚠️ 该部分数据不可用：预测曲线数据缺失。")
            lines.append("")
        else:
            filename = os.path.basename(prediction_fig_path)
            lines.append(f"![Prediction Plots](fig/{filename})")
            lines.append("")

        return "\n".join(lines)
