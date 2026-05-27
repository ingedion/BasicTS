"""Single experiment reporter - main entry point for report generation.

Orchestrates all analysis components (CheckpointValidator, LossCurvePlotter,
MetricCurvePlotter, MetricSummarizer, PredictionPlotter, ReportAssembler)
to generate a complete structured training report for a single experiment.
"""

import os
from typing import List, Optional

import matplotlib
matplotlib.use("Agg")
from matplotlib.figure import Figure

from .checkpoint_validator import CheckpointValidator
from .loss_curve_plotter import LossCurvePlotter
from .metric_curve_plotter import MetricCurvePlotter
from .metric_summarizer import MetricSummarizer
from .models import ExperimentConfig
from .prediction_plotter import PredictionPlotter
from .report_assembler import ReportAssembler


class SingleExperimentReporter:
    """单次实验报告生成器。"""

    def __init__(self, ckpt_dir: str, output_dir: Optional[str] = None, num_samples: int = 200):
        """
        Args:
            ckpt_dir: 实验检查点目录路径（哈希目录或模型目录）
            output_dir: 报告输出目录，默认为 {ckpt_dir}/report/
            num_samples: 预测曲线展示的时间步数量，默认 200

        Raises:
            FileNotFoundError: 如果 ckpt_dir 路径无效或缺少必需文件
            NotADirectoryError: 如果 ckpt_dir 不是目录
            ValueError: 如果 cfg.json 格式无效
        """
        # Resolve checkpoint directory (handles model dir -> latest hash dir)
        resolved_dir = CheckpointValidator.resolve_checkpoint_dir(ckpt_dir)

        # Validate and parse config
        validator = CheckpointValidator(resolved_dir)
        self.config: ExperimentConfig = validator.validate()

        # Store output directory and num_samples
        self.output_dir = output_dir if output_dir else os.path.join(self.config.ckpt_dir, "report")
        self.num_samples = num_samples

    def generate_report(self) -> str:
        """生成完整报告，返回报告文件绝对路径。

        Orchestrates all components: plots loss curves, metric curves,
        prediction curves, summarizes metrics, and assembles the final
        Markdown report.

        Returns:
            Absolute path to the generated report.md file.
        """
        fig_dir = os.path.join(self.output_dir, "fig")
        os.makedirs(fig_dir, exist_ok=True)

        skipped_sections: List[str] = []

        # 1. Plot loss curves
        loss_fig_path: Optional[str] = None
        loss_fig = self.plot_loss_curves()
        if loss_fig is not None:
            loss_fig_path = os.path.join(fig_dir, "loss_curves.png")
            loss_fig.savefig(loss_fig_path, dpi=150, bbox_inches="tight")
            import matplotlib.pyplot as plt
            plt.close(loss_fig)
        else:
            skipped_sections.append("loss_curves")

        # 2. Plot metric curves
        metric_fig_path: Optional[str] = None
        metric_fig = self.plot_metric_curves()
        if metric_fig is not None:
            metric_fig_path = os.path.join(fig_dir, "metric_curves.png")
            metric_fig.savefig(metric_fig_path, dpi=150, bbox_inches="tight")
            import matplotlib.pyplot as plt
            plt.close(metric_fig)
        else:
            skipped_sections.append("metric_curves")

        # 3. Summarize metrics (generate markdown table)
        summarizer = MetricSummarizer(self.config.ckpt_dir, self.config)
        metrics_table = summarizer.to_markdown_table()
        if not metrics_table or "不可用" in metrics_table:
            skipped_sections.append("metrics_table")

        # 4. Plot predictions
        prediction_fig_path: Optional[str] = None
        prediction_fig = self.plot_predictions()
        if prediction_fig is not None:
            prediction_fig_path = os.path.join(fig_dir, "prediction_curves.png")
            prediction_fig.savefig(prediction_fig_path, dpi=150, bbox_inches="tight")
            import matplotlib.pyplot as plt
            plt.close(prediction_fig)
        else:
            skipped_sections.append("prediction_plots")

        # 5. Assemble report
        assembler = ReportAssembler(self.config, self.output_dir)
        report_path = assembler.assemble(
            loss_fig_path=loss_fig_path,
            metric_fig_path=metric_fig_path,
            prediction_fig_path=prediction_fig_path,
            metrics_table=metrics_table,
            skipped_sections=skipped_sections,
        )

        # Print report path to console (Requirement 6.6)
        print(f"Report generated: {report_path}")

        return report_path

    def plot_loss_curves(self) -> Optional[Figure]:
        """绘制损失曲线，返回 matplotlib Figure 对象。

        Returns:
            matplotlib Figure object, or None if loss data is unavailable.
        """
        plotter = LossCurvePlotter(self.config.ckpt_dir, self.config)
        return plotter.plot()

    def plot_metric_curves(self) -> Optional[Figure]:
        """绘制指标曲线，返回 matplotlib Figure 对象。

        Returns:
            matplotlib Figure object, or None if metric data is unavailable.
        """
        plotter = MetricCurvePlotter(self.config.ckpt_dir, self.config)
        return plotter.plot()

    def plot_predictions(self) -> Optional[Figure]:
        """绘制预测对比图，返回 matplotlib Figure 对象。

        Returns:
            matplotlib Figure object, or None if prediction data is unavailable.
        """
        plotter = PredictionPlotter(self.config.ckpt_dir, self.config, self.num_samples)
        return plotter.plot()

    def summarize_metrics(self) -> dict:
        """返回测试集评估指标字典。

        Returns:
            Dictionary containing 'overall' key with metric values and
            'horizon_N' keys with per-horizon metric values. Returns
            empty dict if test_metrics.json is unavailable.
        """
        summarizer = MetricSummarizer(self.config.ckpt_dir, self.config)
        return summarizer.summarize()
