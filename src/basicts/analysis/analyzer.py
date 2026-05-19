"""
Experiment Analyzer — Main Entry Point.

Combines collector, visualizer, and reporter into a unified interface.
"""

from pathlib import Path
from typing import Dict, List, Optional

from .collector import ExperimentResult, ResultCollector
from .reporter import ExperimentReporter
from .visualizer import ExperimentVisualizer


class ExperimentAnalyzer:
    """
    Unified interface for experiment analysis.

    Usage:
        analyzer = ExperimentAnalyzer("checkpoints/Debutanizer_benchmark")
        analyzer.collect()
        analyzer.plot_overall_comparison()
        analyzer.plot_horizon_comparison()
        analyzer.plot_zone_comparison()
        analyzer.generate_report()

    Or use the one-shot method:
        analyzer.run_all()
    """

    def __init__(
        self,
        experiment_dir: str,
        output_dir: Optional[str] = None,
    ):
        """
        Initialize analyzer.

        Args:
            experiment_dir: Path to checkpoint directory containing experiment results.
            output_dir: Path to save analysis outputs. Default: {experiment_dir}/analysis/
        """
        self.experiment_dir = Path(experiment_dir)

        if output_dir is None:
            self.output_dir = self.experiment_dir / "analysis"
        else:
            self.output_dir = Path(output_dir)

        self._collector = ResultCollector(str(self.experiment_dir))
        self._visualizer: Optional[ExperimentVisualizer] = None
        self._reporter: Optional[ExperimentReporter] = None
        self._results: List[ExperimentResult] = []
        self._chart_files: Dict[str, str] = {}

    @property
    def results(self) -> List[ExperimentResult]:
        """Get collected results."""
        return self._results

    def collect(self) -> "ExperimentAnalyzer":
        """
        Scan experiment directory and collect all results.

        Returns:
            self (for chaining)
        """
        self._results = self._collector.collect()
        self._visualizer = ExperimentVisualizer(self._results, str(self.output_dir))
        self._reporter = ExperimentReporter(self._results, str(self.output_dir))
        return self

    def to_dataframe(self):
        """Convert results to pandas DataFrame."""
        return self._collector.to_dataframe()

    def plot_overall_comparison(self, metrics: Optional[List[str]] = None) -> Optional[str]:
        """Generate overall metric comparison chart."""
        self._ensure_collected()
        path = self._visualizer.plot_overall_comparison(metrics=metrics)
        if path:
            self._chart_files["Overall Comparison"] = path
        return path

    def plot_horizon_comparison(self, metric: str = "R2") -> Optional[str]:
        """Generate horizon-wise comparison chart."""
        self._ensure_collected()
        path = self._visualizer.plot_horizon_comparison(metric=metric)
        if path:
            self._chart_files["Horizon Comparison"] = path
        return path

    def plot_zone_comparison(self, metric: str = "R2") -> Optional[str]:
        """Generate estimation vs prediction zone comparison (soft sensor)."""
        self._ensure_collected()
        path = self._visualizer.plot_zone_comparison(metric=metric)
        if path:
            self._chart_files["Zone Comparison"] = path
        return path

    def plot_heatmap(self, metric: str = "R2") -> Optional[str]:
        """Generate model × horizon heatmap."""
        self._ensure_collected()
        path = self._visualizer.plot_heatmap(metric=metric)
        if path:
            self._chart_files["Performance Heatmap"] = path
        return path

    def plot_predictions(self, top_k: int = 3, num_samples: int = 200) -> Optional[str]:
        """Plot prediction curves for top-k models (requires save_results=True)."""
        self._ensure_collected()
        path = self._visualizer.plot_predictions(top_k=top_k, num_samples=num_samples)
        if path:
            self._chart_files["Prediction Curves"] = path
        return path

    def generate_report(self, filename: str = "experiment_report.md") -> str:
        """Generate markdown report with all charts."""
        self._ensure_collected()
        return self._reporter.generate_report(
            chart_files=self._chart_files,
            filename=filename
        )

    def run_all(self, metrics: Optional[List[str]] = None) -> str:
        """
        Run complete analysis pipeline: collect → visualize → report.

        Args:
            metrics: Metrics to include in overall comparison. Default: ["R2", "RMSE"]

        Returns:
            Path to generated report.
        """
        self.collect()

        print(f"\n{'='*60}")
        print(f"Analyzing {len(self._results)} experiments")
        print(f"Output: {self.output_dir}")
        print(f"{'='*60}\n")

        # Generate all charts
        self.plot_overall_comparison(metrics=metrics)
        self.plot_horizon_comparison(metric="R2")
        self.plot_heatmap(metric="R2")

        # Soft sensor specific
        if any(r.is_soft_sensor for r in self._results):
            self.plot_zone_comparison(metric="R2")

        # Prediction curves (if available)
        self.plot_predictions()

        # Generate report
        report_path = self.generate_report()

        print(f"\n{'='*60}")
        print(f"Analysis complete. Report: {report_path}")
        print(f"{'='*60}\n")

        return report_path

    def _ensure_collected(self):
        """Ensure results have been collected."""
        if not self._results:
            raise RuntimeError("No results collected. Call .collect() first.")
