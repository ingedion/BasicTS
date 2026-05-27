"""
BasicTS Analysis Module.

Provides experiment post-processing, visualization, and reporting tools
for multi-model benchmark experiments and single experiment analysis.

Usage:
    from basicts.analysis import ExperimentAnalyzer

    analyzer = ExperimentAnalyzer("checkpoints/Debutanizer_benchmark")
    analyzer.collect()
    analyzer.plot_overall_comparison()
    analyzer.generate_report()

    from basicts.analysis import SingleExperimentReporter

    reporter = SingleExperimentReporter("checkpoints/Dataset/Model/hash_dir")
    reporter.generate_report()

CLI:
    python -m basicts.analysis --experiment_dir checkpoints/Debutanizer_benchmark
    python -m basicts.analysis.single --ckpt_dir checkpoints/Dataset/Model/hash_dir
"""

from .analyzer import ExperimentAnalyzer
from .single import SingleExperimentReporter

__all__ = ["ExperimentAnalyzer", "SingleExperimentReporter"]
