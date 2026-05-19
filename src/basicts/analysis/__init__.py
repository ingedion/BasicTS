"""
BasicTS Analysis Module.

Provides experiment post-processing, visualization, and reporting tools
for multi-model benchmark experiments.

Usage:
    from basicts.analysis import ExperimentAnalyzer

    analyzer = ExperimentAnalyzer("checkpoints/Debutanizer_benchmark")
    analyzer.collect()
    analyzer.plot_overall_comparison()
    analyzer.generate_report()

CLI:
    python -m basicts.analysis --experiment_dir checkpoints/Debutanizer_benchmark
"""

from .analyzer import ExperimentAnalyzer

__all__ = ["ExperimentAnalyzer"]
