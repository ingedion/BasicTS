"""
Single Experiment Analysis Module.

Provides training report generation for individual experiments,
including loss curves, metric curves, prediction plots, and structured reports.

Usage:
    from basicts.analysis.single import SingleExperimentReporter

    reporter = SingleExperimentReporter("checkpoints/Dataset/Model/hash_dir")
    reporter.generate_report()

CLI:
    python -m basicts.analysis.single --ckpt_dir checkpoints/Dataset/Model/hash_dir
"""

try:
    from .reporter import SingleExperimentReporter
except ImportError:
    # reporter.py will be implemented in a later task
    SingleExperimentReporter = None  # type: ignore[assignment, misc]

__all__ = ["SingleExperimentReporter"]
