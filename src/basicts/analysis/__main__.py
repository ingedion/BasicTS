"""
CLI entry point for BasicTS Analysis Module.

Usage:
    python -m basicts.analysis --experiment_dir checkpoints/Debutanizer_benchmark
    python -m basicts.analysis --experiment_dir checkpoints/Debutanizer_benchmark --output_dir results/analysis
    python -m basicts.analysis --experiment_dir checkpoints/Debutanizer_benchmark --metrics R2 RMSE MAE
"""

import argparse
import sys

from .analyzer import ExperimentAnalyzer


def main():
    parser = argparse.ArgumentParser(
        description="BasicTS Experiment Analysis Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m basicts.analysis --experiment_dir checkpoints/Debutanizer_benchmark
  python -m basicts.analysis --experiment_dir checkpoints/Debutanizer_benchmark --output_dir analysis_output
  python -m basicts.analysis --experiment_dir checkpoints/Debutanizer_benchmark --metrics R2 RMSE
        """
    )

    parser.add_argument(
        "--experiment_dir", "-e",
        type=str, required=True,
        help="Path to experiment checkpoint directory."
    )
    parser.add_argument(
        "--output_dir", "-o",
        type=str, default=None,
        help="Path to save analysis outputs. Default: {experiment_dir}/analysis/"
    )
    parser.add_argument(
        "--metrics", "-m",
        nargs="+", default=None,
        help="Metrics to include in comparison. Default: R2 RMSE"
    )

    args = parser.parse_args()

    analyzer = ExperimentAnalyzer(
        experiment_dir=args.experiment_dir,
        output_dir=args.output_dir,
    )

    report_path = analyzer.run_all(metrics=args.metrics)
    print(f"\nDone. Report: {report_path}")


if __name__ == "__main__":
    main()
