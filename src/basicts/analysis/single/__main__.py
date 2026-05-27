"""CLI entry point for single experiment report generation.

Usage:
    python -m basicts.analysis.single --ckpt_dir <path> [--output_dir <path>] [--num_samples <int>]

Exit codes:
    0: Success
    1: Invalid --ckpt_dir (path does not exist or is not a directory)
    2: Invalid --num_samples (value out of range [1, 10000])
"""

import argparse
import os
import sys


def main():
    parser = argparse.ArgumentParser(
        description="Generate a structured training report for a single experiment checkpoint.",
        prog="python -m basicts.analysis.single",
    )

    parser.add_argument(
        "--ckpt_dir",
        type=str,
        required=True,
        help="Path to the experiment checkpoint directory (required). "
             "Can be a hash directory containing cfg.json, or a model directory "
             "containing hash subdirectories.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for the generated report. "
             "Default: {ckpt_dir}/report/",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=200,
        help="Number of time steps to display in prediction curves. "
             "Must be in range [1, 10000]. Default: 200",
    )

    args = parser.parse_args()

    # Validate --ckpt_dir exists and is a directory
    if not os.path.exists(args.ckpt_dir):
        print(f"Error: --ckpt_dir path does not exist: {args.ckpt_dir}", file=sys.stderr)
        sys.exit(1)
    if not os.path.isdir(args.ckpt_dir):
        print(f"Error: --ckpt_dir path is not a directory: {args.ckpt_dir}", file=sys.stderr)
        sys.exit(1)

    # Validate --num_samples range [1, 10000]
    if args.num_samples < 1 or args.num_samples > 10000:
        print(
            f"Error: --num_samples value {args.num_samples} is out of valid range [1, 10000].",
            file=sys.stderr,
        )
        sys.exit(2)

    # Import here to avoid import overhead when just showing help or validation fails
    from .reporter import SingleExperimentReporter

    reporter = SingleExperimentReporter(
        ckpt_dir=args.ckpt_dir,
        output_dir=args.output_dir,
        num_samples=args.num_samples,
    )
    reporter.generate_report()


if __name__ == "__main__":
    main()
