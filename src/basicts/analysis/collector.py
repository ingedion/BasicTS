"""
Experiment Result Collector.

Scans checkpoint directories, reads cfg.json and test_metrics.json,
and organizes results into a structured DataFrame for analysis.
"""

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class ExperimentResult:
    """Single experiment result container."""

    model_name: str
    experiment_dir: str
    config: Dict[str, Any]
    metrics: Dict[str, Any]

    # Derived fields
    num_features: int = 0
    exclude_target: Optional[bool] = None
    measurement_lag: Optional[int] = None
    input_len: Optional[int] = None
    output_len: Optional[int] = None
    dataset_name: str = ""
    taskflow_name: str = ""

    # Short name mapping for display
    _SHORT_NAMES = {
        "PatchTSTForForecasting": "PatchTST",
        "PatchTSTForClassification": "PatchTST",
        "PatchTSTForReconstruction": "PatchTST",
        "iTransformerForForecasting": "iTransformer",
        "iTransformerForClassification": "iTransformer",
        "iTransformerForReconstruction": "iTransformer",
    }

    def __post_init__(self):
        """Extract key fields from config."""
        raw_name = self.config.get("model", {}).get("name", "Unknown")
        self.model_name = self._SHORT_NAMES.get(raw_name, raw_name)
        self.num_features = self.config.get("model_config", {}).get("num_features", 0)
        self.exclude_target = self.config.get("exclude_target_from_input", None)
        self.measurement_lag = self.config.get("measurement_lag", None)
        self.input_len = self.config.get("model_config", {}).get("input_len", None)
        self.output_len = self.config.get("model_config", {}).get("output_len", None)
        self.dataset_name = self.config.get("dataset_name", "")
        self.taskflow_name = self.config.get("taskflow", {}).get("name", "")

    @property
    def is_soft_sensor(self) -> bool:
        """Check if this experiment is a soft sensor task."""
        return "SoftSensor" in self.taskflow_name

    @property
    def config_label(self) -> str:
        """Generate a human-readable label for this configuration."""
        if self.exclude_target is not None:
            return "excl_target" if self.exclude_target else "incl_target"
        return f"feat{self.num_features}"

    @property
    def display_name(self) -> str:
        """Short display name for plots."""
        return f"{self.model_name}\n({self.config_label})"


class ResultCollector:
    """
    Scans experiment directories and collects results.

    Expects directory structure:
        experiment_dir/
        ├── ModelA/
        │   ├── hash1/
        │   │   ├── cfg.json
        │   │   └── test_metrics.json
        │   └── hash2/
        │       ├── cfg.json
        │       └── test_metrics.json
        └── ModelB/
            └── ...
    """

    def __init__(self, experiment_dir: str):
        self.experiment_dir = Path(experiment_dir)
        self.results: List[ExperimentResult] = []

    def collect(self) -> List[ExperimentResult]:
        """
        Scan experiment directory and collect all results.

        Returns:
            List of ExperimentResult objects.
        """
        self.results = []

        if not self.experiment_dir.exists():
            raise FileNotFoundError(f"Experiment directory not found: {self.experiment_dir}")

        # Walk through directory structure
        for model_dir in sorted(self.experiment_dir.iterdir()):
            if not model_dir.is_dir():
                continue

            for hash_dir in sorted(model_dir.iterdir()):
                if not hash_dir.is_dir():
                    continue

                cfg_path = hash_dir / "cfg.json"
                metrics_path = hash_dir / "test_metrics.json"

                if not cfg_path.exists() or not metrics_path.exists():
                    continue

                try:
                    with open(cfg_path, "r", encoding="utf-8") as f:
                        # Handle NaN in JSON (BasicTS uses NaN which is not valid JSON)
                        content = f.read().replace("NaN", "null")
                        config = json.loads(content)

                    with open(metrics_path, "r", encoding="utf-8") as f:
                        metrics = json.load(f)

                    result = ExperimentResult(
                        model_name="",
                        experiment_dir=str(hash_dir),
                        config=config,
                        metrics=metrics,
                    )
                    self.results.append(result)

                except (json.JSONDecodeError, KeyError) as e:
                    print(f"Warning: Failed to parse {hash_dir}: {e}")
                    continue

        print(f"Collected {len(self.results)} experiment results from {self.experiment_dir}")

        # Sort by overall R2 descending (best first)
        self.results.sort(
            key=lambda r: r.metrics.get("overall", {}).get("R2", -999),
            reverse=True
        )

        return self.results

    def to_dataframe(self):
        """
        Convert results to a pandas DataFrame for easy analysis.

        Returns:
            pandas.DataFrame with one row per experiment.
        """
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("pandas is required for DataFrame conversion. Install with: pip install pandas")

        rows = []
        for r in self.results:
            row = {
                "model": r.model_name,
                "config": r.config_label,
                "display_name": r.display_name,
                "dataset": r.dataset_name,
                "exclude_target": r.exclude_target,
                "num_features": r.num_features,
                "measurement_lag": r.measurement_lag,
                "input_len": r.input_len,
                "output_len": r.output_len,
                "is_soft_sensor": r.is_soft_sensor,
            }

            # Flatten overall metrics
            overall = r.metrics.get("overall", {})
            for k, v in overall.items():
                row[f"overall_{k}"] = v

            # Flatten horizon metrics
            for key, values in r.metrics.items():
                if key.startswith("horizon_"):
                    horizon = key.replace("horizon_", "h")
                    for k, v in values.items():
                        row[f"{horizon}_{k}"] = v

            rows.append(row)

        return pd.DataFrame(rows)

    def get_horizon_names(self) -> List[str]:
        """Get all horizon names present in results."""
        horizons = set()
        for r in self.results:
            for key in r.metrics:
                if key.startswith("horizon_"):
                    horizons.add(key)
        return sorted(horizons)

    def get_metric_names(self) -> List[str]:
        """Get all metric names present in results."""
        metrics = set()
        for r in self.results:
            overall = r.metrics.get("overall", {})
            metrics.update(overall.keys())
        return sorted(metrics)
