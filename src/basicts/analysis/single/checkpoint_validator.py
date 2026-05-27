"""Checkpoint directory validation and configuration parsing."""

import json
import os
from pathlib import Path
from typing import Any, Dict

from .models import ExperimentConfig


class CheckpointValidator:
    """验证检查点目录完整性并解析配置。"""

    REQUIRED_FILES = ["cfg.json", "test_metrics.json"]

    def __init__(self, ckpt_dir: str):
        """
        Args:
            ckpt_dir: 实验检查点目录路径（哈希目录或模型目录）
        """
        self.ckpt_dir = ckpt_dir

    def validate(self) -> ExperimentConfig:
        """验证目录并返回解析后的配置。失败时抛出异常。

        Raises:
            FileNotFoundError: 如果必需文件缺失
            ValueError: 如果 cfg.json 无法解析为有效 JSON
        """
        ckpt_path = Path(self.ckpt_dir)

        # Check existence of required files
        for filename in self.REQUIRED_FILES:
            filepath = ckpt_path / filename
            if not filepath.exists():
                raise FileNotFoundError(
                    f"Required file '{filename}' not found at expected path: {filepath}"
                )

        # Parse cfg.json
        cfg_path = ckpt_path / "cfg.json"
        try:
            with open(cfg_path, "r", encoding="utf-8") as f:
                raw_config = json.load(f)
        except (json.JSONDecodeError, ValueError) as e:
            raise ValueError(
                f"Invalid JSON format in file: {cfg_path}"
            ) from e

        # Extract fields from config
        model_name = raw_config.get("model", {}).get("name", "")
        dataset_name = raw_config.get("dataset_name", "")
        input_len = raw_config.get("model_config", {}).get("input_len", 0)
        output_len = raw_config.get("model_config", {}).get("output_len", 0)
        measurement_lag = raw_config.get("measurement_lag", 0)
        metrics = raw_config.get("metrics", [])
        eval_horizons = raw_config.get("eval_horizons", [])
        target_vars = raw_config.get("target_vars", [])
        is_soft_sensor = measurement_lag >= 1

        return ExperimentConfig(
            model_name=model_name,
            dataset_name=dataset_name,
            input_len=input_len,
            output_len=output_len,
            measurement_lag=measurement_lag,
            metrics=metrics,
            eval_horizons=eval_horizons,
            target_vars=target_vars,
            is_soft_sensor=is_soft_sensor,
            ckpt_dir=str(ckpt_path.resolve()),
            raw_config=raw_config,
        )

    @staticmethod
    def resolve_checkpoint_dir(path: str) -> str:
        """
        解析检查点目录路径。
        如果 path 是哈希目录（含 cfg.json），直接返回。
        如果 path 是模型目录（含子目录），返回修改时间最晚的子目录。

        Args:
            path: 检查点目录路径

        Returns:
            解析后的检查点目录路径

        Raises:
            FileNotFoundError: 如果路径不存在或模型目录下无子目录
        """
        dir_path = Path(path)

        if not dir_path.exists():
            raise FileNotFoundError(f"Path does not exist: {path}")

        if not dir_path.is_dir():
            raise NotADirectoryError(f"Path is not a directory: {path}")

        # If path contains cfg.json, it's already a hash directory
        if (dir_path / "cfg.json").exists():
            return str(dir_path)

        # Otherwise, find the subdirectory with the most recent modification time
        subdirs = [d for d in dir_path.iterdir() if d.is_dir()]

        if not subdirs:
            raise FileNotFoundError(
                f"No experiment checkpoints found in model directory: {path}"
            )

        # Select subdirectory with the most recent modification time
        latest_subdir = max(subdirs, key=lambda d: os.path.getmtime(d))
        return str(latest_subdir)
