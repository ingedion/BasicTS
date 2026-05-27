"""Data models for single experiment analysis."""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class ExperimentConfig:
    """从 cfg.json 解析的实验配置。"""

    model_name: str           # model.name
    dataset_name: str         # dataset_name
    input_len: int            # model_config.input_len
    output_len: int           # model_config.output_len
    measurement_lag: int      # measurement_lag（默认 0）
    metrics: List[str]        # metrics 列表（如 ["MAE", "MSE", "RMSE", "R2"]）
    eval_horizons: List[int]  # eval_horizons 列表（如 [1, 3, 6]）
    target_vars: List[int]    # target_vars 列表（如 [6]）
    is_soft_sensor: bool      # 是否为软测量任务
    ckpt_dir: str             # 检查点目录绝对路径
    raw_config: Dict[str, Any]  # 原始 cfg.json 内容


@dataclass
class TrainingCurveData:
    """训练过程中的曲线数据。"""

    epochs: List[int]                          # epoch 编号列表
    losses: Dict[str, List[float]]             # {'train': [...], 'val': [...], 'test': [...]}
    metrics: Dict[str, Dict[str, List[float]]] # {'MAE': {'train': [...], 'val': [...]}, ...}
    early_stop_epoch: Optional[int]            # Early stopping 触发的 epoch（None 表示未触发）
