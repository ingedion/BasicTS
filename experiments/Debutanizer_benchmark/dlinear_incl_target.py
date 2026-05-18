"""
DLinear Predictive Soft Sensor on Debutanizer - Target Included.

Configuration: exclude_target_from_input=False
Input: 8 variables (u1~u7 + y), Target: y (butane content)
Predictive mode: lag=3, output=6 (3 estimation + 3 prediction)
"""

from torch.optim.lr_scheduler import MultiStepLR

from basicts import BasicTSLauncher
from basicts.configs import BasicTSSoftSensorConfig
from basicts.metrics import masked_mse
from basicts.models.DLinear import DLinear, DLinearConfig
from basicts.runners.callback import EarlyStopping, GradientClipping


def main():

    model_config = DLinearConfig(
        input_len=32,
        output_len=6,
        num_features=8,
        moving_avg=13,
        stride=1,
        individual=False
    )

    BasicTSLauncher.launch_training(BasicTSSoftSensorConfig(
        model=DLinear,
        model_config=model_config,
        dataset_name="Debutanizer",

        # Soft sensor specific
        target_vars=-1,
        input_vars=None,
        exclude_target_from_input=False,
        measurement_lag=3,
        input_len=32,
        output_len=6,

        # Checkpoint
        ckpt_save_dir="checkpoints/Debutanizer_benchmark/DLinear",

        # Training
        gpus="0",
        callbacks=[EarlyStopping(patience=15), GradientClipping(1.0)],
        seed=42,
        num_epochs=100,
        batch_size=32,
        metrics=["MAE", "MSE", "RMSE", "R2"],
        loss=masked_mse,
        optimizer_params={
            "lr": 1e-3,
            "weight_decay": 1e-4
        },
        lr_scheduler=MultiStepLR,
        lr_scheduler_params={
            "milestones": [30, 60],
            "gamma": 0.5
        },
        eval_horizons=[1, 3, 6],
    ))


if __name__ == "__main__":
    main()
