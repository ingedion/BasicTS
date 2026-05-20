"""
NLinear Univariate Autoregressive Baseline on Debutanizer.

Purpose: Establish the autoregressive reference — how much prediction power
         can be obtained from the target variable's own history alone.

Configuration:
- input_vars=[7] (only target variable y)
- target_vars=[7] (predict y)
- exclude_target_from_input=False (irrelevant since input_vars is explicit)
- NLinear: per-channel linear mapping with last-value normalization
"""

from torch.optim.lr_scheduler import MultiStepLR

from basicts import BasicTSLauncher
from basicts.configs import BasicTSSoftSensorConfig
from basicts.metrics import masked_mse
from basicts.models.NLinear import NLinear, NLinearConfig
from basicts.runners.callback import EarlyStopping, GradientClipping


def main():

    model_config = NLinearConfig(
        input_len=32,
        output_len=6,
    )

    BasicTSLauncher.launch_training(BasicTSSoftSensorConfig(
        model=NLinear,
        model_config=model_config,
        dataset_name="Debutanizer",

        # Soft sensor specific — single variable autoregressive
        target_vars=-1,
        input_vars=[7],  # only target variable (y is column 7)
        exclude_target_from_input=False,
        measurement_lag=3,
        input_len=32,
        output_len=6,

        # Checkpoint
        ckpt_save_dir="checkpoints/Debutanizer_benchmark/NLinear_univariate",

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
        save_results=True,
    ))


if __name__ == "__main__":
    main()
