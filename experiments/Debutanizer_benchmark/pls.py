"""
PLS (Partial Least Squares) Baseline on Debutanizer.

Purpose: Establish the linear multivariate baseline — how strong is the
         linear correlation between process variables and quality variable.

Configuration:
- exclude_target_from_input=True (pure soft sensor, no autoregression)
- Input: 7 process variables (u1~u7)
- Target: y (butane content)
- Uses NonGradientRunner for one-shot sklearn fit
"""

from basicts import BasicTSLauncher
from basicts.configs import BasicTSSoftSensorConfig
from basicts.metrics import masked_mse
from basicts.models.PLS import PLS, PLSConfig
from basicts.runners import NonGradientRunner


def main():

    model_config = PLSConfig(
        input_len=32,
        output_len=6,
        num_features=7,
        num_targets=1,
        n_components=None,  # auto-determine
    )

    BasicTSLauncher.launch_training(BasicTSSoftSensorConfig(
        model=PLS,
        model_config=model_config,
        dataset_name="Debutanizer",

        # Runner — non-gradient
        runner=NonGradientRunner,

        # Soft sensor specific
        target_vars=-1,
        input_vars=None,
        exclude_target_from_input=True,
        measurement_lag=3,
        input_len=32,
        output_len=6,

        # Checkpoint
        ckpt_save_dir="checkpoints/Debutanizer_benchmark/PLS",

        # Training (mostly ignored for non-gradient, but needed for config validation)
        gpus=None,  # PLS runs on CPU
        seed=42,
        num_epochs=1,
        batch_size=64,
        metrics=["MAE", "MSE", "RMSE", "R2"],
        loss=masked_mse,
        eval_horizons=[1, 3, 6],
        save_results=True,
    ))


if __name__ == "__main__":
    main()
