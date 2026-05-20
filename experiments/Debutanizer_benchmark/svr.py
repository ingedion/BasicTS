"""
SVR (Support Vector Regression) Baseline on Debutanizer.

Purpose: Establish the nonlinear static baseline — how much additional
         prediction power does nonlinear modeling provide over PLS.

Configuration:
- exclude_target_from_input=True (pure soft sensor, no autoregression)
- Input: 7 process variables (u1~u7)
- Target: y (butane content)
- RBF kernel for nonlinear mapping
- Uses NonGradientRunner for one-shot sklearn fit
"""

from basicts import BasicTSLauncher
from basicts.configs import BasicTSSoftSensorConfig
from basicts.metrics import masked_mse
from basicts.models.SVR import SVR, SVRConfig
from basicts.runners import NonGradientRunner


def main():

    model_config = SVRConfig(
        input_len=32,
        output_len=6,
        num_features=7,
        num_targets=1,
        kernel="rbf",
        C=10.0,
        epsilon=0.01,
        gamma="scale",
    )

    BasicTSLauncher.launch_training(BasicTSSoftSensorConfig(
        model=SVR,
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
        ckpt_save_dir="checkpoints/Debutanizer_benchmark/SVR",

        # Training (mostly ignored for non-gradient, but needed for config validation)
        gpus=None,  # SVR runs on CPU
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
