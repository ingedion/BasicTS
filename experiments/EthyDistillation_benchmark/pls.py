"""
PLS (Partial Least Squares) Baseline on EthyDistillation.

Purpose: Establish the linear multivariate baseline — how strong is the
         linear correlation between process variables and quality variable.

Configuration:
- exclude_target_from_input=True (pure soft sensor)
- Input: 37 process variables
- Target: 塔顶乙烷浓度 (index=20)
- Uses NonGradientRunner for one-shot sklearn fit
"""

from basicts import BasicTSLauncher
from basicts.configs import BasicTSSoftSensorConfig
from basicts.metrics import masked_mse
from basicts.models.PLS import PLS, PLSConfig
from basicts.runners import NonGradientRunner


def main():

    model_config = PLSConfig(
        input_len=96,
        output_len=12,
        num_features=37,
        num_targets=1,
        n_components=None,  # auto-determine
    )

    BasicTSLauncher.launch_training(BasicTSSoftSensorConfig(
        model=PLS,
        model_config=model_config,
        dataset_name="EthyDistillation",

        # Runner — non-gradient
        runner=NonGradientRunner,

        # Soft sensor specific
        target_vars=20,
        input_vars=None,
        exclude_target_from_input=True,
        measurement_lag=6,
        input_len=96,
        output_len=12,

        # Checkpoint
        ckpt_save_dir="checkpoints/EthyDistillation_benchmark/PLS",

        # Training (mostly ignored for non-gradient)
        gpus=None,
        seed=42,
        num_epochs=1,
        batch_size=64,
        metrics=["MAE", "MSE", "RMSE", "R2"],
        loss=masked_mse,
        eval_horizons=[1, 6, 12],
        save_results=True,
    ))


if __name__ == "__main__":
    main()
