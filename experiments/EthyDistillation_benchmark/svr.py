"""
SVR (Support Vector Regression) Baseline on EthyDistillation.

Purpose: Establish the nonlinear static baseline — how much additional
         prediction power does nonlinear modeling provide over PLS.

Configuration:
- exclude_target_from_input=True (pure soft sensor)
- Input: 37 process variables
- Target: 塔顶乙烷浓度 (index=20)
- RBF kernel for nonlinear mapping
- Uses NonGradientRunner for one-shot sklearn fit

Note: EthyDistillation has ~4700 training samples with 37*96=3552 input features.
      SVR may be slow due to high dimensionality. C and epsilon tuned for this scale.
"""

from basicts import BasicTSLauncher
from basicts.configs import BasicTSSoftSensorConfig
from basicts.metrics import masked_mse
from basicts.models.SVR import SVR, SVRConfig
from basicts.runners import NonGradientRunner


def main():

    model_config = SVRConfig(
        input_len=96,
        output_len=12,
        num_features=37,
        num_targets=1,
        kernel="rbf",
        C=10.0,
        epsilon=0.01,
        gamma="scale",
    )

    BasicTSLauncher.launch_training(BasicTSSoftSensorConfig(
        model=SVR,
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
        ckpt_save_dir="checkpoints/EthyDistillation_benchmark/SVR",

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
