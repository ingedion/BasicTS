"""
LSTM Baseline on Debutanizer - Target Excluded.

Purpose: Establish the nonlinear temporal baseline — how much additional
         prediction power does temporal modeling provide over static methods.

Configuration:
- exclude_target_from_input=True (pure soft sensor)
- Input: 7 process variables (u1~u7)
- Target: y (butane content)
- Standard LSTM encoder → Linear head
"""

from torch.optim.lr_scheduler import MultiStepLR

from basicts import BasicTSLauncher
from basicts.configs import BasicTSSoftSensorConfig
from basicts.metrics import masked_mse
from basicts.models.LSTM import LSTM, LSTMConfig
from basicts.runners.callback import EarlyStopping, GradientClipping


def main():

    model_config = LSTMConfig(
        input_len=32,
        output_len=6,
        num_features=7,
        num_targets=1,
        hidden_size=64,
        num_layers=2,
        dropout=0.1,
        bidirectional=False,
    )

    BasicTSLauncher.launch_training(BasicTSSoftSensorConfig(
        model=LSTM,
        model_config=model_config,
        dataset_name="Debutanizer",

        # Soft sensor specific
        target_vars=-1,
        input_vars=None,
        exclude_target_from_input=True,
        measurement_lag=3,
        input_len=32,
        output_len=6,

        # Checkpoint
        ckpt_save_dir="checkpoints/Debutanizer_benchmark/LSTM_excl",

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
