"""
LSTM Baseline on EthyDistillation - Target Excluded.

Purpose: Establish the nonlinear temporal baseline — how much additional
         prediction power does temporal modeling provide over static methods.

Configuration:
- exclude_target_from_input=True (pure soft sensor)
- Input: 37 process variables
- Target: 塔顶乙烷浓度 (index=20)
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
        input_len=96,
        output_len=12,
        num_features=37,
        num_targets=1,
        hidden_size=128,
        num_layers=2,
        dropout=0.2,
        bidirectional=False,
    )

    BasicTSLauncher.launch_training(BasicTSSoftSensorConfig(
        model=LSTM,
        model_config=model_config,
        dataset_name="EthyDistillation",

        # Soft sensor specific
        target_vars=20,
        input_vars=None,
        exclude_target_from_input=True,
        measurement_lag=6,
        input_len=96,
        output_len=12,

        # Checkpoint
        ckpt_save_dir="checkpoints/EthyDistillation_benchmark/LSTM_excl",

        # Training
        gpus="0",
        callbacks=[EarlyStopping(patience=15), GradientClipping(1.0)],
        seed=42,
        num_epochs=100,
        batch_size=64,
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
        eval_horizons=[1, 6, 12],
        save_results=True,
    ))


if __name__ == "__main__":
    main()
