"""
TimeXer Predictive Soft Sensor on EthyDistillation - Target Included.

Configuration: exclude_target_from_input=False
Input: 38 variables, Target: 塔顶乙烷浓度 (index=20)
Predictive mode: lag=6, output=12 (6 estimation + 6 prediction)
"""

from torch.optim.lr_scheduler import MultiStepLR

from basicts import BasicTSLauncher
from basicts.configs import BasicTSSoftSensorConfig
from basicts.metrics import masked_mse
from basicts.models.TimeXer import TimeXer, TimeXerConfig
from basicts.runners.callback import EarlyStopping, GradientClipping


def main():

    model_config = TimeXerConfig(
        input_len=96,
        output_len=12,
        num_features=38,
        patch_len=16,
        hidden_size=256,
        n_heads=4,
        intermediate_size=512,
        hidden_act="gelu",
        num_layers=2,
        dropout=0.1,
        use_revin=True,
        output_attentions=False
    )

    BasicTSLauncher.launch_training(BasicTSSoftSensorConfig(
        model=TimeXer,
        model_config=model_config,
        dataset_name="EthyDistillation",

        # Soft sensor specific
        target_vars=20,
        input_vars=None,
        exclude_target_from_input=False,
        measurement_lag=6,
        input_len=96,
        output_len=12,

        # Checkpoint
        ckpt_save_dir="checkpoints/EthyDistillation_benchmark/TimeXer",

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
