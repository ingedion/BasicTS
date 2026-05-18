"""
PatchTST Predictive Soft Sensor on Debutanizer - Target Excluded.

Configuration: exclude_target_from_input=True (pure soft sensor)
Input: 7 process variables (u1~u7), Target: y (butane content)
Predictive mode: lag=3, output=6 (3 estimation + 3 prediction)
"""

from torch.optim.lr_scheduler import MultiStepLR

from basicts import BasicTSLauncher
from basicts.configs import BasicTSSoftSensorConfig
from basicts.metrics import masked_mse
from basicts.models.PatchTST import PatchTSTForForecasting, PatchTSTConfig
from basicts.runners.callback import EarlyStopping, GradientClipping


def main():

    model_config = PatchTSTConfig(
        input_len=32,
        output_len=6,
        num_features=7,
        patch_len=8,
        patch_stride=4,
        padding=True,
        hidden_size=128,
        n_heads=4,
        intermediate_size=256,
        hidden_act="gelu",
        num_layers=2,
        attn_dropout=0.1,
        fc_dropout=0.1,
        head_dropout=0.0,
        norm_type="layer_norm",
        individual_head=False,
        use_revin=True,
        affine=True,
        subtract_last=False,
        decomp=False,
        moving_avg=13,
        output_attentions=False
    )

    BasicTSLauncher.launch_training(BasicTSSoftSensorConfig(
        model=PatchTSTForForecasting,
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
        ckpt_save_dir="checkpoints/Debutanizer_benchmark/PatchTSTForForecasting",

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
