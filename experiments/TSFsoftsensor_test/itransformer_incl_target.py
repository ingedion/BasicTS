"""
iTransformer Soft Sensor Experiment - Target Included.

Configuration: exclude_target_from_input=False
Input: 7 variables (OT included)
"""

from torch.optim.lr_scheduler import MultiStepLR

from basicts import BasicTSLauncher
from basicts.configs import BasicTSSoftSensorConfig
from basicts.metrics import masked_mse
from basicts.models.iTransformer import iTransformerForForecasting, iTransformerConfig
from basicts.runners.callback import EarlyStopping, GradientClipping


def main():

    model_config = iTransformerConfig(
        input_len=96,
        output_len=12,
        num_features=7,
        hidden_size=256,
        n_heads=4,
        intermediate_size=1024,
        hidden_act="gelu",
        num_layers=2,
        dropout=0.1,
        use_revin=True,
        output_attentions=False
    )

    BasicTSLauncher.launch_training(BasicTSSoftSensorConfig(
        model=iTransformerForForecasting,
        model_config=model_config,
        dataset_name="ETTh1",

        # Soft sensor specific
        target_vars=-1,
        input_vars=None,
        exclude_target_from_input=False,
        measurement_lag=6,
        input_len=96,
        output_len=12,

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
    ))


if __name__ == "__main__":
    main()
