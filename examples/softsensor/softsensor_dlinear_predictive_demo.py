"""
Predictive Soft Sensor Demo with DLinear on ETTh1.

This demo demonstrates the "predictive soft sensor" mode where:
- measurement_lag = 6: quality variable has 6-step measurement delay
- output_len = 12: model outputs 12 steps total
  - Steps 1~6 (estimation zone): compensate for measurement lag
  - Steps 7~12 (prediction zone): genuine short-term forecasting

Target variable: OT (last column, index -1) of ETTh1 dataset.
Input variables: all other 6 process variables (exclude_target_from_input=True).
"""

from torch.optim.lr_scheduler import MultiStepLR

from basicts import BasicTSLauncher
from basicts.configs import BasicTSSoftSensorConfig
from basicts.metrics import masked_mse
from basicts.models.DLinear import DLinear, DLinearConfig
from basicts.runners.callback import EarlyStopping, GradientClipping


def main():

    model_config = DLinearConfig(
        input_len=96,
        output_len=12,
        num_features=6,       # 6 input process variables (OT excluded)
        moving_avg=25,
        stride=1,
        individual=False
    )

    BasicTSLauncher.launch_training(BasicTSSoftSensorConfig(
        model=DLinear,
        model_config=model_config,
        dataset_name="ETTh1",

        # Soft sensor specific
        target_vars=-1,                   # predict OT (last variable)
        input_vars=None,                  # auto: all vars except target
        exclude_target_from_input=True,   # use only process variables as input
        measurement_lag=6,                # quality variable has 6-step lag
        input_len=96,
        output_len=12,                    # 6 steps estimation + 6 steps prediction

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

        # Evaluation at specific horizons to compare estimation vs prediction zones
        eval_horizons=[1, 6, 12],         # h1 (estimation start), h6 (lag boundary), h12 (prediction end)
    ))


if __name__ == "__main__":
    main()
