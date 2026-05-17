from torch.optim.lr_scheduler import MultiStepLR

from basicts import BasicTSLauncher
from basicts.configs import BasicTSSoftSensorConfig
from basicts.metrics import masked_mse
from basicts.models.DLinear import DLinear, DLinearConfig
from basicts.runners.callback import EarlyStopping, GradientClipping


def main():

    # train DLinear for soft sensor task on ETTh1
    # target_var=-1 means predicting the last variable (OT) using other variables
    # measurement_lag=1 means quality variable has 1-step measurement lag

    model_config = DLinearConfig(
        input_len=96,
        output_len=1,
        num_features=7,
        moving_avg=25,
        stride=1,
        individual=False
    )

    BasicTSLauncher.launch_training(BasicTSSoftSensorConfig(
        model=DLinear,
        model_config=model_config,
        dataset_name="ETTh1",
        target_vars=-1,
        input_vars=None,
        exclude_target_from_input=False,
        input_len=96,
        output_len=1,
        measurement_lag=1,
        gpus="0",
        callbacks=[EarlyStopping(), GradientClipping(1.0)],
        seed=233,
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
            "milestones": [25, 50],
            "gamma": 0.5
        }
    ))


if __name__ == "__main__":
    main()

