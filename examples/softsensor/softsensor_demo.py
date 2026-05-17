from torch.optim.lr_scheduler import MultiStepLR

from basicts import BasicTSLauncher
from basicts.configs import BasicTSSoftSensorConfig
from basicts.metrics import masked_mse
from basicts.models.iTransformer import iTransformerConfig, iTransformerForForecasting
from basicts.runners.callback import EarlyStopping, GradientClipping


def main():

    # train iTransformer for soft sensor task on ETTh1
    # target_var=0 means predicting the first variable (HUFL) using other variables
    # measurement_lag=12 means quality variable has 12-step measurement lag
    
    model_config = iTransformerConfig(
        num_features=7,  # use all features including target variable for autoregression
        hidden_size=32,
        intermediate_size=32,
        n_heads=1,
        num_layers=1,
        dropout=0.1,
        use_revin=True  # Can use RevIN when including target in input
    )

    BasicTSLauncher.launch_training(BasicTSSoftSensorConfig(
        model=iTransformerForForecasting,
        model_config=model_config,
        dataset_name="ETTh1",
        target_vars=-1,  # predict the last variable (OT)
        input_vars=None,  # automatically determined by exclude_target_from_input
        exclude_target_from_input=False,  # Include target variable in input (autoregression)
        input_len=96,
        output_len=1,  # predict 1 step ahead
        measurement_lag=1,  # 1-step measurement lag
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
