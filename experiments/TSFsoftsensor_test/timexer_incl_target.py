"""
TimeXer Soft Sensor Experiment.

Configuration: exclude_target_from_input=False (必须包含目标变量)
Input: 7 variables (OT included)

NOTE: TimeXer 虽然通过 cross-attention 引入外生变量信息，但本质仍是 encoder-only 架构，
其输出通道数等于输入通道数（N→N）。当 exclude_target_from_input=True 时，目标变量
不在输入中，模型无法为其生成预测通道，且 postprocess 中用原始 target_vars 索引提取
预测会导致索引错位。因此 TimeXer 不支持 exclude_target_from_input=True 模式。
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
        num_features=7,
        patch_len=16,
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
        model=TimeXer,
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
