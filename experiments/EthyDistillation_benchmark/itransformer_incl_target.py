"""
iTransformer Predictive Soft Sensor on EthyDistillation.

Configuration: exclude_target_from_input=False (必须包含目标变量)
Input: 38 variables, Target: 塔顶乙烷浓度 (index=20)
Predictive mode: lag=6, output=12 (6 estimation + 6 prediction)

NOTE: iTransformer 虽然通过 inverted attention 实现了跨通道建模，但其 forecasting head
采用 N 输入通道 → N 输出通道的设计（每个通道共享同一个 Linear 映射），无法在输入中
不包含目标变量的情况下生成目标变量的预测。当 exclude_target_from_input=True 时，
模型输出的通道数等于输入过程变量数，postprocess 中用原始 target_vars 索引提取预测
会导致索引错位，实际取到的是其他过程变量的预测值。因此 iTransformer 不支持
exclude_target_from_input=True 模式。
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
        num_features=38,
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
        model=iTransformerForForecasting,
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
        ckpt_save_dir="checkpoints/EthyDistillation_benchmark/iTransformer_incl",

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
