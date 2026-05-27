"""
PatchTST Soft Sensor Experiment.

Configuration: exclude_target_from_input=False (必须包含目标变量)
Input: 7 variables (OT included)

NOTE: PatchTST 是 channel-independent 的 encoder-only 模型，每个通道独立建模，
无法建立"过程变量 → 目标变量"的跨通道映射关系。因此 PatchTST 不支持
exclude_target_from_input=True 模式。在该模式下模型输出的通道索引与目标变量
索引不对应，且即使修正索引，channel-independent 架构也无法从过程变量推断目标变量。
"""

from torch.optim.lr_scheduler import MultiStepLR

from basicts import BasicTSLauncher
from basicts.configs import BasicTSSoftSensorConfig
from basicts.metrics import masked_mse
from basicts.models.PatchTST import PatchTSTForForecasting, PatchTSTConfig
from basicts.runners.callback import EarlyStopping, GradientClipping


def main():

    model_config = PatchTSTConfig(
        input_len=96,
        output_len=12,
        num_features=7,
        patch_len=16,
        patch_stride=8,
        padding=True,
        hidden_size=256,
        n_heads=4,
        intermediate_size=1024,
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
        moving_avg=25,
        output_attentions=False
    )

    BasicTSLauncher.launch_training(BasicTSSoftSensorConfig(
        model=PatchTSTForForecasting,
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
