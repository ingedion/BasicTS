"""
Informer Encoder-Decoder Soft Sensor on EthyDistillation.

Encoder-Decoder Soft Sensor Design:
    This configuration uses the encoder-decoder architecture of Informer for soft sensing
    by splitting the input along the TIME dimension (not variable dimension):

    - Encoder: receives complete historical data [t_begin, t0) where all variables have
      true values (quality variables are fully measured before t0). The encoder learns
      long-range temporal patterns via self-attention and provides historical context
      to the decoder through cross-attention.

    - Decoder: receives data from [t0 - lookback, t_end + pred_len) with masking:
      * Lookback zone [0, lookback): all variables retain true values (guidance)
      * Lag zone [lookback, lookback + measurement_lag): process vars retain true values,
        quality vars masked to 0 (already occurred but not yet measured)
      * Prediction zone [lookback + measurement_lag, decoder_len): all vars set to 0
        (future unknown values)

    The decoder uses cross-attention to query encoder's historical context, and
    self-attention among process variables to learn "process → quality" mappings,
    enabling soft sensing without any model architecture modifications.

Configuration:
    Dataset: EthyDistillation (38 variables, target: 塔顶乙烷浓度, index=20)
    input_len=96, measurement_lag=6, lookback=12, pred_len=6
    encoder_len = input_len - measurement_lag = 90
    decoder_len = lookback + measurement_lag + pred_len = 24
    label_len = lookback = 12
"""

from torch.optim.lr_scheduler import MultiStepLR

from basicts import BasicTSLauncher
from basicts.configs import EncDecSoftSensorConfig
from basicts.metrics import masked_mse
from basicts.models.Informer import Informer, InformerConfig
from basicts.runners.callback import EarlyStopping, GradientClipping


# Dataset parameters
INPUT_LEN = 96
MEASUREMENT_LAG = 6
LOOKBACK = 12
PRED_LEN = 6
NUM_FEATURES = 38

# Derived parameters (encoder-decoder soft sensor)
ENCODER_LEN = INPUT_LEN - MEASUREMENT_LAG          # 90
DECODER_LEN = LOOKBACK + MEASUREMENT_LAG + PRED_LEN  # 24


def main():

    model_config = InformerConfig(
        input_len=ENCODER_LEN,       # encoder input length
        output_len=DECODER_LEN,      # decoder output length
        label_len=LOOKBACK,          # decoder lookback (label) length
        num_features=NUM_FEATURES,
        hidden_size=256,
        n_heads=4,
        prob_attn=True,
        factor=3,
        intermediate_size=512,
        hidden_act="gelu",
        distill=True,
        num_encoder_layers=2,
        num_decoder_layers=1,
        dropout=0.1,
        output_attentions=False,
        use_timestamps=False,
    )

    BasicTSLauncher.launch_training(EncDecSoftSensorConfig(
        model=Informer,
        model_config=model_config,
        dataset_name="EthyDistillation",

        # Encoder-decoder soft sensor specific
        target_vars=20,
        measurement_lag=MEASUREMENT_LAG,
        lookback=LOOKBACK,
        pred_len=PRED_LEN,
        input_len=INPUT_LEN,

        # Timestamps: disabled for Informer (no timestamp_sizes configured)
        use_timestamps=False,

        # Checkpoint
        ckpt_save_dir="checkpoints/EthyDistillation_benchmark/Informer_encdec_ss",

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
