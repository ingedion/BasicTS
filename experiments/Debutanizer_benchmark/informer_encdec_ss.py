"""
Informer Encoder-Decoder Soft Sensor on Debutanizer.

Encoder-Decoder Soft Sensor Design:
    This configuration uses the Informer model in an encoder-decoder soft sensor setup.
    Unlike encoder-only models (DLinear, LSTM) that treat soft sensing as a standard
    forecasting problem, the encoder-decoder approach leverages the natural two-input
    architecture of Informer to implement a time-dimension splitting strategy:

    - Encoder: Receives the full historical window [t_begin, t0) where all variables
      (including quality variable y) have known true values. The encoder learns long-range
      temporal patterns and encodes historical context via self-attention.

    - Decoder: Receives [t0 - lookback, t_end + pred_len) with masking applied:
      * Lookback zone [t0-lookback, t0): all variables retain true values (guidance)
      * Lag zone [t0, t_end): process variables (u1~u7) retain true values,
        quality variable (y) is masked to 0 (unknown due to measurement delay)
      * Prediction zone [t_end, t_end+pred_len): all variables masked to 0

    The decoder uses cross-attention to query encoder's historical context and
    self-attention among process variables to infer the masked quality variable.

Window Configuration:
    input_len = 32, measurement_lag = 3, lookback = 6, pred_len = 3
    encoder_len = input_len - measurement_lag = 29
    decoder_len = lookback + measurement_lag + pred_len = 12

    Prediction target: quality variable y at [t0, t_end + pred_len), length = 6
      - Estimation zone [t0, t_end): 3 steps (already occurred, not yet measured)
      - Prediction zone [t_end, t_end+pred_len): 3 steps (future values)

Dataset: Debutanizer (8 features: u1~u7 process variables + y quality variable)
Target: y (butane content, index -1 / 7)
"""

from torch.optim.lr_scheduler import MultiStepLR

from basicts import BasicTSLauncher
from basicts.configs import EncDecSoftSensorConfig
from basicts.metrics import masked_mse
from basicts.models.Informer import Informer, InformerConfig
from basicts.runners.callback import EarlyStopping, GradientClipping


def main():

    # Encoder-decoder soft sensor parameters
    input_len = 32
    measurement_lag = 3
    lookback = 6
    pred_len = 3

    # Derived lengths
    encoder_len = input_len - measurement_lag  # 29
    decoder_len = lookback + measurement_lag + pred_len  # 12

    model_config = InformerConfig(
        input_len=encoder_len,       # Encoder input time steps
        output_len=decoder_len,      # Decoder output length
        label_len=lookback,          # Informer's label_len parameter
        num_features=8,              # All variables in Debutanizer dataset
        hidden_size=128,
        n_heads=4,
        intermediate_size=256,
        hidden_act="gelu",
        num_encoder_layers=2,
        num_decoder_layers=1,
        dropout=0.1,
        prob_attn=True,
        factor=3,
        distill=True,
    )

    BasicTSLauncher.launch_training(EncDecSoftSensorConfig(
        model=Informer,
        model_config=model_config,
        dataset_name="Debutanizer",

        # Encoder-decoder soft sensor specific
        target_vars=-1,
        measurement_lag=measurement_lag,
        input_len=input_len,
        lookback=lookback,
        pred_len=pred_len,

        # Timestamps: disabled for Informer (no timestamp_sizes configured)
        use_timestamps=False,

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
        save_results=True,
    ))


if __name__ == "__main__":
    main()
