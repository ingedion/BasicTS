"""
Debutanizer Dataset Test with iTransformer.

Verifies that the Debutanizer industrial dataset can be loaded and trained
with the soft sensor pipeline. Uses iTransformer as the test model.

Dataset: Debutanizer (7 process vars u1~u7, 1 quality var y)
Task: Predict y from u1~u7 (exclude_target_from_input=True)
"""

# pylint: disable=wrong-import-position

import os
import sys

sys.path.append(os.path.abspath(__file__ + "/../../../src/"))
os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from basicts import BasicTSLauncher
from basicts.configs import BasicTSSoftSensorConfig
from basicts.metrics import masked_mse
from basicts.models.iTransformer import iTransformerForForecasting, iTransformerConfig


def test_debutanizer_itransformer():
    """Test Debutanizer dataset with iTransformer soft sensor pipeline."""

    model_config = iTransformerConfig(
        input_len=32,
        output_len=1,
        num_features=7,       # 7 process variables (u1~u7, target excluded)
        hidden_size=64,
        n_heads=2,
        intermediate_size=128,
        hidden_act="gelu",
        num_layers=1,
        dropout=0.1,
        use_revin=True,
        output_attentions=False
    )

    BasicTSLauncher.launch_training(BasicTSSoftSensorConfig(
        model=iTransformerForForecasting,
        model_config=model_config,
        dataset_name="Debutanizer",

        # Soft sensor specific
        target_vars=-1,                   # y (last column)
        input_vars=None,                  # auto: all vars except target
        exclude_target_from_input=True,   # pure soft sensor: u1~u7 -> y
        measurement_lag=1,                # basic estimation (lag=1)
        input_len=32,
        output_len=1,

        # Training (lightweight for testing)
        gpus=None,                        # CPU only for test
        num_epochs=5,
        batch_size=32,
        metrics=["MAE", "MSE", "RMSE", "R2"],
        loss=masked_mse,
        optimizer_params={
            "lr": 1e-3,
            "weight_decay": 1e-4
        },
    ))


if __name__ == "__main__":
    test_debutanizer_itransformer()
