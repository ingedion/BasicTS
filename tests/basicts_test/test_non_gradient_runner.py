# pylint: disable=wrong-import-position
"""
Unit tests for NonGradientRunner.

Tests the complete one-shot fit + evaluation pipeline for non-gradient models.
Uses ETTh1_mini dataset (7 variables) with soft sensor configuration.
"""

import logging
import os
import shutil
import sys
import time

import numpy as np
import pytest
import torch

sys.path.append(os.path.abspath(__file__ + "/../../../src/"))
os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "smoke_test")))

from basicts.configs import BasicTSSoftSensorConfig
from basicts.metrics import masked_mse
from basicts.models.PLS import PLS, PLSConfig
from basicts.models.SVR import SVR, SVRConfig
from basicts.runners import NonGradientRunner


CKPT_DIR_PLS = "checkpoints/_test_non_gradient_pls"
CKPT_DIR_SVR = "checkpoints/_test_non_gradient_svr"


def _safe_rmtree(path):
    """Remove directory tree, handling Windows file lock issues."""
    if not os.path.exists(path):
        return
    # Close all logging handlers that might hold file locks
    for handler in logging.root.handlers[:]:
        handler.close()
        logging.root.removeHandler(handler)
    # Also close handlers on named loggers
    for name in list(logging.Logger.manager.loggerDict.keys()):
        logger = logging.getLogger(name)
        for handler in logger.handlers[:]:
            handler.close()
            logger.removeHandler(handler)
    time.sleep(0.1)  # Brief pause for Windows to release file handles
    try:
        shutil.rmtree(path)
    except PermissionError:
        pass  # Best effort cleanup; CI can handle stale dirs


@pytest.fixture(autouse=True, scope="session")
def cleanup_after_all():
    """Clean up checkpoint directories after all tests complete."""
    yield
    _safe_rmtree(CKPT_DIR_PLS)
    _safe_rmtree(CKPT_DIR_SVR)


class TestNonGradientRunnerPLS:
    """Test NonGradientRunner with PLS model."""

    def _make_config(self):
        model_config = PLSConfig(
            input_len=16,
            output_len=4,
            num_features=6,
            num_targets=1,
            n_components=5,
        )
        return BasicTSSoftSensorConfig(
            model=PLS,
            model_config=model_config,
            dataset_name="ETTh1_mini",
            runner=NonGradientRunner,
            target_vars=-1,
            input_vars=None,
            exclude_target_from_input=True,
            measurement_lag=2,
            input_len=16,
            output_len=4,
            gpus=None,
            seed=42,
            num_epochs=1,
            batch_size=32,
            metrics=["MAE", "MSE", "RMSE", "R2"],
            loss=masked_mse,
            eval_horizons=[1, 4],
            save_results=False,  # Avoid memmap file lock issues on Windows
            ckpt_save_dir=CKPT_DIR_PLS,
        )

    def test_full_pipeline(self):
        """Test that NonGradientRunner completes fit + eval without errors."""
        cfg = self._make_config()
        runner = NonGradientRunner(cfg)
        runner.init_logger(logger_name="test-pls", log_file_name="test_log")
        runner.train()

        # Verify outputs exist
        md5_dirs = [d for d in os.listdir(CKPT_DIR_PLS)
                    if os.path.isdir(os.path.join(CKPT_DIR_PLS, d))]
        assert len(md5_dirs) == 1, f"Expected 1 checkpoint dir, got {md5_dirs}"

        ckpt_dir = os.path.join(CKPT_DIR_PLS, md5_dirs[0])
        assert os.path.exists(os.path.join(ckpt_dir, "test_metrics.json"))

    def test_model_save_load(self):
        """Test that fitted PLS model can be saved and loaded correctly."""
        cfg = self._make_config()
        runner = NonGradientRunner(cfg)
        runner.init_logger(logger_name="test-pls-save", log_file_name="test_log")
        runner.train()

        # Load the saved model
        model_path = runner._get_fitted_model_path()
        assert os.path.exists(model_path)

        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
        assert "sklearn_model" in checkpoint
        assert checkpoint["sklearn_model"] is not None


class TestNonGradientRunnerSVR:
    """Test NonGradientRunner with SVR model."""

    def _make_config(self):
        model_config = SVRConfig(
            input_len=16,
            output_len=4,
            num_features=6,
            num_targets=1,
            kernel="rbf",
            C=1.0,
            epsilon=0.1,
            gamma="scale",
        )
        return BasicTSSoftSensorConfig(
            model=SVR,
            model_config=model_config,
            dataset_name="ETTh1_mini",
            runner=NonGradientRunner,
            target_vars=-1,
            input_vars=None,
            exclude_target_from_input=True,
            measurement_lag=2,
            input_len=16,
            output_len=4,
            gpus=None,
            seed=42,
            num_epochs=1,
            batch_size=32,
            metrics=["MAE", "MSE", "RMSE", "R2"],
            loss=masked_mse,
            eval_horizons=[1, 4],
            save_results=False,  # Avoid memmap file lock issues on Windows
            ckpt_save_dir=CKPT_DIR_SVR,
        )

    def test_full_pipeline(self):
        """Test that NonGradientRunner with SVR completes without errors."""
        cfg = self._make_config()
        runner = NonGradientRunner(cfg)
        runner.init_logger(logger_name="test-svr", log_file_name="test_log")
        runner.train()

        # Verify outputs exist
        md5_dirs = [d for d in os.listdir(CKPT_DIR_SVR)
                    if os.path.isdir(os.path.join(CKPT_DIR_SVR, d))]
        assert len(md5_dirs) == 1

        ckpt_dir = os.path.join(CKPT_DIR_SVR, md5_dirs[0])
        assert os.path.exists(os.path.join(ckpt_dir, "test_metrics.json"))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
