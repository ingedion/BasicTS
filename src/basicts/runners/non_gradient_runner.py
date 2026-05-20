"""
NonGradientRunner: Runner for non-gradient optimization models (PLS, SVR, etc.)

These models use sklearn-style fit/predict instead of iterative gradient descent.
The runner performs:
1. Load all training data
2. Call model.fit(X_train, y_train) once
3. Evaluate on test set using the standard _eval_loop

This allows non-gradient models to be managed within BasicTS's unified experiment
framework while respecting their fundamentally different training paradigm.
"""

import json
import os
import pickle
import time
from typing import TYPE_CHECKING, Optional

import numpy as np
import torch
from easytorch.utils import get_logger, is_master
from torch.utils.data import DataLoader
from tqdm import tqdm

from basicts.runners.basicts_runner import BasicTSRunner
from basicts.runners.builder import Builder
from basicts.utils import BasicTSMode

if TYPE_CHECKING:
    from basicts.configs import BasicTSConfig


class NonGradientRunner(BasicTSRunner):
    """
    Runner for non-gradient optimization models.
    
    Overrides the training pipeline to perform one-shot fit instead of
    iterative gradient optimization. Evaluation pipeline is fully reused.
    
    Models used with this runner must implement:
        - fit(X: np.ndarray, y: np.ndarray): Train the model
        - forward(inputs: torch.Tensor) -> torch.Tensor: Predict (for eval loop)
    """

    def __init__(self, cfg: "BasicTSConfig") -> None:
        # Skip optimizer/lr_scheduler creation in parent __init__
        # Parent __init__ only declares them as None, actual creation is in _init_train
        super().__init__(cfg)

    def _init_train(self):
        """Initialize training for non-gradient models.
        
        Skips optimizer and lr_scheduler creation.
        Only sets up data loader and scaler.
        """
        self.logger.info("Initializing non-gradient training.")

        # We use epoch-based training unit with num_epochs=1 conceptually
        self.num_epochs = 1
        self.num_steps = None
        self.training_unit = "epoch"
        self.best_metrics = {}

        # Build train data loader
        self.train_data_loader = Builder._build_data_loader(self.cfg, BasicTSMode.TRAIN, self.logger)
        self.steps_per_epoch = len(self.train_data_loader)

        # Fit scaler on training data
        if self.scaler is not None:
            self.scaler.fit(self.train_data_loader.dataset.data)

        # No optimizer or lr_scheduler needed
        self.optimizer = None
        self.lr_scheduler = None

    def train(self):
        """One-shot training: collect all data, fit model, then evaluate."""

        # Initialize
        self._init_train()
        self.is_train_initialized = True

        # Initialize test
        self._init_test()
        self.is_test_initialized = True

        # Count parameters (for logging)
        self._count_parameters()

        self.logger.info("=" * 60)
        self.logger.info("Non-gradient model training (one-shot fit)")
        self.logger.info("=" * 60)

        # Collect all training data
        fit_start_time = time.time()
        X_all, y_all = self._collect_training_data()
        collect_time = time.time() - fit_start_time
        self.logger.info(f"Collected training data: X={X_all.shape}, y={y_all.shape} ({collect_time:.2f}s)")

        # Fit model
        fit_start_time = time.time()
        model = self.model.module if hasattr(self.model, 'module') else self.model
        model.fit(X_all, y_all)
        fit_time = time.time() - fit_start_time
        self.logger.info(f"Model fitting completed in {fit_time:.2f}s")

        # Save the fitted model
        self._save_fitted_model()

        # Evaluate on test set
        self.logger.info("Evaluating fitted model on test set.")
        self.eval(ckpt_path=self._get_fitted_model_path())

    def _collect_training_data(self):
        """Collect all training data into numpy arrays.
        
        Applies the same preprocessing (normalization) as the standard pipeline,
        then collects inputs and targets.
        
        Returns:
            tuple: (X_all, y_all) as numpy arrays
                X_all: shape [N, input_len, num_input_features]
                y_all: shape [N, output_len, num_target_features]
        """
        X_list = []
        y_list = []

        for data in tqdm(self.train_data_loader, desc="Collecting training data"):
            # Apply taskflow preprocessing (normalization)
            data = self.taskflow.preprocess(self, data)

            # Move to CPU and convert to numpy
            inputs = data['inputs'].cpu().numpy()   # [B, input_len, C_in]
            targets = data['targets'].cpu().numpy()  # [B, output_len, C_out]

            X_list.append(inputs)
            y_list.append(targets)

        X_all = np.concatenate(X_list, axis=0)
        y_all = np.concatenate(y_list, axis=0)

        return X_all, y_all

    def _save_fitted_model(self):
        """Save the fitted sklearn model using pickle."""
        save_path = self._get_fitted_model_path()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        model = self.model.module if hasattr(self.model, 'module') else self.model
        
        # Save both the PyTorch module state and the sklearn model
        save_dict = {
            "model_state_dict": model.state_dict(),
            "sklearn_model": model.get_sklearn_model(),
            "epoch": 1,
        }
        
        # Save scaler stats
        if self.scaler is not None:
            save_dict["scaler_stats"] = self.scaler.stats

        torch.save(save_dict, save_path)
        self.logger.info(f"Fitted model saved to {save_path}")

    def _get_fitted_model_path(self) -> str:
        """Get the path for saving/loading the fitted model."""
        return os.path.join(
            self.ckpt_save_dir,
            f"{self.model_name}_best_val_{self.target_metric.replace('/', '_')}.pt"
        )

    def _load_model(self, ckpt_path: str = None, strict: bool = True) -> None:
        """Load a fitted non-gradient model.
        
        Overrides parent to handle sklearn model loading in addition to state_dict.
        """
        if ckpt_path is None:
            ckpt_path = self._get_fitted_model_path()

        if not os.path.exists(ckpt_path):
            raise OSError(f"Checkpoint file does not exist: {ckpt_path}")

        self.logger.info(f"Loading model from {ckpt_path}")
        checkpoint_dict = torch.load(ckpt_path, map_location="cpu", weights_only=False)

        model = self.model.module if hasattr(self.model, 'module') else self.model

        # Load state dict (for dummy parameters)
        if "model_state_dict" in checkpoint_dict:
            model.load_state_dict(checkpoint_dict["model_state_dict"], strict=strict)

        # Load sklearn model
        if "sklearn_model" in checkpoint_dict:
            model.set_sklearn_model(checkpoint_dict["sklearn_model"])

    def eval(self, ckpt_path: Optional[str] = None) -> None:
        """Evaluate the fitted model on the test set."""
        self.on_eval_start(ckpt_path)
        self._test(None, None, BasicTSMode.EVAL)
        self.on_eval_end()

    def on_eval_start(self, ckpt_path: str) -> None:
        """Callback at the start of evaluation.
        
        For non-gradient models, we need to ensure scaler is fitted
        and the model is loaded.
        """
        if not self.is_train_initialized and self.scaler is not None:
            train_dataset = Builder._build_dataset(self.cfg, BasicTSMode.TRAIN)
            self.scaler.fit(train_dataset.data)

        if not self.is_test_initialized:
            self._init_test()
            self.is_test_initialized = True

        if ckpt_path is not None:
            self._load_model(ckpt_path=ckpt_path, strict=True)
