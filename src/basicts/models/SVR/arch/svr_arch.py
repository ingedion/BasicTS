"""
SVR (Support Vector Regression) model for soft sensor tasks.

Wraps sklearn's SVR as a PyTorch nn.Module for integration with BasicTS's
unified experiment management framework.

SVR with RBF kernel provides a nonlinear static baseline:
- Captures nonlinear relationships between process variables and quality variables
- No temporal modeling (treats flattened window as feature vector)
- Positioned between PLS (linear) and LSTM (nonlinear + temporal) in the baseline hierarchy

Uses MultiOutputRegressor to handle multi-step output, since sklearn's SVR
only supports single-output regression.

Input: [B, input_len, num_features] → flattened to [B, input_len * num_features]
Output: [B, output_len, num_targets]
"""

import numpy as np
import torch
from torch import nn

from ..config.svr_config import SVRConfig


class SVR(nn.Module):
    """
    SVR (Support Vector Regression) model.
    
    This model wraps sklearn's SVR for use within BasicTS.
    It requires NonGradientRunner for training (one-shot fit).
    
    Uses MultiOutputRegressor to handle multi-dimensional output
    (output_len * num_targets outputs).
    """

    def __init__(self, config: SVRConfig):
        super().__init__()
        self.input_len = config.input_len
        self.output_len = config.output_len
        self.num_features = config.num_features
        self.num_targets = config.num_targets
        self.kernel = config.kernel
        self.C = config.C
        self.epsilon = config.epsilon
        self.gamma = config.gamma

        # Dummy parameter so PyTorch doesn't complain about empty module
        self.dummy = nn.Parameter(torch.zeros(1), requires_grad=False)

        # sklearn model (set during fit or load)
        self._svr_model = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit the SVR model on training data.
        
        Args:
            X: Input data, shape [N, input_len, num_features]
            y: Target data, shape [N, output_len, num_targets]
        """
        from sklearn.multioutput import MultiOutputRegressor
        from sklearn.svm import SVR as SklearnSVR

        N = X.shape[0]
        # Flatten input: [N, input_len, C] → [N, input_len * C]
        X_flat = X.reshape(N, -1)
        # Flatten target: [N, output_len, T] → [N, output_len * T]
        y_flat = y.reshape(N, -1)

        # Create base SVR estimator
        base_svr = SklearnSVR(
            kernel=self.kernel,
            C=self.C,
            epsilon=self.epsilon,
            gamma=self.gamma
        )

        # Use MultiOutputRegressor for multi-dimensional output
        if y_flat.shape[1] == 1:
            # Single output: use SVR directly (faster)
            self._svr_model = base_svr
            self._svr_model.fit(X_flat, y_flat.ravel())
        else:
            # Multi-output: wrap with MultiOutputRegressor
            # Note: n_jobs=1 to avoid joblib multiprocessing conflicts with PyTorch on Windows
            self._svr_model = MultiOutputRegressor(base_svr, n_jobs=1)
            self._svr_model.fit(X_flat, y_flat)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Predict using the fitted SVR model.
        
        Args:
            inputs: Input tensor, shape [B, input_len, num_features]
            
        Returns:
            Prediction tensor, shape [B, output_len, num_targets]
        """
        if self._svr_model is None:
            raise RuntimeError("SVR model has not been fitted. Call fit() first or load a checkpoint.")

        device = inputs.device
        B = inputs.shape[0]

        # Flatten input: [B, input_len, C] → [B, input_len * C]
        X_flat = inputs.reshape(B, -1).cpu().numpy()

        # Predict
        y_pred = self._svr_model.predict(X_flat)  # [B, output_len * num_targets] or [B,]

        # Reshape to [B, output_len, num_targets]
        if y_pred.ndim == 1:
            y_pred = y_pred.reshape(B, self.output_len, self.num_targets)
        else:
            y_pred = y_pred.reshape(B, self.output_len, self.num_targets)

        return torch.tensor(y_pred, dtype=inputs.dtype, device=device)

    def get_sklearn_model(self):
        """Get the sklearn model for serialization."""
        return self._svr_model

    def set_sklearn_model(self, model) -> None:
        """Set the sklearn model (for deserialization)."""
        self._svr_model = model
