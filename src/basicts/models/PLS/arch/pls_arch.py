"""
PLS (Partial Least Squares) model for soft sensor tasks.

Wraps sklearn's PLSRegression as a PyTorch nn.Module for integration
with BasicTS's unified experiment management framework.

PLS is a classic linear baseline in industrial soft sensing:
- Projects high-dimensional correlated process variables into latent space
- Finds directions that maximize covariance between X and y
- Equivalent to linear regression when n_components = rank(X)

Input: [B, input_len, num_features] → flattened to [B, input_len * num_features]
Output: [B, output_len, num_targets]
"""

import numpy as np
import torch
from torch import nn

from ..config.pls_config import PLSConfig


class PLS(nn.Module):
    """
    PLS (Partial Least Squares) regression model.
    
    This model wraps sklearn's PLSRegression for use within BasicTS.
    It requires NonGradientRunner for training (one-shot fit).
    
    The model flattens the input time series window into a single feature vector,
    then performs PLS regression to predict the target variables at all output horizons.
    """

    def __init__(self, config: PLSConfig):
        super().__init__()
        self.input_len = config.input_len
        self.output_len = config.output_len
        self.num_features = config.num_features
        self.num_targets = config.num_targets
        self.n_components = config.n_components

        # Dummy parameter so PyTorch doesn't complain about empty module
        self.dummy = nn.Parameter(torch.zeros(1), requires_grad=False)

        # sklearn model (set during fit or load)
        self._pls_model = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit the PLS model on training data.
        
        Args:
            X: Input data, shape [N, input_len, num_features]
            y: Target data, shape [N, output_len, num_targets]
        """
        from sklearn.cross_decomposition import PLSRegression

        N = X.shape[0]
        # Flatten input: [N, input_len, C] → [N, input_len * C]
        X_flat = X.reshape(N, -1)
        # Flatten target: [N, output_len, T] → [N, output_len * T]
        y_flat = y.reshape(N, -1)

        # Determine n_components
        n_components = self.n_components
        if n_components is None:
            # Default: min(n_samples, n_features, n_targets), capped at reasonable value
            n_components = min(N, X_flat.shape[1], y_flat.shape[1], 20)
        # Ensure n_components doesn't exceed constraints
        n_components = min(n_components, N, X_flat.shape[1])

        self._pls_model = PLSRegression(n_components=n_components, max_iter=1000)
        self._pls_model.fit(X_flat, y_flat)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Predict using the fitted PLS model.
        
        Args:
            inputs: Input tensor, shape [B, input_len, num_features]
            
        Returns:
            Prediction tensor, shape [B, output_len, num_targets]
        """
        if self._pls_model is None:
            raise RuntimeError("PLS model has not been fitted. Call fit() first or load a checkpoint.")

        device = inputs.device
        B = inputs.shape[0]

        # Flatten input: [B, input_len, C] → [B, input_len * C]
        X_flat = inputs.reshape(B, -1).cpu().numpy()

        # Predict
        y_pred = self._pls_model.predict(X_flat)  # [B, output_len * num_targets]

        # Reshape to [B, output_len, num_targets]
        y_pred = y_pred.reshape(B, self.output_len, self.num_targets)

        return torch.tensor(y_pred, dtype=inputs.dtype, device=device)

    def get_sklearn_model(self):
        """Get the sklearn model for serialization."""
        return self._pls_model

    def set_sklearn_model(self, model) -> None:
        """Set the sklearn model (for deserialization)."""
        self._pls_model = model
