"""
LSTM model for soft sensor / time series forecasting tasks.

Standard LSTM encoder with a linear projection head.
Serves as the classic nonlinear temporal baseline:
- Captures nonlinear relationships in multivariate time series
- Models temporal dependencies through recurrent hidden states
- Positioned as the strongest traditional baseline before Transformer models

Architecture:
    Input [B, input_len, num_features]
    → LSTM encoder (multi-layer)
    → Take last hidden state [B, hidden_size * num_directions]
    → Linear projection
    → Output [B, output_len, num_targets]
"""

import torch
from torch import nn

from ..config.lstm_config import LSTMConfig


class LSTM(nn.Module):
    """
    LSTM model for time series prediction.
    
    Uses the standard gradient-based training pipeline (BasicTSRunner).
    """

    def __init__(self, config: LSTMConfig):
        super().__init__()
        self.input_len = config.input_len
        self.output_len = config.output_len
        self.num_features = config.num_features
        self.num_targets = config.num_targets
        self.hidden_size = config.hidden_size
        self.num_layers = config.num_layers
        self.bidirectional = config.bidirectional

        num_directions = 2 if config.bidirectional else 1

        # LSTM encoder
        self.lstm = nn.LSTM(
            input_size=config.num_features,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            dropout=config.dropout if config.num_layers > 1 else 0.0,
            bidirectional=config.bidirectional,
            batch_first=True
        )

        # Linear projection head: last hidden state → output
        self.projection = nn.Linear(
            config.hidden_size * num_directions,
            config.output_len * config.num_targets
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            inputs: Input tensor, shape [B, input_len, num_features]
            
        Returns:
            Prediction tensor, shape [B, output_len, num_targets]
        """
        B = inputs.shape[0]

        # LSTM encoding
        # output: [B, input_len, hidden_size * num_directions]
        # h_n: [num_layers * num_directions, B, hidden_size]
        output, (h_n, c_n) = self.lstm(inputs)

        # Use the last time step's output as the representation
        # [B, hidden_size * num_directions]
        last_output = output[:, -1, :]

        # Project to output space
        # [B, output_len * num_targets]
        prediction = self.projection(last_output)

        # Reshape to [B, output_len, num_targets]
        prediction = prediction.reshape(B, self.output_len, self.num_targets)

        return prediction
