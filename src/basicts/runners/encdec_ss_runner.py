"""
EncDecSoftSensorRunner: Runner for encoder-decoder soft sensor tasks.

Overrides the default behavior of replacing `targets` with empty tensors
during non-training phases. For encoder-decoder soft sensor, the decoder input
(stored in `targets`) contains:
- Lookback zone: historical true values (known, not answers)
- Lag zone: process variable true values (known) + quality vars masked to 0
- Prediction zone: all zeros

Since all answer information (quality variables in lag+pred zones) has already
been masked to 0 by the TaskFlow preprocess, there is no information leakage.
The process variable values in the lag zone are essential for soft sensing
and must be preserved during inference.
"""

from typing import TYPE_CHECKING, Dict, Optional

import torch
from torch import nn

from basicts.utils import RunnerStatus

from .basicts_runner import BasicTSRunner

if TYPE_CHECKING:
    from basicts.configs import BasicTSConfig


class EncDecSoftSensorRunner(BasicTSRunner):
    """
    Runner for encoder-decoder soft sensor tasks.

    Overrides the default behavior of replacing `targets` with empty tensors
    during non-training phases. For encoder-decoder soft sensor, the decoder input
    (stored in `targets`) contains:
    - Lookback zone: historical true values (known, not answers)
    - Lag zone: process variable true values (known) + quality vars masked to 0
    - Prediction zone: all zeros

    Since all answer information (quality variables in lag+pred zones) has already
    been masked to 0 by the TaskFlow preprocess, there is no information leakage.
    The process variable values in the lag zone are essential for soft sensing
    and must be preserved during inference.
    """

    def _forward(self, model: nn.Module, data: Dict, step: int, epoch: Optional[int] = None) -> Dict:
        """Forward pass for encoder-decoder soft sensor.

        Key difference from BasicTSRunner._forward:
            Does NOT replace `targets` with `torch.empty_like()` during non-training
            phases. For encoder-decoder soft sensor, targets is the decoder input with
            quality vars already masked to 0 by TaskFlow preprocess. Process variable
            values in the lag zone are essential for soft sensing inference.

        Args:
            model (nn.Module): the model
            data (Dict): data dictionary containing at least 'inputs'
            step (int): current training step
            epoch (Optional[int]): current epoch

        Returns:
            Dict: forward return dictionary with 'prediction' and other data keys
        """

        # Move data to running device
        for k in data.keys():
            data[k] = self.to_running_device(data[k]) if isinstance(data[k], torch.Tensor) else data[k]

        # data must contain "inputs"
        assert "inputs" in data, "data must contain key \"inputs\"."
        inputs = data["inputs"]
        kwargs = {k: data[k] for k in self.forward_params if k in data}

        # KEY DIFFERENCE: Do NOT replace targets with empty during inference.
        # For encoder-decoder soft sensor, targets is the decoder input with
        # quality vars already masked to 0. Process vars must be preserved.

        if "step" in self.forward_params:
            kwargs["step"] = step
        if "epoch" in self.forward_params:
            kwargs["epoch"] = epoch
        if "train" in self.forward_params:
            kwargs["train"] = self.status == RunnerStatus.TRAINING

        # Forward pass through the model
        forward_return = model(inputs, **kwargs)

        # Parse forward return
        if isinstance(forward_return, torch.Tensor):
            forward_return = {"prediction": forward_return}
        # add other keys and values in `data` to `forward_return`
        for k, v in data.items():
            if k not in forward_return:
                forward_return[k] = v

        # KEY: Override 'targets' with ground_truth for loss/metric computation.
        # At this point, forward_return['targets'] is the decoder input (masked),
        # but the loss function expects 'targets' to be the ground truth.
        # The ground_truth was extracted and stored by TaskFlow.preprocess.
        if 'ground_truth' in forward_return:
            forward_return['targets'] = forward_return['ground_truth']

        # Extract prediction for loss computation:
        # Skip lookback steps and select target variables so that prediction
        # shape matches ground_truth shape for the loss function.
        # Full prediction: [batch, lookback + measurement_lag + pred_len, num_features]
        # After extraction: [batch, measurement_lag + pred_len, len(target_vars)]
        lookback = getattr(self.cfg, 'lookback', 0)
        target_vars = getattr(self.cfg, 'target_vars', None)
        if target_vars is not None and 'prediction' in forward_return:
            forward_return['prediction'] = forward_return['prediction'][:, lookback:, target_vars]

        return forward_return
