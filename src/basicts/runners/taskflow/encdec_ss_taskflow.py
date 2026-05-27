from typing import TYPE_CHECKING, Any, Dict

import torch

from basicts.utils.mask import null_val_mask

from .softsensor_taskflow import BasicTSSoftSensorTaskFlow

if TYPE_CHECKING:
    from basicts.runners.basicts_runner import BasicTSRunner


class EncDecSoftSensorTaskFlow(BasicTSSoftSensorTaskFlow):
    """
    TaskFlow for encoder-decoder soft sensor tasks.

    Handles normalization, masking, and postprocessing for encoder-decoder
    architectures (Informer, Autoformer, etc.) in soft sensor scenarios.

    Key differences from BasicTSSoftSensorTaskFlow:
    - Uses full-channel scaler (all variables share the same scaler stats)
      since encoder and decoder inputs contain all num_features channels.
    - Applies temporal masking to decoder input:
      * [lookback:lookback+measurement_lag] target_vars → 0
      * [lookback+measurement_lag:] all vars → 0
    - Extracts ground truth from raw targets before masking for loss computation.
    - Postprocess skips lookback steps and extracts only target_vars from prediction.
    """

    def preprocess(self, runner: 'BasicTSRunner', data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Preprocess data for encoder-decoder soft sensor task.

        Steps:
            1. Extract ground truth targets (before normalization) for loss computation.
            2. Z-score normalize inputs and targets using full-channel scaler.
            3. Apply masking to decoder input (after normalization):
               - [lookback:lookback+measurement_lag] target_vars → 0
               - [lookback+measurement_lag:] all vars → 0
            4. Store ground truth and targets_mask for postprocessing.

        Args:
            runner: BasicTSRunner instance with cfg and scaler attributes.
            data: Dictionary with 'inputs' and 'targets' tensors.

        Returns:
            Preprocessed data dictionary with added 'ground_truth' and 'targets_mask'.
        """
        # Get config parameters
        lookback = runner.cfg.lookback
        measurement_lag = runner.cfg.measurement_lag
        pred_len = runner.cfg.pred_len
        target_vars = runner.cfg.target_vars  # List[int]

        # 1. Extract ground truth before normalization
        #    ground_truth = targets[:, lookback:, target_vars]
        #    shape: [batch, measurement_lag + pred_len, len(target_vars)]
        ground_truth = data['targets'][:, lookback:, target_vars].clone()

        # Create null value mask for ground truth (True = valid, False = null)
        ground_truth_mask = null_val_mask(ground_truth, runner.cfg.null_val)

        # 2. Normalize inputs and targets using full-channel scaler
        if runner.scaler is not None:
            # For encoder-decoder soft sensor, both inputs and targets contain
            # all num_features channels, so we use the full scaler directly.
            data['inputs'] = runner.scaler.transform(data['inputs'])
            data['targets'] = runner.scaler.transform(data['targets'])

        # Replace null values with configured value (default: 0.0) after normalization
        inputs_mask = null_val_mask(data['inputs'], runner.cfg.null_val)
        targets_null_mask = null_val_mask(data['targets'], runner.cfg.null_val)

        data['inputs'] = torch.where(
            inputs_mask,
            data['inputs'],
            torch.tensor(runner.cfg.null_to_num, device=data['inputs'].device, dtype=data['inputs'].dtype)
        )
        data['targets'] = torch.where(
            targets_null_mask,
            data['targets'],
            torch.tensor(runner.cfg.null_to_num, device=data['targets'].device, dtype=data['targets'].dtype)
        )

        # 3. Apply masking to decoder input (after normalization)
        #    Mask target_vars in lag zone: [lookback:lookback+measurement_lag]
        data['targets'][:, lookback:lookback + measurement_lag, target_vars] = 0

        #    Mask all vars in prediction zone: [lookback+measurement_lag:]
        if pred_len > 0:
            data['targets'][:, lookback + measurement_lag:, :] = 0

        # 4. Store ground truth and mask for postprocessing
        data['ground_truth'] = ground_truth
        data['targets_mask'] = ground_truth_mask

        return data

    def postprocess(self, runner: 'BasicTSRunner', forward_return: Dict[str, Any]) -> Dict[str, Any]:
        """
        Postprocess model outputs for encoder-decoder soft sensor task.

        Steps:
            1. Inverse transform prediction to original scale using target_vars scaler stats.
            2. Set forward_return['targets'] = ground_truth (already in original scale).

        Note: Prediction extraction (skip lookback, select target vars) is already done
        in EncDecSoftSensorRunner._forward to ensure correct shapes for loss computation.

        Args:
            runner: BasicTSRunner instance with cfg and scaler attributes.
            forward_return: Dictionary with 'prediction' tensor (already extracted).

        Returns:
            Postprocessed dictionary with prediction and targets in original scale.
        """
        target_vars = runner.cfg.target_vars  # List[int]

        prediction = forward_return['prediction']
        # prediction shape: [batch, measurement_lag + pred_len, len(target_vars)]
        # (already extracted by runner._forward)

        # 1. Inverse transform to original scale
        if runner.cfg.rescale and runner.scaler is not None:
            mean = runner.scaler.stats['mean'].to(prediction.device)
            std = runner.scaler.stats['std'].to(prediction.device)

            # Slice scaler stats for target variables only
            target_mean = mean[..., target_vars]
            target_std = std[..., target_vars]

            prediction = prediction * target_std + target_mean

        # 2. Set outputs
        forward_return['prediction'] = prediction
        forward_return['targets'] = forward_return['ground_truth']

        return forward_return

    def get_weight(self, forward_return: Dict[str, Any]) -> float:
        """
        Get weight for loss calculation based on valid target mask.

        The weight is the number of valid (non-null) elements in the ground truth,
        which ensures that null values don't contribute to the loss.

        Args:
            forward_return: Dictionary containing 'targets_mask'.

        Returns:
            Weight value (number of valid target elements).
        """
        return forward_return['targets_mask'].sum().item()
