from typing import TYPE_CHECKING, Any, Dict

import torch

from basicts.utils.mask import null_val_mask

from .basicts_taskflow import BasicTSTaskFlow

if TYPE_CHECKING:
    from basicts.runners.basicts_runner import BasicTSRunner


class BasicTSSoftSensorTaskFlow(BasicTSTaskFlow):
    """
    Task flow for soft sensor tasks.
    
    Key differences from forecasting:
    - Focuses on single/multiple target variables (quality variables)
    - Handles measurement lag appropriately
    - Compatible with typical forecasting models (DLinear, PatchTST, iTransformer, TimeXer, etc.)
    - Supports spatial-temporal modeling features
    
    Important: The scaler is fit on the full dataset (all channels), but inputs/targets
    may only contain a subset of channels. This taskflow slices the scaler stats to match
    the actual channel indices used in inputs and targets.
    """

    def _get_channel_stats(self, runner: 'BasicTSRunner'):
        """
        Extract per-channel scaler statistics (mean/std) for input and target variables.
        
        The scaler is fit on the full dataset with all channels, so we need to slice
        the mean/std to match the actual input_vars and target_vars used by the dataset.
        
        Returns:
            tuple: (input_mean, input_std, target_mean, target_std) or None if scaler is None
        """
        if runner.scaler is None:
            return None
        
        mean = runner.scaler.stats['mean']  # shape: [1, num_all_channels] when norm_each_channel=True
        std = runner.scaler.stats['std']

        # Get channel indices from config
        input_vars = runner.cfg.input_vars
        target_vars = runner.cfg.target_vars

        # If input_vars is None in config, get the resolved list from the dataset
        if input_vars is None:
            # The dataset resolves input_vars in __init__ based on exclude_target_from_input
            dataset = runner.train_data_loader.dataset if runner.train_data_loader is not None \
                else runner.val_data_loader.dataset if runner.val_data_loader is not None \
                else runner.test_data_loader.dataset
            input_vars = dataset.input_vars

        # Slice stats for input channels
        input_mean = mean[..., input_vars]
        input_std = std[..., input_vars]

        # Slice stats for target channels
        target_mean = mean[..., target_vars]
        target_std = std[..., target_vars]

        return input_mean, input_std, target_mean, target_std

    def _transform(self, data: torch.Tensor, mean: torch.Tensor, std: torch.Tensor,
                   mask: torch.Tensor = None) -> torch.Tensor:
        """Apply z-score normalization with given mean/std."""
        mean = mean.to(data.device)
        std = std.to(data.device)
        normed = (data - mean) / std
        if mask is not None:
            normed = torch.where(mask, normed, data)
        return normed

    def _inverse_transform(self, data: torch.Tensor, mean: torch.Tensor, std: torch.Tensor,
                           mask: torch.Tensor = None) -> torch.Tensor:
        """Apply inverse z-score normalization with given mean/std."""
        mean = mean.to(data.device)
        std = std.to(data.device)
        denormed = data * std + mean
        if mask is not None:
            denormed = torch.where(mask, denormed, data)
        return denormed

    def preprocess(self, runner: 'BasicTSRunner', data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Preprocess data for soft sensor task.
        
        Uses channel-aligned scaler statistics so that each variable subset is
        normalized with its own mean/std rather than broadcasting incorrectly.
        
        Args:
            runner: BasicTSRunner instance
            data: Raw data dictionary
            
        Returns:
            Preprocessed data dictionary
        """
        # Create masks for null values
        inputs_mask = null_val_mask(data['inputs'], runner.cfg.null_val)
        targets_mask = null_val_mask(data['targets'], runner.cfg.null_val)

        # Normalize data using channel-aligned scaler stats
        stats = self._get_channel_stats(runner)
        if stats is not None:
            input_mean, input_std, target_mean, target_std = stats
            data['inputs'] = self._transform(data['inputs'], input_mean, input_std, inputs_mask)
            data['targets'] = self._transform(data['targets'], target_mean, target_std, targets_mask)

        # Replace null values with configured value (default: 0.0)
        data['inputs'] = torch.where(
            inputs_mask,
            data['inputs'],
            torch.tensor(runner.cfg.null_to_num, device=data['inputs'].device)
        )
        data['targets'] = torch.where(
            targets_mask,
            data['targets'],
            torch.tensor(runner.cfg.null_to_num, device=data['targets'].device)
        )

        # Store masks for later use
        data['targets_mask'] = targets_mask

        return data

    def postprocess(self, runner: 'BasicTSRunner', forward_return: Dict[str, Any]) -> Dict[str, Any]:
        """
        Postprocess model outputs for soft sensor task.
        
        Extracts target variable predictions from multi-channel model output,
        then applies channel-aligned inverse normalization.
        
        Args:
            runner: BasicTSRunner instance
            forward_return: Model forward return dictionary
            
        Returns:
            Postprocessed data dictionary
        """
        # Extract target variables from prediction
        prediction = forward_return['prediction']
        target_vars = runner.cfg.target_vars

        # If model outputs all features, extract only target variables
        if prediction.shape[-1] != forward_return['targets'].shape[-1]:
            prediction = prediction[..., target_vars]

        forward_return['prediction'] = prediction

        # Inverse transform predictions and targets to original scale
        if runner.cfg.rescale:
            stats = self._get_channel_stats(runner)
            if stats is not None:
                _, _, target_mean, target_std = stats
                forward_return['prediction'] = self._inverse_transform(
                    forward_return['prediction'], target_mean, target_std
                )
                forward_return['targets'] = self._inverse_transform(
                    forward_return['targets'], target_mean, target_std,
                    forward_return['targets_mask']
                )

        return forward_return

    def get_weight(self, forward_return: Dict[str, Any]) -> float:
        """
        Get weight for loss calculation.
        
        Args:
            forward_return: Model forward return dictionary
            
        Returns:
            Weight value based on valid samples
        """
        return forward_return['targets_mask'].sum().item()
