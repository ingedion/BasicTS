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
    """

    def preprocess(self, runner: 'BasicTSRunner', data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Preprocess data for soft sensor task.
        
        Args:
            runner: BasicTSRunner instance
            data: Raw data dictionary
            
        Returns:
            Preprocessed data dictionary
        """
        # Create masks for null values
        inputs_mask = null_val_mask(data['inputs'], runner.cfg.null_val)
        targets_mask = null_val_mask(data['targets'], runner.cfg.null_val)
        
        # Normalize data using scaler
        if runner.scaler is not None:
            data['inputs'] = runner.scaler.transform(data['inputs'], inputs_mask)
            data['targets'] = runner.scaler.transform(data['targets'], targets_mask)
        
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
        
        Args:
            runner: BasicTSRunner instance
            forward_return: Model forward return dictionary
            
        Returns:
            Postprocessed data dictionary
        """
        # Extract target variables from prediction
        # For soft sensor, model outputs all variables but we only need target variables
        prediction = forward_return['prediction']
        
        # If model outputs all features, extract only target variables
        target_vars = runner.cfg.target_vars
        if prediction.shape[-1] != forward_return['targets'].shape[-1]:
            # Extract target variable predictions
            prediction = prediction[..., target_vars]
        
        forward_return['prediction'] = prediction
        
        # Inverse transform predictions and targets to original scale
        # This is crucial for soft sensor to get actual quality variable values
        if runner.cfg.rescale and runner.scaler is not None:
            forward_return['prediction'] = runner.scaler.inverse_transform(
                forward_return['prediction']
            )
            forward_return['targets'] = runner.scaler.inverse_transform(
                forward_return['targets'], 
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
