import torch


def masked_r2(prediction: torch.Tensor, targets: torch.Tensor, targets_mask: torch.Tensor = None) -> torch.Tensor:
    """
    Calculate the Masked R-squared (coefficient of determination) between predicted and target values,
    while ignoring entries marked as null/missing.

    R² is always computed globally across the entire batch (and all time steps if present).
    This avoids the degenerate case where per-sample variance is near zero (e.g., short
    output sequences or single-channel targets), which would cause R² to explode.

    Args:
        prediction (torch.Tensor): Predicted values, shape [B, T, C], [B, T], or [B, C].
        targets (torch.Tensor): Ground truth values, same shape as prediction.
        targets_mask (torch.Tensor, optional): Boolean mask of valid entries.

    Returns:
        torch.Tensor: Scalar R² value computed globally.
    """

    mask = targets_mask if targets_mask is not None else torch.ones_like(targets)

    mask = mask.float()
    prediction = torch.nan_to_num(prediction) * mask
    targets = torch.nan_to_num(targets) * mask

    # Flatten all dimensions to compute a single global R²
    pred_flat = prediction.reshape(-1)
    tgt_flat = targets.reshape(-1)
    mask_flat = mask.reshape(-1)

    # Only consider valid (masked) entries
    n_valid = mask_flat.sum()
    if n_valid < 2:
        return torch.tensor(0.0, device=prediction.device)

    tgt_mean = (tgt_flat * mask_flat).sum() / n_valid
    ss_res = (mask_flat * torch.pow(tgt_flat - pred_flat, 2)).sum()
    ss_tot = (mask_flat * torch.pow(tgt_flat - tgt_mean, 2)).sum()

    r2 = 1 - (ss_res / (ss_tot + 1e-6))
    return torch.nan_to_num(r2)
