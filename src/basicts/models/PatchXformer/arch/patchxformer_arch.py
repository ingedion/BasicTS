from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from torch import nn

from basicts.modules.embed import PatchEmbedding
from basicts.modules.norm import RevIN

from ..config.patchxformer_config import PatchXformerConfig
from .layers import DualAxisBlock


class PatchXformerBackbone(nn.Module):
    """
    PatchXformer Backbone: Feature extraction via dual-axis attention blocks.

    Applies patch embedding independently to each variable, adds lag-aware
    positional encoding when measurement_lag > 0, then stacks dual-axis blocks
    (temporal attention + variate attention) to progressively refine both
    temporal and cross-variable representations.

    Paper: PatchXformer - A dual-axis attention Transformer for soft sensing tasks.
    """

    def __init__(self, config: PatchXformerConfig):
        """
        Initialize the PatchXformerBackbone.

        Args:
            config: PatchXformerConfig with all model hyperparameters.
        """
        super().__init__()

        self.num_features = config.num_features
        self.hidden_size = config.hidden_size
        self.measurement_lag = config.measurement_lag
        self.input_len = config.input_len
        self.patch_len = config.patch_len
        self.patch_stride = config.patch_stride
        self.output_attentions = config.output_attentions
        self.target_var_index = config.target_var_index
        self.use_asymmetric_attn = config.use_asymmetric_attn

        # Compute num_patches (same formula as PatchTST)
        self.num_patches = int((config.input_len - config.patch_len) / config.patch_stride + 1)
        if config.padding:
            self.num_patches += 1

        # Patch embedding (from basicts.modules.embed)
        # PatchEmbedding handles padding internally via ReplicationPad1d
        padding = (0, config.patch_stride) if config.padding else None
        self.patch_embedding = PatchEmbedding(
            config.hidden_size,
            config.patch_len,
            config.patch_stride,
            padding,
            config.dropout,
        )

        # Lag-aware positional encoding
        # Learnable embedding added to patches overlapping the lag boundary
        if self.measurement_lag > 0:
            self.lag_embedding = nn.Parameter(
                torch.zeros(1, 1, 1, config.hidden_size)
            )
            nn.init.normal_(self.lag_embedding, mean=0.0, std=0.02)
            # Precompute which patch indices overlap the lag boundary
            self._lag_patch_indices = self._compute_lag_patch_indices(config)
        else:
            self.lag_embedding = None
            self._lag_patch_indices = []

        # Dual-axis blocks
        self.dual_axis_blocks = nn.ModuleList([
            DualAxisBlock(config) for _ in range(config.num_layers)
        ])

        # Final layer normalization
        self.final_norm = nn.LayerNorm(config.hidden_size)

    def _compute_lag_patch_indices(self, config: PatchXformerConfig) -> List[int]:
        """
        Compute which patch indices overlap the lag boundary.

        The lag boundary is at time position (input_len - measurement_lag).
        A patch overlaps the boundary if its start is before the boundary
        and its end is at or after the boundary.

        Args:
            config: PatchXformerConfig with input_len, measurement_lag,
                patch_len, patch_stride, and padding.

        Returns:
            List of patch indices that overlap the lag boundary.
        """
        lag_boundary_time = config.input_len - config.measurement_lag
        lag_indices = []

        for patch_idx in range(self.num_patches):
            patch_start = patch_idx * config.patch_stride
            patch_end = patch_start + config.patch_len
            if patch_start < lag_boundary_time <= patch_end:
                lag_indices.append(patch_idx)

        return lag_indices

    def forward(
        self, inputs: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[List[torch.Tensor]]]:
        """
        Forward pass of the PatchXformerBackbone.

        Args:
            inputs: Input tensor of shape [batch_size, input_len, num_features].

        Returns:
            hidden_states: Tensor of shape [batch_size, num_features, num_patches, hidden_size].
            attn_weights: Optional list of attention weight tuples when
                output_attentions=True, else None.
        """
        B, L, N = inputs.shape

        # Apply patch embedding per variable
        # PatchEmbedding expects [B, L, N] and outputs [B*N, P, H]
        hidden_states = self.patch_embedding(inputs)

        # Reshape to [B, N, P, H]
        hidden_states = hidden_states.reshape(B, N, self.num_patches, self.hidden_size)

        # Apply lag embedding to patches overlapping the lag boundary
        if self.measurement_lag > 0 and self.lag_embedding is not None and len(self._lag_patch_indices) > 0:
            for patch_idx in self._lag_patch_indices:
                hidden_states[:, :, patch_idx, :] = (
                    hidden_states[:, :, patch_idx, :] + self.lag_embedding.squeeze(2)
                )

        # Stack dual-axis blocks
        all_attn_weights = [] if self.output_attentions else None
        for block in self.dual_axis_blocks:
            hidden_states, attn_weights = block(
                hidden_states,
                target_var_index=self.target_var_index,
                use_asymmetric=self.use_asymmetric_attn,
                output_attentions=self.output_attentions,
            )
            if self.output_attentions and attn_weights is not None:
                all_attn_weights.append(attn_weights)

        # Apply final layer norm
        hidden_states = self.final_norm(hidden_states)

        return hidden_states, all_attn_weights


class PatchXformerForForecasting(nn.Module):
    """
    PatchXformer for soft sensor forecasting.

    Wraps PatchXformerBackbone with optional RevIN normalization and a linear
    forecasting head that maps the target variable's patch representations
    to the prediction horizon.

    Paper: PatchXformer - A dual-axis attention Transformer for soft sensing tasks.
    """

    def __init__(self, config: PatchXformerConfig):
        """
        Initialize PatchXformerForForecasting.

        Args:
            config: PatchXformerConfig with all model hyperparameters.

        Raises:
            ValueError: If num_features < 2 with use_asymmetric_attn=True.
            ValueError: If patch_len > input_len.
            ValueError: If measurement_lag >= input_len.
            ValueError: If hidden_size % n_heads != 0.
        """
        super().__init__()

        # Input validation
        if config.use_asymmetric_attn and config.num_features < 2:
            raise ValueError(
                f"Asymmetric attention requires at least 2 variables "
                f"(1 process + 1 target), got num_features={config.num_features}."
            )
        if config.patch_len > config.input_len:
            raise ValueError(
                f"patch_len ({config.patch_len}) must be <= input_len ({config.input_len})."
            )
        if config.measurement_lag >= config.input_len:
            raise ValueError(
                f"measurement_lag ({config.measurement_lag}) must be < input_len ({config.input_len})."
            )
        if config.hidden_size % config.n_heads != 0:
            raise ValueError(
                f"hidden_size ({config.hidden_size}) must be divisible by "
                f"n_heads ({config.n_heads})."
            )

        self.input_len = config.input_len
        self.output_len = config.output_len
        self.num_features = config.num_features
        self.target_var_index = config.target_var_index
        self.output_attentions = config.output_attentions

        # Backbone
        self.backbone = PatchXformerBackbone(config)
        num_patches = self.backbone.num_patches

        # Optional RevIN normalization
        self.use_revin = config.use_revin
        if self.use_revin:
            self.revin = RevIN(config.num_features, affine=False)

        # Forecasting head components
        self.head_dropout = nn.Dropout(config.dropout)
        self.flatten = nn.Flatten(start_dim=-2)
        self.forecasting_head = nn.Linear(num_patches * config.hidden_size, config.output_len)

    def forward(
        self,
        inputs: torch.Tensor,
        inputs_timestamps: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Dict[str, Any]]:
        """
        Forward pass of PatchXformerForForecasting.

        Args:
            inputs: Input tensor of shape [batch_size, input_len, num_features].
            inputs_timestamps: Optional timestamps of shape [batch_size, input_len, num_timestamps].
                Currently unused but accepted for API compatibility.

        Returns:
            If output_attentions=False: prediction tensor of shape [batch_size, output_len, 1].
            If output_attentions=True: dict with keys "prediction" and "attn_weights".

        Raises:
            ValueError: If inputs.shape[1] != input_len.
        """
        B, L, N = inputs.shape

        # Validate input sequence length
        if L != self.input_len:
            raise ValueError(
                f"Expected input sequence length {self.input_len}, got {L}."
            )

        # Resolve target variable index (handle negative indexing)
        target_idx = self.target_var_index if self.target_var_index >= 0 else N + self.target_var_index

        # RevIN normalization
        if self.use_revin:
            inputs = self.revin(inputs, "norm")

        # Backbone forward: [B, N, P, H]
        hidden_states, attn_weights = self.backbone(inputs)

        # Extract target variable: [B, 1, P, H]
        target_hidden = hidden_states[:, target_idx:target_idx + 1, :, :]

        # Flatten patches and hidden dim: [B, P*H]
        target_flat = self.flatten(target_hidden.squeeze(1))  # [B, P, H] -> [B, P*H]

        # Dropout
        target_flat = self.head_dropout(target_flat)

        # Linear projection: [B, output_len]
        prediction = self.forecasting_head(target_flat)

        # Reshape to [B, output_len, 1]
        prediction = prediction.unsqueeze(-1)

        # RevIN denormalization (only for target variable)
        if self.use_revin:
            # Extract target variable statistics from RevIN
            # RevIN stores mean [B, 1, N] and stdev [B, 1, N]
            target_stdev = self.revin.stdev[:, :, target_idx:target_idx + 1]  # [B, 1, 1]
            if self.revin.subtract_last:
                target_last = self.revin.last[:, :, target_idx:target_idx + 1]  # [B, 1, 1]
            else:
                target_mean = self.revin.mean[:, :, target_idx:target_idx + 1]  # [B, 1, 1]

            # Manual denormalization for single target variable
            prediction = prediction * target_stdev
            if self.revin.subtract_last:
                prediction = prediction + target_last
            else:
                prediction = prediction + target_mean

        if self.output_attentions:
            return {"prediction": prediction, "attn_weights": attn_weights}
        else:
            return prediction
