from typing import Optional, Tuple

import torch
from torch import nn

from basicts.modules.mlps import MLPLayer
from basicts.modules.transformer import MultiHeadAttention


class VariableImportanceGate(nn.Module):
    """
    Variable Importance Gate for PatchXformer.

    Computes per-variable importance scores from embedded representations
    and modulates variable contributions via element-wise gating before
    variate attention.

    The gate pools across patches to produce a single importance weight
    per variable, applies a linear projection followed by sigmoid activation,
    and multiplies the result element-wise with the input representations.
    """

    def __init__(self, hidden_size: int, num_patches: int):
        """
        Initialize the Variable Importance Gate.

        Args:
            hidden_size (int): Hidden dimension size (H).
            num_patches (int): Number of patches (P), used for documentation
                purposes; pooling is performed via mean reduction.
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.num_patches = num_patches
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Compute gated variable representations.

        Args:
            hidden_states: Tensor of shape [B, N, P, H] where
                B = batch size, N = num variables, P = num patches,
                H = hidden size.

        Returns:
            gated_states: Tensor of shape [B, N, P, H], the input
                multiplied element-wise by per-variable importance weights.
        """
        # Pool across patches: [B, N, P, H] -> [B, N, H]
        pooled = hidden_states.mean(dim=2)

        # Linear projection: [B, N, H] -> [B, N, 1]
        importance = self.linear(pooled)

        # Sigmoid activation and reshape to broadcast: [B, N, 1] -> [B, N, 1, 1]
        importance = torch.sigmoid(importance).unsqueeze(-1)

        # Element-wise multiply with input: [B, N, P, H] * [B, N, 1, 1] -> [B, N, P, H]
        return hidden_states * importance


class DualAxisBlock(nn.Module):
    """
    Dual-Axis Block for PatchXformer.

    One layer of temporal attention (within each variable's patch sequence)
    followed by variate attention (across variables at each patch position).
    Uses residual connections and post-layer-normalization pattern.

    Optionally includes a Variable Importance Gate between temporal and
    variate attention to modulate variable contributions.
    """

    def __init__(self, config):
        """
        Initialize the Dual-Axis Block.

        Args:
            config: PatchXformerConfig with fields hidden_size, n_heads,
                intermediate_size, hidden_act, dropout, use_variable_gate,
                and num_patches.
        """
        super().__init__()

        hidden_size = config.hidden_size
        n_heads = config.n_heads
        intermediate_size = config.intermediate_size
        hidden_act = config.hidden_act
        dropout = config.dropout

        # Temporal attention sublayer (self-attention along patches per variable)
        self.temporal_attn = MultiHeadAttention(
            hidden_size=hidden_size,
            n_heads=n_heads,
            dropout=dropout,
        )
        self.temporal_ffn = MLPLayer(
            input_size=hidden_size,
            intermediate_size=intermediate_size,
            output_size=hidden_size,
            hidden_act=hidden_act,
            dropout=dropout,
        )
        self.temporal_norm1 = nn.LayerNorm(hidden_size)
        self.temporal_norm2 = nn.LayerNorm(hidden_size)

        # Variate attention sublayer (attention across variables per patch)
        self.variate_attn = MultiHeadAttention(
            hidden_size=hidden_size,
            n_heads=n_heads,
            dropout=dropout,
        )
        self.variate_ffn = MLPLayer(
            input_size=hidden_size,
            intermediate_size=intermediate_size,
            output_size=hidden_size,
            hidden_act=hidden_act,
            dropout=dropout,
        )
        self.variate_norm1 = nn.LayerNorm(hidden_size)
        self.variate_norm2 = nn.LayerNorm(hidden_size)

        # Optional variable importance gate
        self.use_variable_gate = config.use_variable_gate
        if self.use_variable_gate:
            # Compute num_patches from config
            num_patches = (config.input_len - config.patch_len) // config.patch_stride + 1
            if config.padding:
                num_patches += 1
            self.variable_gate = VariableImportanceGate(
                hidden_size=hidden_size,
                num_patches=num_patches,
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        target_var_index: int = -1,
        use_asymmetric: bool = True,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Forward pass of the Dual-Axis Block.

        Args:
            hidden_states: Tensor of shape [B, N, P, H] where
                B = batch size, N = num variables, P = num patches,
                H = hidden size.
            target_var_index: Index of the target variable for asymmetric
                attention. -1 means last variable.
            use_asymmetric: Whether to use asymmetric cross-variate attention
                (target as Q, process vars as K/V).
            output_attentions: Whether to return attention weights.

        Returns:
            hidden_states: Tensor of shape [B, N, P, H].
            attn_weights: Optional tuple of (temporal_attn_weights, variate_attn_weights)
                when output_attentions=True, else None.
        """
        B, N, P, H = hidden_states.shape

        # === Temporal Attention ===
        # Reshape [B, N, P, H] -> [B*N, P, H] for per-variable temporal attention
        temporal_input = hidden_states.reshape(B * N, P, H)

        # Self-attention along patches
        residual = temporal_input
        attn_output, temporal_attn_weights, _ = self.temporal_attn(
            temporal_input,
            output_attentions=output_attentions,
        )
        # Residual + post-layer-norm
        temporal_input = self.temporal_norm1(residual + attn_output)

        # FFN with residual + post-layer-norm
        residual = temporal_input
        ffn_output = self.temporal_ffn(temporal_input)
        temporal_input = self.temporal_norm2(residual + ffn_output)

        # Reshape back [B*N, P, H] -> [B, N, P, H]
        hidden_states = temporal_input.reshape(B, N, P, H)

        # === Variable Importance Gate ===
        if self.use_variable_gate:
            hidden_states = self.variable_gate(hidden_states)

        # === Variate Attention ===
        # Reshape [B, N, P, H] -> [B*P, N, H] for per-patch variate attention
        variate_input = hidden_states.permute(0, 2, 1, 3).reshape(B * P, N, H)

        if use_asymmetric and N > 1:
            # Asymmetric cross-attention: target as Q, process vars as K/V
            # Resolve target_var_index
            tgt_idx = target_var_index if target_var_index >= 0 else N + target_var_index

            # Build process variable indices (all except target)
            process_indices = [i for i in range(N) if i != tgt_idx]

            # Extract target and process representations
            # variate_input shape: [B*P, N, H]
            target_repr = variate_input[:, tgt_idx:tgt_idx + 1, :]  # [B*P, 1, H]
            process_repr = variate_input[:, process_indices, :]  # [B*P, N-1, H]

            # Cross-attention: target as Q, process as K/V
            residual_target = target_repr
            cross_output, variate_attn_weights, _ = self.variate_attn(
                target_repr,
                key_value_states=process_repr,
                output_attentions=output_attentions,
            )
            # Residual + post-layer-norm for target
            target_repr = self.variate_norm1(residual_target + cross_output)

            # FFN with residual + post-layer-norm for target
            residual_target = target_repr
            ffn_output = self.variate_ffn(target_repr)
            target_repr = self.variate_norm2(residual_target + ffn_output)

            # Reconstruct full variate tensor: process vars unchanged, target updated
            # Build output tensor
            variate_output = variate_input.clone()
            variate_output[:, tgt_idx:tgt_idx + 1, :] = target_repr
        else:
            # Standard self-attention across all variables
            residual = variate_input
            attn_output, variate_attn_weights, _ = self.variate_attn(
                variate_input,
                output_attentions=output_attentions,
            )
            # Residual + post-layer-norm
            variate_output = self.variate_norm1(residual + attn_output)

            # FFN with residual + post-layer-norm
            residual = variate_output
            ffn_output = self.variate_ffn(variate_output)
            variate_output = self.variate_norm2(residual + ffn_output)

        # Reshape back [B*P, N, H] -> [B, P, N, H] -> [B, N, P, H]
        hidden_states = variate_output.reshape(B, P, N, H).permute(0, 2, 1, 3)

        # Collect attention weights
        attn_weights = None
        if output_attentions:
            attn_weights = (temporal_attn_weights, variate_attn_weights)

        return hidden_states, attn_weights
