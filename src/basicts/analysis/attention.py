"""
Attention Visualization (Reserved Interface).

This module provides interfaces for visualizing attention weights
from Transformer-based models. Currently a placeholder for future implementation.

Planned features:
- Attention heatmap for individual samples
- Cross-variable attention patterns (iTransformer)
- Temporal attention patterns (PatchTST)
- Layer-wise attention comparison
"""

from typing import Optional

import numpy as np


class AttentionVisualizer:
    """
    Placeholder for attention weight visualization.

    Future usage:
        from basicts.analysis import AttentionVisualizer

        vis = AttentionVisualizer(model, output_dir="analysis/attention")
        vis.plot_attention_heatmap(sample_input, layer=0, head=0)
        vis.plot_cross_variable_attention(sample_input)
    """

    def __init__(self, model=None, output_dir: str = "analysis/attention"):
        """
        Initialize attention visualizer.

        Args:
            model: A trained model with output_attentions=True support.
            output_dir: Directory to save attention plots.
        """
        self.model = model
        self.output_dir = output_dir

    def plot_attention_heatmap(
        self,
        attention_weights: np.ndarray,
        layer: int = 0,
        head: int = 0,
        filename: Optional[str] = None,
    ) -> Optional[str]:
        """
        Plot attention weight heatmap for a single layer/head.

        Args:
            attention_weights: Attention weights array (layers, heads, seq, seq).
            layer: Layer index.
            head: Head index.
            filename: Output filename.

        Returns:
            Path to saved figure.
        """
        raise NotImplementedError(
            "AttentionVisualizer is a reserved interface for future implementation. "
            "To use, set output_attentions=True in model config and implement this method."
        )

    def plot_cross_variable_attention(
        self,
        attention_weights: np.ndarray,
        variable_names: Optional[list] = None,
        filename: Optional[str] = None,
    ) -> Optional[str]:
        """
        Plot cross-variable attention patterns (e.g., for iTransformer).

        Args:
            attention_weights: Variable-level attention weights.
            variable_names: Names of variables for axis labels.
            filename: Output filename.

        Returns:
            Path to saved figure.
        """
        raise NotImplementedError(
            "Cross-variable attention visualization is planned for future implementation."
        )
