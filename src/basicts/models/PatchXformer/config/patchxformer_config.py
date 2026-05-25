from dataclasses import dataclass, field

from basicts.configs import BasicTSModelConfig


@dataclass
class PatchXformerConfig(BasicTSModelConfig):

    """
    Config class for PatchXformer model.

    PatchXformer is a dual-axis attention Transformer for soft sensing tasks,
    combining temporal patch attention with variate attention and soft-sensor-specific
    mechanisms (variable importance gating, lag-aware positional encoding, asymmetric attention).
    """

    input_len: int = field(default=None, metadata={"help": "Input sequence length."})
    output_len: int = field(default=None, metadata={"help": "Output sequence length for forecasting task."})
    num_features: int = field(default=None, metadata={"help": "Number of input features (variables)."})
    patch_len: int = field(default=16, metadata={"help": "Patch length."})
    patch_stride: int = field(default=8, metadata={"help": "Stride for patching."})
    padding: bool = field(default=True, metadata={"help": "Whether to pad the input sequence before patching."})
    hidden_size: int = field(default=256, metadata={"help": "Hidden size."})
    n_heads: int = field(default=4, metadata={"help": "Number of heads in multi-head attention."})
    intermediate_size: int = field(default=512, metadata={"help": "Intermediate size of FFN layers."})
    hidden_act: str = field(default="gelu", metadata={"help": "Activation function."})
    num_layers: int = field(default=2, metadata={"help": "Number of dual-axis encoder blocks."})
    dropout: float = field(default=0.1, metadata={"help": "Dropout rate."})
    use_revin: bool = field(default=True, metadata={"help": "Whether to use RevIN normalization."})
    output_attentions: bool = field(default=False, metadata={"help": "Whether to output attention weights."})

    # Soft-sensor-specific fields
    measurement_lag: int = field(default=0, metadata={"help": "Measurement delay in time steps. When 0, all positions are treated equally without lag-aware encoding."})
    use_variable_gate: bool = field(default=True, metadata={"help": "Whether to use variable importance gating."})
    use_asymmetric_attn: bool = field(default=True, metadata={"help": "Whether to use asymmetric cross-variate attention (process vars as K/V, target as Q)."})
    target_var_index: int = field(default=-1, metadata={"help": "Target variable position index. -1 means last variable."})
