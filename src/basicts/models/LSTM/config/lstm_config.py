from dataclasses import dataclass, field

from basicts.configs import BasicTSModelConfig


@dataclass
class LSTMConfig(BasicTSModelConfig):
    """
    Config class for LSTM model.
    
    Standard LSTM encoder with a linear projection head for multi-step prediction.
    Serves as the nonlinear temporal baseline in the soft sensor benchmark.
    """

    input_len: int = field(default=None, metadata={"help": "Input sequence length."})
    output_len: int = field(default=None, metadata={"help": "Output sequence length."})
    num_features: int = field(default=None, metadata={"help": "Number of input features."})
    num_targets: int = field(default=1, metadata={"help": "Number of target features."})
    hidden_size: int = field(default=64, metadata={"help": "LSTM hidden state size."})
    num_layers: int = field(default=2, metadata={"help": "Number of LSTM layers."})
    dropout: float = field(default=0.1, metadata={"help": "Dropout rate between LSTM layers."})
    bidirectional: bool = field(default=False, metadata={"help": "Whether to use bidirectional LSTM."})
