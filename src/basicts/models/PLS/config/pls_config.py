from dataclasses import dataclass, field

from basicts.configs import BasicTSModelConfig


@dataclass
class PLSConfig(BasicTSModelConfig):
    """
    Config class for PLS (Partial Least Squares) model.
    
    PLS is a linear regression method that projects input and output variables
    into a lower-dimensional latent space. It is particularly effective when
    input variables are highly collinear (common in industrial process data).
    """

    input_len: int = field(default=None, metadata={"help": "Input sequence length."})
    output_len: int = field(default=None, metadata={"help": "Output sequence length."})
    num_features: int = field(default=None, metadata={"help": "Number of input features."})
    num_targets: int = field(default=1, metadata={"help": "Number of target features."})
    n_components: int = field(default=None, metadata={
        "help": "Number of PLS components. If None, defaults to min(n_samples, n_features, n_targets)."
    })
