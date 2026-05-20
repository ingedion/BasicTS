from dataclasses import dataclass, field

from basicts.configs.model_config import BasicTSModelConfig


@dataclass
class SVRConfig(BasicTSModelConfig):
    """
    Config class for SVR (Support Vector Regression) model.
    
    SVR with RBF kernel is a classic nonlinear baseline in soft sensing.
    It captures nonlinear relationships between process variables and quality
    variables without requiring temporal modeling.
    """

    input_len: int = field(default=None, metadata={"help": "Input sequence length."})
    output_len: int = field(default=None, metadata={"help": "Output sequence length."})
    num_features: int = field(default=None, metadata={"help": "Number of input features."})
    num_targets: int = field(default=1, metadata={"help": "Number of target features."})
    kernel: str = field(default="rbf", metadata={"help": "SVR kernel type: 'rbf', 'linear', 'poly'."})
    C: float = field(default=1.0, metadata={"help": "Regularization parameter."})
    epsilon: float = field(default=0.1, metadata={"help": "Epsilon in the epsilon-SVR model."})
    gamma: str = field(default="scale", metadata={"help": "Kernel coefficient: 'scale', 'auto', or float."})
