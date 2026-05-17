from dataclasses import dataclass, field
from typing import Callable, List, Literal, Tuple, Union

import numpy as np
from torch.optim import Adam

from basicts.runners.callback import BasicTSCallback
from basicts.runners.taskflow import BasicTSTaskFlow, BasicTSSoftSensorTaskFlow
from basicts.scaler import ZScoreScaler
from basicts.data import BasicTSSoftSensorDataset

from .base_config import BasicTSConfig
from .model_config import BasicTSModelConfig


@dataclass(init=False)
class BasicTSSoftSensorConfig(BasicTSConfig):
    """
    BasicTS Soft Sensor Config, designed for soft sensor tasks.
    
    Soft sensor tasks predict key quality variables using correlated process variables,
    which is similar to multivariate input single-variable output forecasting.
    However, soft sensor tasks focus more on spatio-temporal relationship modeling
    and typically involve short-term prediction rather than long-term forecasting.
    
    **Key Characteristics of Soft Sensor Tasks:**
    - **Measurement Lag**: Quality variables typically have a measurement lag compared to process variables.
      For example, if lab analysis takes 1 hour and process data is sampled every 5 minutes, the lag is 12 steps.
    - **Estimation vs Prediction**: Soft sensor tasks involve estimating current quality values using available
      process variables, not purely predicting future values.
    - **Lag Constraint**: The prediction horizon must be >= measurement lag to ensure meaningful estimation.
    - **Short-term Focus**: Typically involves short-term prediction (1-24 steps) rather than long-term forecasting.
    - **Single Target**: Usually predicts a single quality variable from multiple process variables.
    
    **Required Fields:** These fields must be specified for running BasicTS soft sensor tasks.
    - `dataset_name` (str): Dataset name.
    - `model` (cls): Model class. You can pass its class name as string and it will be transformed into class type automatically.
    - `model_config` (BasicTSModelConfig): Model parameters.
    
    **Hot Fields:** Though these parameters have default settings, they are likely to be modified frequently.
    - `target_vars` (List[int]|int): Target variable indices for soft sensor prediction. Default: 0.
    - `input_vars` (List[int]|None): Input variable indices. None means use all variables. Default: None.
    - `input_len` (int): Input sequence length. Default: 96.
    - `output_len` (int): Output sequence length. Default: 1.
    - `gpus` (str|None): The used GPU devices. Default: None (on CPU).
    - `num_epochs` (int): Number of training epochs. Default: 100.
    - `batch_size` (int): Batch size. Default: 64.
    - `loss` (cls): Loss function. Default: MAE.
    - `seed` (int): Random seed. Default: 42.
    """

    ################################# Required Fields #################################

    model: type = field(metadata={"help": "Model class. Must be specified."})

    model_config: BasicTSModelConfig = field(metadata={"help": "Model configuration. Must be specified."})

    dataset_name: str = field(default=None, metadata={"help": "Dataset name. Must be specified if it is not in `dataset_params`."})

    ############################## Soft Sensor Specific Configuration ##############################

    # Measurement lag - critical parameter for soft sensor
    measurement_lag: int = field(
        default=1,
        metadata={"help": "Measurement lag of quality variables (in time steps). "
                  "For example, if lab analysis takes 1 hour and process data is sampled every 5 minutes, "
                  "the lag is 12 steps. This represents the delay between process variables and quality measurements. "
                  "Must be <= output_len. Default: 1 (at least 1 step lag)."})

    # Target and input variables
    target_vars: Union[List[int], int] = field(
        default=0,
        metadata={"help": "Target variable indices for soft sensor prediction. Can be a single index or a list of indices."})

    input_vars: Union[List[int], None] = field(
        default=None,
        metadata={"help": "Input variable indices. None means use all variables."})

    exclude_target_from_input: bool = field(
        default=True,
        metadata={"help": "Whether to exclude target variables from input features. "
                  "If True (default), only use other variables as input (no autoregression). "
                  "If False, include target variables in input (supports autoregression). "
                  "Set to False for autoregressive models or ablation studies."})

    # Feature engineering for soft sensor
    use_spatial_features: bool = field(
        default=False,
        metadata={"help": "Whether to use spatial features (e.g., sensor locations, process topology)."})

    use_temporal_features: bool = field(
        default=True,
        metadata={"help": "Whether to use temporal features (e.g., timestamps, time differences)."})

    ############################## General Configuration ##############################

    gpus: Union[str, None] = field(
        default=None,
        metadata={"help": "Whether to use GPUs. The default is None (on CPU). For example, '0,1' is using 'cuda:0' and 'cuda:1'."})

    gpu_num: int = field(default=0, metadata={"help": "Post-init. Number of GPUs."})

    seed: int = field(default=42, metadata={"help": "Random seed."})

    taskflow: BasicTSTaskFlow = field(default_factory=lambda: BasicTSSoftSensorTaskFlow(), metadata={"help": "Taskflow. Default: BasicTSSoftSensorTaskFlow."})

    callbacks: List[BasicTSCallback] = field(default_factory=list, metadata={"help": "Callbacks."})

    ddp_find_unused_parameters: bool = field(
        default=False,
        metadata={"help": "Controls the `find_unused_parameters parameter` of `torch.nn.parallel.DistributedDataParallel`."})

    compile_model: bool = field(default=False, metadata={"help": "Whether to compile model."})

    ############################## Dataset and Scaler Configuration ##############################

    # Dataset settings - adjusted for soft sensor tasks
    dataset_type: type = field(default=BasicTSSoftSensorDataset, metadata={"help": "Dataset type. Default: BasicTSSoftSensorDataset."})
    
    dataset_params: Union[dict, None] = field(
        default_factory=lambda: {
            "input_len": 96,
            "output_len": 1,
            "use_timestamps": True,
            "memmap": False,
        }, metadata={"help": "Dataset parameters."})

    # shortcuts
    input_len: int = field(default=96, metadata={"help": "Input length. Default: 96 for soft sensor tasks."})
    output_len: int = field(default=1, metadata={"help": "Output length. Default: 1 for immediate prediction."})
    use_timestamps: bool = field(default=True, metadata={"help": "Whether to use timestamps as supplementary."})
    memmap: bool = field(default=None, metadata={"help": "Whether to use memmap to load datasets."})
    batch_size: Union[int, None] = field(
        default=None, metadata={"help": "Batch size. If setted, all dataloaders will be setted to the same batch size."})

    null_val: float = field(default=np.nan, metadata={"help": "Null value."})
    null_to_num: float = field(default=0.0, metadata={"help": "Null value to number."})

    # Scaler settings
    scaler: type = field(default=ZScoreScaler, metadata={"help": "Scaler type."})
    norm_each_channel: bool = field(default=True, metadata={"help": "Whether to normalize data for each channel independently."})
    rescale: bool = field(default=True, metadata={"help": "Whether to rescale data. Default: True for soft sensor tasks."})

    ############################## Metrics Configuration ##############################

    metrics: List[Union[str, Tuple[str, Callable]]] = field(
        default_factory=lambda: ["MAE", "MSE", "RMSE", "MAPE", "R2"],
        metadata={"help": "Metric names for soft sensor evaluation. R2 is commonly used for soft sensor tasks."})

    target_metric: str = field(
        default="RMSE",
        metadata={"help": "Target metric, used for saving best checkpoints. Default: RMSE for soft sensor tasks."})

    best_metric: Literal["min", "max"] = field(
        default="min",
        metadata={"help": "Best metric, used for saving best checkpoints. Should be 'min' or 'max'."})

    ############################## Training Configuration ##############################

    num_epochs: int = field(
        default=100, metadata={"help": "Number of epochs. If not None, the training will stop after `num_epochs` epochs."})

    num_steps: Union[int, None] = field(
        default=None, metadata={"help": "Number of steps. If not None, the training will stop after `num_steps` steps."})

    loss: Union[str, Callable] = field(
        default="MAE", metadata={"help": "Loss function. If a string, it should be in `basicts.metrics.ALL_METRICS`."})

    # Optimizer - slightly different defaults for soft sensor
    optimizer: type = field(default=Adam, metadata={"help": "Optimizer class."})
    optimizer_params: dict = field(
        default_factory=lambda: {"lr": 1e-3, "weight_decay": 1e-4},
        metadata={"help": "Optimizer parameters. Higher learning rate for soft sensor tasks."})
    lr: float = field(default=None, metadata={"help": "Learning rate."})

    # Learning rate scheduler
    lr_scheduler: Union[type, None] = field(default=None, metadata={"help": "Learning rate scheduler type."})
    lr_scheduler_params: Union[dict, None] = field(default=None, metadata={"help": "Learning rate scheduler parameters."})

    # Checkpoint loading and saving settings
    ckpt_save_dir: str = field(
        default=None,
        metadata={"help": "Directory to save checkpoints." \
                  "Default: 'checkpoints/{model_name}/{dataset_name}_{num_epochs}_{input_len}_{output_len}', which will be post-initialized."})

    ckpt_save_strategy: Union[int, List[int], Tuple[int]] = field(
        default_factory=lambda: None,
        metadata={"help": "Checkpoint save strategy. " \
                  "None: remove last checkpoint file every epoch. " \
                  "Int: save checkpoint every `CFG.TRAIN.CKPT_SAVE_STRATEGY` epoch. " \
                  "List or Tuple: save checkpoint when epoch in `CFG.TRAIN.CKPT_SAVE_STRATEGY`, " \
                    "remove last checkpoint file when last_epoch not in ckpt_save_strategy."})

    finetune_from: Union[str, None] = field(
        default=None,
        metadata={"help": "Checkpoint path for fine-tuning. If not specified, the model will be trained from scratch."})

    strict_load: bool = field(default=True, metadata={"help": "Whether to strictly load the checkpoint."})

    # Train data loader settings
    train_batch_size: int = field(default=64, metadata={"help": "Batch size for training."})
    train_data_prefetch: bool = field(default=False, metadata={"help": "Whether to use dataloader with prefetch."})
    train_data_shuffle: bool = field(default=True, metadata={"help": "Whether to shuffle the training data."})
    train_data_collate_fn: Union[Callable, None] = field(default=None, metadata={"help": "Collate function for the training dataloader."})
    train_data_num_workers: int = field(default=0, metadata={"help": "Number of workers for the training dataloader."})
    train_data_pin_memory: bool = field(default=False, metadata={"help": "Whether to pin memory for the training dataloader."})

    ############################## Validation Configuration ##############################

    val_batch_size: int = field(default=64, metadata={"help": "Batch size for validation."})
    val_interval: int = field(default=1, metadata={"help": "Conduct validation every `val_interval` epochs."})
    val_data_prefetch: bool = field(default=False, metadata={"help": "Whether to use dataloader with prefetch."})
    val_data_shuffle: bool = field(default=False, metadata={"help": "Whether to shuffle the validation data."})
    val_data_collate_fn: Union[Callable, None] = field(default=None, metadata={"help": "Collate function for the validation dataloader."})
    val_data_num_workers: int = field(default=0, metadata={"help": "Number of workers for the validation dataloader."})
    val_data_pin_memory: bool = field(default=False, metadata={"help": "Whether to pin memory for the validation dataloader."})

    ############################## Test Configuration ##############################

    test_batch_size: int = field(default=64, metadata={"help": "Batch size for testing."})
    test_interval: int = field(default=1, metadata={"help": "Conduct testing every `test_interval` epochs."})
    test_data_prefetch: bool = field(default=False, metadata={"help": "Whether to use dataloader with prefetch."})
    test_data_shuffle: bool = field(default=False, metadata={"help": "Whether to shuffle the testing data."})
    test_data_collate_fn: Union[Callable, None] = field(default=None, metadata={"help": "Collate function for the testing dataloader."})
    test_data_num_workers: int = field(default=0, metadata={"help": "Number of workers for the testing dataloader."})
    test_data_pin_memory: bool = field(default=False, metadata={"help": "Whether to pin memory for the testing dataloader."})

    ########################### Evaluation Configuration ##########################

    eval_horizons: Union[List[int], None] = field(
        default=None, metadata={"help": "Horizons for evaluation. For soft sensor, typically evaluates at specific time steps."})

    eval_after_train: bool = field(default=True, metadata={"help": "Whether to evaluate the model after training."})

    save_results: bool = field(default=False, metadata={"help": "Whether to save evaluation results in a numpy file."})

    ############################## Environment Configuration ##############################

    tf32: bool = field(default=False, metadata={"help": "Whether to use TensorFloat-32 in GPU."})

    deterministic: bool = field(default=False, metadata={"help": "Whether to set the random seed to get deterministic results."})

    cudnn_enabled: bool = field(default=True, metadata={"help": "Whether to enable cuDNN."})

    cudnn_benchmark: bool = field(default=True, metadata={"help": "Whether to enable cuDNN benchmark."})

    cudnn_determinstic: bool = field(default=False, metadata={"help": "Whether to set cuDNN to deterministic mode."})

    ##################################### Post Init #######################################

    def __post_init__(self):
        # Validate measurement lag
        if self.measurement_lag < 1:
            raise ValueError(f"measurement_lag must be >= 1, got {self.measurement_lag}. "
                           "Quality variables should have at least 1 step lag compared to process variables.")
        
        if self.output_len < self.measurement_lag:
            raise ValueError(f"output_len ({self.output_len}) must be >= measurement_lag ({self.measurement_lag}). "
                           "Soft sensor prediction horizon should cover the measurement lag.")
        
        # Convert single target variable to list
        if isinstance(self.target_vars, int):
            self.target_vars = [self.target_vars]
        
        # Set default checkpoint save directory
        if self.ckpt_save_dir is None:
            self.ckpt_save_dir = \
                f"checkpoints/{self.model.__name__}/{self.dataset_name}_{self.num_epochs}_{self.input_len}_{self.output_len}_lag{self.measurement_lag}"
        
        # Add exclude_target_from_input to dataset_params if not already set
        if self.dataset_params is not None and 'exclude_target_from_input' not in self.dataset_params:
            self.dataset_params['exclude_target_from_input'] = self.exclude_target_from_input
