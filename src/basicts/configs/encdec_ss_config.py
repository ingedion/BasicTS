from dataclasses import dataclass, field
from typing import List, Union

from .ss_config import BasicTSSoftSensorConfig


@dataclass(init=False)
class EncDecSoftSensorConfig(BasicTSSoftSensorConfig):
    """
    Config for encoder-decoder soft sensor tasks.

    Extends BasicTSSoftSensorConfig with parameters specific to encoder-decoder
    architectures (Informer, Autoformer, etc.) that use time-dimension splitting
    to construct encoder/decoder input pairs for soft sensing.

    **New Parameters:**
    - `lookback` (int): Decoder lookback window length. The decoder receives
      `lookback` steps of fully-known historical data as guidance. Default: 0.
    - `pred_len` (int): Number of prediction steps beyond the input window.
      When > 0, the model performs soft sensing + short-term prediction. Default: 0.

    **Auto-computed:**
    - `output_len` is automatically set to `measurement_lag + pred_len + lookback`,
      which equals the decoder output length.

    **Defaults overridden:**
    - `dataset_type`: EncDecSoftSensorDataset
    - `taskflow`: EncDecSoftSensorTaskFlow()
    - `runner`: EncDecSoftSensorRunner
    """

    # New parameters for encoder-decoder soft sensor
    lookback: int = field(
        default=0,
        metadata={"help": "Decoder lookback window length. "
                  "The decoder receives `lookback` steps of fully-known historical data as guidance. "
                  "Must be >= 0 and <= input_len - measurement_lag. Default: 0."})

    pred_len: int = field(
        default=0,
        metadata={"help": "Number of prediction steps beyond the input window. "
                  "When > 0, the model performs soft sensing + short-term prediction. "
                  "Must be >= 0. Default: 0."})

    def __post_init__(self):
        # --- Validation specific to encoder-decoder soft sensor ---
        # Validate lookback
        if self.lookback < 0:
            raise ValueError(
                f"lookback must be >= 0, got {self.lookback}.")
        if self.lookback > self.input_len - self.measurement_lag:
            raise ValueError(
                f"lookback must be <= input_len - measurement_lag "
                f"({self.input_len} - {self.measurement_lag} = {self.input_len - self.measurement_lag}), "
                f"got {self.lookback}.")
        # Validate pred_len
        if self.pred_len < 0:
            raise ValueError(
                f"pred_len must be >= 0, got {self.pred_len}.")

        # Auto-compute output_len for model config
        # output_len = decoder output length = lookback + measurement_lag + pred_len
        self.output_len = self.measurement_lag + self.pred_len + self.lookback

        # --- Call parent __post_init__ for measurement_lag validation, target_vars resolution, etc. ---
        super().__post_init__()

        # --- Override defaults for encoder-decoder soft sensor ---
        # Use lazy imports to avoid circular imports since these classes
        # are created in later tasks.
        try:
            from basicts.data import EncDecSoftSensorDataset
            if self.dataset_type.__name__ == 'BasicTSSoftSensorDataset':
                self.dataset_type = EncDecSoftSensorDataset
        except ImportError:
            pass

        try:
            from basicts.runners.taskflow import EncDecSoftSensorTaskFlow
            if type(self.taskflow).__name__ == 'BasicTSSoftSensorTaskFlow':
                self.taskflow = EncDecSoftSensorTaskFlow()
        except ImportError:
            pass

        try:
            from basicts.runners import EncDecSoftSensorRunner
            if not hasattr(self, 'runner') or self.runner is None:
                self.runner = EncDecSoftSensorRunner
        except ImportError:
            pass

        # --- Override ckpt_save_dir to include lookback and pred_len ---
        # Pattern: checkpoints/{model}/{dataset}_{epochs}_{input_len}_{output_len}_lag{lag}_lb{lookback}_pred{pred_len}
        self.ckpt_save_dir = (
            f"checkpoints/{self.model.__name__}/"
            f"{self.dataset_name}_{self.num_epochs}_{self.input_len}_{self.output_len}"
            f"_lag{self.measurement_lag}_lb{self.lookback}_pred{self.pred_len}"
        )

        # Add lookback and pred_len to dataset_params for the dataset constructor
        if self.dataset_params is not None:
            self.dataset_params['lookback'] = self.lookback
            self.dataset_params['pred_len'] = self.pred_len
            # Remove output_len from dataset_params since EncDecSoftSensorDataset
            # doesn't accept it (it uses measurement_lag + lookback + pred_len instead)
            self.dataset_params.pop('output_len', None)
            # Remove exclude_target_from_input since it's not applicable to enc-dec
            self.dataset_params.pop('exclude_target_from_input', None)
