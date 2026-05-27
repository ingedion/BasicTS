import os
from typing import Union

import numpy as np

from basicts.utils.constants import BasicTSMode

from .base_dataset import BasicTSDataset


class EncDecSoftSensorDataset(BasicTSDataset):
    """
    Dataset for encoder-decoder soft sensor tasks.

    Splits data along the TIME dimension (not variable dimension) to produce
    encoder/decoder input pairs suitable for models like Informer and Autoformer.

    Key differences from BasicTSSoftSensorDataset:
    - Returns `inputs` (encoder input) and `targets` (decoder input) split by time
    - Both inputs and targets contain ALL variables (num_features), no variable splitting
    - Decoder input has masking applied: quality vars zeroed in lag zone, all vars zeroed in prediction zone
    - Supports lookback (decoder sees some known history) and pred_len (future prediction beyond lag)

    Time window layout:
        Raw data:   [t_begin ............... t0 ............... t_end ... t_end+pred_len)
        Encoder:    [t_begin, t0)                   length = input_len - measurement_lag
        Decoder:    [t0 - lookback, t_end + pred_len)   length = lookback + measurement_lag + pred_len

        Where t0 = t_begin + input_len - measurement_lag (first unknown quality variable timestep)

    Decoder masking zones:
        [0, lookback)                           → all variables keep true values
        [lookback, lookback+measurement_lag)    → process vars keep true values, quality vars = 0
        [lookback+measurement_lag, decoder_len) → all variables = 0

    Attributes:
        dataset_name (str): The name of the dataset.
        input_len (int): Total input window length.
        measurement_lag (int): Measurement lag of quality variables in time steps.
        target_vars (list): Indices of target (quality) variables.
        lookback (int): Number of known timesteps prepended to decoder input.
        pred_len (int): Number of prediction steps beyond the input window.
        encoder_len (int): Length of encoder input = input_len - measurement_lag.
        decoder_len (int): Length of decoder input = lookback + measurement_lag + pred_len.
    """

    def __init__(
            self,
            dataset_name: str,
            input_len: int,
            measurement_lag: int,
            target_vars: Union[int, list],
            mode: Union[BasicTSMode, str],
            lookback: int = 0,
            pred_len: int = 0,
            use_timestamps: bool = False,
            data_file_path: Union[str, None] = None,
            memmap: bool = False) -> None:
        """
        Initializes the EncDecSoftSensorDataset.

        Args:
            dataset_name (str): Dataset name.
            input_len (int): Total input window length.
            measurement_lag (int): Measurement lag of quality variables.
            target_vars (Union[int, list]): Target variable indices (quality variables).
            mode (Union[BasicTSMode, str]): Dataset mode (train/val/test).
            lookback (int): Decoder lookback window length. Defaults to 0.
            pred_len (int): Prediction steps beyond input window. Defaults to 0.
            use_timestamps (bool): Whether to load and return timestamps.
            data_file_path (str | None): Path to data directory. Defaults to datasets/{dataset_name}.
            memmap (bool): Whether to use memory-mapped file loading.

        Raises:
            ValueError: If parameter validation fails.
            FileNotFoundError: If data files cannot be found.
        """
        super().__init__(dataset_name, mode, memmap)

        # Validate parameters
        if measurement_lag < 1:
            raise ValueError(f"measurement_lag must be >= 1, got {measurement_lag}")
        if input_len <= measurement_lag:
            raise ValueError(
                f"input_len must be > measurement_lag, got input_len={input_len}, "
                f"measurement_lag={measurement_lag}")
        if lookback < 0:
            raise ValueError(f"lookback must be >= 0, got {lookback}")
        if lookback > input_len - measurement_lag:
            raise ValueError(
                f"lookback must be <= input_len - measurement_lag, got lookback={lookback}, "
                f"input_len - measurement_lag={input_len - measurement_lag}")
        if pred_len < 0:
            raise ValueError(f"pred_len must be >= 0, got {pred_len}")

        self.input_len = input_len
        self.measurement_lag = measurement_lag
        self.lookback = lookback
        self.pred_len = pred_len
        self.use_timestamps = use_timestamps

        # Compute encoder/decoder lengths
        self.encoder_len = input_len - measurement_lag
        self.decoder_len = lookback + measurement_lag + pred_len

        # Convert target_vars to list
        self.target_vars = [target_vars] if isinstance(target_vars, int) else list(target_vars)

        # Load data
        if data_file_path is None:
            data_file_path = f"datasets/{dataset_name}"

        try:
            self._data = np.load(
                os.path.join(data_file_path, f"{mode}_data.npy"),
                mmap_mode="r" if memmap else None)
            if use_timestamps:
                self.timestamps = np.load(
                    os.path.join(data_file_path, f"{mode}_timestamps.npy"),
                    mmap_mode="r" if memmap else None)
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"Cannot load dataset from {data_file_path}. "
                f"Please set a correct local path."
            ) from e

        # Resolve negative indices and validate target_vars range
        num_vars = self._data.shape[-1]
        self.target_vars = [v % num_vars for v in self.target_vars]
        for v in self.target_vars:
            if v < 0 or v >= num_vars:
                raise ValueError(
                    f"target_vars index {v} out of range [0, {num_vars})")

        self.num_features = num_vars

    def __getitem__(self, index: int) -> dict:
        """
        Retrieves a sample from the dataset.

        Encoder-decoder soft sensor semantics:
        - Encoder receives the full history before the first unknown quality measurement (t0).
        - Decoder receives data from t0-lookback onwards, with masking applied to prevent
          information leakage of quality variables.

        Data layout for a sample at `index`:
            t_begin = index
            t0 = index + encoder_len  (= index + input_len - measurement_lag)
            t_end = index + input_len
            
            Encoder input: raw_data[t_begin : t0]           shape: [encoder_len, num_features]
            Decoder input: raw_data[t0 - lookback : t_end + pred_len]  shape: [decoder_len, num_features]
            
            Decoder masking:
              [0, lookback)                           → true values (all vars)
              [lookback, lookback+measurement_lag)    → process vars true, quality vars = 0
              [lookback+measurement_lag, decoder_len) → all vars = 0

        Args:
            index (int): Sample index.

        Returns:
            dict: Contains:
                - 'inputs': [encoder_len, num_features] encoder input (all vars, true values)
                - 'targets': [decoder_len, num_features] decoder input (with masking)
                - 'inputs_timestamps': [encoder_len, num_timestamps] (if use_timestamps)
                - 'targets_timestamps': [decoder_len, num_timestamps] (if use_timestamps)
        """
        item = {}

        # Compute time indices
        t_begin = index
        t0 = index + self.encoder_len  # first unknown quality variable timestep
        t_end = index + self.input_len

        # Encoder input: [t_begin, t0) - all variables, true values
        encoder_data = self._data[t_begin: t0]

        # Decoder input: [t0 - lookback, t_end + pred_len) - with masking
        decoder_start = t0 - self.lookback
        decoder_end = t_end + self.pred_len
        decoder_data = self._data[decoder_start: decoder_end].copy()

        # Apply masking to decoder input
        # Lag zone: [lookback, lookback+measurement_lag) - quality vars = 0
        decoder_data[self.lookback: self.lookback + self.measurement_lag, self.target_vars] = 0

        # Prediction zone: [lookback+measurement_lag, decoder_len) - all vars = 0
        pred_zone_start = self.lookback + self.measurement_lag
        if pred_zone_start < self.decoder_len:
            decoder_data[pred_zone_start:, :] = 0

        item["inputs"] = encoder_data.copy() if self.memmap else encoder_data
        item["targets"] = decoder_data

        if self.use_timestamps:
            encoder_timestamps = self.timestamps[t_begin: t0]
            decoder_timestamps = self.timestamps[decoder_start: decoder_end]
            item["inputs_timestamps"] = encoder_timestamps.copy() if self.memmap else encoder_timestamps
            item["targets_timestamps"] = decoder_timestamps.copy() if self.memmap else decoder_timestamps

        return item

    def __len__(self) -> int:
        """
        Returns the number of samples in the dataset.

        Total samples = data_len - input_len - pred_len + 1
        """
        return len(self._data) - self.input_len - self.pred_len + 1

    @property
    def data(self) -> np.ndarray:
        """Returns the raw data array."""
        return self._data
