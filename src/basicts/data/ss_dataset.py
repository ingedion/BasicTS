import os
from typing import Union

import numpy as np

from basicts.utils.constants import BasicTSMode

from .base_dataset import BasicTSDataset


class BasicTSSoftSensorDataset(BasicTSDataset):
    """
    Dataset class for soft sensor tasks.
    
    Key difference from forecasting: 
    - Soft sensor predicts target variables (quality variables) that have measurement lag
    - Uses process variables (fast-sampled) to estimate quality variables (slow-sampled/lagged)
    - The prediction horizon must cover the measurement lag
    
    Attributes:
        dataset_name (str): The name of the dataset.
        input_len (int): Length of input sequence (historical process data).
        output_len (int): Length of output sequence (prediction horizon, must >= measurement_lag).
        measurement_lag (int): Measurement lag of quality variables in time steps.
        target_vars (Union[int, list]): Indices of target variables (quality variables).
        input_vars (Union[list, None]): Indices of input variables (process variables). None means use all vars.
        mode (Union[BasicTSMode, str]): Dataset mode (train/val/test).
        use_timestamps (bool): Whether to use timestamps.
        data_file_path (str | None): Path to data file.
        memmap (bool): Whether to use memory mapping.
    """

    def __init__(
            self,
            dataset_name: str,
            input_len: int,
            output_len: int,
            measurement_lag: int,
            target_vars: Union[int, list],
            mode: Union[BasicTSMode, str],
            input_vars: Union[list, None] = None,
            exclude_target_from_input: bool = True,
            use_timestamps: bool = False,
            local: bool = True,
            data_file_path: Union[str, None] = None,
            memmap: bool = False) -> None:
        """
        Initializes the BasicTSSoftSensorDataset.
        
        Args:
            dataset_name (str): Dataset name.
            input_len (int): Length of input sequence.
            output_len (int): Length of output sequence (must >= measurement_lag).
            measurement_lag (int): Measurement lag of quality variables.
            target_vars (Union[int, list]): Target variable indices.
            mode (Union[BasicTSMode, str]): Dataset mode.
            input_vars (Union[list, None]): Input variable indices. None = all variables.
            exclude_target_from_input (bool): Whether to exclude target vars from input.
            use_timestamps (bool): Whether to use timestamps.
            local (bool): Whether dataset is local.
            data_file_path (str | None): Path to data file.
            memmap (bool): Whether to use memory mapping.
        """
        super().__init__(dataset_name, mode, memmap)
        
        # Validate parameters
        if measurement_lag < 1:
            raise ValueError(f"measurement_lag must be >= 1, got {measurement_lag}")
        if output_len < measurement_lag:
            raise ValueError(f"output_len ({output_len}) must be >= measurement_lag ({measurement_lag}). "
                           "Output horizon must at least cover the measurement lag.")
        
        self.input_len = input_len
        self.output_len = output_len
        self.measurement_lag = measurement_lag
        self.exclude_target_from_input = exclude_target_from_input
        
        # Convert target_vars to list if single int
        self.target_vars = [target_vars] if isinstance(target_vars, int) else list(target_vars)
        self.input_vars = input_vars
        
        # Load data
        if not local:
            pass  # TODO: support remote download
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
                f"Please set a correct local path. "
                f"If you want to download the dataset, please set `local=False`."
            ) from e
        
        self.use_timestamps = use_timestamps
        
        # Determine input variable indices
        if self.input_vars is None:
            num_vars = self._data.shape[-1]
            # Resolve negative indices in target_vars to positive
            self.target_vars = [v % num_vars for v in self.target_vars]
            if self.exclude_target_from_input:
                # Use all variables except target variables
                self.input_vars = [i for i in range(num_vars) if i not in self.target_vars]
            else:
                # Use all variables including target variables
                self.input_vars = list(range(num_vars))
        else:
            num_vars = self._data.shape[-1]
            # Resolve negative indices
            self.target_vars = [v % num_vars for v in self.target_vars]
            self.input_vars = [v % num_vars for v in self.input_vars]

    def __getitem__(self, index: int) -> dict:
        """
        Retrieves a sample from the dataset.
        
        Predictive soft sensor semantics:
        - At time t, process variables are immediately available, but quality variables
          measured at time t won't be available until t + measurement_lag (lab analysis delay).
        - The model predicts output_len steps of quality variables starting from t+1.
        - The first measurement_lag steps are "estimation" (compensating for measurement delay):
          these quality values have already occurred but haven't been measured yet.
        - Steps beyond measurement_lag are genuine "prediction" of future quality values.
        
        Data layout:
            input window:  [index, index + input_len)                    → process variables
            target window: [index + input_len, index + input_len + output_len)  → quality variables
            
            Within the target window:
            [0, measurement_lag)        → estimation zone (已发生未测量)
            [measurement_lag, output_len) → prediction zone (未来预测)
        
        When output_len == measurement_lag: pure estimation (traditional soft sensor)
        When output_len >  measurement_lag: estimation + short-term prediction
        
        Args:
            index (int): Sample index.
            
        Returns:
            dict: Contains 'inputs' (process vars) and 'targets' (quality vars).
        """
        item = {}
        
        # Historical process data: [index, index + input_len)
        history_data = self._data[index: index + self.input_len]
        
        # Target quality variables: [index + input_len, index + input_len + output_len)
        # First measurement_lag steps = estimation, remaining = prediction
        future_start = index + self.input_len
        future_data = self._data[future_start: future_start + self.output_len]
        
        # Extract input variables (process variables)
        inputs = history_data[..., self.input_vars]
        
        # Extract target variables (quality variables)
        targets = future_data[..., self.target_vars]
        
        item["inputs"] = inputs.copy() if self.memmap else inputs
        item["targets"] = targets.copy() if self.memmap else targets
        
        if self.use_timestamps:
            history_timestamps = self.timestamps[index: index + self.input_len]
            future_timestamps = self.timestamps[future_start: future_start + self.output_len]
            item["inputs_timestamps"] = history_timestamps.copy() if self.memmap else history_timestamps
            item["targets_timestamps"] = future_timestamps.copy() if self.memmap else future_timestamps
        
        return item

    def __len__(self) -> int:
        """Returns the number of samples in the dataset."""
        return len(self._data) - self.input_len - self.output_len + 1

    @property
    def data(self) -> np.ndarray:
        """Returns the raw data array."""
        return self._data
