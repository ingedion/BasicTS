"""
Debutanizer Dataset Preparation for Soft Sensor Task.

Source: Fortuna L., Graziani S., Rizzo A., Xibilia M.G. (2007).
"Soft Sensors for Monitoring and Control of Industrial Processes".

Description:
    Industrial debutanizer column data with 7 process variables (u1~u7)
    and 1 quality variable (y, butane content at the bottom of the column).

    The quality variable y is the LAST column, suitable for soft sensor tasks
    where the goal is to estimate/predict y from u1~u7.

File format:
    - Whitespace-separated text file
    - Header line: "u1 u2 u3 u4 u5 u6 u7 y"
    - ~2394 samples (no timestamps)

Note: This dataset has NO real timestamps. Synthetic step-based timestamps
are generated for compatibility with BasicTS data loading pipeline.
"""

import json
import os

import numpy as np
import pandas as pd

# Current path
current_dir = os.path.dirname(os.path.abspath(__file__))

# target path
base_dir = os.path.abspath(os.path.dirname(os.path.join(current_dir, '../..', '../..')))

# Hyperparameters
dataset_name = 'Debutanizer'
data_file_path = base_dir + f'/datasets/raw_data/debutanizer/Debutanizer.txt'
graph_file_path = None
output_dir = base_dir + f'/datasets/{dataset_name}'
target_channel = [-1]  # y (quality variable) is the last column
add_time_of_day = False  # No real timestamps in this dataset
add_day_of_week = False
add_day_of_month = False
add_day_of_year = False
add_step_index = True   # Use normalized step index as a synthetic time feature
domain = 'industrial debutanizer column (soft sensor)'
timestamps_desc = ['step_index_normalized']
regular_settings = {
    'train_val_test_ratio': [0.7, 0.1, 0.2],
    'norm_each_channel': True,
    'rescale': False,
    'metrics': ['MAE', 'MSE', 'RMSE', 'R2'],
    'null_val': np.nan
}


def load_and_preprocess_data():
    '''Load and preprocess raw data from whitespace-separated txt file.'''

    # Read whitespace-separated file with header
    df = pd.read_csv(data_file_path, sep=r'\s+', engine='python')

    # Strip any whitespace in column names (header has trailing space: 'y ')
    df.columns = df.columns.str.strip()

    print(f'Raw time series shape: {df.shape}')
    print(f'Columns: {list(df.columns)}')
    print(f'Quality variable (target): {df.columns[-1]}')
    return df


def add_temporal_features(df):
    '''Add synthetic step-based features (no real timestamps available).'''
    l = df.shape[0]
    timestamps = []

    if add_step_index:
        # Normalized step index in [0, 1)
        step_idx = np.arange(l) / l
        timestamps.append(step_idx)

    timestamps = np.stack(timestamps, axis=-1)
    return timestamps


def split_and_save_data(data, timestamps):
    '''Save the preprocessed data to a binary file.'''
    train_ratio, val_ratio, _ = regular_settings['train_val_test_ratio']
    train_len = int(data.shape[0] * train_ratio)
    val_len = int(data.shape[0] * val_ratio)

    train_data = data[:train_len].astype(np.float32)
    val_data = data[train_len : train_len + val_len].astype(np.float32)
    test_data = data[train_len + val_len :].astype(np.float32)
    train_timestamps = timestamps[:train_len].astype(np.float32)
    val_timestamps = timestamps[train_len : train_len + val_len].astype(np.float32)
    test_timestamps = timestamps[train_len + val_len :].astype(np.float32)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"train_data shape: {train_data.shape}")
    np.save(os.path.join(output_dir, 'train_data.npy'), train_data)
    print(f"val_data shape: {val_data.shape}")
    np.save(os.path.join(output_dir, 'val_data.npy'), val_data)
    print(f"test_data shape: {test_data.shape}")
    np.save(os.path.join(output_dir, 'test_data.npy'), test_data)
    print(f"train_timestamps shape: {train_timestamps.shape}")
    np.save(os.path.join(output_dir, 'train_timestamps.npy'), train_timestamps)
    print(f"val_timestamps shape: {val_timestamps.shape}")
    np.save(os.path.join(output_dir, 'val_timestamps.npy'), val_timestamps)
    print(f"test_timestamps shape: {test_timestamps.shape}")
    np.save(os.path.join(output_dir, 'test_timestamps.npy'), test_timestamps)
    print(f'Data saved to {output_dir}')


def save_description(data, timestamps):
    '''Save a description of the dataset to a JSON file.'''
    description = {
        'name': dataset_name,
        'domain': domain,
        'frequency (minutes)': None,
        'shape': data.shape,
        'timestamps_shape': timestamps.shape,
        'timestamps_description': timestamps_desc,
        'num_time_steps': data.shape[0],
        'num_vars': data.shape[1],
        'variable_names': ['u1', 'u2', 'u3', 'u4', 'u5', 'u6', 'u7', 'y'],
        'target_variable': 'y',
        'target_index': -1,
        'has_graph': graph_file_path is not None,
        'regular_settings': regular_settings,
    }
    description_path = os.path.join(output_dir, 'meta.json')
    with open(description_path, 'w') as f:
        json.dump(description, f, indent=4)
    print(f'Description saved to {description_path}')
    print(description)
    print('\n')


def main():
    print(f"---------- Generating {dataset_name} data ----------")

    # Load and preprocess data
    df = load_and_preprocess_data()

    # Add temporal features (synthetic, since no real timestamps)
    timestamps = add_temporal_features(df)

    # Save processed data
    split_and_save_data(df.values, timestamps)

    # Save dataset description
    save_description(df.values, timestamps)


if __name__ == '__main__':
    main()
