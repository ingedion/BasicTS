"""
Ethylene Distillation Column Dataset Preparation for Soft Sensor Task.

Source: Industrial ethylene distillation column process data.

Description:
    38 process variables sampled at 1-minute intervals.
    Target variable: 塔顶乙烷浓度 (ethane concentration at column top),
    which is a key product quality indicator.

    Time range: 2016-10-13 13:00 ~ 2016-10-20 12:59 (7 days, 10080 samples)
    Sampling frequency: 1 minute
    Missing values: None

File format:
    - Excel (.xlsx) with datetime index column
    - 38 numeric process variable columns
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
dataset_name = 'EthyDistillation'
data_file_path = base_dir + '/datasets/raw_data/ethy/ethy.xlsx'
graph_file_path = None
output_dir = base_dir + f'/datasets/{dataset_name}'
target_column = '塔顶乙烷浓度'  # Ethane concentration at column top
add_time_of_day = True
add_day_of_week = True
add_day_of_month = False
add_day_of_year = False
steps_per_day = 1440  # 1-minute sampling: 60 * 24 = 1440 steps per day
frequency = 1  # 1 minute
domain = 'industrial ethylene distillation column (soft sensor)'
timestamps_desc = ['time of day', 'day of week']
regular_settings = {
    'train_val_test_ratio': [0.7, 0.1, 0.2],
    'norm_each_channel': True,
    'rescale': False,
    'metrics': ['MAE', 'MSE', 'RMSE', 'R2'],
    'null_val': np.nan
}


def load_and_preprocess_data():
    '''Load and preprocess raw data from Excel file.'''

    df = pd.read_excel(data_file_path)

    # Set datetime index
    time_col = df.columns[0]  # 'Unnamed: 0' is the datetime column
    df_index = pd.to_datetime(df[time_col].values)
    df = df.drop(columns=[time_col])
    df.index = df_index

    # Strip whitespace from column names
    df.columns = df.columns.str.strip()

    print(f'Raw time series shape: {df.shape}')
    print(f'Columns ({len(df.columns)}): {list(df.columns)}')
    print(f'Target variable: {target_column}')
    print(f'Target column index: {list(df.columns).index(target_column)}')
    print(f'Time range: {df.index[0]} ~ {df.index[-1]}')
    print(f'Sampling frequency: {frequency} min')

    return df


def add_temporal_features(df):
    '''Add time of day and day of week as features to the data.'''
    timestamps = []

    if add_time_of_day:
        # Normalized time of day [0, 1)
        tod = (df.index.hour * 60 + df.index.minute) / 1440.0
        timestamps.append(tod.values)

    if add_day_of_week:
        # Normalized day of week [0, 1)
        dow = df.index.dayofweek / 7.0
        timestamps.append(dow.values)

    if add_day_of_month:
        dom = (df.index.day - 1) / 31.0
        timestamps.append(dom.values)

    if add_day_of_year:
        doy = (df.index.dayofyear - 1) / 366.0
        timestamps.append(doy.values)

    timestamps = np.stack(timestamps, axis=-1)
    return timestamps


def split_and_save_data(data, timestamps):
    '''Save the preprocessed data to binary files.'''
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

    print(f"\ntrain_data shape: {train_data.shape}")
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


def save_description(data, timestamps, column_names):
    '''Save a description of the dataset to a JSON file.'''
    target_index = list(column_names).index(target_column)

    description = {
        'name': dataset_name,
        'domain': domain,
        'frequency (minutes)': frequency,
        'shape': data.shape,
        'timestamps_shape': timestamps.shape,
        'timestamps_description': timestamps_desc,
        'num_time_steps': data.shape[0],
        'num_vars': data.shape[1],
        'variable_names': list(column_names),
        'target_variable': target_column,
        'target_index': target_index,
        'has_graph': graph_file_path is not None,
        'regular_settings': regular_settings,
    }
    description_path = os.path.join(output_dir, 'meta.json')
    with open(description_path, 'w', encoding='utf-8') as f:
        json.dump(description, f, indent=4, ensure_ascii=False)
    print(f'Description saved to {description_path}')
    print(f'Target: {target_column} (index={target_index})')
    print('\n')


def main():
    print(f"---------- Generating {dataset_name} data ----------")

    # Load and preprocess data
    df = load_and_preprocess_data()

    # Add temporal features
    timestamps = add_temporal_features(df)

    # Save processed data
    split_and_save_data(df.values, timestamps)

    # Save dataset description
    save_description(df.values, timestamps, df.columns)


if __name__ == '__main__':
    main()
