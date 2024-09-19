import os
import urllib.request
import numpy as np
import pandas as pd


def get_datafile(url, data_dir, destination_file)-> None:
  """
  Function to download time series datafile from url and save it to destination.
  Args:
    url: url to download from
    data_dir: Folder to save to
    destination_file: file name to save as
  Returns:
    None
  """
  os.makedirs(data_dir, exist_ok=True)
  destination = data_dir+"/"+destination_file
  block_size = 1024
  with urllib.request.urlopen(url) as response:
    with open(destination, "wb") as file:
      while True:
        chunk = response.read(block_size)
        if not chunk:
          break
        file.write(chunk)


def get_data(data_file, index_col, target_col, new_target_col_name, ratio: float=0.9):
  """
  Accesses datafile with time series data, selects the target column.
  Returns a tuple of x_train, x_val, y_train, y_val.
  """
  df = pd.read_csv(data_file,parse_dates=[index_col],index_col=[index_col])
  final_df = pd.DataFrame(df[target_col]).rename(columns={target_col: new_target_col_name})
  time_steps = final_df.index.to_numpy()
  target = final_df[new_target_col_name].to_numpy()
  N  = int(len(time_steps)*ratio)
  return target[:N], target[N:], time_steps[:N], time_steps[N:]


def get_labeled_windows(x, window_size: int=7, horizon: int=1)->tuple:
  """
  Takes a sequence of values as a 1-D numpy array.
  Returns a tuple of sliding windows and sliding horizons to be used as
  inputs and targets for sequence modeling. Similar to 
  tf.keras.preprocessing.timeseries_dataset_from_array().
  Args:
    x: sequence of values, 1-D numpy array.
    window_size: length of sliding windows for the input features.
    horizon: length of sliding windows for the target.
  Returns:
    tuple of 
  """
  windows = np.lib.stride_tricks.sliding_window_view(x[:-horizon], window_size)
  horizons = np.lib.stride_tricks.sliding_window_view(x[window_size:], window_shape=horizon)

  return windows, horizons


def make_train_test_splits(windows, labels, test_split=0.2):
  """
  Splits the tuple of windowed inputs and targets into train and test/val tuples.
  """
  split_idx = 1 - int(len(windows)*test_split)
  return windows[:split_idx], windows[split_idx:], labels[:split_idx], labels[split_idx:]

  
