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
  Args:
    data_file: path to file with time series data
    index_col: name of cloumn to order by
    target_col: Name of target column
    new_target_col_name: new name of target
    ratio: Ratio for split between train and validation data.
  Returns:
    x_train, x_val, y_train, y_val
  """
  df = pd.read_csv(data_file,parse_dates=[index_col],index_col=[index_col])
  final_df = pd.DataFrame(df[target_col]).rename(columns={target_col: new_target_col_name})
  time_steps = final_df.index.to_numpy()
  target = final_df[new_target_col_name].to_numpy()
  N  = int(len(time_steps)*ratio)
  return target[:N], target[N:], time_steps[:N], time_steps[N:]

