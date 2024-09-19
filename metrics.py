import numpy as np
import tensorflow as tf
from tensorflow import keras


def my_sMAPE(y_true: tf.Tensor, y_pred: tf.Tensor) -> float:
  """
  Implementation of symmetric Mean Absolute Percentage Error, based off of
  Armstrong, J.(1978), Long-range forecasting: From crystal ball to computer.
  Args:
    y_true: true sequence of values as tf.Tensor or np.array.
    y_pred: Prediction of sequence.
  Returns:
    float value of mean reduced sMAPE.
  """
  abs_error = tf.math.abs(y_true - y_pred)
  scale = y_true + y_pred 
  return tf.reduce_mean(200.0*abs_error/scale).numpy().item()


def my_MASE(y_true: tf.Tensor, y_pred: tf.Tensor) -> float:
  """
  Implementation of Scaled Mean Absolute Error, based off of
  Hyndman, R., & Koehler, A.(2006),"Another look at measures of forecast 
  accuracy", International Journal of Forecasting.
  NOTE: We assume no seasonality in the time series.
  Args:
    y_true: true sequence of values as tf.Tensor or np.array.
    y_pred: Prediction of sequence.
  Returns:
    float value of mean reduced MASE.
  """
  mae = tf.reduce_mean(tf.math.abs(y_true - y_pred))
  scale = tf.reduce_mean(tf.math.abs(y_true[1:] - y_true[:-1]))
  return (mae / scale).numpy().item()


def my_MAPE(y_true: tf.Tensor, y_pred: tf.Tensor) -> float:
  """
  Implementation of Mean Absolute Percentage Error
  Args:
    y_true: true sequence of values as tf.Tensor or np.array.
    y_pred: Prediction of sequence.
  Returns:
    float value of mean reduced MAPE.
  """
  error = y_true - y_pred 
  percentage_error = error / y_true
  return tf.reduce_mean(tf.math.abs(percentage_error)).numpy().item() 


def my_MAE(y_true: tf.Tensor, y_pred: tf.Tensor) -> float:
  """
  Basic implementation of Mean Absolute Error.
  Args:
    y_true: true sequence of values as tf.Tensor or np.array.
    y_pred: Prediction of sequence.
  Returns:
    float value of mean reduced MAE.
  """
  return tf.reduce_mean(tf.math.abs(y_true - y_pred)).numpy().item()


def my_MSE(y_true: tf.Tensor, y_pred: tf.Tensor) -> float:
  """
  Basic implementation of Mean Squared Error.
  Args:
    y_true: true sequence of values as tf.Tensor or np.array.
    y_pred: Prediction of sequence.
  Returns:
    float value of mean reduced MSE.
  """
  return tf.reduce_mean(tf.square(y_true - y_pred)).numpy().item()


def my_RMSE(y_true: tf.Tensor, y_pred: tf.Tensor) -> float:
  """
  Basic implementation of Root Mean Squared Error.
  Args:
    y_true: true sequence of values as tf.Tensor or np.array.
    y_pred: Prediction of sequence.
  Returns:
    float value of mean reduced RMSE.
  """
  return tf.math.sqrt(tf.reduce_mean(tf.square(y_true - y_pred))).numpy().item()


def evaluate_preds(y_true: tf.Tensor, y_pred: tf.Tensor) -> dict:
  """
  Takes true values of sequence and predicted values, evaluates predictions on 
  MAE, RMSE, MAPE, sMAPE, MASE and returns metrics as a dict.
  Args:
    y_true: true sequence of values as tf.Tensor or np.array.
    y_pred: Prediction of sequence.
  Returns:
    dict with keys "mae", "rmse", "mape", "smape", "mase".
  """
  y_true = tf.cast(y_true, dtype=tf.float32)
  y_pred = tf.cast(y_pred, dtype=tf.float32)

  mae = my_MAE(y_true, y_pred)
  rmse = my_RMSE(y_true, y_pred)
  mape = my_MAPE(y_true, y_pred)
  smape = my_sMAPE(y_true, y_pred)
  mase = my_MASE(y_true, y_pred)

  return {"mae": mae,
          "rmse": rmse,
          "mape": mape,
          "smape": smape,
          "mase": mase}

