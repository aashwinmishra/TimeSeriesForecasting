import numpy as np
import os
import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.layers as tfkl
from tensorflow.keras.models import Model, Sequential


class BasicBlock(tf.keras.layers.Layer):
  """
  Basic Building Block for N-BEATS, outlined in 
  Oreshkin et al (2020).
  """
  def __init__(self, 
               input_size: int,
               theta_size: int,
               horizon: int,
               n_neurons: int,
               n_layers: int=4,
               **kwargs):
    """
    Initializes the Basic Building Block for N-BEATS.
    Args:
      input_size: size of window to predict from
      theta_size: dimensionality of the parameters for the bases
      horizon: steps to predict for
      n_neurons: neurons in the FC layers
      n_layers: number of layers in the model
    """
    super().__init__(**kwargs)
    self.FC_Stack = Sequential([tfkl.Dense(n_neurons, activation='relu') 
                    for _ in range(n_layers)
                    ])
    self.FC_b = tfkl.Dense(theta_size, activation='linear')
    self.FC_f = tfkl.Dense(theta_size, activation='linear')
    self.g_f = tfkl.Dense(horizon, activation='linear')
    self.g_b = tfkl.Dense(input_size, activation='linear')

  def call(self, inputs):
    x = self.FC_Stack(inputs)
    theta_b = self.FC_b(x)
    theta_f = self.FC_f(x)
    backcast = self.g_b(theta_b)
    forecast = self.g_f(theta_f)
    return backcast, forecast


def create_model_checkpoint(model_name, save_path: str="Experiments"):
  return tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join(save_path, model_name+".keras"),
                                            verbose=0, 
                                            save_best_only=True)



def naive_forecast(input: np.array, horizon: int=1):
  return np.repeat(input[:,-1:], horizon, axis=-1)

