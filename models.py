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


class Stack(tf.keras.layers.Layer):
  """
  Organizing K-blocks into Stacks using the doubly residual stacking principle 
  for N-BEATS, outlined in Oreshkin et al (2020).
  """
  def __init__(self, 
               num_blocks: int, 
               input_size: int, 
               theta_size: int, 
               horizon: int, 
               n_neurons: int, 
               n_layers: int, 
               **kwargs):
    """
    Initializes the Stack for N-BEATS.
    Args:
      num_blocks: Number of blocks constituting the stack
      input_size: size of window to predict from
      theta_size: dimensionality of the parameters for the bases
      horizon: steps to predict for
      n_neurons: neurons in the FC layers
      n_layers: number of layers in the FC section of the block
    """
    super().__init__(**kwargs)
    self.blocks = [BasicBlock(input_size, theta_size, horizon, n_neurons, n_layers) for _ in range(num_blocks)]

  def call(self, inputs):
    forecast = []
    for Block in self.blocks:
      x_hat, y_hat = Block(inputs)
      inputs = tfkl.subtract([inputs, x_hat])
      forecast.append(y_hat)
    forecast = tfkl.add(forecast)
    return inputs, forecast
    

class NBeats(tf.keras.models.Model):
  """
  Uses the BasicBlock and Stack classes to define the N-BEATS model (generic), 
  outlined in Oreshkin et al (2020).
  """
  def __init__(self, 
               num_stacks: int,
               num_blocks: int, 
               input_size: int, 
               theta_size: int, 
               horizon: int, 
               n_neurons: int, 
               n_layers: int, 
               **kwargs):
    """
    Initializes the model for N-BEATS.
    Args:
      num_stacks: Number of stacks in the model
      num_blocks: Number of blocks constituting the stack
      input_size: size of window to predict from
      theta_size: dimensionality of the parameters for the bases
      horizon: steps to predict for
      n_neurons: neurons in the FC layers
      n_layers: number of layers in the FC section of the block
    """
    super().__init__(**kwargs)
    self.stacks = [Stack(num_blocks, input_size, theta_size, horizon, n_neurons, n_layers) for _ in range(num_stacks)]

  def call(self, inputs):
    forecast = []
    for Stack in self.stacks:
      x_hat, y_hat = Stack(inputs)
      inputs = tfkl.subtract([inputs, x_hat])
      forecast.append(y_hat)
    forecast = tfkl.add(forecast)
    return forecast


def create_model_checkpoint(model_name, save_path: str="Experiments"):
  return tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join(save_path, model_name+".keras"),
                                            verbose=0, 
                                            save_best_only=True)



def naive_forecast(input: np.array, horizon: int=1):
  return np.repeat(input[:,-1:], horizon, axis=-1)

