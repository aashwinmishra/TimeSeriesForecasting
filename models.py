import numpy as np
import os
import tensorflow as tf
from tensorflow import keras
import keras.layers as tfkl
from keras.models import Model, Sequential


def create_model_checkpoint(model_name, save_path: str="Experiments"):
  return tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join(save_path, model_name+".keras"),
                                            verbose=0, 
                                            save_best_only=True)



def naive_forecast(input: np.array, horizon: int=1):
  return np.repeat(input[:,-1:], horizon, axis=-1)

