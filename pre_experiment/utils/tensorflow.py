import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model as tensorflow_load_model


def load_model(data_dir, model_name):
    return tensorflow_load_model(f"{data_dir}{model_name}.h5")


def generate_random_input(model):
    shape = list(model.layers[0].input_shape[0])
    shape[0] = 1

    return tf.random.normal(shape)


def inference(model, input):
    output = model(input).numpy()[0]

    return np.max(output)
