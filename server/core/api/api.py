import time
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model as tensorflow_load_model
from tf2cv.model_provider import get_model as tf2cv_get_model

from .model_transform import model_structure_transformation

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


def load_model(data_dir, model_name):
    return tensorflow_load_model(f"{data_dir}{model_name}.h5")


def get_model(model_name):
    return tf2cv_get_model(model_name)


def load_weights(data_dir, model):
    model.load_weights(f"{data_dir}{model._name}_weights.h5")


def switch_model(data_dir, parent_model, child_model_name):
    parent_model_name = parent_model._name
    with open(f"{data_dir}{child_model_name}_info.json") as input_file:
        child_model_info = json.load(input_file)
    with open(
        f"{data_dir}{parent_model_name}_to_{child_model_name}_solution.json"
    ) as input_file:
        solution = json.load(input_file)

    child_model = model_structure_transformation(
        parent_model, child_model_info, solution
    )
    child_model._name = child_model_name
    load_weights(data_dir, child_model)
    # child_model.compile(loss="categorical_crossentropy")

    return child_model


def generate_random_input(model):
    shape = list(model.layers[0].input_shape[0])
    shape[0] = 1

    return tf.random.normal(shape)


def inference(model, input):
    output = model(input).numpy()[0]

    return np.max(output)
