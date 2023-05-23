import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model as tensorflow_load_model
from .model_transform import model_structure_transformation as transform


tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


def load_model(data_dir, model_name):
    model = tensorflow_load_model(f"{data_dir}{model_name}.h5")
    model._name = model_name

    return model


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

    child_model = transform(parent_model, child_model_info, solution)

    child_model._name = child_model_name
    load_weights(data_dir, child_model)

    return child_model


def inference(model, input):
    output = model.predict(input).numpy()[0]

    return np.max(output)
