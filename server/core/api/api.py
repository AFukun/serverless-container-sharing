import json
import numpy as np
from tensorflow.keras.models import load_model as tensorflow_load_model
from .model_transform import model_structure_transformation as transform
from .nasbench_model_transform import (
    model_structure_transformation as nasbench_transform,
)
from .save_information import compute_node_to_node_mapping


def load_model(data_dir, model_name):
    model = tensorflow_load_model(f"{data_dir}{model_name}.h5")
    model._name = model_name

    return model


def load_weights(data_dir, model):
    model.load_weights(f"{data_dir}{model._name}_weights.h5")


def generate_solution(data_dir, parent_model_name, child_model_name):
    parent_model = load_model(data_dir, parent_model_name)
    child_model = load_model(data_dir, child_model_name)
    with open(
        f"{data_dir}{parent_model_name}_to_{child_model_name}_solution.json",
        "w",
    ) as outfile:
        node_to_node_mapping = compute_node_to_node_mapping(parent_model, child_model)
        json.dump(node_to_node_mapping, outfile)


def switch_model(
    data_dir, parent_model, child_model_name, use_nasbench_transform=False
):
    parent_model_name = parent_model._name
    with open(f"{data_dir}{child_model_name}_info.json") as input_file:
        child_model_info = json.load(input_file)
    with open(
        f"{data_dir}{parent_model_name}_to_{child_model_name}_solution.json"
    ) as input_file:
        solution = json.load(input_file)

    if use_nasbench_transform:
        child_model = nasbench_transform(parent_model, child_model_info, solution)
    else:
        child_model = transform(parent_model, child_model_info, solution)

    child_model._name = child_model_name
    load_weights(data_dir, child_model)

    return child_model


def inference(model, input):
    output = model(input).numpy()[0]

    return np.max(output)
