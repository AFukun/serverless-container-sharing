import time
import json
import numpy as np

from tensorflow.keras.models import load_model as tensorflow_load_model
from tensorflow.keras.applications.imagenet_utils import (
    preprocess_input,
    decode_predictions,
)
from tensorflow.keras.preprocessing import image

from .model_transform import model_structure_transformation


def load_model(data_dir, model_name):
    return tensorflow_load_model(f"{data_dir}{model_name}.h5")


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
    start = time.time()
    child_model._name = child_model_name
    load_weights(data_dir, child_model)
    child_model.compile(loss="categorical_crossentropy")
    end = time.time()

    return child_model


def _preprocess_image(image_path):
    img = image.load_img(image_path, target_size=(224, 224))
    img = image.img_to_array(img)

    return np.expand_dims(img, axis=0)


def inference(data_dir, model, input_file):
    img = _preprocess_image(data_dir + input_file)
    input = preprocess_input(img)
    predicts = model.predict(input)
    output = decode_predictions(predicts, top=1)[0]

    return output
