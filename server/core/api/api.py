import time
import json
import numpy as np
from tensorflow.keras.models import load_model as tensorflow_load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import (
    preprocess_input as resnet50_preprocess_input,
    decode_predictions as resnet50_decode_predictions,
)
from tensorflow.keras.applications.mobilenet import (
    preprocess_input as mobilenet_preprocess_input,
    decode_predictions as mobilenet_decode_predictions,
)
from tensorflow.keras.applications.vgg16 import (
    preprocess_input as vgg16_preprocess_input,
    decode_predictions as vgg16_decode_predictions,
)
from tensorflow.keras.applications.vgg19 import (
    preprocess_input as vgg19_preprocess_input,
    decode_predictions as vgg19_decode_predictions,
)

from .transform import transform


def load_model(data_dir, model_name):
    return tensorflow_load_model(f"{data_dir}{model_name}_imagenet.h5")


def load_weights(data_dir, model):
    model.load_weights(f"{data_dir}{model._name}_imagenet_weights.h5")


def switch_model(data_dir, parent_model, child_model_name):
    parent_model_name = parent_model._name
    with open(f"{data_dir}{child_model_name}_info.json") as input_file:
        child_model_info = json.load(input_file)
    with open(
        f"{data_dir}{parent_model_name}_to_{child_model_name}_solution.json"
    ) as input_file:
        solution = json.load(input_file)
    child_model, transform_log = transform(
        parent_model,
        child_model_info,
        solution["munkres"],
        solution["n"],
        solution["m"],
    )

    start = time.time()
    child_model._name = child_model_name
    load_weights(data_dir, child_model)
    child_model.compile(loss="categorical_crossentropy")
    end = time.time()
    return child_model, f"{transform_log},reload in {end - start}s)"


def _preprocess_image(image_path):
    img = image.load_img(image_path, target_size=(224, 224))
    img = image.img_to_array(img)
    return np.expand_dims(img, axis=0)


def inference(data_dir, model, input_file):
    img = _preprocess_image(data_dir + input_file)
    if model._name == "vgg16":
        input = vgg16_preprocess_input(img)
    elif model._name == "vgg19":
        input = vgg19_preprocess_input(img)
    elif model._name == "resnet50":
        input = resnet50_preprocess_input(img)
    elif model._name == "mobilenet":
        input = mobilenet_preprocess_input(img)

    predicts = model.predict(input)

    if model._name == "vgg16":
        output = vgg16_decode_predictions(predicts, top=1)[0]
    elif model._name == "vgg19":
        output = vgg19_decode_predictions(predicts, top=1)[0]
    elif model._name == "resnet50":
        output = resnet50_decode_predictions(predicts, top=1)[0]
    elif model._name == "mobilenet":
        output = mobilenet_decode_predictions(predicts, top=1)[0]

    return output
