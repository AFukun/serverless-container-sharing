import numpy as np
from tensorflow.keras.models import load_model as tensorflow_load_model
from tensorflow.keras.applications.imagenet_utils import (
    preprocess_input,
    decode_predictions,
)
from tensorflow.keras.preprocessing import image

input_shape = {
    "vgg11": (224, 224),
    "vgg16": (224, 224),
    "vgg19": (224, 224),
    "resnet50": (224, 224),
    "resnet101": (224, 224),
    "resnet152": (224, 224),
    "inceptionv3": (299, 299),
    "densenet121": (224, 224),
    "mobilenet": (224, 224),
}


def load_model(data_dir, model_name):
    return tensorflow_load_model(f"{data_dir}{model_name}.h5")


def _preprocess_image(image_path, model_name):
    img = image.load_img(image_path, target_size=input_shape[model_name])
    img = image.img_to_array(img)
    return np.expand_dims(img, axis=0)


def inference(model, input_file_path):
    img = _preprocess_image(input_file_path, model._name)
    input = preprocess_input(img)
    predicts = model.predict(input)
    output = decode_predictions(predicts, top=1)[0]

    return output
