import time
import json
from tensorflow.keras import models

from .main import transform


def load_vgg16(data_dir):
    return models.load_model(data_dir + "vgg16_imagenet.h5")


def vgg19_to_vgg16(data_dir, vgg19):
    with open(data_dir + "vgg16_info.json") as inputfile:
        vgg16_info = json.load(inputfile)
    with open(data_dir + "vgg19_to_vgg16_solution.json") as inputfile:
        solution = json.load(inputfile)
    vgg16 = transform(
        vgg19, vgg16_info, solution["matrix"], solution["n"], solution["m"]
    )
    vgg16.load_weights(data_dir + "vgg16_imagenet_weights.h5")
    return vgg16


def load_vgg19(data_dir):
    return models.load_model(data_dir + "vgg19_imagenet.h5")
