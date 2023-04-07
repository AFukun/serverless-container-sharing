import json
from os.path import exists
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.mobilenet import MobileNet

from .build_cost_matrix import build_solution, build_child_info


def gen_resnet50_data(data_dir):
    if not exists(data_dir + "resnet50_imagenet.h5"):
        resnet50 = ResNet50(weights="imagenet")
        resnet50.save(data_dir + "resnet50_imagenet.h5")
        resnet50.save_weights(data_dir + "resnet50_imagenet_weights.h5")
        with open(data_dir + "resnet50_info.json", "w") as outfile:
            info = build_child_info(resnet50)
            json.dump(info, outfile)
    else:
        resnet50 = load_model(data_dir + "resnet50_imagenet.h5")
    return resnet50


def gen_mobilenet_data(data_dir):
    if not exists(data_dir + "mobilenet_imagenet.h5"):
        mobilenet = MobileNet(weights="imagenet")
        mobilenet._name = "mobilenet"
        mobilenet.save(data_dir + "mobilenet_imagenet.h5")
        mobilenet.save_weights(data_dir + "mobilenet_imagenet_weights.h5")
        with open(data_dir + "mobilenet_info.json", "w") as outfile:
            info = build_child_info(mobilenet)
            json.dump(info, outfile)
    else:
        mobilenet = load_model(data_dir + "mobilenet_imagenet.h5")
    return mobilenet


def gen_vgg16_data(data_dir):
    if not exists(data_dir + "vgg16_imagenet.h5"):
        vgg16 = VGG16(weights="imagenet")
        vgg16.save(data_dir + "vgg16_imagenet.h5")
        vgg16.save_weights(data_dir + "vgg16_imagenet_weights.h5")
        with open(data_dir + "vgg16_info.json", "w") as outfile:
            info = build_child_info(vgg16)
            json.dump(info, outfile)
    else:
        vgg16 = load_model(data_dir + "vgg16_imagenet.h5")
    return vgg16


def gen_vgg19_data(data_dir):
    if not exists(data_dir + "vgg19_imagenet.h5"):
        vgg19 = VGG19(weights="imagenet")
        vgg19.save(data_dir + "vgg19_imagenet.h5")
        vgg19.save_weights(data_dir + "vgg19_imagenet_weights.h5")
        with open(data_dir + "vgg19_info.json", "w") as outfile:
            info = build_child_info(vgg19)
            json.dump(info, outfile)
    else:
        vgg19 = load_model(data_dir + "vgg19_imagenet.h5")
    return vgg19


def gen_solutions(data_dir, models):
    vgg16 = models[0]
    vgg19 = models[1]
    with open(data_dir + "vgg19_to_vgg16_solution.json", "w") as outfile:
        solution = build_solution(vgg19, vgg16)
        json.dump(solution, outfile)
    with open(data_dir + "vgg16_to_vgg19_solution.json", "w") as outfile:
        solution = build_solution(vgg16, vgg19)
        json.dump(solution, outfile)


def gen_model_data(data_dir):
    models = []
    models.append(gen_vgg16_data(data_dir))
    models.append(gen_vgg19_data(data_dir))
    models.append(gen_resnet50_data(data_dir))
    models.append(gen_mobilenet_data(data_dir))
    gen_solutions(data_dir, models)
