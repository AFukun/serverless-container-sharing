import json
from os.path import exists
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.resnet import ResNet101, ResNet152
from tensorflow.keras.applications.densenet import DenseNet121
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.mobilenet import MobileNet

from .custom_models import VGG11
from .build_cost_matrix import build_solution, build_child_info

model_name_list = [
    "vgg11",
    "vgg16",
    "vgg19",
    "resnet50",
    "resnet101",
    "resnet152",
    "densenet121",
    "inceptionv3",
    "mobilenet",
]
tf_applications = {
    "vgg11": VGG11,
    "vgg16": VGG16,
    "vgg19": VGG19,
    "resnet50": ResNet50,
    "resnet101": ResNet101,
    "resnet152": ResNet152,
    "densenet121": DenseNet121,
    "inceptionv3": InceptionV3,
    "mobilenet": MobileNet,
}


def gen_model_data(data_dir):
    models = []
    for model_name in model_name_list:
        if not exists(f"{data_dir}{model_name}.h5"):
            model = tf_applications[model_name]()
            model._name = model_name
            model.save(f"{data_dir}{model_name}.h5")
            model.save_weights(f"{data_dir}{model_name}_weights.h5")
            with open(f"{data_dir}{model_name}_info.json", "w") as outfile:
                info = build_child_info(model)
                json.dump(info, outfile)
        else:
            models.append(load_model(f"{data_dir}{model_name}.h5"))
    return models


def gen_solutions(data_dir, models):
    vgg16 = models[0]
    vgg19 = models[1]
    with open(data_dir + "vgg19_to_vgg16_solution.json", "w") as outfile:
        solution = build_solution(vgg19, vgg16)
        json.dump(solution, outfile)
    with open(data_dir + "vgg16_to_vgg19_solution.json", "w") as outfile:
        solution = build_solution(vgg16, vgg19)
        json.dump(solution, outfile)
