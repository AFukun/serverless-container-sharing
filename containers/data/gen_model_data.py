import json
from os.path import exists
from tensorflow.keras import models
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.applications.resnet50 import ResNet50

from build_cost_matrix import build_solution, build_child_info

data_dir = "../model_data/"

# if not exists(data_dir + "resnet50_imagenet.h5"):
#     resnet50 = ResNet50(weights="imagenet")
#     resnet50.save(data_dir + "resnet50_imagenet.h5")
#     resnet50.save_weights(data_dir + "resnet50_imagenet_weights.h5")
#     with open(data_dir + "resnet50_info.json", "w") as outfile:
#         info = build_child_info(resnet50)
#         json.dump(info, outfile)

# else:
#     resnet50 = models.load_model(data_dir + "resnet50_imagenet.h5")

if not exists(data_dir + "vgg16_imagenet.h5"):
    vgg16 = VGG16(weights="imagenet")
    vgg16.save(data_dir + "vgg16_imagenet.h5")
    vgg16.save_weights(data_dir + "vgg16_imagenet_weights.h5")
    with open(data_dir + "vgg16_info.json", "w") as outfile:
        info = build_child_info(vgg16)
        json.dump(info, outfile)
else:
    vgg16 = models.load_model(data_dir + "vgg16_imagenet.h5")

if not exists(data_dir + "vgg19_imagenet.h5"):
    vgg19 = VGG19(weights="imagenet")
    vgg19.save(data_dir + "vgg19_imagenet.h5")
    vgg19.save_weights(data_dir + "vgg19_imagenet_weights.h5")
    with open(data_dir + "vgg19_info.json", "w") as outfile:
        info = build_child_info(vgg19)
        json.dump(info, outfile)
else:
    vgg19 = models.load_model(data_dir + "vgg19_imagenet.h5")

with open(data_dir + "vgg19_to_vgg16_solution.json", "w") as outfile:
    solution = build_solution(vgg19, vgg16)
    json.dump(solution, outfile)
