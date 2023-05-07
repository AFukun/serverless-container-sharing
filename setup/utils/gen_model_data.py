import json
from os.path import exists
from tensorflow.keras.models import load_model
import numpy as np
import torch
from torch.autograd import Variable
from pytorchcv.model_provider import get_model
from pytorch2keras import pytorch_to_keras


from .json_encoder import NumpyEncoder
from .save_information import build_childmodel_info, compute_node_to_node_mapping


from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.resnet import ResNet101, ResNet152
from tensorflow.keras.applications.densenet import DenseNet121
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.mobilenet import MobileNet

tf_applications = {
    "vgg16": VGG16,
    "vgg19": VGG19,
    "resnet50": ResNet50,
    "resnet101": ResNet101,
    "resnet152": ResNet152,
    "densenet121": DenseNet121,
    "inceptionv3": InceptionV3,
    "mobilenet": MobileNet,
}

TF_NATIVE = False


def gen_model_data(data_dir, model_name_list):
    models = []
    for model_name in model_name_list:
        if not exists(f"{data_dir}{model_name}.h5"):
            if TF_NATIVE:
                model = tf_applications[model_name]()
            else:
                pt_model = get_model(model_name)
                input_var = Variable(
                    torch.FloatTensor(np.random.uniform(0, 1, (1, 3, 224, 224)))
                )
                model = pytorch_to_keras(
                    pt_model,
                    input_var,
                    change_ordering=True,
                    name_policy="renumerate",
                )
            model._name = model_name
            model.save(f"{data_dir}{model_name}.h5")
            model.save_weights(f"{data_dir}{model_name}_weights.h5")
            with open(f"{data_dir}{model_name}_info.json", "w") as outfile:
                info = build_childmodel_info(model)
                json.dump(info, outfile, cls=NumpyEncoder)
            models.append(model)
        else:
            models.append(load_model(f"{data_dir}{model_name}.h5"))

    for model_a in models:
        for model_b in models:
            if model_a._name != model_b._name:
                with open(
                    f"{data_dir}{model_a._name}_to_{model_b._name}_solution.json", "w"
                ) as outfile:
                    node_to_node_mapping = compute_node_to_node_mapping(
                        model_a, model_b
                    )
                    json.dump(node_to_node_mapping, outfile)
