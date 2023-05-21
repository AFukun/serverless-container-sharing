import json
import numpy as np
from os.path import exists
import tensorflow as tf
import torch
from torch.autograd import Variable
from pytorchcv.model_provider import get_model as get_pt_model
from pytorch2keras import pytorch_to_keras


from nasbench.lib import ModelSpec, build_module
from .json_encoder import NumpyEncoder
from .save_information import build_childmodel_info, compute_node_to_node_mapping


from tensorflow.keras.models import load_model
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

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


def get_pt_model_input(model_name):
    if "cifar" in model_name or "svhn" in model_name:
        return Variable(torch.FloatTensor(np.random.uniform(0, 1, (1, 3, 32, 32))))
    elif "ception" in model_name:
        return Variable(torch.FloatTensor(np.random.uniform(0, 1, (1, 3, 299, 299))))
    else:
        return Variable(torch.FloatTensor(np.random.uniform(0, 1, (1, 3, 224, 224))))


def gen_model_data(
    data_dir,
    model_name_list,
    use_tf_native_app=False,
):
    models = []
    generated_model_count = 0
    total_model_count = len(model_name_list)
    for model_name in model_name_list:
        generated_model_count += 1
        if not exists(f"{data_dir}{model_name}.h5"):
            if use_tf_native_app:
                model = tf_applications[model_name]()
            else:
                pt_model = get_pt_model(model_name)
                input_var = get_pt_model_input(model_name)
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
            print(f"generate {model_name}({generated_model_count}/{total_model_count})")

        else:
            models.append(load_model(f"{data_dir}{model_name}.h5"))
            print(f"load {model_name}({generated_model_count}/{total_model_count})")

    generated_solution_count = 0
    total_solution_count = len(models) * (len(models) - 1)

    for model_a in models:
        for model_b in models:
            if model_a._name != model_b._name:
                if not exists(
                    f"{data_dir}{model_a._name}_to_{model_b._name}_solution.json"
                ):
                    with open(
                        f"{data_dir}{model_a._name}_to_{model_b._name}_solution.json",
                        "w",
                    ) as outfile:
                        node_to_node_mapping = compute_node_to_node_mapping(
                            model_a, model_b
                        )
                        json.dump(node_to_node_mapping, outfile)
                generated_solution_count = generated_solution_count + 1
                print(
                    f"{model_a._name} to {model_b.name}({generated_solution_count}/{total_solution_count})"
                )


def gen_nasbench_model_data(data_dir, model_graphs, sample_set_size=423624):
    config = {
        "available_ops": ["conv3x3-bn-relu", "conv1x1-bn-relu", "maxpool3x3"],
        "stem_filter_size": 128,
        "data_format": "channels_last",
        "num_stacks": 3,
        "num_modules_per_stack": 2,
        "num_labels": 1000,
    }
    index = 1

    for _, graph_info in model_graphs.items():
        outfile_name = f"nasbench_{index:06d}"
        print(f"{outfile_name}({index}/{sample_set_size})")
        if not exists(f"{data_dir}{outfile_name}.h5"):
            matrix, labels_list = graph_info
            labels = (
                ["input"]
                + [config["available_ops"][l] for l in labels_list[1:-1]]
                + ["output"]
            )
            spec = ModelSpec(matrix, labels, data_format="channels_last")
            inputs = tf.keras.layers.Input((224, 224, 3), 1)
            outputs = build_module(
                spec=spec, inputs=inputs, channels=128, is_training=True
            )
            model = tf.keras.Model(inputs=inputs, outputs=outputs)
            model.save(f"{data_dir}{outfile_name}.h5")
            model.save_weights(f"{data_dir}{outfile_name}_weights.h5")
            # model.summary()
            model = load_model(f"{data_dir}{outfile_name}.h5")
            info = build_childmodel_info(model)
            with open(f"{data_dir}{outfile_name}_info.json", "w") as f:
                json.dump(info, f)
        index += 1
        if index > sample_set_size:
            break
