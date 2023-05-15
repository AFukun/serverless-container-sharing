# sys.path.insert(1, "server")
# from core.api import *

import json
import numpy as np
import torch
import tensorflow as tf
from torch.autograd import Variable
from torchsummary import summary
from pytorchcv.model_provider import get_model as get_pt_model
from pytorch2keras import pytorch_to_keras


import sys
import traceback


sys.path.insert(1, "setup")
from utils import *

model_name_list = []
with open("tests/local_test_models.json") as file:
    # with open("tests/pt_models.json") as file:
    model_name_list = json.load(file)


def get_model_and_convert(model_name):
    pt_model = get_pt_model(model_name)
    input_var = get_pt_model_input(model_name)
    # summary(pt_model, (3, 224, 224))
    return pytorch_to_keras(
        pt_model,
        input_var,
        # verbose=True,
        change_ordering=True,
        name_policy="renumerate",
    )


for model_name in model_name_list:
    print(f"[{model_name}]")
    try:
        model = get_model_and_convert(model_name)
        build_childmodel_info(model)
        print("----------------Pass----------------")
    except Exception:
        traceback.print_exc()
        print("----------------Error----------------")
    finally:
        print("\n\n\n")
