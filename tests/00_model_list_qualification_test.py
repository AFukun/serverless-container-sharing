# sys.path.insert(1, "server")
# from core.api import *

import json
import numpy as np
import torch
import tensorflow as tf
from torch.autograd import Variable
from torchsummary import summary
from pytorchcv.model_provider import get_model
from pytorch2keras import pytorch_to_keras


import sys
import traceback


sys.path.insert(1, "setup")
from utils import *

model_name_list = []
with open("tests/local_test_models.json") as file:
    # with open("tests/pt_models.json") as file:
    model_name_list = json.load(file)


for model_name in model_name_list:
    print(f"[{model_name}]")
    try:
        pt_model = get_model(model_name)
        input_var = Variable(
            torch.FloatTensor(np.random.uniform(0, 1, (1, 3, 224, 224)))
        )
        # summary(pt_model, (3, 224, 224))
        model = pytorch_to_keras(
            pt_model,
            input_var,
            # verbose=True,
            change_ordering=True,
            name_policy="renumerate",
        )
        # model(tf.random.uniform((1, 224, 224, 3)))
        build_childmodel_info(model)
        print("----------------Pass----------------")
    except Exception:
        traceback.print_exc()
        print("----------------Error----------------")
    finally:
        print("\n\n\n")
