import time
import sys

sys.path.insert(1, "server")
from core.api import *

data_dir = "./tmp/"

test_model_list = [
    "resnet18",
    "resnet34",
    "resnet50",
    "vgg11",
    "vgg16",
    "vgg19",
]

input = None

for model_a_name in test_model_list:
    for model_b_name in test_model_list:
        model_a = load_model(data_dir, model_a_name)
        if input is None:
            input = generate_random_input(model_a)
        print(f"{model_a_name} result:", inference(model_a, input))
        start = time.time()
        if model_a_name == model_b_name:
            # load_weights(data_dir, model_a)
            end = time.time()
            result = inference(model_a, input)
            print(f"{model_a_name} result:", result)
        else:
            model_b = switch_model(data_dir, model_a, model_b_name)
            end = time.time()
            result = inference(model_b, input)
        print(f"{model_b_name} result:", result)
        print(f"switch time: {end - start}s")
