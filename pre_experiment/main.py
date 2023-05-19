import time

start_time = time.time()

from argparse import ArgumentParser

parser = ArgumentParser(
    prog="pre-tester",
    description="Pre-Experiments exec in docker container",
)
parser.add_argument("-D", "--data-dir")
parser.add_argument("-M", "--model-name")
args = parser.parse_args()

import importlib.util
import sys


def lazy_import(name):
    spec = importlib.util.find_spec(name)
    loader = importlib.util.LazyLoader(spec.loader)
    spec.loader = loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    loader.exec_module(module)
    return module


utils = lazy_import("utils")


start = time.time()
model = utils.load_model(args.data_dir, args.model_name)
end = time.time()
load_time = end - start

input = utils.generate_random_input(model)
start = time.time()
output = utils.inference(model, input)
end = time.time()
inference_time = end - start

print(inference_time)
print(load_time)
print(start_time)
