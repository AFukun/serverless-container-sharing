import time
from argparse import ArgumentParser
from utils import *

parser = ArgumentParser(
    prog="pre-tester",
    description="Pre-Experiments exec in docker container",
)
parser.add_argument("-D", "--data-dir")
parser.add_argument("-M", "--model-name")
args = parser.parse_args()

logs = []

start = time.time()
model = load_model(args.data_dir, args.model_name)
end = time.time()
logs.append(f"model load time: {end - start}")

start = time.time()
input = generate_random_input(model)
output = inference(model, input)
end = time.time()
logs.append(f"inference time: {end - start}")
logs.append(f"result: {output}")

print(logs)
