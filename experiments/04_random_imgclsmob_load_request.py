import time
import json
import numpy as np
import random


import sys

sys.path.insert(1, "client")

REPEATS = 500

from core import Client

client = Client("ssh://luosf@blockchain2")

with open("setup/imgclsmob_models.json") as file:
    available_model_list = json.load(file)

model_list_size = len(available_model_list)

for i in range(0, REPEATS):
    model_name = available_model_list[random.randint(0, model_list_size - 1)]
    client.manual_run_container()
    client.wait_for_container_to_setup()
    try:
        load_time = client.manual_load_model(model_name)
        with open("output04.txt", "a") as f:
            print(load_time, file=f)
        print(f"[ITERATION {i}] {model_name} loaded in {load_time}s")
    except:
        print(f"[ITERATION {i}] {model_name} failed")
    client.reset()
