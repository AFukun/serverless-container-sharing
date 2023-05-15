import time
import sys
import json
import numpy as np
import random

from prettytable import PrettyTable

sys.path.insert(1, "client")
from core import Client

REPEATS = 1000

client = Client("ssh://luosf@blockchain2")

with open("setup/test_models.json") as file:
    available_model_list = json.load(file)

model_list_size = len(available_model_list)

for i in range(0, REPEATS):
    model_a_name = available_model_list[random.randint(0, model_list_size - 1)]
    model_b_name = available_model_list[random.randint(0, model_list_size - 1)]
    client.manual_run_container()
    client.wait_for_container_to_setup()
    try:
        client.manual_load_model(model_a_name)
        switch_time = client.manual_switch_model(model_b_name)
        with open("output.txt", "a") as f:
            print(switch_time, file=f)
        print(f"[ITERATION {i}] {model_a_name} to {model_b_name} in {switch_time}s")
    except:
        print(f"[ITERATION {i}] {model_a_name} to {model_b_name} failed")
    client.reset()
