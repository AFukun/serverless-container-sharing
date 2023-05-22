import time
import json
import numpy as np
import random


import sys

sys.path.insert(1, "client")

REPEATS = 500

from core import Client

client = Client("ssh://luosf@blockchain2")

sample_set_size = 1000

for i in range(0, REPEATS):
    model_a_name = f"nasbench_{random.randint(1,sample_set_size):06d}"
    model_b_name = f"nasbench_{random.randint(1,sample_set_size):06d}"
    client.manual_run_container()
    client.wait_for_container_to_setup()
    try:
        client.manual_load_model(model_a_name)
        switch_time = client.manual_switch_nasbench_model(model_b_name)
        with open("output05.txt", "a") as f:
            print(switch_time, file=f)
        print(f"[ITERATION {i}] {model_a_name} to {model_b_name} in {switch_time}s")
    except:
        print(f"[ITERATION {i}] {model_a_name} to {model_b_name} failed")
    client.reset()
