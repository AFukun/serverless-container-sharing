import requests
import time
import sys
import json
import numpy as np

from prettytable import PrettyTable

sys.path.insert(1, "client")
from core import Client

REPEATS = 10

client = Client("ssh://luosf@blockchain2")

with open("setup/test_models.json") as file:
    test_model_list = json.load(file)

table = PrettyTable(["", "setup time", "load time", "inference time"])
for model_name in test_model_list:
    rows = []
    for iteration in range(0, REPEATS):
        start = time.time()
        client.manual_run_container()
        client.wait_for_container_to_setup()
        end = time.time()
        setup_time = end - start
        load_time = client.manual_load_model(model_name)
        inference_time = client.manual_inference()
        client.reset()
        rows.append([float(setup_time), float(load_time), float(inference_time)])
        print(
            f"[ITERATION {iteration}] {model_name} ({setup_time},{load_time},{inference_time})"
        )
    avg_row = np.average(np.array(rows), axis=0).tolist()
    table.add_row([model_name] + ["{:.2f}".format(x) for x in avg_row])
print(table)
