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

grids = []
for iteration in range(0, REPEATS):
    grid = []
    for model_a_name in test_model_list:
        row = []
        for model_b_name in test_model_list:
            client.manual_run_container()
            client.wait_for_container_to_setup()
            client.manual_load_model(model_a_name)
            switch_time = client.manual_switch_model(model_b_name)
            print(
                f"[ITERATION {iteration+1}] {model_a_name} to {model_b_name} in {switch_time}s"
            )
            row.append(float(switch_time))
            client.reset()
        grid.append(row)
        with open("transform_grid.txt", "a") as f:
            print(row, file=f)
    grids.append(grid)
avg_grid = np.average(np.array(grids), axis=0).tolist()
table = PrettyTable([""] + test_model_list)
for index, model_name in enumerate(test_model_list):
    table.add_row([model_name] + ["{:.2f}".format(x) for x in avg_grid[index]])
print(table)


rows = []
for iteration in range(0, REPEATS):
    row = []
    for model_name in test_model_list:
        client.manual_run_container()
        client.wait_for_container_to_setup()
        load_time = client.manual_load_model(model_name)
        client.reset()
        row.append(float(load_time))
        print(f"[ITERATION {iteration+1}] load {model_name} in {load_time}s")
    rows.append(row)
    with open("load_list.txt", "a") as f:
        print(row, file=f)
avg_row = np.average(np.array(rows), axis=0).tolist()
table = PrettyTable(test_model_list)
table.add_row(["{:.2f}".format(x) for x in avg_row])
print(table)
