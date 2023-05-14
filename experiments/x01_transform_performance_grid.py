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

patch_models = ["xception", "inceptionv3"]

grids = []
for iteration in range(0, REPEATS):
    grid = []
    for model_a_name in test_model_list:
        row = []
        for model_b_name in test_model_list:
            if model_a_name in patch_models or model_b_name in patch_models:
                client.manual_run_container()
                client.wait_for_container_to_setup()
                client.manual_load_model(model_a_name)
                switch_time = client.manual_switch_model(model_b_name)
                client.reset()
                print(
                    f"[ITERATION {iteration}] {model_a_name} to {model_b_name} in {switch_time}s"
                )

                row.append(float(switch_time))
            else:
                row.append(-1)
        grid.append(row)
    grids.append(grid)
avg_grid = np.average(np.array(grids), axis=0).tolist()
table = PrettyTable([""] + test_model_list)
for index, model_name in enumerate(test_model_list):
    table.add_row([model_name] + ["{:.2f}".format(x) for x in avg_grid[index]])
print(table)
