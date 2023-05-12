import requests
import time
import sys
import json

from prettytable import PrettyTable

sys.path.insert(1, "client")
from core import Client

client = Client("ssh://luosf@blockchain2")

client.manual_run_container()
time.sleep(5)

with open("setup/test_models.json") as file:
    test_model_list = json.load(file)

table = PrettyTable([""] + test_model_list)
for model_a_name in test_model_list:
    row = [model_a_name]
    for model_b_name in test_model_list:
        client.manual_load_model(model_a_name)
        switch_time = client.manual_switch_model(model_b_name)
        row.append(switch_time)
    table.add_row(row)
print(table)

table = PrettyTable(test_model_list)
row = []
for model_name in test_model_list:
    load_time = client.manual_load_model(model_name)
    row.append(load_time)
    # load_time1 = client.manual_load_model(model_name)
    # load_time2 = client.manual_load_model(model_name)
    # load_time3 = client.manual_load_model(model_name)
    # row.append(f"{load_time1},{load_time2},{load_time3}")
table.add_row(row)
print(table)

client.reset()
