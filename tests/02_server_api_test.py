import time
import sys

sys.path.insert(1, "server")
from core.api import *
from prettytable import PrettyTable

data_dir = "./tmp/"

with open("tests/local_test_models.json") as file:
    test_model_list = json.load(file)

table = PrettyTable([""] + test_model_list)
for model_a_name in test_model_list:
    row = [model_a_name]
    for model_b_name in test_model_list:
        model_a = load_model(data_dir, model_a_name)
        if input is None:
            input = generate_random_input(model_a)
        start = time.time()
        if model_a_name == model_b_name:
            load_weights(data_dir, model_a)
        else:
            model_b = switch_model(data_dir, model_a, model_b_name)
        end = time.time()
        row.append("{:.2f}".format(end - start))
    table.add_row(row)
print(table)

table = PrettyTable(test_model_list)
row = []
for model_name in test_model_list:
    start = time.time()
    model = load_model(data_dir, model_name)
    end = time.time()
    row.append("{:.2f}".format(end - start))
table.add_row(row)
print(table)
