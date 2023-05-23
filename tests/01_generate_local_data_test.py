import sys

sys.path.insert(1, "setup")
import os
import shutil
import json
import utils


data_dir = "tmp/"
shutil.rmtree(data_dir)
os.mkdir(data_dir)

with open("tests/local_test_models.json") as file:
    test_model_list = json.load(file)
utils.gen_model_data(data_dir, test_model_list)

# with open("setup/nasbench_models.json", "rb") as f:
#     nasbench_model_graphs = json.load(f)
# utils.gen_nasbench_model_data(data_dir, nasbench_model_graphs, sample_set_size=10)
