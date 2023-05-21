import sys

sys.path.insert(1, "setup")
sys.path.insert(1, "server")

import json
import random
from core.api import *
from utils import compute_node_to_node_mapping


TOTAL_SAMPLE_COUNT = 10
TOTAL_TEST_CASE = 10

data_dir = "tmp/"


for _ in range(0, TOTAL_TEST_CASE):
    model_a_name = f"nasbench_{random.randint(1,TOTAL_TEST_CASE):06d}"
    model_b_name = f"nasbench_{random.randint(1,TOTAL_TEST_CASE):06d}"
    print(model_b_name, model_a_name)

    generate_solution(data_dir, model_a_name, model_b_name)
    # model_a = load_model(data_dir, model_a_name)
    # model_a.summary()
    # model_b = switch_model(data_dir, model_a, model_b_name, use_nasbench_transform=True)
    # model_b.summary()
