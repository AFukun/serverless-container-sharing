import sys

sys.path.insert(1, "setup")
sys.path.insert(1, "server")

import json
import time
import random
from core.api import *
from core import Server


TOTAL_SAMPLE_COUNT = 10
TOTAL_TEST_CASE = 10

data_dir = "tmp/"
server = Server(data_dir)


for _ in range(0, TOTAL_TEST_CASE):
    model_a_name = f"nasbench_{random.randint(1,TOTAL_TEST_CASE):06d}"
    model_b_name = f"nasbench_{random.randint(1,TOTAL_TEST_CASE):06d}"
    server.manual_generate_solution(model_b_name)
    server.manual_load_model(model_a_name)

    start = time.time()
    server.manual_switch_nasbench_model(model_b_name)
    end = time.time()
    switch_structure_time = end - start

    server.manual_load_model(model_b_name)
    start = time.time()
    server.manual_load_weights()
    end = time.time()
    load_weights_time = end - start

    print(
        f"{model_b_name} to {model_a_name} in {(switch_structure_time+ load_weights_time):.2f}s"
    )
