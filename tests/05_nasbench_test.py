import sys

sys.path.insert(1, "setup")
sys.path.insert(1, "server")

import json
import random
from core.api import *
from utils import compute_node_to_node_mapping


TOTAL_SAMPLE_COUNT = 10
TOTAL_TEST_CASE = 10


for _ in range(0, TOTAL_TEST_CASE):
    model_a_name = f"nasbench_{random.randint(1,TOTAL_TEST_CASE):06d}"
    model_b_name = f"nasbench_{random.randint(1,TOTAL_TEST_CASE):06d}"
    model_a = load_model(data_dir, model_a_name)
