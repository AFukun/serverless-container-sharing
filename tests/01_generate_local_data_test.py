import os
import sys

sys.path.insert(1, "setup")
import utils

os.chdir("setup")
data_dir = "../tmp/"
utils.gen_model_data(data_dir)
