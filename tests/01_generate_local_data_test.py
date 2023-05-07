import os
import sys

sys.path.insert(1, "setup")
import utils

os.chdir("setup")
data_dir = "../tmp/"
model_name_list = ["resnet50", "vgg16", "vgg19"]
utils.gen_model_data(data_dir, model_name_list)
