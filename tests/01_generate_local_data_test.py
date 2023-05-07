import os
import sys
import shutil

sys.path.insert(1, "setup")
import utils

os.chdir("setup")

data_dir = "../tmp/"
# shutil.rmtree(data_dir)
# os.mkdir(data_dir)
model_name_list = ["resnet50", "vgg16", "vgg19"]
utils.gen_model_data(data_dir, model_name_list)
