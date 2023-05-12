import utils
import json

from argparse import ArgumentParser

parser = ArgumentParser(
    prog="Tensorflow Setup Process",
    description="Tensorflow generate model data",
)
parser.add_argument("-D", "--data-dir")
args = parser.parse_args()

with open("test_models.json") as file:
    test_model_list = json.load(file)

if __name__ == "__main__":
    utils.gen_model_data(args.data_dir, test_model_list)
