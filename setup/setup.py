import utils

from argparse import ArgumentParser

parser = ArgumentParser(
    prog="Tensorflow Setup Process",
    description="Tensorflow generate model data",
)
parser.add_argument("-D", "--data-dir")
args = parser.parse_args()

if __name__ == "__main__":
    model_list = ["resnet50", "vgg16", "vgg19"]

    utils.gen_model_data(args.data_dir, model_list)
