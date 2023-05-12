import utils

from argparse import ArgumentParser

parser = ArgumentParser(
    prog="Tensorflow Setup Process",
    description="Tensorflow generate model data",
)
parser.add_argument("-D", "--data-dir")
args = parser.parse_args()

if __name__ == "__main__":
    model_list = [
        "resnet18",
        "resnet50",
        "resnet101",
        "vgg13",
        "vgg16",
        "vgg19",
        # "densenet121",
        # "densenet169",
        # "sparsenet121",
        # "sparsenet169",
        "mobilenet_w1",
    ]

    utils.gen_model_data(args.data_dir, model_list)
