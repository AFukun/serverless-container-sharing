import os
import shutil


def copy_samples(data_dir):
    for file_name in os.listdir("samples/"):
        shutil.copy(f"samples/{file_name}", data_dir)
