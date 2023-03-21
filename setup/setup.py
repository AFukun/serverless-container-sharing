import utils

if __name__ == "__main__":
    data_dir = "/data/"
    utils.gen_model_data(data_dir)
    utils.copy_samples(data_dir)
