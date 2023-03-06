import time
from tensorflow.keras import models


def load_model_a():
    start = time.time()
    model = models.load_model("model_data/model_a.h5")
    end = time.time()
    model.summary()
    print("loading time of whole model A: %ss" % str(end - start))
    return model


def load_model_b():
    start = time.time()
    model = models.load_model("model_data/model_b.h5")
    end = time.time()
    model.summary()
    print("loading time of whole model B: %ss" % str(end - start))
    return model


def switch_to_model_a(model):
    start = time.time()
    model.load_weights("model_data/model_a_weights.h5")
    end = time.time()
    print(
        "loading time of model A using model B and model A's parameters: %ss"
        % str(end - start)
    )
    return model


def switch_to_model_b(model):
    start = time.time()
    model.load_weights("model_data/model_b_weights.h5")
    end = time.time()
    print(
        "loading time of model B using model A and model B's parameters: %ss"
        % str(end - start)
    )
    return model
