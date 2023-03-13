import time
from tensorflow.keras import models


def load_model_a():
    model = models.load_model("/data/model_a.h5")
    return model


def load_model_b():
    model = models.load_model("/data/model_b.h5")
    return model


def switch_to_model_a(model):
    model.load_weights("/data/model_a_weights.h5")
    return model


def switch_to_model_b(model):
    model.load_weights("/data/model_b_weights.h5")
    return model
