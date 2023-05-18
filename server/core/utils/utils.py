from tf2cv.model_provider import get_model as tf2cv_get_model
import tensorflow as tf


def get_model(model_name):
    return tf2cv_get_model(model_name)


def generate_random_input(model):
    shape = list(model.layers[0].input_shape[0])
    shape[0] = 1

    return tf.random.normal(shape)
