import tensorflow as tf


def generate_random_input(model):
    shape = list(model.layers[0].input_shape[0])
    shape[0] = 1

    return tf.random.normal(shape)
