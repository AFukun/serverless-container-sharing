import tensorflow as tf

from .layers import (
    conv3_unit,
    conv1_unit,
    maxpool_unit,
    flatten_unit,
    fc_unit,
    fc_final,
)


def VGG11(shape=(224, 224, 3), classes=1000):
    # Input Layer
    inputs = tf.keras.Input(shape=shape)

    # Block 1
    hidden = conv3_unit(64, inputs)
    hidden = maxpool_unit(hidden)

    # Block 2
    hidden = conv3_unit(128, hidden)
    hidden = maxpool_unit(hidden)

    # Block 3
    hidden = conv3_unit(256, hidden)
    hidden = conv3_unit(256, hidden)
    hidden = maxpool_unit(hidden)

    # Block 4
    hidden = conv3_unit(512, hidden)
    hidden = conv3_unit(512, hidden)
    hidden = maxpool_unit(hidden)

    # Block 5
    hidden = conv3_unit(512, hidden)
    hidden = conv3_unit(512, hidden)
    hidden = maxpool_unit(hidden)

    # FC Block
    hidden = flatten_unit(hidden)
    hidden = fc_unit(4096, hidden)
    hidden = fc_unit(4096, hidden)
    outputs = fc_final(classes, hidden)

    return tf.keras.models.Model(inputs=inputs, outputs=outputs)
