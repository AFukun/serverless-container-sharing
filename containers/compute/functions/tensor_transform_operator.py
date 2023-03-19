import tensorflow as tf
import time
import numpy as np


def resize_tensorflow(oldTensorShape, newTensorShape, oldTensor):
    weight = oldTensor.weights[0]
    bias = oldTensor.weights[1]
    oldTensorLength = 1
    newTensorLength = 1
    for _ in oldTensorShape:
        oldTensorLength *= _
    for _ in newTensorShape:
        newTensorLength *= _

    tempOldTensor = tf.reshape(
        weight,
        [
            -1,
        ],
    )

    if oldTensorLength > newTensorLength:
        tempOldTensor = tempOldTensor[:newTensorLength]
        newWeight = tf.reshape(tempOldTensor, newTensorShape)
    else:
        tempOldTensor = tf.pad(tempOldTensor, [[newTensorLength - oldTensorLength, 0]])
        newWeight = tf.reshape(tempOldTensor, newTensorShape)

    if oldTensorShape[-1] > newTensorShape[-1]:
        newBias = bias[: newTensorShape[-1]]
    else:
        newBias = tf.pad(bias, [[newTensorShape[-1] - oldTensorShape[-1], 0]])
    oldTensor.kernel, oldTensor.bias = (tf.Variable(newWeight), tf.Variable(newBias))


def transform_by_tensor_info(tensor, childInfo):
    _type = type(tensor)

    if _type == tf.keras.layers.Conv2D:
        oldTensorShape = tensor.weights[0].shape
        newTensorShape = childInfo[-1]
        resize_tensorflow(oldTensorShape, newTensorShape, tensor)
        tensor.filters = childInfo[1]
        tensor.kernel_size = childInfo[2]
        tensor.strides = childInfo[3]
        tensor.padding = childInfo[4]
        tensor._name = childInfo[5]
        if childInfo[6] == "relu":
            tensor.activation = tf.keras.activations.relu
        elif childInfo[6] == "softmax":
            tensor.activation = tf.keras.activations.softmax
        tensor.input_spec.axes[-1] = childInfo[-1][2]

    elif _type == tf.keras.layers.Dense:
        oldTensorShape = tensor.weights[0].shape
        newTensorShape = childInfo[-1]

        resize_tensorflow(oldTensorShape, newTensorShape, tensor)
        tensor.units = childInfo[1]
        tensor._name = childInfo[2]
        if childInfo[3] == "relu":
            tensor.activation = tf.keras.activations.relu
        elif childInfo[3] == "softmax":
            tensor.activation = tf.keras.activations.softmax
        tensor.input_spec.axes[-1] = childInfo[-1][0]

    elif _type == tf.keras.layers.MaxPool2D:
        tensor.pool_size = childInfo[1]
        tensor.strides = childInfo[2]
        tensor._name = childInfo[3]
