from tensorflow.keras.layers import Input, DepthwiseConv2D
from tensorflow.keras.layers import Conv2D, BatchNormalization
from tensorflow.keras.layers import ReLU, AvgPool2D, Flatten, Dense, MaxPool2D


def _group_by_tensor_type():
    inMatrixSet = set()
    exMatrixSet = set()
    preGoup = set()

    inMatrixSet.add(Conv2D)
    inMatrixSet.add(Dense)

    exMatrixSet.add(AvgPool2D)
    exMatrixSet.add(ReLU)
    exMatrixSet.add(MaxPool2D)
    exMatrixSet.add(Flatten)

    return inMatrixSet, exMatrixSet
