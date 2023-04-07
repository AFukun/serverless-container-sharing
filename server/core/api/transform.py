import time

from .tensor_transform_operator import transform_by_tensor_info

import tensorflow as tf
from tensorflow.keras.layers import (
    Conv2D,
    MaxPool2D,
    Dropout,
    BatchNormalization,
    Activation,
    Flatten,
    Dense,
    InputLayer,
)
import tensorflow as tf
import numpy as np


def swap_module(n, m, tempParentModel, solution):
    _map = {}
    for _ in range(len(solution)):
        _map[solution[_][0]] = [solution[_][1], False]

    for _ in range(len(solution)):
        tempIndex = _
        stack = []
        while _ != _map[tempIndex][0] and _map[tempIndex][1] == False:
            _map[tempIndex][1] = True
            stack.append((tempIndex, _map[tempIndex][0]))
            tempIndex = _map[tempIndex][0]
        if _map[tempIndex][1] == False:
            stack.append((tempIndex, _map[tempIndex][0]))
            _map[tempIndex][1] = True

        tempModule = tempParentModel[_]
        while len(stack) > 1:
            popElement = stack.pop()
            tempParentModel[popElement[1]] = tempParentModel[popElement[0]]
        if len(stack) == 1:
            tempParentModel[stack[0][1]] = tempModule

    childModel = tf.keras.Sequential()
    for _, layer in enumerate(tempParentModel[0:m]):
        childModel.add(layer)

    return childModel


def transform(parentModel, childStructureInfo, solution, n, m):
    # step 1: transform by node
    start = time.time()
    model = parentModel
    for _, layer in enumerate(model.layers):
        if solution[_][1] < m:
            transform_by_tensor_info(layer, childStructureInfo[solution[_][1]])

    end = time.time()
    reshape_time = end - start

    # add_modules

    start = time.time()
    tempParentModel = model.layers[:]
    for _ in range(n, n + m):
        if solution[_][1] >= m:
            tempParentModel.append([])
        else:
            typeInfo = childStructureInfo[solution[_][1]]
            if typeInfo[0] == "InputLayer":
                tensor = InputLayer(
                    input_shape=(
                        224,
                        224,
                        3,
                    ),
                    name="input_1",
                )
            elif typeInfo[0] == "Flatten":
                tensor = Flatten()
            elif typeInfo[0] == "Conv2D":
                tensor = Conv2D(
                    filters=typeInfo[1],
                    kernel_size=typeInfo[2],
                    strides=typeInfo[3],
                    padding=typeInfo[4],
                    name=typeInfo[5],
                    activation=typeInfo[6],
                )
            elif typeInfo[0] == "Dense":
                tensor = Dense(
                    units=typeInfo[1], name=typeInfo[2], activation=typeInfo[3]
                )
            elif typeInfo[0] == "MaxPool2D":
                tensor = MaxPool2D(
                    pool_size=typeInfo[1], strides=typeInfo[2], name=typeInfo[3]
                )

            tempParentModel.append(tensor)
    end = time.time()
    add_and_fail_time = end - start

    # step 2: transform by edge
    start = time.time()
    childModel = swap_module(n, m, tempParentModel, solution)
    end = time.time()
    swap_time = end - start

    return (
        childModel,
        f"(reshape in {reshape_time}s,add in {add_and_fail_time}s,swap in {swap_time}s",
    )
