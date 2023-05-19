import tensorflow as tf
import tensorflow as tf
import time
import numpy as np
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import activations
from tensorflow.keras.layers import (
    Conv2D,
    MaxPool2D,
    Dropout,
    Activation,
    Flatten,
    Dense,
    InputLayer,
    Add,
    ZeroPadding2D,
    GlobalAveragePooling2D,
    AveragePooling2D,
    DepthwiseConv2D,
    ReLU,
    Reshape,
    Dropout,
    SeparableConv2D,
    Lambda,
)
import tensorflow as tf
from tensorflow.keras.models import Model

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

import time


def build_childmodel_info(childmodel):
    childmodel_info = []
    for layer in childmodel.layers:
        layer_info = {}
        layer_type = type(layer)
        ######################
        inputTensorName = []
        # if type(layer.input) == list:
        #     for input in layer.input:
        #         tempStr = input.name.split('/')
        #         if len(tempStr) == 3:
        #              tempStr[0] += '/' + tempStr[1]
        #         tempStr[0] = tempStr[0].split('_')[0]
        #         # inputTensorName.append(tempStr[0])
        # else:
        #     tempStr = layer.input.name.split('/')
        #     if len(tempStr) == 3:
        #         tempStr[0] += '/' + tempStr[1]
        #     # tempStr[0] = tempStr[0].split('_')[0]
        #     inputTensorName.append(tempStr[0])
        if type(layer.inbound_nodes[0].inbound_layers) == list:
            if len(layer.inbound_nodes[0].inbound_layers) == 0:
                inputTensorName.append(layer.inbound_nodes[0].outbound_layer.name)
            else:
                for inbound_layer in layer.inbound_nodes[0].inbound_layers:
                    inputTensorName.append(inbound_layer.name)
        else:
            inputTensorName.append(layer.inbound_nodes[0].inbound_layers.name)
        layer_info["input_tensor_name"] = inputTensorName
        layer_info["node_output_tensors_shape"] = tuple(
            layer.inbound_nodes[0].output_tensors[0].shape
        )
        layer_info["node_input_shapes"] = layer.inbound_nodes[0].input_shapes
        layer_info["node_output_shapes"] = layer.inbound_nodes[0].output_shapes
        #######################
        if layer_type == tf.keras.layers.Conv2D:
            layer_info["layer_type"] = "Conv2D"
            layer_info["layer_name"] = layer.name
            layer_info["layer_filters"] = layer.filters
            layer_info["layer_kernel_size"] = layer.kernel_size
            layer_info["layer_strides"] = layer.strides
            layer_info["layer_padding"] = layer.padding
            layer_info["layer_activation_name"] = layer.activation.__name__
            layer_info["layer_use_bias"] = layer.use_bias
            layer_info["layer_kernel_shape"] = layer.kernel.shape
            # tempChildInfo.append(layer.weights[0].shape)
        elif layer_type == tf.keras.layers.Dense:
            layer_info["layer_type"] = "Dense"
            layer_info["layer_name"] = layer.name
            layer_info["layer_units"] = layer.units
            layer_info["layer_activation_name"] = layer.activation.__name__
            layer_info["layer_kernel_shape"] = layer.kernel.shape
        elif layer_type == tf.keras.layers.MaxPool2D:
            layer_info["layer_type"] = "MaxPool2D"
            layer_info["layer_name"] = layer.name
            layer_info["layer_pool_size"] = layer.pool_size
            layer_info["layer_strides"] = layer.strides
        elif layer_type == tf.keras.layers.Flatten:
            layer_info["layer_type"] = "Flatten"
            layer_info["layer_name"] = layer.name
        elif layer_type == tf.keras.layers.InputLayer:
            layer_info["layer_type"] = "InputLayer"
            layer_info["layer_name"] = layer.name
        elif layer_type == tf.compat.v1.keras.layers.BatchNormalization:
            layer_info["layer_type"] = "BatchNormalization"
            layer_info["layer_name"] = layer.name
            layer_info["layer_axis"] = layer.axis
            layer_info["layer_scale"] = layer.scale
            layer_info["layer_epsilon"] = layer.epsilon
            layer_info["layer_beta_shape"] = layer.beta.shape
        elif layer_type == tf.keras.layers.Add:
            layer_info["layer_type"] = "Add"
            layer_info["layer_name"] = layer.name
        elif layer_type == tf.keras.layers.ZeroPadding2D:
            layer_info["layer_type"] = "ZeroPadding2D"
            layer_info["layer_name"] = layer.name
            layer_info["layer_padding"] = layer.padding
        elif layer_type == tf.keras.layers.Activation:
            layer_info["layer_type"] = "Activation"
            layer_info["layer_name"] = layer.name
            layer_info["layer_activation_name"] = layer.activation_name
        elif layer_type == tf.keras.layers.GlobalAveragePooling2D:
            layer_info["layer_type"] = "GlobalAveragePooling2D"
            layer_info["layer_name"] = layer.name
        elif layer_type == tf.compat.v1.keras.layers.Concatenate:
            layer_info["layer_type"] = "Concatenate"
            layer_info["layer_name"] = layer.name
            layer_info["layer_axis"] = layer.axis
        elif layer_type == tf.keras.layers.AveragePooling2D:
            layer_info["layer_type"] = "AveragePooling2D"
            layer_info["layer_name"] = layer.name
            layer_info["layer_pool_size"] = layer.pool_size
            layer_info["layer_strides"] = layer.strides
            layer_info["layer_padding"] = layer.padding
        elif layer_type == tf.keras.layers.DepthwiseConv2D:
            layer_info["layer_type"] = "DepthwiseConv2D"
            layer_info["layer_name"] = layer.name
            layer_info["layer_kernel_size"] = layer.kernel_size
            layer_info["depth_multiplier"] = layer.depth_multiplier
            layer_info["layer_strides"] = layer.strides
            layer_info["layer_padding"] = layer.padding
            layer_info["layer_use_bias"] = layer.use_bias
            layer_info["layer_depthwise_kernel_shape"] = layer.depthwise_kernel.shape
        elif layer_type == tf.keras.layers.ReLU:
            layer_info["layer_type"] = "ReLU"
            layer_info["layer_name"] = layer.name
            layer_info["layer_max_value"] = layer.max_value
        elif layer_type == tf.keras.layers.Reshape:
            layer_info["layer_type"] = "Reshape"
            layer_info["layer_name"] = layer.name
            layer_info["layer_target_shape"] = layer.target_shape
        elif layer_type == tf.keras.layers.Dropout:
            layer_info["layer_type"] = "Dropout"
            layer_info["layer_name"] = layer.name
            layer_info["layer_rate"] = layer.rate
        elif layer_type == tf.keras.layers.SeparableConv2D:
            layer_info["layer_type"] = "SeparableConv2D"
            layer_info["layer_name"] = layer.name
            layer_info["layer_kernel_size"] = layer.kernel_size
            layer_info["layer_filters"] = layer.filters
            layer_info["layer_padding"] = layer.padding
            layer_info["layer_use_bias"] = layer.use_bias
            layer_info["layer_depthwise_kernel_shape"] = layer.depthwise_kernel.shape
        else:
            raise Exception("This type:{} has not been added".format(type(layer)))

        childmodel_info.append(layer_info)

    return childmodel_info


def resize_by_layer_info(oldTensorShape, newTensorShape, oldTensor, use_bias):
    weight = oldTensor.weights[0]
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
    oldTensor.kernel = tf.Variable(newWeight)

    if oldTensor.bias is None and use_bias:
        newBias = tf.zeros(shape=(newTensorShape[-1],))
        oldTensor.use_bias = True
    elif not oldTensor.bias is None and use_bias:
        bias = oldTensor.bias
        if oldTensorShape[-1] > newTensorShape[-1]:
            newBias = bias[: newTensorShape[-1]]
        else:
            newBias = tf.pad(bias, [[newTensorShape[-1] - oldTensorShape[-1], 0]])
    elif not oldTensor.bias is None and use_bias == False:
        oldTensor.use_bias = False
        oldTensor.bias = None
        return
    else:
        return
    oldTensor.bias = tf.Variable(newBias)


def resize_bn_by_layer_info(old_layer_shape, new_layer_shape, old_layer, scale):
    old_layer_length = 1
    new_layer_length = 1
    for _ in old_layer_shape:
        old_layer_length *= _
    for _ in new_layer_shape:
        new_layer_length *= _
    if old_layer_length > new_layer_length:
        if old_layer.scale and scale:
            old_layer.gamma = tf.Variable(old_layer.gamma[:new_layer_length])
        elif old_layer.scale and scale == False:
            old_layer.gamma = None
            old_layer._gamma_const = K.constant(1.0, shape=(new_layer_shape[-1],))
        elif old_layer.scale == False and scale:
            old_layer.gamma = tf.Variable(tf.zeros(shape=(new_layer_shape[-1],)))

        old_layer.beta = tf.Variable(old_layer.beta[:new_layer_length])
        old_layer.moving_mean = tf.Variable(old_layer.moving_mean[:new_layer_length])
        old_layer.moving_variance = tf.Variable(
            old_layer.moving_variance[:new_layer_length]
        )
    else:
        if old_layer.scale and scale:
            old_layer.gamma = tf.Variable(
                tf.pad(old_layer.gamma, [[new_layer_length - old_layer_length, 0]])
            )
        elif old_layer.scale and scale == False:
            old_layer.gamma = None
            old_layer._gamma_const = K.constant(1.0, shape=(new_layer_shape[-1],))
        elif old_layer.scale == False and scale:
            old_layer.gamma = tf.Variable(tf.zeros(shape=(new_layer_shape[-1],)))
        # oldTensor.gamma = tf.Variable(tf.pad(oldTensor.gamma, [[newTensorLength - oldTensorLength, 0]]))
        old_layer.beta = tf.Variable(
            tf.pad(old_layer.beta, [[new_layer_length - old_layer_length, 0]])
        )
        old_layer.moving_mean = tf.Variable(
            tf.pad(old_layer.moving_mean, [[new_layer_length - old_layer_length, 0]])
        )
        old_layer.moving_variance = tf.Variable(
            tf.pad(
                old_layer.moving_variance, [[new_layer_length - old_layer_length, 0]]
            )
        )


def transform_by_layer_info(layer, layer_info):
    layer_type = type(layer)
    if layer_type == tf.keras.layers.Conv2D:
        old_layer_shape = layer.kernel.shape
        new_layer_shape = layer_info["layer_kernel_shape"]
        resize_by_layer_info(
            old_layer_shape, new_layer_shape, layer, layer_info["layer_use_bias"]
        )
        layer.name = layer_info["layer_name"]
        layer.filters = layer_info["layer_filters"]
        layer.kernel_size = layer_info["layer_kernel_size"]
        layer.strides = layer_info["layer_strides"]
        layer.padding = layer_info["layer_padding"]
        layer.input_spec.axes[-1] = layer_info["layer_kernel_shape"][2]
        if layer_info["layer_activation_name"] == "relu":
            layer.activation = tf.keras.activations.relu
        elif layer_info["layer_activation_name"] == "softmax":
            layer.activation = tf.keras.activations.softmax
        elif layer_info["layer_activation_name"] == "linear":
            layer.activation = tf.keras.activations.linear

    elif layer_type == tf.keras.layers.Dense:
        old_layer_shape = layer.kernel.shape
        new_layer_shape = layer_info["layer_kernel_shape"]
        resize_by_layer_info(old_layer_shape, new_layer_shape, layer, True)
        layer.name = layer_info["layer_name"]
        layer.units = layer_info["layer_units"]
        if layer_info["layer_activation_name"] == "relu":
            layer.activation = tf.keras.activations.relu
        elif layer_info["layer_activation_name"] == "softmax":
            layer.activation = tf.keras.activations.softmax
        elif layer_info["layer_activation_name"] == "linear":
            layer.activation = tf.keras.activations.linear
        layer.input_spec.axes[-1] = layer_info["layer_kernel_shape"][0]
    elif layer_type == tf.keras.layers.MaxPool2D:
        layer.name = layer_info["layer_name"]
        layer.pool_size = layer_info["layer_pool_size"]
        layer.strides = layer_info["layer_strides"]
    elif layer_type == tf.compat.v1.keras.layers.BatchNormalization:
        old_layer_shape = layer.beta.shape
        new_layer_shape = layer_info["layer_beta_shape"]
        resize_bn_by_layer_info(
            old_layer_shape, new_layer_shape, layer, layer_info["layer_scale"]
        )
        layer.name = layer_info["layer_name"]
        layer.axis = layer_info["layer_axis"]
        layer.scale = layer_info["layer_scale"]
        layer.epsilon = layer_info["layer_epsilon"]
    elif layer_type == tf.keras.layers.Add:
        layer.name = layer_info["layer_name"]
    elif layer_type == tf.keras.layers.ZeroPadding2D:
        layer.name = layer_info["layer_name"]
        layer.padding = layer_info["layer_padding"]
    elif layer_type == tf.compat.v1.keras.layers.Concatenate:
        layer.name = layer_info["layer_name"]
        layer.axis = layer_info["layer_axis"]
    elif layer_type == tf.keras.layers.Activation:
        layer.name = layer_info["layer_name"]
        layer.activation = activations.get(layer_info["layer_activation_name"])
    elif layer_type == tf.keras.layers.AveragePooling2D:
        layer.name = layer_info["layer_name"]
        layer.pool_size = layer_info["layer_pool_size"]
        layer.strides = layer_info["layer_strides"]
        layer.padding = layer_info["layer_padding"]
    elif layer_type == tf.keras.layers.DepthwiseConv2D:
        old_layer_shape = layer.depthwise_kernel.shape
        new_layer_shape = layer_info["layer_depthwise_kernel_shape"]
        resize_by_layer_info(
            old_layer_shape, new_layer_shape, layer, layer_info["layer_use_bias"]
        )
        layer.name = layer_info["layer_name"]
        layer.depth_multiplier = layer_info["depth_multiplier"]
        layer.kernel_size = layer_info["layer_kernel_size"]
        layer.strides = layer_info["layer_strides"]
        layer.padding = layer_info["layer_padding"]
        layer.input_spec.axes[-1] = layer_info["layer_depthwise_kernel_shape"][2]
    elif layer_type == tf.keras.layers.ReLU:
        layer.name = layer_info["layer_name"]
        max_value = layer_info["layer_max_value"]
        if max_value is not None:
            layer.max_value = K.cast_to_floatx(max_value)
    elif layer_type == tf.keras.layers.Reshape:
        layer.name = layer_info["layer_name"]
        layer.target_shape = layer_info["layer_target_shape"]
    elif layer_type == tf.keras.layers.Dropout:
        layer.name = layer_info["layer_name"]
        layer.rate = layer_info["layer_rate"]
    elif layer_type == tf.keras.layers.SeparableConv2D:
        old_layer_shape = layer.depthwise_kernel.shape
        new_layer_shape = layer_info["layer_depthwise_kernel_shape"]
        resize_by_layer_info(
            old_layer_shape, new_layer_shape, layer, layer_info["layer_use_bias"]
        )
        layer.name = layer_info["layer_name"]
        layer.filters = layer_info["layer_filters"]
        layer.kernel_size = layer_info["layer_kernel_size"]
        layer.padding = layer_info["layer_padding"]
        layer.input_spec.axes[-1] = layer_info["layer_depthwise_kernel_shape"][2]


def test_resnet50_diverse_operation_load_time(model, model_info):
    record = []
    for _, layer_info in enumerate(model_info):
        if _ == 0 or _ == 1:
            continue
        start = time.time()
        if layer_info["layer_type"] == "InputLayer":
            layer = InputLayer(
                input_shape=(
                    224,
                    224,
                    3,
                ),
                name="input_1",
            )
        elif layer_info["layer_type"] == "Flatten":
            layer = Flatten()
        elif layer_info["layer_type"] == "Conv2D":
            layer = Conv2D(
                name=layer_info["layer_name"],
                filters=layer_info["layer_filters"],
                kernel_size=layer_info["layer_kernel_size"],
                strides=layer_info["layer_strides"],
                padding=layer_info["layer_padding"],
                activation=layer_info["layer_activation_name"],
                use_bias=layer_info["layer_use_bias"],
            )
        elif layer_info["layer_type"] == "Dense":
            layer = Dense(
                name=layer_info["layer_name"],
                units=layer_info["layer_units"],
                activation=layer_info["layer_activation_name"],
            )
        elif layer_info["layer_type"] == "MaxPool2D":
            layer = MaxPool2D(
                name=layer_info["layer_name"],
                pool_size=layer_info["layer_pool_size"],
                strides=layer_info["layer_strides"],
            )
        elif layer_info["layer_type"] == "AveragePooling2D":
            layer = AveragePooling2D(
                name=layer_info["layer_name"],
                pool_size=layer_info["layer_pool_size"],
                strides=layer_info["layer_strides"],
                padding=layer_info["layer_padding"],
            )
        elif layer_info["layer_type"] == "BatchNormalization":
            layer = tf.compat.v1.keras.layers.BatchNormalization(
                name=layer_info["layer_name"],
                axis=layer_info["layer_axis"],
                scale=layer_info["layer_scale"],
                epsilon=layer_info["layer_epsilon"],
            )
        elif layer_info["layer_type"] == "Add":
            layer = Add(name=layer_info["layer_name"])
        elif layer_info["layer_type"] == "ZeroPadding2D":
            layer = ZeroPadding2D(
                name=layer_info["layer_name"], padding=layer_info["layer_padding"]
            )
        elif layer_info["layer_type"] == "Activation":
            layer = Activation(
                layer_info["layer_activation_name"], name=layer_info["layer_name"]
            )
        elif layer_info["layer_type"] == "GlobalAveragePooling2D":
            layer = GlobalAveragePooling2D(name=layer_info["layer_name"])
        elif layer_info["layer_type"] == "Concatenate":
            layer = tf.compat.v1.keras.layers.Concatenate(
                name=layer_info["layer_name"], axis=layer_info["layer_axis"]
            )
        elif layer_info["layer_type"] == "DepthwiseConv2D":
            layer = DepthwiseConv2D(
                name=layer_info["layer_name"],
                kernel_size=layer_info["layer_kernel_size"],
                padding=layer_info["layer_padding"],
                depth_multiplier=layer_info["depth_multiplier"],
                strides=layer_info["layer_strides"],
                use_bias=layer_info["layer_use_bias"],
            )
        elif layer_info["layer_type"] == "ReLU":
            layer = ReLU(
                name=layer_info["layer_name"], max_value=layer_info["layer_max_value"]
            )
        elif layer_info["layer_type"] == "Reshape":
            layer = Reshape(
                name=layer_info["layer_name"],
                target_shape=layer_info["layer_target_shape"],
            )
        elif layer_info["layer_type"] == "Dropout":
            layer = Dropout(
                name=layer_info["layer_name"], rate=layer_info["layer_rate"]
            )
        elif layer_info["layer_type"] == "SeparableConv2D":
            layer = SeparableConv2D(
                name=layer_info["layer_name"],
                kernel_size=layer_info["layer_kernel_size"],
                padding=layer_info["layer_padding"],
                filters=layer_info["layer_filters"],
                use_bias=layer_info["layer_use_bias"],
            )
        else:
            raise Exception("This type has not been added")

        layer(model.layers[_].inbound_nodes[0].input_tensors)
        end = time.time()
        during = end - start
        if type(layer) == Conv2D or type(layer) == Dense:
            shape = layer.kernel.shape
        else:
            shape = "no shape"
        record.append([str(type(layer)), shape, during])
    return record


import csv


def record_data(file_name, meta_operation_type, record):
    with open(file_name, "w", encoding="utf-8", newline="") as file_obj:
        # create object
        writer = csv.writer(file_obj)
        # write table header
        writer.writerow(meta_operation_type)
        # write the data of each row to the table
        for _ in record:
            writer.writerow(_)


model = tf.keras.models.load_model("/data/resnet50.h5")
model_info = build_childmodel_info(model)
record = test_resnet50_diverse_operation_load_time(model, model_info)
record_data("fig4_loading_latency_for_varing_operations.csv", "load_time", record)
