import tensorflow as tf
import time
import numpy as np
from tensorflow.python.keras import backend as K
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
)
from tensorflow.python.keras import activations


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


def generate_edge(model, layers, model_info):
    # initialize
    childmodel_layers_length = len(layers)
    inbound_layers = {}
    inbound_layers[model_info[0]["input_tensor_name"][0]] = layers[0]
    for _ in range(1, childmodel_layers_length):
        # only assign value by order
        if _ == 1:
            layers[_ - 1].inbound_nodes[0].output_tensors = model.input
        else:
            layers[_ - 1].inbound_nodes[0].output_tensors = tf.keras.Input(
                shape=model_info[_ - 1]["node_output_tensors_shape"]
            )
        # get input_tensors/inbound_layers by layers_name
        inbound_layers_name = model_info[_]["input_tensor_name"]
        if len(inbound_layers_name) == 1:
            temp_inbound_layers = inbound_layers[inbound_layers_name[0]]
            temp_inbound_layers_output_tensors = temp_inbound_layers.inbound_nodes[
                0
            ].output_tensors
        else:
            temp_inbound_layers_output_tensors = []
            temp_inbound_layers = []
            for name in inbound_layers_name:
                temp_layer = inbound_layers[name]
                temp_inbound_layers.append(temp_layer)
            for layer in temp_inbound_layers:
                temp_inbound_layers_output_tensors.append(
                    layer.inbound_nodes[0].output_tensors
                )
        # create nodes by assign input_tensors and inbound_layers
        if len(layers[_].inbound_nodes) == 0:
            layers[_](temp_inbound_layers_output_tensors)
            layers[_].inbound_nodes[0].inbound_layers = temp_inbound_layers
        else:
            layers[_].inbound_nodes[
                0
            ].input_tensors = temp_inbound_layers_output_tensors
            layers[_].inbound_nodes[0].inbound_layers = temp_inbound_layers
            if len(inbound_layers_name) > 1:
                layers[_].inbound_nodes[0].node_indices = [
                    0 for _ in range(len(inbound_layers_name))
                ]
                layers[_].inbound_nodes[0].tensor_indices = [
                    0 for _ in range(len(inbound_layers_name))
                ]
        # add new dict item
        inbound_layers[model_info[_]["layer_name"]] = layers[_]
    # model's output_tensors setting
    x = tf.keras.Input(
        shape=model_info[childmodel_layers_length - 1]["node_output_tensors_shape"]
    )
    layers[childmodel_layers_length - 1].inbound_nodes[0].output_tensors = x
    for _ in range(childmodel_layers_length):
        layers[_].inbound_nodes[0].input_shapes = model_info[_]["node_input_shapes"]
        layers[_].inbound_nodes[0].output_shapes = model_info[_]["node_output_shapes"]
    model._init_graph_network(
        inputs=model.input, outputs=x, _layer=layers[childmodel_layers_length - 1]
    )
    return model


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
        layer.pool_size = tuple(layer_info["layer_pool_size"])
        layer.strides = tuple(layer_info["layer_strides"])
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
        layer.pool_size = tuple(layer_info["layer_pool_size"])
        layer.strides = tuple(layer_info["layer_strides"])
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
        layer.target_shape = tuple(layer_info["layer_target_shape"])
    elif layer_type == tf.keras.layers.Dropout:
        layer.name = layer_info["layer_name"]
        layer.rate = layer_info["layer_rate"]


def swap_node_location(parentmodel_layers, node_to_node_mapping, model_info, model):
    _map = {}
    for _ in range(len(node_to_node_mapping)):
        _map[node_to_node_mapping[_][1]] = True
    for left in range(len(node_to_node_mapping)):
        stack = []
        if (
            left != node_to_node_mapping[left][1]
            and _map[node_to_node_mapping[left][1]]
        ):
            templeft = left
            _map[templeft] = False
            while left != node_to_node_mapping[templeft][1]:
                stack.append(node_to_node_mapping[templeft][1])
                templeft = node_to_node_mapping[templeft][1]
                _map[templeft] = False
            tempmodule = parentmodel_layers[left]
            while len(stack) > 0:
                pop_element = stack.pop()
                parentmodel_layers[
                    node_to_node_mapping[pop_element][1]
                ] = parentmodel_layers[pop_element]
            parentmodel_layers[node_to_node_mapping[left][1]] = tempmodule
    childmodel = generate_edge(
        model, parentmodel_layers[0 : len(model_info)], model_info
    )
    return childmodel


def model_structure_transformation(parentModel, childmodel_info, node_to_node_mapping):
    # step 1: transform by node
    parentmodel_layers_length = len(parentModel.layers)
    childmodel_layers_length = len(childmodel_info)
    model = parentModel
    for _, layer in enumerate(model.layers):
        if node_to_node_mapping[_][1] < childmodel_layers_length:
            transform_by_layer_info(layer, childmodel_info[node_to_node_mapping[_][1]])
    parentmodel_layers = model.layers[:]

    # add_modules
    for _ in range(
        parentmodel_layers_length, parentmodel_layers_length + childmodel_layers_length
    ):
        if node_to_node_mapping[_][1] >= childmodel_layers_length:
            parentmodel_layers.append([])
        else:
            layer_info = childmodel_info[node_to_node_mapping[_][1]]
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
                    pool_size=tuple(layer_info["layer_pool_size"]),
                    strides=tuple(layer_info["layer_strides"]),
                )
            elif layer_info["layer_type"] == "AveragePooling2D":
                layer = AveragePooling2D(
                    name=layer_info["layer_name"],
                    pool_size=tuple(layer_info["layer_pool_size"]),
                    strides=tuple(layer_info["layer_strides"]),
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
                    name=layer_info["layer_name"],
                    max_value=layer_info["layer_max_value"],
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
            else:
                raise Exception("This type has not been added")
            parentmodel_layers.append(layer)

    # step 2: transformation by edge
    childModel = swap_node_location(
        parentmodel_layers, node_to_node_mapping, childmodel_info, model
    )

    # step3: load childModel weights
    # childModel.load_weights(r'model_data/save_weights.h5')
    return childModel
