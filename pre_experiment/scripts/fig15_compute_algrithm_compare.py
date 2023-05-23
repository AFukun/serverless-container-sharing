# from ged_munkres import munkres
# from save_information import build_childmodel_info, compute_node_to_node_mapping
# from model_transform import model_structure_transformation, slow_model_structure_transformation
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
import time
import csv
from tensorflow.python.keras.engine.base_layer import TensorFlowOpLayer

INFINITY = 1e1000
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
    Permute,
    LeakyReLU,
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
    elif layer_type == tf.keras.layers.Permute:
        layer.name = layer_info["layer_name"]
        layer.dims = layer_info["layer_dims"]
        # layer.input_spec = layer_info["layer_input_spec"]
    elif layer_type == tf.keras.layers.LeakyReLU:
        layer.name = layer_info["layer_name"]
        layer.alpha = layer_info["layer_alpha"]


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


def slow_model_structure_transformation(
    parentModel, childmodel_info, node_to_node_mapping
):
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
            elif layer_info["layer_type"] == "Permute":
                layer = Permute(
                    name=layer_info["layer_name"], dims=layer_info["layer_dims"]
                )
            elif layer_info["layer_type"] == "LeakyReLU":
                layer = LeakyReLU(
                    name=layer_info["layer_name"], alpha=layer_info["layer_alpha"]
                )
            else:
                raise Exception("This type has not been added")
            parentmodel_layers.append(layer)

    # step 2: transformation by edge
    childModel = fast_swap_node_location(
        parentmodel_layers,
        node_to_node_mapping,
        childmodel_info,
        model,
        len(parentmodel_layers),
    )

    # step3: load childModel weights
    # childModel.load_weights(r'model_data/save_weights.h5')
    return childModel


def fast_swap_node_location(
    parentmodel_layers,
    node_to_node_mapping,
    model_info,
    model,
    parentmodel_layers_length,
):
    child_layers_length = len(model_info)
    child_layers = [[] for _ in range(child_layers_length)]
    Add_index = 0
    for _, mapping in enumerate(node_to_node_mapping):
        if mapping[1] < child_layers_length:
            if mapping[0] < parentmodel_layers_length:
                child_layers[mapping[1]] = parentmodel_layers[mapping[0]]
            else:
                child_layers[mapping[1]] = parentmodel_layers[
                    parentmodel_layers_length + Add_index
                ]
                Add_index += 1
    childmodel = generate_edge(model, child_layers, model_info)
    return childmodel


def model_structure_transformation(parentModel, childmodel_info, node_to_node_mapping):
    # step 1: transform by node
    parentmodel_layers_length = len(parentModel.layers)
    childmodel_layers_length = len(childmodel_info)
    model = parentModel
    parentmodel_layers = model.layers[:]
    for _, mapping in enumerate(node_to_node_mapping):
        if (
            mapping[0] < parentmodel_layers_length
            and mapping[1] < childmodel_layers_length
        ):
            transform_by_layer_info(
                model.layers[mapping[0]], childmodel_info[mapping[1]]
            )
        elif mapping[0] > parentmodel_layers_length:
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
            elif layer_info["layer_type"] == "Permute":
                layer = Permute(
                    name=layer_info["layer_name"], dims=layer_info["layer_dims"]
                )
            elif layer_info["layer_type"] == "LeakyReLU":
                layer = LeakyReLU(
                    name=layer_info["layer_name"], alpha=layer_info["layer_alpha"]
                )
            else:
                raise Exception("This type has not been added")
            parentmodel_layers.append(layer)

    # step 2: transformation by edge
    childModel = fast_swap_node_location(
        parentmodel_layers,
        node_to_node_mapping,
        childmodel_info,
        model,
        parentmodel_layers_length,
    )

    # # step3: load childModel weights
    # childModel.load_weights(r'model_data/save_weights.h5')
    return childModel


def ignore_TFop(model):
    layers = []
    for layer in model.layers:
        if type(layer) != TensorFlowOpLayer:
            layers.append(layer)
    return layers


def node_group(layers):
    group_type_index = {}
    for _, layer in enumerate(layers):
        layer_type = type(layer)
        layer_name = str(layer_type)
        if layer_name in group_type_index:
            group_type_index[layer_name].append(_)
        else:
            group_type_index[layer_name] = [_]
    return group_type_index


def find_solution(
    parent_node_group, child_node_group, parent_model_layer_size, child_model_layer_size
):
    solution = []
    for key, value_list in child_node_group.items():
        if key in parent_node_group:
            for _, value in enumerate(value_list):
                if len(parent_node_group[key]) > 0:
                    solution.append((parent_node_group[key][0], value))
                    del parent_node_group[key][0]
                else:
                    solution.append((parent_model_layer_size + value, value))
        else:
            for _, value in enumerate(value_list):
                solution.append((parent_model_layer_size + value, value))
    for key, value_list in parent_node_group.items():
        for value in value_list:
            solution.append((value, child_model_layer_size + value))
    return solution


def compute_node_to_node_mapping(parentmodel, childmodel):
    """
    design strategy:
    step 1: group
    Step 2: match
    """
    # parent_node_group = node_group(parentmodel)
    # child_node_group = node_group(childmodel)
    # parent_model_layer_size = len(parentmodel.layers)
    # child_model_layer_size = len(childmodel.layers)
    parentmodel_layers = ignore_TFop(parentmodel)
    childmodel_layers = ignore_TFop(childmodel)
    parent_node_group = node_group(parentmodel_layers)
    child_node_group = node_group(childmodel_layers)
    parent_model_layer_size = len(parentmodel_layers)
    child_model_layer_size = len(childmodel_layers)
    node_to_node_mapping = find_solution(
        parent_node_group,
        child_node_group,
        parent_model_layer_size,
        child_model_layer_size,
    )

    return node_to_node_mapping


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
        # layer_info["node_input_shapes"] = layer.inbound_nodes[0].input_shapes
        if type(layer.inbound_nodes[0].input_tensors) == list:
            layer_info["node_input_shapes"] = [
                tuple(input_tensor[0].shape)
                for _, input_tensor in enumerate(layer.inbound_nodes[0].input_tensors)
            ]
        else:
            layer_info["node_input_shapes"] = tuple(
                layer.inbound_nodes[0].input_tensors[0].shape
            )
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
            layer_info["layer_kernel_shape"] = layer.kernel.shape.as_list()
            # tempChildInfo.append(layer.weights[0].shape)
        elif layer_type == tf.keras.layers.Dense:
            layer_info["layer_type"] = "Dense"
            layer_info["layer_name"] = layer.name
            layer_info["layer_units"] = layer.units
            layer_info["layer_activation_name"] = layer.activation.__name__
            layer_info["layer_kernel_shape"] = layer.kernel.shape.as_list()
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
            layer_info["layer_beta_shape"] = list(layer.beta.shape)
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
            layer_info[
                "layer_depthwise_kernel_shape"
            ] = layer.depthwise_kernel.shape.as_list()
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
        elif layer_type == tf.keras.layers.Permute:
            layer_info["layer_type"] = "Permute"
            layer_info["layer_name"] = layer.name
            layer_info["layer_dims"] = layer.dims
            # layer_info["layer_input_spec"] = layer.input_spec
        elif layer_type == tf.keras.layers.LeakyReLU:
            layer_info["layer_type"] = "LeakyReLU"
            layer_info["layer_name"] = layer.name
            layer_info["layer_alpha"] = layer.alpha
        elif layer_type == TensorFlowOpLayer:
            continue
        else:
            raise Exception("This type:{} has not been added".format(type(layer)))

        childmodel_info.append(layer_info)

    return childmodel_info


class MunkresMatrix(object):
    """
    Auxiliary class.
    Stores the sparse representation of the munkres matrix.
    The rows and columns are remapped (to avoid having emtpy rows and columns).
    """

    def __init__(self, values):
        assert all(value >= 0 for i, j, value in values)
        bigvalue = 1e10
        rowindices = list(set(i for (i, j, value) in values))
        colindices = list(set(j for (i, j, value) in values))
        rowmap = dict((k, v) for v, k in enumerate(rowindices))
        colmap = dict((k, v) for v, k in enumerate(colindices))
        self.transposed = False
        self.values = [(rowmap[i], colmap[j], value) for (i, j, value) in values]
        self.values.sort()
        self.rowmap = rowindices
        self.colmap = colindices
        self.nrows = len(rowindices)
        self.real_columns = len(colindices)
        self.ncols = len(colindices) + self.nrows
        # Ensure there is a feasible but very undesireable solution
        # covering the rows
        for i in range(self.nrows):
            self.values.append((i, self.real_columns + i, bigvalue))
        self.K = self.nrows
        self.rowindices = range(self.nrows)
        self.colindices = range(self.ncols)
        self.row_adds = [0] * self.nrows
        self.column_adds = [0] * self.ncols

    def remap(self, indices):
        """
        Transform the list of indices back to the original
        domain.
        """
        return [
            (self.rowmap[i], self.colmap[j])
            for i, j in indices
            if j < self.real_columns
        ]

    def row(self, rowindex):
        """
        Returns the list of (value, column) in row rowindex
        """
        return (
            (value + self.column_adds[j] + self.row_adds[i], j)
            for i, j, value in self.values
            if i == rowindex
        )

    def get_values(self):
        """
        Returns the current values of the matrix.
        """
        return (
            (i, j, value + self.column_adds[j] + self.row_adds[i])
            for i, j, value in self.values
        )

    def add_column(self, colindex, value):
        """
        Adds value to all the elements of column colindex.
        """
        self.column_adds[colindex] += value

    def add_row(self, rowindex, value):
        """
        Adds value to all the elements of row rowindex.
        """
        self.row_adds[rowindex] += value

    def zeros(self):
        """
        Returns the indices (row, col) of all zero elements in the
        matrix. An element is considered to be zero if abs(value) <= 1e-6
        """
        return [(i, j) for (i, j, value) in self.get_values() if abs(value) <= 1e-6]


class Munkres(object):
    """
    Auxiliary class. Use the top level munkres method instead.
    """

    def __init__(self, values):
        """
        Initialize the munkres.
        values: list of non-infinite values entries of the cost matrix
                [(i,j,value)...]
        """
        self.matrix = MunkresMatrix(values)
        self.starred = set()
        self.primed = set()
        self.covered_columns = [False] * self.matrix.ncols
        self.covered_rows = [False] * self.matrix.nrows
        self.last_primed = None

    def munkres(self):
        """
        Executes the munkres algorithm.
        Returns the optimal matching.
        """
        next_step = self._step_1
        while next_step:
            next_step = next_step()

        # Transform the mapping back to the input domain
        return self.matrix.remap(self.starred)

    def _step_1(self):
        """
        For each row of the matrix, find the smallest element and subtract it
        from every element in its row.  Go to Step 2.
        """
        # TODO: This can probably be done much better than using .row(i),
        # but it is executed only once, so the performance penalty is low.
        for i in self.matrix.rowindices:
            minimum = min(self.matrix.row(i))[0]
            self.matrix.add_row(i, -minimum)
        return self._step_2

    def _step_2(self):
        """
        Find a zero (Z) in the resulting matrix.  If there is no starred zero
        in its row or column, star Z.
        Repeat for each element in the matrix. Go to Step 3.
        """
        zeros = self.matrix.zeros()
        for i, j in zeros:
            for i1, j1 in self.starred:
                if i1 == i or j1 == j:
                    break
            else:
                self.starred.add((i, j))
        return self._step_3

    def _step_3(self):
        """
        Cover each column containing a starred zero.  If K columns are covered,
        the starred zeros describe a complete set of unique assignments.  In
        this case, Go to DONE, otherwise, Go to Step 4.
        """
        for _, j in self.starred:
            self.covered_columns[j] = True
        if sum(self.covered_columns) == self.matrix.K:
            return None
        else:
            return self._step_4

    def _find_uncovered_zero(self):
        """
        Returns the (row, column) of one of the uncovered zeros in the matrix.
        If there are no uncovered zeros, returns None
        """
        zeros = self.matrix.zeros()
        for i, j in zeros:
            if not self.covered_columns[j] and not self.covered_rows[i]:
                return (i, j)
        return None

    def _step_4(self):
        """
        Find a noncovered zero and prime it.  If there is no starred zero in
        the row containing this primed zero, Go to Step 5.  Otherwise, cover
        this row and uncover the column containing the starred zero. Continue
        in this manner until there are no uncovered zeros left. Save the
        smallest uncovered value and Go to Step 6.
        """
        done = False
        while not done:
            zero = self._find_uncovered_zero()
            if zero:
                i, j = zero
                self.primed.add((i, j))
                self.last_primed = (i, j)
                st = [(i1, j1) for (i1, j1) in self.starred if i1 == i]
                if not st:
                    return self._step_5
                assert len(st) == 1
                i1, j1 = st[0]
                self.covered_rows[i] = True
                self.covered_columns[j1] = False
            else:
                done = True
        return self._step_6

    def _step_5(self):
        """
        Construct a series of alternating primed and starred zeros as follows.
        Let Z0 represent the uncovered primed zero found in Step 4. Let Z1
        denote the starred zero in the column of Z0 (if any). Let Z2 denote
        the primed zero in the row of Z1 (there will always be one). Continue
        until the series terminates at a primed zero that has no starred zero
        in its column.  Unstar each starred zero of the series, star each
        primed zero of the series, erase all primes and uncover every line in
        the matrix. Return to Step 3.
        """
        last_primed = self.last_primed
        last_starred = None
        primed = [last_primed]
        starred = []
        while True:
            # find the starred zero in the same column of last_primed
            t = [(i, j) for (i, j) in self.starred if j == last_primed[1]]
            if not t:
                break
            assert len(t) == 1
            last_starred = t[0]
            starred.append(last_starred)
            t = [(i, j) for (i, j) in self.primed if i == last_starred[0]]
            assert len(t) == 1
            last_primed = t[0]
            primed.append(last_primed)
        for s in starred:
            self.starred.remove(s)
        for p in primed:
            self.starred.add(p)
        self.primed.clear()
        for i in range(len(self.covered_rows)):
            self.covered_rows[i] = False

        return self._step_3

    def _step_6(self):
        """
        Add the value found in Step 4 to every element of each covered row, and
        subtract it from every element of each uncovered column.  Return to
        Step 4 without altering any stars, primes, or covered lines.
        """
        minval = INFINITY
        for i, j, value in self.matrix.get_values():
            covered = self.covered_rows[i] or self.covered_columns[j]
            if not covered and minval > value:
                minval = value
        assert 1e-6 < abs(minval) < INFINITY
        for i in self.matrix.rowindices:
            if self.covered_rows[i]:
                self.matrix.add_row(i, minval)
        for j in self.matrix.colindices:
            if not self.covered_columns[j]:
                self.matrix.add_column(j, -minval)
        return self._step_4


import random
import itertools


def random_test_munkres(nrows, ncols):
    """
    Naive test for the munkres implementation.
    Generates a random sparse cost matrix, applies munkres, and compares the
    result with the exahustive search.
    nrows, ncols: number of rows and columns of the generated matrix
    """
    values = [
        (i, j, random.random())
        for i in range(nrows)
        for j in range(ncols)
        if random.random() > 0.8
    ]
    values_dict = dict(((i, j), v) for i, j, v in values)
    print(values)
    munkres_match = munkres(values)
    munkres_weight = sum(values_dict[p] for p in munkres_match)
    print(len(munkres_match))
    print(munkres_match)
    print(munkres_weight)
    minimum = min(nrows, ncols)
    rows = set(i for i, j, v in values)
    cols = set(j for i, j, v in values)
    for part_row in itertools.combinations(rows, minimum):
        for part_col in itertools.combinations(cols, minimum):
            matching = zip(part_row, part_col)
            weight = sum(values_dict.get(p, INFINITY) for p in matching)
            if weight < munkres_weight:
                print("Munkres failed")
                print(values)
                print(weight)
                print(matching)
                print("munkres weight", munkres_weight)
                raise Exception()
    return munkres_weight


def munkres(costs):
    """
    Entry method to solve the assignment problem.
    costs: list of non-infinite values entries of the cost matrix
            [(i,j,value)...]
    output: output the result as a list
            [(i_1,j_1, value_1), (i_2,j_2, value_2)...]
    """
    solver = Munkres(costs)
    result = solver.munkres()
    result.sort()
    return result


def test_childmodel_add(model, model_info):
    record = {}
    for _, layer_info in enumerate(model_info):
        # if _ == 0 or _ == 1:
        #     continue
        if _ == 0:
            record[0] = 0
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
        # if len(layer.weights) > 0:
        #     record.append([str(type(layer)), during, layer.weights[0].shape])
        # else:
        #    record.append([str(type(layer)), during])
        record[_] = during

    return record


def test_model_reshape(parentmodel, childmodel, parentmodel_info, childmodel_info):
    weights_set = {Dense, Conv2D, tf.compat.v1.keras.layers.BatchNormalization}
    record = {}
    for i, parentlayer in enumerate(parentmodel.layers):
        for j, childlayer in enumerate(childmodel.layers):
            if type(parentlayer) in weights_set and type(parentlayer) == type(
                childlayer
            ):
                start = time.time()
                transform_by_layer_info(parentlayer, childmodel_info[j])
                end = time.time()
                during = end - start
                record[(i, j)] = during
                transform_by_layer_info(parentlayer, parentmodel_info[i])
    return record


def record_operation_time(parentmodel, childmodel, parentmodel_info, childmodel_info):
    add_operations_time = test_childmodel_add(childmodel, childmodel_info)
    reshape_operations_time = test_model_reshape(
        parentmodel, childmodel, parentmodel_info, childmodel_info
    )
    return add_operations_time, reshape_operations_time


# with open('add.csv', newline='') as csvfile:
#     reader = csv.reader(csvfile, delimiter=',', quotechar='"')
#     records = []
#     add_operations_time = {}
#     for i, record in enumerate(reader):
#         if i == 0:
#             continue
#         add_operations_time[record[0]] = float(record[1])

meta_operation_execute_time = {
    "reshape": 0.000,
    "add": 0,
    "fail": 0,
    "connect": 0.0008,
    "replace": 0,
    "INFINITY": 1e1000,
}


def group_tensor_type():
    inMatrixSet = set()
    inMatrixSet.add(Conv2D)
    inMatrixSet.add(Dense)
    inMatrixSet.add(tf.compat.v1.keras.layers.BatchNormalization)
    return inMatrixSet


def build_cost_matrix(
    parentModel, childModel, add_operations_time, reshape_operations_time
):
    inMatrixSet = group_tensor_type()
    costMatrix = []
    n = len(parentModel.layers)
    m = len(childModel.layers)
    # step 1: compute cost of scaling
    for i in range(n):
        tensorTypeOfParent = type(parentModel.layers[i])
        for j in range(m):
            tensorTypeOfChild = type(childModel.layers[j])
            if (
                tensorTypeOfParent == tensorTypeOfChild
                and tensorTypeOfParent in inMatrixSet
            ):
                cost = (i, j, reshape_operations_time[(i, j)])
            elif (
                tensorTypeOfParent == tensorTypeOfChild
                and tensorTypeOfParent not in inMatrixSet
            ):
                cost = (i, j, 0)
            else:
                cost = (i, j, meta_operation_execute_time["INFINITY"])
            costMatrix.append(cost)
    # step 2: compute cost of deleting
    for i in range(n):
        for j in range(n):
            if i == j:
                cost = (i, j + m, meta_operation_execute_time["fail"])
            else:
                cost = (i, j + m, meta_operation_execute_time["INFINITY"])
            costMatrix.append(cost)
    # step 3: compute cost of adding
    for i in range(m):
        for j in range(m):
            if i == j:
                if i > 0 and type(childModel.layers[i]) != Flatten:
                    cost = (i + n, j, add_operations_time[j])
                    # cost = (i + n, j, 0.5)
                else:
                    cost = (i + n, j, 0.0)
            else:
                cost = (i + n, j, meta_operation_execute_time["INFINITY"])
            costMatrix.append(cost)
    # step 4
    for i in range(m):
        for j in range(n):
            cost = (i + n, j + m, 0)
            costMatrix.append(cost)
    return costMatrix


import pickle


def test_slow():
    slow_test_list = [
        [
            # tf.keras.applications.VGG16(weights="imagenet"),
            # tf.keras.applications.VGG19(weights="imagenet"),
            tf.keras.models.load_model("/data/vgg16.h5"),
            tf.keras.models.load_model("/data/vgg19.h5"),
        ],
        [
            # tf.keras.applications.VGG16(weights="imagenet"),
            # tf.keras.applications.ResNet50(weights="imagenet"),
            tf.keras.models.load_model("/data/vgg16.h5"),
            tf.keras.models.load_model("/data/resnet50.h5"),
        ],
        [
            # tf.keras.applications.ResNet50(weights="imagenet"),
            # tf.keras.applications.VGG19(weights="imagenet"),
            tf.keras.models.load_model("/data/resnet50.h5"),
            tf.keras.models.load_model("/data/vgg19.h5"),
        ],
    ]
    # slow_test_list = [
    #     [tf.keras.applications.VGG16(weights='imagenet'), tf.keras.applications.VGG19(weights='imagenet')],
    #     ]
    for model_pair in slow_test_list:
        parentmodel = model_pair[0]
        childmodel = model_pair[1]
        parentmodel_info = build_childmodel_info(model_pair[0])
        childmodel_info = build_childmodel_info(model_pair[1])
        childmodel.save_weights("save_weights.h5")
        add_operations_time, reshape_operations_time = record_operation_time(
            parentmodel, childmodel, parentmodel_info, childmodel_info
        )
        start = time.time()
        cost_matrix = build_cost_matrix(
            parentmodel, childmodel, add_operations_time, reshape_operations_time
        )
        node_to_node_mapping = munkres(cost_matrix)
        end = time.time()
        print("Time for slow computing algorithms (Munkres): ", end - start)

        start = time.time()
        childmodel = slow_model_structure_transformation(
            parentmodel, childmodel_info, node_to_node_mapping
        )
        childmodel.load_weights("save_weights.h5")
        end = time.time()
        print("Transformation time in slow computing algorithms: ", end - start)


def test_rapid():
    rapid_test_list = [
        [
            # tf.keras.applications.VGG16(weights="imagenet"),
            # tf.keras.applications.VGG19(weights="imagenet"),
            tf.keras.models.load_model("/data/vgg16.h5"),
            tf.keras.models.load_model("/data/vgg19.h5"),
        ],
        [
            # tf.keras.applications.VGG16(weights="imagenet"),
            # tf.keras.applications.ResNet50(weights="imagenet"),
            tf.keras.models.load_model("/data/vgg16.h5"),
            tf.keras.models.load_model("/data/resnet50.h5"),
        ],
        [
            # tf.keras.applications.ResNet50(weights="imagenet"),
            # tf.keras.applications.VGG19(weights="imagenet"),
            tf.keras.models.load_model("/data/resnet50.h5"),
            tf.keras.models.load_model("/data/vgg19.h5"),
        ],
    ]
    for model_pair in rapid_test_list:
        parentmodel = model_pair[0]
        childmodel = model_pair[1]
        parentmodel_info = build_childmodel_info(model_pair[0])
        childmodel_info = build_childmodel_info(model_pair[1])
        childmodel.save_weights("save_weights.h5")
        start = time.time()
        node_to_node_mapping = compute_node_to_node_mapping(parentmodel, childmodel)
        end = time.time()
        print("Time for rapid computing algorithms (Ours): ", end - start)

        start = time.time()
        childModel = model_structure_transformation(
            parentmodel, childmodel_info, node_to_node_mapping
        )
        childModel.load_weights("save_weights.h5")
        end = time.time()
        print("Transformation time in rapid computing algorithms (Ours): ", end - start)


def test():
    test_slow()
    test_rapid()


test()
