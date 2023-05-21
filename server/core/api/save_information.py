import tensorflow as tf

# from options import args_parser
# from ged_munkres import munkres
#
# from group_by_tensor_type import _group_by_tensor_type


# two layers


# def build_cost_matrix(parentModel, childModel):
#     args = args_parser()
#     inMatrixSet, exMatrixSet = _group_by_tensor_type()
#     costMatrix = []
#
#     n = len(parentModel.layers)
#     m = len(childModel.layers)
#     print('n={},m={}'.format(n, m))
#     # step 1: compute cost of scaling
#     for i in range(n):
#         tensorTypeOfParent = type(parentModel.layers[i])
#         for j in range(m):
#             tensorTypeOfChild = type(childModel.layers[j])
#             if tensorTypeOfParent == tensorTypeOfChild and tensorTypeOfParent in inMatrixSet:
#                 cost = (i, j, args.timeScale)
#             elif tensorTypeOfParent == tensorTypeOfChild:
#                 cost = (i, j, 0)
#             else:
#                 cost = (i, j, args.INFINITY)
#             costMatrix.append(cost)
#     # step 2: compute cost of deleting
#     for i in range(n):
#         for j in range(n):
#             if i == j:
#                 cost = (i, j + m, args.timeDeleteTensor)
#             else:
#                 cost = (i, j + m, args.INFINITY)
#             costMatrix.append(cost)
#     # step 3: compute cost of adding
#     for i in range(m):
#         for j in range(m):
#             if i == j:
#                 cost = (i + n, j, args.timeAddTensor)
#             else:
#                 cost = (i + n, j, args.INFINITY)
#             costMatrix.append(cost)
#     # step 4
#     for i in range(m):
#         for j in range(n):
#             cost = (i + n, j + m, 0)
#             costMatrix.append(cost)
#
#     return n, m, costMatrix


def slow_compute_node_to_node_mapping(parentmodel, childmodel):
    finded_solution = set()
    node_to_node_mapping = []
    parentmodel_layers_length = len(parentmodel.layers)
    childmodel_layers_length = len(childmodel.layers)

    # step 1: compute cost of scaling
    for ind_parent in range(parentmodel_layers_length):
        tensor_type_of_parent = type(parentmodel.layers[ind_parent])
        for ind_child in range(parentmodel_layers_length + childmodel_layers_length):
            if ind_child < childmodel_layers_length:
                tensor_type_of_child = type(childmodel.layers[ind_child])
                if (
                    tensor_type_of_parent == tensor_type_of_child
                    and ind_child not in finded_solution
                ):
                    finded_solution.add(ind_child)
                    node_to_node_mapping.append((ind_parent, ind_child))
                    break
            else:
                if ind_child not in finded_solution:
                    finded_solution.add(ind_child)
                    node_to_node_mapping.append((ind_parent, ind_child))
                    break
    # step 2:
    for ind_child in range(childmodel_layers_length):
        if ind_child not in finded_solution:
            finded_solution.add(ind_child)
            node_to_node_mapping.append(
                (ind_child + parentmodel_layers_length, ind_child)
            )
        else:
            for _ in range(
                childmodel_layers_length,
                childmodel_layers_length + parentmodel_layers_length,
            ):
                if _ not in finded_solution:
                    finded_solution.add(_)
                    node_to_node_mapping.append(
                        (ind_child + parentmodel_layers_length, _)
                    )
                    break
    return node_to_node_mapping


from tensorflow.python.keras.engine.base_layer import TensorFlowOpLayer


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


from tensorflow.python.keras.engine.base_layer import TensorFlowOpLayer


def ignore_TFop(model):
    layers = []
    for layer in model.layers:
        if type(layer) != TensorFlowOpLayer:
            layers.append(layer)
    return layers


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
