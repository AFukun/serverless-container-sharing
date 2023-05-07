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


def compute_node_to_node_mapping(parentmodel, childmodel):
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
        else:
            raise Exception("This type:{} has not been added".format(type(layer)))

        childmodel_info.append(layer_info)

    return childmodel_info
