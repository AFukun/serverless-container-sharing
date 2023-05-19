def node_group(model):
    group_type_index = {}
    for _, layer in enumerate(model.layers):
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
    parent_node_group = node_group(parentmodel)
    child_node_group = node_group(childmodel)
    parent_model_layer_size = len(parentmodel.layers)
    child_model_layer_size = len(childmodel.layers)

    node_to_node_mapping = find_solution(
        parent_node_group,
        child_node_group,
        parent_model_layer_size,
        child_model_layer_size,
    )
    return node_to_node_mapping
