import tensorflow as tf
from group_by_tensor_type import _group_by_tensor_type
from options import args_parser
from GED_Munkres import munkres

# two layers


def _cost_matrix(parentModel, childModel):
    args = args_parser()
    inMatrixSet, exMatrixSet = _group_by_tensor_type()
    costMatrix = []

    n = len(parentModel.layers)
    m = len(childModel.layers)
    # print("n={},m={}".format(n, m))
    # step 1: compute cost of scaling
    for i in range(n):
        tensorTypeOfParent = type(parentModel.layers[i])
        for j in range(m):
            tensorTypeOfChild = type(childModel.layers[j])
            if (
                tensorTypeOfParent == tensorTypeOfChild
                and tensorTypeOfParent in inMatrixSet
            ):
                cost = (i, j, args.timeScale)
            elif (
                tensorTypeOfParent == tensorTypeOfChild
                and tensorTypeOfParent in exMatrixSet
            ):
                cost = (i, j, 0)
            else:
                cost = (i, j, args.INFINITY)
            costMatrix.append(cost)
    # step 2: compute cost of deleting
    for i in range(n):
        for j in range(n):
            if i == j:
                cost = (i, j + m, args.timeDeleteTensor)
            else:
                cost = (i, j + m, args.INFINITY)
            costMatrix.append(cost)
    # step 3: compute cost of adding
    for i in range(m):
        for j in range(m):
            if i == j:
                cost = (i + n, j, args.timeAddTensor)
            else:
                cost = (i + n, j, args.INFINITY)
            costMatrix.append(cost)
    # step 4
    for i in range(m):
        for j in range(n):
            cost = (i + n, j + m, 0)
            costMatrix.append(cost)
    # print(costMatrix)
    # print(len(munkres(costMatrix)))
    return n, m, costMatrix


# parentModel = models.alexnet(pretrained=True)
# # print(parentModel)
# childModel = models.vgg16(pretrained=True)
# _cost_matrix(parentModel, childModel)


def build_child_info(childModel):
    childInfo = []
    for layer in childModel.layers:
        _type = type(layer)
        tempChildInfo = []

        if _type == tf.keras.layers.Conv2D:
            tempChildInfo.append("Conv2D")
            tempChildInfo.append(layer.filters)
            tempChildInfo.append(layer.kernel_size)
            tempChildInfo.append(layer.strides)
            tempChildInfo.append(layer.padding)
            tempChildInfo.append(layer.name)
            tempChildInfo.append(layer.activation.__name__)

            tempChildInfo.append(layer.weights[0].shape.as_list())

        elif _type == tf.keras.layers.Dense:
            tempChildInfo.append("Dense")
            tempChildInfo.append(layer.units)

            # tempChildInfo.append(layer.activation.__name__)
            tempChildInfo.append(layer.name)
            tempChildInfo.append(layer.activation.__name__)
            tempChildInfo.append(layer.weights[0].shape.as_list())

        elif _type == tf.keras.layers.MaxPool2D:
            tempChildInfo.append("MaxPool2D")
            tempChildInfo.append(layer.pool_size)
            tempChildInfo.append(layer.strides)
            tempChildInfo.append(layer.name)
        elif _type == tf.keras.layers.Flatten:
            tempChildInfo.append("Flatten")
        elif _type == tf.keras.layers.InputLayer:
            tempChildInfo.append("InputLayer")

        childInfo.append(tempChildInfo)

    return childInfo


def build_solution(parent_model, child_model):
    n, m, cost_matrix = _cost_matrix(parent_model, child_model)
    return {"n": n, "m": m, "munkres": munkres(cost_matrix)}
