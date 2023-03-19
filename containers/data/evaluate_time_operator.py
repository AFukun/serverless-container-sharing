from options import args_parser
import tensorflow as tf
import time 
from tensorflow.keras.applications.vgg16 import VGG16
args = args_parser() 
old_model = VGG16(weights='imagenet')
layers = old_model.layers[:]
print("layers",layers)
print("lenth:",len(old_model.layers))
print(old_model.summary())

# evaluate time of swap operator
tempList = [ _ for _ in range(2)]
startTime = time.time()
layers = old_model.layers[:]
tempList[0] = old_model.layers[1]
tempList[1] = old_model.layers[0]
endTime = time.time()
print(old_model.layers[1],tempList)
during = endTime - startTime
print("copy",during)
startTime = time.time()
temp = old_model.get_layer('block4_pool')

print(old_model.layers[1])
print(old_model.layers[0])
old_model.layers[1] = temp

endTime = time.time()
during = endTime - startTime
# print("time of swaping tensor {:.20f}".format(during))
# print(old_model.layers[0])

#evaluate time of delete operator
# startTime = time.time()
# # print(old_model.get_layer(index=1))
# old_model = tf.keras.Sequential(name='my_sequential')
# old_model.add(tf.keras.layers.Dense(5,activation='softmax',name='dense_output'))
# print(old_model)
# endTime = time.time()
# during = endTime - startTime
# print("time of deleting tensor {:.20f}".format(during))
# print(old_model.get_layer(index=1))
# print(old_model.summary())
# import tensorflow.keras as keras
# model = keras.Sequential(
#     [
#         keras.layers.Dense(2, activation="softmax", input_shape=(10,)),
#         keras.layers.Dense(3, activation="relu"),
#         keras.layers.Dense(4),
#         keras.layers.Dropout(0.5)
#     ]
# ) 
# print('model weights',model.layers[0].activation.__name__)
# model.layers[0].activation = keras.layers.ReLU()
# print('model weights',model.layers[0].activation)
# # 此时输入model.summary()，model.weights都会报错
# # 必须给模型一个输入参数
# x = tf.ones((1, 3, 10))  #参数1理解为batch
# y = model(x)
# a = model.layers[3]
# print("pointer",a,model.layers[3])


# import tensorflow as tf

# class MyModel(tf.keras.Model):
#     def __init__(self):
#         super(MyModel, self).__init__()
#         self.dense1 = tf.keras.layers.Dense(64, activation='relu')
#         self.dense2 = tf.keras.layers.Dense(10)

#     def call(self, inputs):
#         x = self.dense1(inputs)
#         return self.dense2(x)

# # 创建模型并进行训练
# model = MyModel()

# # 替换Dense层
# new_dense_layer = tf.keras.layers.Dense(128, activation='relu')
# model.dense1 = new_dense_layer
# print(model.layers[0].get_weights())


