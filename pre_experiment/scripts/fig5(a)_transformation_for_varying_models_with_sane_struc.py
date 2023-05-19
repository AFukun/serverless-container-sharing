import tensorflow.keras.applications
import tensorflow as tf
import time

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


start = time.time()
model = tf.keras.models.load_model("/data/resnet18.h5")
print("resnet18 load model:", time.time() - start)
model.save_weights("my_model_weights.h5")
start = time.time()
model.load_weights("my_model_weights.h5")
print("resnet18 load weight:", time.time() - start)

start = time.time()
model = tf.keras.models.load_model("/data/resnet50.h5")
print("resnet50 load model:", time.time() - start)
model.save_weights("my_model_weights.h5")
start = time.time()
model.load_weights("my_model_weights.h5")
print("resnet50 load weight:", time.time() - start)

start = time.time()
model = tf.keras.models.load_model("/data/resnet101.h5")
print("resnet101 load model:", time.time() - start)
model.save_weights("my_model_weights.h5")
start = time.time()
model.load_weights("my_model_weights.h5")
print("resnet101 load weight:", time.time() - start)


start = time.time()
model = tf.keras.models.load_model("/data/vgg11.h5")
print("VGG11 load model:", time.time() - start)
model.save_weights("my_model_weights.h5")
start = time.time()
model.load_weights("my_model_weights.h5")
print("VGG11 load weight:", time.time() - start)

start = time.time()
model = tf.keras.models.load_model("/data/vgg16.h5")
print("VGG16 load model:", time.time() - start)
model.save_weights("my_model_weights.h5")
start = time.time()
model.load_weights("my_model_weights.h5")
print("VGG16 load weight:", time.time() - start)


start = time.time()
model = tf.keras.models.load_model("/data/vgg19.h5")
print("VGG19 load model:", time.time() - start)
model.save_weights("my_model_weights.h5")
start = time.time()
model.load_weights("my_model_weights.h5")
print("VGG19 load weight:", time.time() - start)
