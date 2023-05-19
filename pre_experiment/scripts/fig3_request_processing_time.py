import tensorflow as tf

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

model = tf.keras.models.load_model("/data/vgg16.h5")
model = tf.keras.models.load_model("/data/vgg19.h5")
model = tf.keras.models.load_model("/data/resnet50.h5")
model = tf.keras.models.load_model("/data/resnet101.h5")
