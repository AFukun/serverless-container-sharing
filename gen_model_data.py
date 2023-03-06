from tensorflow.keras import Sequential
import tensorflow.keras.models
from tensorflow.keras.layers import (
    Conv2D,
    MaxPool2D,
    Dropout,
    BatchNormalization,
    Activation,
    Flatten,
    Dense,
)
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.resnet50 import ResNet50


# model = Sequential(
#     [
#         Flatten(input_shape=(28, 28)),
#         Dense(1, activation="relu"),
#         Dense(10, activation="softmax"),
#     ]
# )


model = ResNet50(weights="imagenet")
model.save("model_data/model_a.h5")
model.save_weights("model_data/model_a_weights.h5")

model = ResNet50(weights=None)
model.save("model_data/model_b.h5")
model.save_weights("model_data/model_b_weights.h5")
