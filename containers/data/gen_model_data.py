from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.resnet50 import ResNet50

model = ResNet50(weights="imagenet")
model.save("/data/model_a.h5")
model.save_weights("/data/model_a_weights.h5")

model = ResNet50(weights=None)
model.save("/data/model_b.h5")
model.save_weights("/data/model_b_weights.h5")
