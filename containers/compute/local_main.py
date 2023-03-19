import numpy as np
import time
import tensorflow as tf
import json
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input, decode_predictions
import sys

sys.path.insert(1, "../data/")

import functions
from build_cost_matrix import build_solution, build_child_info


img_path = "elephant.jpg"
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

data_dir = "../model_data/"

start = time.time()
vgg16 = functions.load_vgg16(data_dir)
end = time.time()
preds = vgg16.predict(x)
print("result: ", decode_predictions(preds, top=3)[0])
print("load time: ", end - start)

vgg19 = functions.load_vgg19(data_dir)
with open(data_dir + "vgg16_info.json") as file:
    vgg16_info = json.load(file)
print(vgg16_info)
with open(data_dir + "vgg19_to_vgg16_solution.json") as file:
    solution = json.load(file)

vgg16_info = build_child_info(vgg16)
print(vgg16_info)
# solution = build_solution(vgg19, vgg16)


start = time.time()
vgg16 = functions.transform(
    vgg19, vgg16_info, solution["munkres"], solution["n"], solution["m"]
)
vgg16.load_weights(data_dir + "vgg16_imagenet_weights.h5")
vgg16.compile(loss="categorical_crossentropy")
end = time.time()
preds = vgg16.predict(x)
print("result: ", decode_predictions(preds, top=3)[0])
print("switch time: ", end - start)
