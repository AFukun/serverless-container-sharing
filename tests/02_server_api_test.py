import time
import tensorflow as tf
import sys

sys.path.insert(1, "server")
from core.api import *

data_dir = "tmp/"
input_file = "elephant.jpg"

start = time.time()
resnet50 = load_model(data_dir, "resnet50")
end = time.time()
print(inference(data_dir, resnet50, input_file))
print("resnet50 load time: ", end - start)

start = time.time()
mobilenet = load_model(data_dir, "mobilenet")
end = time.time()
print(inference(data_dir, mobilenet, input_file))
print("mobilenet load time: ", end - start)

start = time.time()
vgg16 = load_model(data_dir, "vgg16")
end = time.time()
print(inference(data_dir, vgg16, input_file))
print("vgg16 load time: ", end - start)

vgg19 = load_model(data_dir, "vgg19")
start = time.time()
vgg16, _ = switch_model(data_dir, vgg19, "vgg16")
end = time.time()
print(inference(data_dir, vgg16, input_file))
print("vgg19 to vgg16 switch time: ", end - start)

start = time.time()
vgg19 = load_model(data_dir, "vgg19")
end = time.time()
print(inference(data_dir, vgg19, input_file))
print("vgg19 load time: ", end - start)

vgg16 = load_model(data_dir, "vgg16")
start = time.time()
vgg19, _ = switch_model(data_dir, vgg16, "vgg19")
end = time.time()
print(inference(data_dir, vgg19, input_file))
print("vgg16 to vgg19 switch time: ", end - start)
