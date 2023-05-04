import requests
import time
import sys


sys.path.insert(1, "client")
from core import Client

client = Client("ssh://luosf@blockchain2")


# start = time.time()
# print(client.inference("vgg19", "elephant.jpg"))
# end = time.time()
# print("invoke time: " + str(end - start))

# start = time.time()
# print(client.inference("vgg19", "elephant.jpg"))
# end = time.time()
# print("invoke time: " + str(end - start))

start = time.time()
print(client.inference("resnet50", "elephant.jpg"))
end = time.time()
print("invoke time: " + str(end - start))

start = time.time()
print(client.inference("vgg16", "elephant.jpg"))
end = time.time()
print("invoke time: " + str(end - start))

start = time.time()
print(client.inference("vgg19", "elephant.jpg"))
end = time.time()
print("invoke time: " + str(end - start))

# client.reset()
