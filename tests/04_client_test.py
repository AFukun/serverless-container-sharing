import requests
import time
import sys


sys.path.insert(1, "client")
from core import Client

client = Client("ssh://luosf@blockchain2")


start = time.time()
print(client.inference("resnet50"))
end = time.time()
print("invoke time: " + str(end - start))

start = time.time()
print(client.inference("vgg16"))
end = time.time()
print("invoke time: " + str(end - start))

start = time.time()
print(client.inference("vgg19"))
end = time.time()
print("invoke time: " + str(end - start))

client.print_logs()

client.reset()
