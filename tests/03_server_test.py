import time
import sys


sys.path.insert(1, "server")
from core import Server

server = Server("tmp/")


start = time.time()
status, result = server.inference("resnet50")
end = time.time()
print(f"status: {status}", result, f"inference time:{end - start}s", sep="\n")

start = time.time()
status, result = server.inference("vgg16")
end = time.time()
print(f"status: {status}", result, f"inference time:{end - start}s", sep="\n")

start = time.time()
status, result = server.inference("vgg19")
end = time.time()
print(f"status: {status}", result, f"inference time:{end - start}s", sep="\n")
