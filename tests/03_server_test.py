import time
import sys


sys.path.insert(1, "server")
from core import Server

server = Server("tmp/")

start = time.time()
status, result = server.inference("vgg16", "elephant.jpg")
end = time.time()
print(f"status: {status}", result, f"inference time:{end - start}s", sep="\n")

start = time.time()
status, result = server.inference("vgg19", "elephant.jpg")
end = time.time()
print(f"status: {status}", result, f"inference time:{end - start}s", sep="\n")

# start = time.time()
# status, result = server.inference("vgg16", "elephant.jpg")
# end = time.time()
# print(f"status: {status}", result, f"inference time:{end - start}s", sep="\n")

start = time.time()
status, result = server.inference("resnet50", "elephant.jpg")
end = time.time()
print(f"status: {status}", result, f"inference time:{end - start}s", sep="\n")

start = time.time()
status, result = server.inference("mobilenet", "elephant.jpg")
end = time.time()
print(f"status: {status}", result, f"inference time:{end - start}s", sep="\n")
