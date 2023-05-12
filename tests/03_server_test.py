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


start = time.time()
server.manual_load_model("vgg16")
end = time.time()
print("load {:.2f}s".format(end - start))
start = time.time()
server.manual_switch_model("vgg19")
end = time.time()
print("switch {:.2f}s".format(end - start))
