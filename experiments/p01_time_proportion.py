import time
import docker


HOST_URL = "ssh://luosf@blockchain2"

model_name_list = [
    # "mobilenet",
    # "vgg11",
    "vgg16",
    "vgg19",
    "resnet50",
    # "resnet101",
    # "resnet152",
]

client = docker.DockerClient(base_url=HOST_URL)

for model_name in model_name_list:
    start = time.time()
    container = client.containers.create(
        "pre-experiment",
        command=f"python main.py -D=/data/ -M={model_name}",
        volumes=["data:/data"],
    )
    end = time.time()
    print(model_name)
    print(f"container setup time: {end - start}")

    start = time.time()
    container.start()
    container.wait()
    end = time.time()
    print(container.logs().decode("utf-8"))
    print(f"total exe time: {end - start}")
    print("\n")
    container.remove()
