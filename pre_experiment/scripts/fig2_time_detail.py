import time
import docker


model_name_list = [
    "vgg11",
    "vgg16",
    "vgg19",
    "resnet18",
    "resnet50",
    "resnet101",
]

REPEATS = 5

client = docker.DockerClient()

for model_name in model_name_list:
    for _ in range(0, REPEATS):
        print(f"[{model_name}]")
        request_time = time.time()
        logs = client.containers.run(
            "pre-experiment",
            command=f"python main.py -D=/data/ -M={model_name}",
            volumes=["data:/data"],
            auto_remove=True,
        )
        print(logs.decode())
        print(request_time)
