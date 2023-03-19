import docker
from .environment import *


def _print_logs(logs):
    for chunk in logs:
        if "stream" in chunk:
            for line in chunk["stream"].splitlines():
                print(line)


def deploy_functions():
    client = docker.DockerClient(base_url=HOST_URL)
    _, logs = client.images.build(
        rm=True, path="containers/compute", tag="tensorflow-with-functions"
    )
    _print_logs(logs)


def deploy_models():
    client = docker.DockerClient(base_url=HOST_URL)
    _, logs = client.images.build(
        rm=True, path="containers/data", tag="tensorflow-generate-model-data"
    )
    _print_logs(logs)
    client.volumes.create(name="data")
    print(
        client.containers.run(
            "tensorflow-generate-model-data",
            command="python gen_model_data.py",
            volumes=["data:/data"],
            auto_remove=True,
        ).decode("utf-8")
    )
