import docker
from .environment import *


def deploy_functions():
    client = docker.DockerClient(base_url=HOST_URL)
    client.images.build(
        rm=True, path="containers/compute", tag="tensorflow-with-functions"
    )


def deploy_models():
    client = docker.DockerClient(base_url=HOST_URL)
    client.images.build(
        rm=True, path="containers/data", tag="tensorflow-generate-model-data"
    )
    client.volumes.create(name="data")
    client.containers.run(
        "tensorflow-generate-model-data",
        command="python gen_model_data.py",
        volumes=["data:/data"],
        auto_remove=True,
    )
