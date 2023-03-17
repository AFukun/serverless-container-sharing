import time
import requests
import docker
from .environment import *


class Engine:
    def __init__(self):
        self.client = docker.DockerClient(base_url=HOST_URL)
        self.container = None

    def handle_request(self, model_name):
        if self.container == None:
            self.container = self.client.containers.run(
                "tensorflow-with-functions",
                volumes=["data:/data"],
                ports={"5000/tcp": 5000},
                detach=True,
            )

        status = 1
        while status != 0:
            try:
                response = requests.get(CONTAINER_URL + model_name)
                status = 0
            except:
                status = 1

        return response.text
