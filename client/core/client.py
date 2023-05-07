import time
import requests
import docker
from urllib.parse import urlparse


class Client:
    def __init__(self, host_url):
        self.client = docker.DockerClient(base_url=host_url)
        self.host = urlparse(host_url).hostname
        self.container = None
        self.container_port = None

    def inference(self, model_name):
        status = "reuse container"
        if self.container == None:
            self.container = self.client.containers.run(
                "server",
                volumes=["data:/data"],
                command="python app.py -D /data/",
                ports={"5000/tcp": 5000},
                detach=True,
            )
            self.container_port = 5000
            status = f"container cold start"

        response = None
        while response == None:
            try:
                response = requests.get(
                    f"http://{self.host}:{self.container_port}/inference",
                    params={"model-name": model_name},
                )
            except:
                response = None

        return status, response.text

    def print_logs(self):
        print(self.container.logs().decode())

    def reset(self):
        try:
            self.container.stop()
            self.container.remove()
            self.container = None
            return "container removed"
        except:
            return "unexpected errror"
