import time
import requests
import docker


class Environment:
    CONTAINER_URL = "http://blockchain2:5000/"
    HOST_URL = "ssh://luosf@blockchain2"


class Engine:
    def __init__(self):
        self.client = docker.DockerClient(base_url=Environment.HOST_URL)
        self.container = None

    def deploy_functions(self):
        self.client.images.build(
            rm=True, path="containers/compute", tag="tensorflow-with-functions"
        )

    def deploy_models(self):
        self.client.images.build(
            rm=True, path="containers/data", tag="tensorflow-generate-model-data"
        )
        self.client.volumes.create(name="data")
        self.client.containers.run(
            "tensorflow-generate-model-data",
            command="python gen_model_data.py",
            volumes=["data:/data"],
            auto_remove=True,
        )

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
                response = requests.get(Environment.CONTAINER_URL + model_name)
                status = 0
            except:
                status = 1

        return response.text
