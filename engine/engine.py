import time
import docker


class Engine:
    def __init__(self):
        self.container = ""
        self.client = docker.DockerClient(base_url="tcp://172.18.166.229:2375")

    def handle_mock_request(self):
        time.sleep(1)
        policy = "warm"
        if self.container == "":
            self.container = "mock"
            time.sleep(2)
            policy = "cold"
        return policy

    def handle_request(self, model_name):
        try:
            self.container = self.client.containers.run(
                "tensorflow-with-models", detach=True, tty=True
            )
            if model_name == "A":
                output = self.container.exec_run(
                    'python -c "from functions import *\nmodel = load_model_a()\nmodel = switch_to_model_b(model)"',
                ).output.decode("utf-8")
            if model_name == "B":
                output = self.container.exec_run(
                    'python -c "from functions import *\nmodel = load_model_b()\nmodel = switch_to_model_a(model)"',
                ).output.decode("utf-8")
        finally:
            self.container.stop()
            self.container.remove()
        return output
