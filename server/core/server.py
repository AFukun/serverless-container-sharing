import time

import core.api as api


class Server:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.model = None

    def _setup_model(self, model_name):
        # temp logic
        if self.model != None:
            if self.model._name == model_name:
                api.load_weights(self.data_dir, self.model)
                return f"load {model_name} weights"
            if (
                self.model._name == "resnet50"
                or model_name == "resnet50"
                or self.model._name == "mobilenet"
                or model_name == "model_name"
            ):
                self.model = api.load_model(self.data_dir, model_name)
                return f"load {model_name}"
            self.model, switch_log = api.switch_model(
                self.data_dir, self.model, model_name
            )
            return f"switch to {model_name}[{switch_log}]"
        else:
            self.model = api.load_model(self.data_dir, model_name)
            return f"load {model_name}"

    def inference(self, model_name, input_file):
        status = self._setup_model(model_name)
        start = time.time()
        result = api.inference(self.data_dir, self.model, input_file)
        end = time.time()

        return status, f"{result} in {end - start}s"
