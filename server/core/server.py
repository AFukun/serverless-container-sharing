import time

import core.api as api


class Server:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.model = None
        self.input = None

    def _set_input(self):
        self.input = api.generate_random_input(self.model)

    def manual_load_model(self, model_name):
        self.model = None
        self.model = api.load_model(self.data_dir, model_name)

    def manual_switch_model(self, model_name):
        if self.model._name == model_name:
            api.load_weights(self.data_dir, self.model)
        else:
            api.switch_model(self.data_dir, self.model, model_name)

    def _setup_model(self, model_name):
        # temp logic
        if self.model != None:
            if self.model._name == model_name:
                api.load_weights(self.data_dir, self.model)
                return f"load {model_name} weights"
            else:
                api.switch_model(self.data_dir, self.model, model_name)
                self._set_input()
                return f"switch to {model_name}"
        else:
            self.model = api.load_model(self.data_dir, model_name)
            self._set_input()
            return f"load {model_name}"

    def inference(self, model_name):
        status = self._setup_model(model_name)
        start = time.time()
        result = api.inference(self.model, self.input)
        end = time.time()

        return status, f"{result} in {end - start}s"
