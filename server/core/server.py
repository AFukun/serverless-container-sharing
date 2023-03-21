import core.api as api


class Server:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.model = None

    def _setup_model(self, model_name):
        # temp logic
        if (
            self.model == None
            or self.model._name == "resnet50"
            or model_name == "resnet50"
        ):
            self.model = api.load_model(self.data_dir, model_name)
            return f"load {model_name}"
        elif self.model._name == model_name:
            api.load_weights(self.data_dir, self.model)
            return f"load {model_name} weights"
        else:
            self.model = api.switch_model(self.data_dir, self.model, model_name)
            return f"switch to {model_name}"

    def inference(self, model_name, input_file):
        status = self._setup_model(model_name)
        result = api.inference(self.data_dir, self.model, input_file)
        return status, result
