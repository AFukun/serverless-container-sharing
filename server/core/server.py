class Server:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.model = None
        self.input = None

    def _set_input(self):
        from core.utils import generate_random_input

        self.input = generate_random_input(self.model)

    def manual_load_model(self, model_name):
        from core.api import (
            load_model,
            load_weights,
            switch_model,
        )

        self.model = None
        self.model = load_model(self.data_dir, model_name)

    def manual_switch_model(self, model_name):
        if self.model._name == model_name:
            load_weights(self.data_dir, self.model)
        else:
            switch_model(self.data_dir, self.model, model_name)

    def manual_inference(self):
        if self.input is None:
            self._set_input()

        return self.model(self.input)

    def manual_get_model(self):
        from core.utils import get_model

        self.model = get_model(self.model._name)
        self.model(self.input)

    def _setup_model(self, model_name):
        # temp logic
        if self.model != None:
            if self.model._name == model_name:
                load_weights(self.data_dir, self.model)
                return f"load {model_name} weights"
            else:
                switch_model(self.data_dir, self.model, model_name)
                self._set_input()
                return f"switch to {model_name}"
        else:
            self.model = load_model(self.data_dir, model_name)
            self._set_input()
            return f"load {model_name}"

    def inference(self, model_name):
        from core.api import inference
        import time

        status = self._setup_model(model_name)
        start = time.time()
        result = inference(self.model, self.input)
        end = time.time()

        return status, f"{result} in {end - start}s"
