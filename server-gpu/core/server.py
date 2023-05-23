import importlib.util
import sys


def lazy_import(name):
    spec = importlib.util.find_spec(name)
    loader = importlib.util.LazyLoader(spec.loader)
    spec.loader = loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    loader.exec_module(module)
    return module


api = lazy_import("core.api")
utils = lazy_import("core.utils")


class Server:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.model = None
        self.input = None

    def _set_input(self):
        self.input = utils.generate_random_input(self.model)

    def manual_load_model(self, model_name):
        self.model = None
        self.model = api.load_model(self.data_dir, model_name)

    def manual_load_weights(self):
        api.load_weights(self.data_dir, self.model)

    def manual_generate_solution(self, child_model_name):
        api.generate_solution(self.data_dir, self.model._name, child_model_name)

    def manual_switch_model(self, model_name):
        if self.model._name == model_name:
            api.load_weights(self.data_dir, self.model)
        else:
            api.switch_model(self.data_dir, self.model, model_name)

    def manual_switch_nasbench_model(self, model_name):
        api.switch_nasbench_model(self.data_dir, self.model, model_name)

    def manual_inference(self):
        if self.input is None:
            self._set_input()

        return self.model(self.input)

    def manual_get_model(self):
        self.model = utils.get_model(self.model._name)
        self.model(self.input)

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
