from flask import Flask

import core

app = Flask(__name__)
engine = core.Engine()


@app.route("/vgg16")
def vgg16():
    engine.setup_model("vgg16_imagenet")
    return engine.inference("vgg16_imagenet", "elephant.jpg")


@app.route("/vgg19")
def b():
    global model
    if model == None:
        model = load_model_b()
        status = "Load Model B\n"
    else:
        model = switch_to_model_b(model)
        status = "Switch to Model B\n"
    return status


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
