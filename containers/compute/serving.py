from functions import *
from flask import Flask

app = Flask(__name__)
model = None


@app.route("/")
def hello():
    return "Hello World!\n"


@app.route("/a")
def a():
    global model
    if model == None:
        model = load_model_a()
        status = "Load Model A\n"
    else:
        model = switch_to_model_a(model)
        status = "Switch to Model A\n"
    return status


@app.route("/b")
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
