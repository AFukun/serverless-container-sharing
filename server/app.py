import time


from argparse import ArgumentParser


parser = ArgumentParser(
    prog="Tensorflow Server",
    description="Tensorflow serving for requests",
)
parser.add_argument("-D", "--data-dir")
args = parser.parse_args()

from core import Server

server = Server(args.data_dir)


from flask import Flask, request

app = Flask(__name__)


@app.route("/greet")
def greet():
    print(time.time())
    return "ready to go"


@app.route("/inference")
def inference():
    args = request.args
    start = time.time()
    status, result = server.inference(args.get("model-name"))
    end = time.time()
    return f"({status}, {result}, {end - start}s)"


@app.route("/manual/load-model")
def manual_load_model():
    args = request.args
    start = time.time()
    server.manual_load_model(args.get("model-name"))
    end = time.time()
    return "{:.2f}".format(end - start)


@app.route("/manual/switch-model")
def manual_switch_model():
    args = request.args
    start = time.time()
    server.manual_switch_model(args.get("model-name"))
    end = time.time()
    return "{:.2f}".format(end - start)


@app.route("/manual/inference")
def manual_inference():
    start = time.time()
    try:
        server.manual_inference()
    except:
        server.manual_get_model()
        start = time.time()
        server.manual_inference()
    end = time.time()
    return "{:.2f}".format(end - start)


if __name__ == "__main__":
    print(time.time())
    app.run(host="0.0.0.0", port=5000)
