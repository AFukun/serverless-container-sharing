import time
from flask import Flask, request

from core import Client

app = Flask(__name__)
client = Client("ssh://luosf@blockchain2")


@app.route("/")
def index():
    return "Hello World!\n"


@app.route("/inference")
def inference():
    args = request.args
    start = time.time()
    status, response = client.inference(args.get("model-name"), args.get("input-file"))
    end = time.time()
    return f"container status: {status}\nserver response: {response}\ninvoke time: {end - start}"


@app.route("/reset")
def reset():
    return client.reset()


if __name__ == "__main__":
    app.run(port=2333)
