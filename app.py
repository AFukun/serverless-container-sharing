from flask import Flask
from engine import Engine
import time

app = Flask(__name__)
engine = Engine()


@app.route("/")
def index():
    return "Hello World!\n"


@app.route("/a")
def a():
    start = time.time()
    output = engine.handle_request("a")
    end = time.time()
    return output + "invoke time: %ss\n" % str(end - start)


@app.route("/b")
def b():
    start = time.time()
    output = engine.handle_request("b")
    end = time.time()
    return output + "invoke time: %ss\n" % str(end - start)


if __name__ == "__main__":
    app.run(debug=True, port=23333)
