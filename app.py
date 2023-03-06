from flask import Flask
from engine import Engine
import time

app = Flask(__name__)
engine = Engine()


@app.route("/")
def index():
    return "Hello World!\n"


@app.route("/test")
def test():
    start = time.time()
    status = engine.handle_mock_request()
    end = time.time()
    return "status: %s start\ninvoke time: %ss\n" % (status, str(end - start))


@app.route("/a-to-b")
def a_to_b():
    start = time.time()
    output = engine.handle_request("A")
    end = time.time()
    return output + "invoke time: %ss\n" % str(end - start)


@app.route("/b-to-a")
def b_to_a():
    start = time.time()
    output = engine.handle_request("B")
    end = time.time()
    return output + "invoke time: %ss\n" % str(end - start)


if __name__ == "__main__":
    app.run(debug=True, port=23333)
