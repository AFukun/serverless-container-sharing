from flask import Flask
import core
import time

app = Flask(__name__)
engine = core.Engine()


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
    app.run(debug=True, port=2333)
