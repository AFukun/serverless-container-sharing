from flask import Flask
from engine import Engine
import time

app = Flask(__name__)
engine = Engine()


@app.route("/")
def index():
    return "Hello World!"


@app.route("/test")
def test():
    start = time.time()
    status = engine.mock_request()
    end = time.time()
    return "status: %s<br>invoke time: %ss" % (status, str(end - start))


@app.route("/model-a")
def model_a():
    return


if __name__ == "__main__":
    app.run(debug=True, port=23333)
