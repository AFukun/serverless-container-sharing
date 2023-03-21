from argparse import ArgumentParser
from flask import Flask, request

from core import Server

parser = ArgumentParser(
    prog="Tensorflow Server",
    description="Tensorflow serving for requests",
)
parser.add_argument("-D", "--data-dir")
args = parser.parse_args()


app = Flask(__name__)
server = Server(args.data_dir)


@app.route("/inference")
def inference():
    args = request.args
    status, result = server.inference(args.get("model-name"), args.get("input-file"))
    return f"({status}, {result})"


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
