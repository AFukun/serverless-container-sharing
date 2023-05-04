FROM tensorflow/tensorflow:2.0.0-py3

RUN apt-get update \
    && apt-get -y install libjpeg-dev zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

COPY core/ core/
copy patch/ patch/
COPY app.py .
COPY requirements.txt .

RUN pip install -r requirements.txt
RUN cp "patch/base_layer.py" "/usr/local/lib/python3.6/dist-packages/tensorflow_core/python/keras/engine"
RUN cp "patch/convolutional.py" "/usr/local/lib/python3.6/dist-packages/tensorflow_core/python/keras/layers"
RUN cp "patch/network.py" "/usr/local/lib/python3.6/dist-packages/tensorflow_core/python/keras/engine"
RUN cp "patch/nn_ops.py" "/usr/local/lib/python3.6/dist-packages/tensorflow_core/python/ops"
