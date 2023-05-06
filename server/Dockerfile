FROM tensorflow/tensorflow:2.0.0-py3

COPY core/ core/
COPY patch/ patch/
COPY app.py .

RUN pip install flask
RUN cp "patch/base_layer.py" "/usr/local/lib/python3.6/dist-packages/tensorflow_core/python/keras/engine"
RUN cp "patch/convolutional.py" "/usr/local/lib/python3.6/dist-packages/tensorflow_core/python/keras/layers"
RUN cp "patch/network.py" "/usr/local/lib/python3.6/dist-packages/tensorflow_core/python/keras/engine"
RUN cp "patch/nn_ops.py" "/usr/local/lib/python3.6/dist-packages/tensorflow_core/python/ops"
