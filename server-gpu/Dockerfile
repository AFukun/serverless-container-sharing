FROM tensorflow/tensorflow:2.0.0-gpu-py3

RUN pip install flask

COPY core/           core/
# COPY patch/          patch/
COPY app.py .

# RUN cp "patch/base_layer.py"    "/usr/local/lib/python3.6/dist-packages/tensorflow_core/python/keras/engine"
# RUN cp "patch/network.py"       "/usr/local/lib/python3.6/dist-packages/tensorflow_core/python/keras/engine"
# RUN cp "patch/convolutional.py" "/usr/local/lib/python3.6/dist-packages/tensorflow_core/python/keras/layers"
# RUN cp "patch/core.py"          "/usr/local/lib/python3.6/dist-packages/tensorflow_core/python/keras/layers"
# RUN cp "patch/merge.py"         "/usr/local/lib/python3.6/dist-packages/tensorflow_core/python/keras/layers"
# RUN cp "patch/hdf5_format.py"   "/usr/local/lib/python3.6/dist-packages/tensorflow_core/python/keras/saving"
# RUN cp "patch/nn_ops.py"        "/usr/local/lib/python3.6/dist-packages/tensorflow_core/python/ops"
