FROM python:3.7-bullseye

RUN pip install protobuf==3.20.3
RUN pip install tensorflow==2.0.0
RUN pip install keras==2.3.1
RUN pip install torch
RUN pip install onnx==1.8.1
RUN pip install onnx2keras
RUN pip install pytorch2keras
RUN pip install pytorchcv
RUN pip install --upgrade h5py==2.10.0

COPY utils/ utils/
COPY patch/ patch/
COPY setup.py .
COPY imgclsmob_models.json .

RUN cp "patch/reshape_layers.py"      "/usr/local/lib/python3.7/site-packages/onnx2keras"
RUN cp "patch/converter.py"           "/usr/local/lib/python3.7/site-packages/onnx2keras"
RUN cp "patch/operation_layers.py"    "/usr/local/lib/python3.7/site-packages/onnx2keras"
RUN cp "patch/core.py"                "/usr/local/lib/python3.7/site-packages/tensorflow_core/python/keras/layers"
