FROM tensorflow/tensorflow:latest

COPY utils/ utils/
COPY main.py .

COPY requirements.txt .
RUN pip install -r requirements.txt
