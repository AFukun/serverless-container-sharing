FROM tensorflow/tensorflow:latest

COPY functions.py .
COPY serving.py .
COPY requirements.txt   .
RUN pip install -r requirements.txt

CMD ["python", "serving.py"]
