FROM tensorflow/tensorflow:latest

COPY functions/ functions/
COPY app.py .
COPY requirements.txt .
COPY elephant.jpg .
RUN pip install -r requirements.txt

CMD ["python", "app.py"]
