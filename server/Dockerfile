FROM tensorflow/tensorflow:latest

COPY core/ core/
COPY app.py .
COPY requirements.txt .
RUN pip install -r requirements.txt

CMD ["python", "app.py","-D","/data/"]
