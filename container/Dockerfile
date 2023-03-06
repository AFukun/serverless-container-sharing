FROM tensorflow/tensorflow:latest

RUN groupadd app && useradd -r -g app app

WORKDIR /home/app/
COPY functions.py .
COPY model_data/ model_data/
RUN chown -R app /home/app
