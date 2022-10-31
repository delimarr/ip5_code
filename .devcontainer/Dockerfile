FROM ubuntu:22.10

WORKDIR /app

COPY ./requirements.txt /app

RUN apt-get update && apt-get install -y python3 pip git vim && \
    pip install --upgrade -r requirements.txt