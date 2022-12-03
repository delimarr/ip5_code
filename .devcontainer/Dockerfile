FROM ubuntu:22.10

WORKDIR /app

COPY ./requirements.txt /app
COPY ./.ssh /root/.ssh

RUN chmod  0700 /root/.ssh/*

RUN apt-get update && apt-get upgrade -y && apt-get install -y git vim python3 pip

RUN pip install --upgrade -r requirements.txt

RUN python3 -m deepxde.backend.set_default_backend tensorflow
  