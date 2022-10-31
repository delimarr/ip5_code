FROM ubuntu:22.10

WORKDIR /app

COPY ./requirements.txt /app
COPY ./.ssh /root/.ssh

RUN apt-get update && apt-get install -y python3 pip git vim && \
    pip install --upgrade -r requirements.txt && \
    chmod  0700 /root/.ssh/*