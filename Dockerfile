# This the base image for running iOpt in container
FROM ubuntu:20.04

RUN apt-get update && \
    apt-get install -y python3.8 python3-pip && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt /home/iOpt/requirements.txt
RUN pip3 install pip==19.3.1 && \
    pip install --trusted-host pypi.python.org -r /home/iOpt/requirements.txt

WORKDIR /home/iOpt
COPY . /home/iOpt

ENV PYTHONPATH /home/iOpt