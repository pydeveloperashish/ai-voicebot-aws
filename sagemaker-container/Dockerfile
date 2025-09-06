# Use NVIDIA TensorRT base image
FROM nvcr.io/nvidia/tensorrt:22.04-py3

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y python3 python3-pip

WORKDIR /opt/program
COPY inference.py server.py requirements.txt /opt/program/

RUN pip3 install --upgrade pip
RUN pip3 install --no-cache-dir -r requirements.txt

EXPOSE 8080
ENTRYPOINT ["python3", "/opt/program/server.py"]
