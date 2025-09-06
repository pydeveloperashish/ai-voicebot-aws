# Dockerfile (GPU-friendly, CUDA 11.4)
FROM nvidia/cuda:11.4.3-cudnn8-runtime-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential wget ca-certificates git python3 python3-pip python3-venv \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/program

# copy app
COPY inference.py server.py requirements.txt /opt/program/

# install Python deps -- prefer a wheel for onnxruntime-gpu that matches cuda11.4
RUN pip3 install --upgrade pip
# If a matching onnxruntime-gpu wheel is available on PyPI for cuda11.4, requirements.txt is fine.
# Otherwise you'll need to install a specific wheel or build ONNX Runtime with CUDA/TensorRT support.
RUN pip3 install --no-cache-dir -r requirements.txt

ENV MODEL_DIR=/opt/ml/model
EXPOSE 8080
# CMD ["python3", "/opt/program/server.py"]
ENTRYPOINT ["python3", "/opt/program/server.py"]
