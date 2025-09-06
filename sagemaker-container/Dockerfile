# Use CPU base for quick testing. Replace with GPU base if you need CUDA/TRT.
FROM python:3.11-slim

# Install small system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/program

# Copy app code and requirements
COPY inference.py server.py requirements.txt /opt/program/

# Install python deps
RUN pip install --no-cache-dir -r requirements.txt

# Expose the default SageMaker port
EXPOSE 8080

# SageMaker expects the server process to run as PID 1 and keep running.
ENV MODEL_DIR=/opt/ml/model
CMD ["python", "/opt/program/server.py"]
