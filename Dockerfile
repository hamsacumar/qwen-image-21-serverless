FROM pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/workspace/hf_cache

RUN apt-get update && apt-get install -y git git-lfs curl && rm -rf /var/lib/apt/lists/*

WORKDIR /opt

RUN git clone https://github.com/comfyanonymous/ComfyUI.git
RUN git -C /opt/ComfyUI checkout e8665ca8

WORKDIR /opt/ComfyUI

RUN pip install --no-cache-dir -r requirements.txt



RUN pip install --no-cache-dir runpod huggingface_hub requests

WORKDIR /app

COPY handler.py .
COPY workflow_api.json .


CMD ["python", "-u", "handler.py"]
