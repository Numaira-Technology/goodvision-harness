FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    SAM_SCENE_CHECKPOINT_PATH=/models/sam3.pt \
    HF_HUB_OFFLINE=1 \
    HF_HOME=/tmp/huggingface \
    TORCH_HOME=/tmp/torch

RUN apt-get update \
    && apt-get install --no-install-recommends -y git build-essential libgomp1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /worker
COPY requirements.txt ./
# CUDA 12.8 wheels include the runtime; the host supplies a compatible NVIDIA
# driver and container toolkit. The pinned official README recommends torch 2.10.
RUN python -m pip install --no-cache-dir \
      torch==2.10.0 torchvision==0.25.0 \
      --index-url https://download.pytorch.org/whl/cu128 \
    && python -m pip install --no-cache-dir -r requirements.txt \
    && python -m pip install --no-cache-dir \
      git+https://github.com/facebookresearch/sam3.git@660a5e9e1b8b4c02c0ad97229b88a09a6e4ff5b7 \
    && python -m pip check

# Import verification needs neither a GPU nor gated model weights.
RUN python -c "from sam3.model_builder import build_sam3_image_model; from sam3.model.sam3_image_processor import Sam3Processor"

COPY app.py ./app.py
COPY smoke.py ./smoke.py
RUN groupadd --system --gid 10003 samscene \
    && useradd --system --uid 10003 --gid samscene --home-dir /nonexistent samscene \
    && mkdir -p /models \
    && chown -R samscene:samscene /worker /models

USER 10003:10003
EXPOSE 8092
HEALTHCHECK --interval=15s --timeout=3s --start-period=180s --retries=4 \
  CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8092/health/tracking', timeout=2).read()"]
# A single process owns all per-camera state and one GPU model.
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8092", "--workers", "1", "--timeout-graceful-shutdown", "20"]
