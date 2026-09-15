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
# CUDA wheels include the runtime; the host supplies the NVIDIA driver and
# container toolkit. Select cu126 for a host whose driver reports CUDA 12.6.
ARG TORCH_CUDA=cu128
RUN case "${TORCH_CUDA}" in cu126|cu128) ;; \
      *) echo "TORCH_CUDA must be cu126 or cu128" >&2; exit 1 ;; esac \
    && python -m pip install --no-cache-dir \
      torch==2.10.0 torchvision==0.25.0 \
      --index-url "https://download.pytorch.org/whl/${TORCH_CUDA}" \
    && python -m pip install --no-cache-dir -r requirements.txt \
    && python -m pip install --no-cache-dir \
      git+https://github.com/facebookresearch/sam3.git@660a5e9e1b8b4c02c0ad97229b88a09a6e4ff5b7 \
    && python -m pip check

# Import verification needs neither a GPU nor gated model weights.
RUN python -c "from sam3.model_builder import build_sam3_image_model, build_sam3_multiplex_video_predictor; from sam3.model.sam3_image_processor import Sam3Processor"

COPY app.py ./app.py
COPY sam31.py ./sam31.py
COPY continuous.py ./continuous.py
COPY streaming_state.py ./streaming_state.py
COPY h264_transport.py ./h264_transport.py
COPY perception_routes.py ./perception_routes.py
COPY sam_live.py ./sam_live.py
COPY hybrid.py ./hybrid.py
COPY hybrid_tracker.py ./hybrid_tracker.py
COPY smoke.py ./smoke.py
COPY video_smoke.py ./video_smoke.py
RUN groupadd --system --gid 10003 samscene \
    && useradd --system --uid 10003 --gid samscene --home-dir /nonexistent samscene \
    && mkdir -p /models \
    && chown -R samscene:samscene /worker /models

USER 10003:10003
EXPOSE 8092
HEALTHCHECK --interval=15s --timeout=3s --start-period=180s --retries=4 \
  CMD ["python", "-c", "import os,urllib.request; path='/health/video/session' if os.getenv('SAM31_SESSION_ENABLED','false').lower()=='true' else '/health/video' if os.getenv('SAM31_ENABLED','false').lower()=='true' else '/health/tracking'; urllib.request.urlopen('http://127.0.0.1:8092'+path, timeout=2).read()"]
# A single process owns all per-camera state and one GPU model.
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8092", "--workers", "1", "--timeout-graceful-shutdown", "20"]
