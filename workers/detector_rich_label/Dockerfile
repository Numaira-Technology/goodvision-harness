# GPU detector rich-label worker (YOLO / RF-DETR). Mirrors the SAM/Paddle workers:
# one process, /v1/labels contract, weights mounted or baked under /models.
FROM nvidia/cuda:12.6.0-runtime-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip libgl1 libglib2.0-0 ca-certificates \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN python3 -m pip install --no-cache-dir -r requirements.txt
COPY app.py tracker.py detectors.py sam3_onnx.py ./
ENV DETECTOR_MODEL=yolo26-s DETECTOR_DEVICE=cuda DETECTOR_WEIGHTS_DIR=/models
EXPOSE 8092
CMD ["python3","-m","uvicorn","app:app","--host","0.0.0.0","--port","8092","--workers","1"]
