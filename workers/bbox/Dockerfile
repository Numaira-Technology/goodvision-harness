ARG GPU_RUNTIME_IMAGE=goodvision-gpu-runtime:torch2.10-cu128
FROM ${GPU_RUNTIME_IMAGE}
WORKDIR /app
COPY workers/bbox/requirements.txt /tmp/module-requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install -r /tmp/module-requirements.txt && python -m pip check
COPY workers/__init__.py /app/workers/__init__.py
COPY workers/detector_rich_label /app/workers/detector_rich_label
ENV DETECTOR_MODEL=yolo26-s DETECTOR_DEVICE=cuda DETECTOR_WEIGHTS_DIR=/models \
    YOLO_CONFIG_DIR=/data/ultralytics
LABEL io.goodvision.module="bbox" io.goodvision.device="cuda"
EXPOSE 8096
CMD ["python", "-m", "uvicorn", "workers.detector_rich_label.app:app", "--host", "0.0.0.0", "--port", "8096", "--workers", "1"]
