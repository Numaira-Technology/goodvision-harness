# Optional CPU-only controller; the default final stage remains Harness.
FROM python:3.11-slim AS sam-remote

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update \
    && apt-get install --no-install-recommends -y ca-certificates openssh-client \
    && rm -rf /var/lib/apt/lists/* \
    && python -m pip install --no-cache-dir vastai==1.5.6 'python-dotenv>=1,<2'

WORKDIR /app
COPY tools/start_sam.py ./tools/start_sam.py
COPY workers/sam_scene/app.py workers/sam_scene/sam31.py \
    workers/sam_scene/continuous.py workers/sam_scene/streaming_state.py \
    workers/sam_scene/h264_transport.py workers/sam_scene/requirements.txt \
    ./workers/sam_scene/

STOPSIGNAL SIGTERM
CMD ["python", "-u", "tools/start_sam.py"]

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HARNESS_CONFIG=/app/config.yaml \
    HARNESS_DEPLOYMENT_SETTINGS_PATH=/data/deployment-settings.json \
    HARNESS_STREAM_URL=webrtc+http://192.168.20.199:1985/rtc/v1/whep/?app=live&stream=livestream \
    HARNESS_DRY_RUN=true \
    HARNESS_AUTOSTART=true \
    HARNESS_HOST=0.0.0.0 \
    HARNESS_PORT=8080 \
    HARNESS_INFERENCE_PROFILE=customer-standard \
    FACE_PROVIDER=CPUExecutionProvider \
    RICH_LABEL_ENABLED=false

RUN apt-get update \
    && apt-get install --no-install-recommends -y \
        ca-certificates \
        ffmpeg \
        libgl1 \
        libgomp1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
RUN python -m pip install --no-cache-dir '.[face,webrtc,vlm,video]'

# InsightFace also requests GUI OpenCV; both distributions overwrite cv2.
# Ship one server variant so dependency updates cannot silently select OpenCV 5.
# Preserve upstream package metadata, including its GUI-name-only dependency.
RUN python -m pip uninstall -y opencv-python opencv-python-headless \
    && python -m pip install --no-cache-dir --no-deps opencv-python-headless==4.14.0.94 \
    && python -c "import cv2; assert cv2.__version__.startswith('4.14.')"

COPY config.yaml ./config.yaml

RUN groupadd --system --gid 10001 harness \
    && useradd --system --uid 10001 --gid harness --home-dir /nonexistent harness \
    && mkdir -p /models /data \
    && chown -R harness:harness /app /models /data

USER 10001:10001

EXPOSE 8080
STOPSIGNAL SIGTERM

HEALTHCHECK --interval=15s --timeout=3s --start-period=120s --retries=4 \
  CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/health/live', timeout=2).read()"]

CMD ["goodvision-harness"]
