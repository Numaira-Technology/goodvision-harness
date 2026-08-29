FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PADDLE_PDX_CACHE_HOME=/models \
    PADDLEX_HOME=/models \
    MPLCONFIGDIR=/tmp/matplotlib \
    RICH_DEVICE=cpu

RUN apt-get update \
    && apt-get install --no-install-recommends -y libgomp1 libglib2.0-0 libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /worker
COPY requirements.txt ./
# Paddle 3.3.x regresses CPU inference through oneDNN for affected detectors. Keep the
# official 3.2.2 workaround until a released wheel contains the upstream fix.
RUN python -m pip install --no-cache-dir \
      --index-url https://www.paddlepaddle.org.cn/packages/stable/cpu/ \
      paddlepaddle==3.2.2 \
    && python -m pip install --no-cache-dir -r requirements.txt

RUN python -c "import cv2, paddlex; assert cv2.__version__.startswith('4.10.')"

COPY app.py ./app.py

RUN groupadd --system --gid 10002 richlabel \
    && useradd --system --uid 10002 --gid richlabel --home-dir /nonexistent richlabel \
    && mkdir -p /models \
    && chown -R richlabel:richlabel /worker /models

USER 10002:10002
EXPOSE 8091
HEALTHCHECK --interval=15s --timeout=3s --start-period=120s --retries=4 \
  CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8091/health/ready', timeout=2).read()"]
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8091"]
