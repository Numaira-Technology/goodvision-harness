FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HARNESS_CONFIG=/app/config.yaml \
    HARNESS_HOST=0.0.0.0 \
    HARNESS_PORT=8080

RUN apt-get update \
    && apt-get install --no-install-recommends -y \
        ca-certificates \
        ffmpeg \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
RUN python -m pip install --no-cache-dir .

COPY config.yaml ./config.yaml

RUN groupadd --system --gid 10001 harness \
    && useradd --system --uid 10001 --gid harness --home-dir /nonexistent harness \
    && chown -R harness:harness /app

USER 10001:10001

EXPOSE 8080
STOPSIGNAL SIGTERM

HEALTHCHECK --interval=15s --timeout=3s --start-period=120s --retries=4 \
  CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/health/ready', timeout=2).read()"]

CMD ["goodvision-harness"]
