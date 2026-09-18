FROM python:3.12-slim-bookworm
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /opt
COPY requirements.txt /opt/requirements.txt
RUN python -m pip install --no-cache-dir -r /opt/requirements.txt && python -m pip check \
    && groupadd --system --gid 10004 contour \
    && useradd --system --uid 10004 --gid contour --home-dir /nonexistent contour
COPY __init__.py settings.py contracts.py engine_adapter.py app.py /opt/contour/
COPY static /opt/contour/static
COPY THIRD_PARTY_NOTICES.md /opt/notices/THIRD_PARTY_NOTICES.md
USER 10004:10004
EXPOSE 8093
HEALTHCHECK --interval=15s --timeout=3s --retries=3 \
  CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8093/health/live', timeout=2).read()"]
CMD ["uvicorn", "contour.app:app", "--host", "0.0.0.0", "--port", "8093", "--workers", "1", "--no-server-header", "--no-access-log"]
