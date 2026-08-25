"""HTTP delivery service for the independent streaming harness."""

from __future__ import annotations

import asyncio
import copy
import json
import logging
import os
import time
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from datetime import UTC
from pathlib import Path
from threading import Lock
from typing import Any

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response, StreamingResponse

from .config import (
    DEFAULT_CONFIG_PATH,
    HarnessConfig,
    ServiceSettings,
    load_config,
)
from .contracts import AnalysisResult
from .delivery import WebhookDelivery
from .grounding import build_grounding_client
from .overlay import draw_overlay
from .pipeline import SessionCallbacks, StreamingSession
from .result_bus import FrameBus, FramePacket, ResultBus
from .stream_adapter import parse_stream_source

LOGGER = logging.getLogger(__name__)
MJPEG_BOUNDARY = "goodvision-frame"


class HarnessRuntime:
    """Own the RTSP session and expose thread-safe delivery state."""

    def __init__(
        self,
        config: HarnessConfig | None,
        *,
        config_error: str | None = None,
    ) -> None:
        try:
            dry_run = _env_bool("HARNESS_DRY_RUN", False)
        except ValueError as exc:
            dry_run = False
            config_error = config_error or f"configuration error: {exc}"
        self.config = config
        self.config_error = config_error
        self.dry_run = dry_run
        self.settings = config.service if config is not None else ServiceSettings()
        self.source_id = (
            os.environ.get("HARNESS_SOURCE_ID", "").strip() or self.settings.source_id
        )
        self.results = ResultBus(self.settings.result_history)
        # ``frames`` carries one exact analyzed frame per result. The separate
        # preview bus remains smooth and intentionally has no delayed boxes.
        self.frames = FrameBus()
        self.preview_frames = FrameBus()
        self._state_lock = Lock()
        self._session: StreamingSession | None = None
        self._delivery: WebhookDelivery | None = None
        self._last_frame_monotonic: float | None = None
        self._last_result_monotonic: float | None = None
        self._stopping = False
        self._state: dict[str, Any] = {
            "service_started_at": _utc_now(),
            "source_id": self.source_id,
            "pipeline_state": (
                "configuration_error" if config_error else "not_started"
            ),
            "ready": False,
            "last_frame_at": None,
            "last_result_at": None,
            "source_frames_received": 0,
            "analysis_frames_published": 0,
            "results_published": 0,
            "last_error": config_error,
            "end_reason": None,
            "stream": None,
            "stream_session_id": None,
            "model": config.inference.model if config is not None else None,
            "dry_run": self.dry_run,
            "webhook_enabled": False,
            "pipeline": None,
        }

    @classmethod
    def from_environment(cls) -> HarnessRuntime:
        config_path = Path(os.environ.get("HARNESS_CONFIG", str(DEFAULT_CONFIG_PATH)))
        try:
            config = load_config(config_path)
        except Exception as exc:
            LOGGER.exception("could not load harness configuration")
            return cls(None, config_error=f"configuration error: {exc}")
        return cls(config)

    def start(self) -> None:
        if self.config is None or self.config_error is not None:
            return
        stream_url = (
            os.environ.get("RTSP_URL", "").strip()
            or os.environ.get("HARNESS_STREAM_URL", "").strip()
            or self.config.stream.url
        )
        if not stream_url:
            self._update_state(
                pipeline_state="configuration_error",
                last_error=(
                    "no video source configured; set RTSP_URL, "
                    "HARNESS_STREAM_URL, or stream.url"
                ),
            )
            return

        try:
            source = parse_stream_source(stream_url)
            client = build_grounding_client(
                self.config.inference,
                self.config.openrouter,
                dry_run=self.dry_run,
            )
            self._delivery = self._build_delivery()
            if self._delivery is not None:
                self._delivery.start()
            callbacks = SessionCallbacks(
                on_display_frame=self._on_display_frame,
                on_state=self._on_pipeline_state,
                on_end=self._on_end,
                on_analysis_frame=self._on_analysis_frame,
                on_analysis_result=self._on_analysis_result,
            )
            self._session = StreamingSession.for_stream(
                self.config,
                client,
                callbacks,
                stream_url=stream_url,
                whep_headers=self.config.stream.whep_headers,
                source_id=self.source_id,
            )
            self._update_state(
                pipeline_state="starting",
                last_error=None,
                stream={
                    "protocol": source.protocol,
                    "host": source.host,
                },
                stream_session_id=self._session.stream_session_id,
                webhook_enabled=self._delivery is not None,
            )
            self._session.start()
        except Exception as exc:
            LOGGER.exception("could not start harness runtime")
            if self._delivery is not None:
                self._delivery.close()
                self._delivery = None
            self._update_state(
                pipeline_state="error",
                last_error=str(exc),
                ready=False,
            )

    def stop(self) -> None:
        self._stopping = True
        session = self._session
        if session is not None:
            session.stop()
            session.join(timeout=5.0)
        if self._delivery is not None:
            self._delivery.close(timeout_seconds=5.0)
        self._update_state(pipeline_state="stopped", ready=False)

    def status(self) -> dict[str, Any]:
        with self._state_lock:
            payload = copy.deepcopy(self._state)
        ready, reason = self.readiness()
        payload["ready"] = ready
        payload["readiness_reason"] = reason
        if self._delivery is not None:
            payload["webhook"] = self._delivery.stats()
        return payload

    def readiness(self) -> tuple[bool, str]:
        with self._state_lock:
            state = self._state["pipeline_state"]
            last_frame = self._last_frame_monotonic
            last_result = self._last_result_monotonic
        if state in {"configuration_error", "error", "stopped"}:
            return False, str(state)
        session = self._session
        if session is None or not session.running:
            return False, "pipeline_not_running"
        if last_frame is None:
            return False, "waiting_for_first_frame"
        now = time.monotonic()
        if now - last_frame > self.settings.ready_max_frame_age_seconds:
            return False, "frame_stale"
        if last_result is None:
            return False, "waiting_for_first_analysis"
        if now - last_result > self.settings.ready_max_result_age_seconds:
            return False, "analysis_stale"
        return True, "ready"

    def _build_delivery(self) -> WebhookDelivery | None:
        assert self.config is not None
        webhook = self.settings.webhook
        url = os.environ.get("RESULT_WEBHOOK_URL", "").strip() or webhook.url
        if not url:
            return None
        secret = os.environ.get(webhook.secret_env)
        return WebhookDelivery(
            url,
            secret=secret,
            timeout_seconds=webhook.timeout_seconds,
            max_retries=webhook.max_retries,
            retry_backoff_seconds=webhook.retry_backoff_seconds,
            queue_size=webhook.queue_size,
        )

    def _on_analysis_frame(self, result: AnalysisResult, jpeg: bytes) -> None:
        source_jpeg = bytes(jpeg)
        try:
            rendered = _render_analysis_jpeg(
                source_jpeg,
                result,
                quality=self.settings.overlay_jpeg_quality,
            )
        except (cv2.error, TypeError, ValueError):
            # Preserve time correlation even if visualization rendering fails.
            LOGGER.warning("could not render analysis overlay", exc_info=True)
            rendered = source_jpeg
        self.frames.publish(
            FramePacket(
                jpeg=rendered,
                media_timestamp=result.frame.timestamp_seconds,
                frame_sequence=result.frame.source_frame_index,
                observed_at=result.frame.observed_at,
                analysis_message_id=result.message_id,
            )
        )
        with self._state_lock:
            frame_count = int(self._state["analysis_frames_published"]) + 1
        self._update_state(
            analysis_frames_published=frame_count,
            last_analysis_frame_at=_utc_now(),
        )

    def _on_analysis_result(self, result: AnalysisResult) -> None:
        payload = result.as_dict()
        self.results.publish(payload)
        with self._state_lock:
            result_count = int(self._state["results_published"]) + 1
            self._last_result_monotonic = time.monotonic()
        self._update_state(
            last_result_at=_utc_now(),
            results_published=result_count,
            stream_session_id=result.stream_session_id,
        )
        if self._delivery is not None:
            self._delivery.submit(payload)

    def _on_display_frame(
        self,
        jpeg: bytes,
        media_timestamp: float,
        frame_sequence: int,
    ) -> None:
        observed_at = _utc_now()
        self.preview_frames.publish(
            FramePacket(
                jpeg=bytes(jpeg),
                media_timestamp=float(media_timestamp),
                frame_sequence=int(frame_sequence),
                observed_at=observed_at,
            )
        )
        with self._state_lock:
            frame_count = int(self._state["source_frames_received"]) + 1
            self._last_frame_monotonic = time.monotonic()
        self._update_state(
            pipeline_state="running",
            last_frame_at=observed_at,
            source_frames_received=frame_count,
            last_media_timestamp=round(float(media_timestamp), 3),
            last_frame_sequence=int(frame_sequence),
        )

    def _on_pipeline_state(self, state: dict[str, Any]) -> None:
        self._update_state(pipeline=copy.deepcopy(state))

    def _on_end(self, reason: str, error: str | None) -> None:
        next_state = "stopped" if self._stopping or reason == "stopped" else "ended"
        if error:
            next_state = "error"
        self._update_state(
            pipeline_state=next_state,
            ready=False,
            end_reason=reason,
            last_error=error,
        )

    def _update_state(self, **values: Any) -> None:
        with self._state_lock:
            self._state.update(values)


def _render_analysis_jpeg(
    jpeg: bytes,
    analysis: AnalysisResult,
    *,
    quality: int,
) -> bytes:
    encoded = np.frombuffer(jpeg, dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("display callback did not contain a valid JPEG")
    judgment = analysis.judgments[0]
    referenced = set(judgment.object_ids)
    objects = tuple(item for item in analysis.objects if item.object_id in referenced)
    rendered = draw_overlay(
        image,
        event_label=judgment.event_label,
        detected=judgment.detected,
        confidence=judgment.confidence,
        description=judgment.description,
        objects=objects,
    )
    ok, output = cv2.imencode(
        ".jpg",
        rendered,
        [cv2.IMWRITE_JPEG_QUALITY, quality],
    )
    if not ok:
        raise ValueError("could not encode annotated JPEG")
    return output.tobytes()


def create_app(
    runtime_factory: Callable[[], HarnessRuntime] | None = None,
) -> FastAPI:
    factory = runtime_factory or HarnessRuntime.from_environment

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        runtime = factory()
        application.state.runtime = runtime
        runtime.start()
        try:
            yield
        finally:
            runtime.stop()

    application = FastAPI(
        title="Goodvision Streaming Harness",
        version="1.0.0",
        lifespan=lifespan,
    )

    @application.get("/", response_class=HTMLResponse, include_in_schema=False)
    async def index() -> str:
        return _DASHBOARD_HTML

    @application.get("/health/live")
    async def health_live() -> dict[str, str]:
        return {"status": "live"}

    @application.get("/health/ready")
    async def health_ready(request: Request) -> JSONResponse:
        runtime = _runtime(request)
        ready, reason = runtime.readiness()
        return JSONResponse(
            {"status": "ready" if ready else "not_ready", "reason": reason},
            status_code=200 if ready else 503,
        )

    @application.get("/v1/status")
    async def status(request: Request) -> dict[str, Any]:
        return _runtime(request).status()

    @application.get("/v1/results/latest")
    async def latest_result(request: Request) -> JSONResponse:
        record = _runtime(request).results.latest()
        if record is None:
            raise HTTPException(status_code=404, detail="no analysis result yet")
        return JSONResponse(
            record.payload,
            headers={
                "Cache-Control": "no-store",
                "X-Result-Cursor": str(record.cursor),
                "X-Result-Stream": _runtime(request).results.epoch,
            },
        )

    @application.get("/v1/results/events")
    async def result_events(
        request: Request,
        after: int = Query(default=0, ge=0),
    ) -> StreamingResponse:
        runtime = _runtime(request)
        header_cursor = request.headers.get("last-event-id")
        cursor = after
        if header_cursor and after == 0:
            epoch, separator, raw_cursor = header_cursor.rpartition(":")
            if separator and epoch == runtime.results.epoch:
                try:
                    cursor = max(0, int(raw_cursor))
                except ValueError:
                    cursor = 0
            else:
                cursor = 0

        async def generate() -> AsyncIterator[bytes]:
            current = cursor
            while not await request.is_disconnected():
                records = await asyncio.to_thread(
                    runtime.results.records_after,
                    current,
                    timeout_seconds=15.0,
                )
                if not records:
                    yield b": heartbeat\n\n"
                    continue
                for record in records:
                    current = record.cursor
                    data = json.dumps(
                        record.payload,
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                    yield (
                        f"id: {runtime.results.epoch}:{record.cursor}\n"
                        f"event: analysis_result\ndata: {data}\n\n"
                    ).encode()

        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache, no-transform",
                "X-Accel-Buffering": "no",
            },
        )

    @application.get("/v1/video/latest.jpg")
    async def latest_frame(request: Request) -> Response:
        return _snapshot_response(
            _runtime(request).frames,
            missing_detail="no analyzed frame yet",
        )

    @application.get("/v1/video/stream.mjpeg")
    async def video_stream(request: Request) -> StreamingResponse:
        return _mjpeg_response(request, _runtime(request).frames)

    @application.get("/v1/video/live/latest.jpg")
    async def latest_live_frame(request: Request) -> Response:
        return _snapshot_response(
            _runtime(request).preview_frames,
            missing_detail="no live source frame yet",
        )

    @application.get("/v1/video/live/stream.mjpeg")
    async def live_video_stream(request: Request) -> StreamingResponse:
        return _mjpeg_response(request, _runtime(request).preview_frames)

    return application


def _runtime(request: Request) -> HarnessRuntime:
    return request.app.state.runtime


def _frame_headers(packet: FramePacket) -> dict[str, str]:
    headers = {
        "Cache-Control": "no-store",
        "X-Frame-Sequence": str(packet.frame_sequence),
        "X-Media-Timestamp": f"{packet.media_timestamp:.3f}",
        "X-Observed-At": packet.observed_at,
    }
    if packet.analysis_message_id is not None:
        headers["X-Analysis-Message-Id"] = packet.analysis_message_id
    return headers


def _snapshot_response(bus: FrameBus, *, missing_detail: str) -> Response:
    latest = bus.latest()
    if latest is None:
        raise HTTPException(status_code=404, detail=missing_detail)
    _, packet = latest
    return Response(
        packet.jpeg,
        media_type="image/jpeg",
        headers=_frame_headers(packet),
    )


def _mjpeg_response(request: Request, bus: FrameBus) -> StreamingResponse:
    async def generate() -> AsyncIterator[bytes]:
        version = 0
        latest = bus.latest()
        if latest is not None:
            version, packet = latest
            yield _mjpeg_part(packet)
        while not await request.is_disconnected():
            update = await asyncio.to_thread(
                bus.wait_after,
                version,
                timeout_seconds=15.0,
            )
            if update is None:
                continue
            version, packet = update
            yield _mjpeg_part(packet)

    return StreamingResponse(
        generate(),
        media_type=f"multipart/x-mixed-replace; boundary={MJPEG_BOUNDARY}",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "X-Accel-Buffering": "no",
        },
    )


def _mjpeg_part(packet: FramePacket) -> bytes:
    analysis_header = ""
    if packet.analysis_message_id is not None:
        analysis_header = f"X-Analysis-Message-Id: {packet.analysis_message_id}\r\n"
    headers = (
        f"--{MJPEG_BOUNDARY}\r\n"
        "Content-Type: image/jpeg\r\n"
        f"Content-Length: {len(packet.jpeg)}\r\n"
        f"X-Frame-Sequence: {packet.frame_sequence}\r\n"
        f"X-Media-Timestamp: {packet.media_timestamp:.3f}\r\n"
        f"X-Observed-At: {packet.observed_at}\r\n"
        f"{analysis_header}\r\n"
    ).encode("ascii")
    return headers + packet.jpeg + b"\r\n"


def _utc_now() -> str:
    from datetime import datetime

    return datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    rendered = value.strip().lower()
    if rendered in {"1", "true", "yes", "on"}:
        return True
    if rendered in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be true or false")


def main() -> None:
    host = os.environ.get("HARNESS_HOST", "0.0.0.0")
    port = int(os.environ.get("HARNESS_PORT", "8080"))
    uvicorn.run(app, host=host, port=port)


_DASHBOARD_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>Goodvision Harness</title>
  <style>
    :root { color-scheme: dark; font-family: system-ui, sans-serif; }
    body { margin: 0; padding: 24px; background: #101318; color: #edf2f7; }
    main { max-width: 1200px; margin: auto; display: grid; gap: 20px; }
    h1 { margin: 0; font-size: 24px; }
    .grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 20px;
    }
    .card {
      background: #191f28;
      border: 1px solid #2b3441;
      border-radius: 12px;
      padding: 16px;
    }
    img {
      width: 100%;
      min-height: 320px;
      object-fit: contain;
      background: #080a0d;
      border-radius: 8px;
    }
    pre {
      white-space: pre-wrap;
      overflow-wrap: anywhere;
      max-height: 520px;
      overflow: auto;
    }
    .muted { color: #9aabba; }
    @media (max-width: 800px) { .grid { grid-template-columns: 1fr; } }
  </style>
</head>
<body><main>
  <h1>Goodvision Streaming Harness</h1>
  <div class="muted">RTSP in → public LLM API → JSON + annotated MJPEG out</div>
  <div class="grid">
    <section class="card">
      <h2>Analyzed frame</h2>
      <div class="muted">One time-aligned frame per JSON result</div>
      <img src="/v1/video/stream.mjpeg" alt="Waiting for analyzed frame">
    </section>
    <section class="card">
      <h2>Live input</h2>
      <div class="muted">Continuous preview without delayed boxes</div>
      <img src="/v1/video/live/stream.mjpeg" alt="Waiting for live stream">
    </section>
    <section class="card">
      <h2>Status</h2><pre id="status">Loading…</pre>
      <h2>Latest result</h2><pre id="result">Waiting…</pre>
    </section>
  </div>
</main>
<script>
const statusNode = document.querySelector("#status");
const resultNode = document.querySelector("#result");
async function updateStatus() {
  try {
    const response = await fetch("/v1/status", {cache: "no-store"});
    statusNode.textContent = JSON.stringify(await response.json(), null, 2);
  } catch (error) { statusNode.textContent = String(error); }
}
setInterval(updateStatus, 2000); updateStatus();
const events = new EventSource("/v1/results/events");
events.addEventListener("analysis_result", event => {
  resultNode.textContent = JSON.stringify(JSON.parse(event.data), null, 2);
});
</script></body></html>"""


app = create_app()


if __name__ == "__main__":
    main()
