"""FastAPI server for the live perception demo.

One WebSocket carries everything for a run, so the browser needs no clock of
its own: binary messages are display JPEG frames (13-byte header: type byte,
uint32 sequence, float64 media timestamp), text messages are JSON events
(``session``, ``window``, ``state``, ``ended``, ``error``). In camera mode
the browser pushes binary frames upstream (uint16 width, uint16 height,
JPEG) and the same pipeline consumes them.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import struct
import tempfile
import uuid
from dataclasses import replace
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, UploadFile, WebSocket
from fastapi.responses import FileResponse, JSONResponse
from starlette.websockets import WebSocketDisconnect

from ..config import DEFAULT_CONFIG_PATH, HarnessConfig, load_config
from ..grounding import build_grounding_client
from ..openrouter import OpenRouterError, api_key_present
from ..pipeline import (
    CAMERA_MODE,
    FILE_MODE,
    STREAM_MODE,
    SessionCallbacks,
    StreamingSession,
)
from ..video_to_stream import VideoInfo, probe_video

INDEX_PATH = Path(__file__).with_name("index.html")

DISPLAY_FRAME_HEADER = struct.Struct(">BId")  # type, sequence, media timestamp
DISPLAY_FRAME_TYPE = 1
CAMERA_FRAME_HEADER = struct.Struct(">HH")  # width, height

# Outbound frames beyond this backlog are dropped; JSON is never dropped.
FRAME_BACKLOG_LIMIT = 60
OUTBOUND_QUEUE_SIZE = 512

_CLOSE = object()

EVENT_LABEL_MAX_CHARS = 80
UPLOAD_CHUNK_BYTES = 1 << 20


def _video_payload(info: VideoInfo) -> dict[str, Any]:
    return {
        "duration": round(info.duration_seconds, 3),
        "fps": round(info.source_fps, 3),
        "width": info.width,
        "height": info.height,
    }


class _SocketSession:
    """Bridge one WebSocket connection to at most one streaming session."""

    def __init__(
        self,
        websocket: WebSocket,
        config: HarnessConfig,
        uploads: dict[str, Path],
    ) -> None:
        self.websocket = websocket
        self.config = config
        self.uploads = uploads
        self.loop = asyncio.get_running_loop()
        self.outbound: asyncio.Queue[Any] = asyncio.Queue(maxsize=OUTBOUND_QUEUE_SIZE)
        self.session: StreamingSession | None = None
        self.dropped_outbound_frames = 0

    # ------------------------------------------------------------- outbound

    def send_threadsafe(self, item: Any) -> None:
        """Queue an outbound item from any thread; never blocks the pipeline."""

        try:
            self.loop.call_soon_threadsafe(self._enqueue, item)
        except RuntimeError:
            pass  # Loop already closed while a worker was finishing.

    def _enqueue(self, item: Any) -> None:
        is_frame = isinstance(item, (bytes, bytearray))
        if is_frame and self.outbound.qsize() >= FRAME_BACKLOG_LIMIT:
            self.dropped_outbound_frames += 1
            return
        while True:
            try:
                self.outbound.put_nowait(item)
                return
            except asyncio.QueueFull:
                try:
                    self.outbound.get_nowait()
                except asyncio.QueueEmpty:
                    pass

    async def drain_outbound(self) -> None:
        while True:
            item = await self.outbound.get()
            if item is _CLOSE:
                return
            if isinstance(item, (bytes, bytearray)):
                await self.websocket.send_bytes(bytes(item))
            else:
                await self.websocket.send_text(json.dumps(item, ensure_ascii=False))

    def send_error(self, message: str) -> None:
        self._enqueue({"type": "error", "message": message})

    # -------------------------------------------------------------- control

    def handle_control(self, payload: dict[str, Any]) -> None:
        command = payload.get("type")
        if command == "start":
            self._handle_start(payload)
        elif command == "stop":
            if self.session is not None:
                self.session.stop()
        else:
            self.send_error(f"unknown command: {command!r}")

    def handle_camera_frame(self, data: bytes) -> None:
        if self.session is None or self.session.mode != CAMERA_MODE:
            return
        if len(data) <= CAMERA_FRAME_HEADER.size:
            return
        width, height = CAMERA_FRAME_HEADER.unpack_from(data)
        if width < 1 or height < 1:
            return
        self.session.push_camera_frame(
            data[CAMERA_FRAME_HEADER.size :], width=width, height=height
        )

    def shutdown(self) -> None:
        if self.session is not None:
            self.session.stop()
        self._enqueue(_CLOSE)

    def _handle_start(self, payload: dict[str, Any]) -> None:
        if self.session is not None and self.session.running:
            self.send_error("a session is already running on this connection")
            return

        mode = str(payload.get("mode") or FILE_MODE)
        dry_run = bool(payload.get("dry_run"))
        event_label = str(
            payload.get("event_label") or self.config.inference.event_label
        ).strip()[:EVENT_LABEL_MAX_CHARS]
        if not event_label:
            event_label = self.config.inference.event_label
        inference_settings = replace(self.config.inference, event_label=event_label)

        try:
            client = build_grounding_client(
                inference_settings, self.config.openrouter, dry_run=dry_run
            )
        except OpenRouterError as exc:
            self.send_error(f"{exc} — export the key or enable dry run")
            return

        callbacks = SessionCallbacks(
            on_event=self.send_threadsafe,
            on_state=self.send_threadsafe,
            on_end=self._on_end,
            on_analysis_result=lambda result: self.send_threadsafe(result.as_dict()),
        )

        session_info: dict[str, Any] = {
            "type": "session",
            "status": "started",
            "mode": mode,
            "event_label": event_label,
            "model": inference_settings.model,
            "dry_run": dry_run,
            "sample_fps": self.config.video.fps,
            "window": {
                "size_frames": self.config.sliding_window.size_frames,
                "stride_frames": self.config.sliding_window.stride_frames,
            },
        }

        if mode == FILE_MODE:
            video_path = self._resolve_file(payload.get("file_id"))
            if video_path is None:
                self.send_error("upload a video or load the example first")
                return
            try:
                info = probe_video(video_path)
            except ValueError as exc:
                self.send_error(str(exc))
                return
            start_seconds = _safe_float(payload.get("start_seconds"), 0.0)
            start_seconds = min(
                max(0.0, start_seconds), max(0.0, info.duration_seconds - 0.5)
            )
            callbacks.on_display_frame = self._send_display_frame
            try:
                self.session = StreamingSession.for_file(
                    self.config,
                    client,
                    callbacks,
                    video_path=video_path,
                    start_seconds=start_seconds,
                )
            except (FileNotFoundError, ValueError) as exc:
                self.send_error(str(exc))
                return
            session_info["video"] = _video_payload(info)
            session_info["start_seconds"] = round(start_seconds, 3)
        elif mode == STREAM_MODE:
            stream_url = str(payload.get("stream_url") or "").strip()
            if not stream_url:
                self.send_error("provide an RTSP, RTMP, or WebRTC/WHEP stream URL")
                return
            callbacks.on_display_frame = self._send_display_frame
            try:
                self.session = StreamingSession.for_stream(
                    self.config,
                    client,
                    callbacks,
                    stream_url=stream_url,
                )
            except ValueError as exc:
                self.send_error(str(exc))
                return
            session_info["stream_protocol"] = self.session.video_stream.protocol
        elif mode == CAMERA_MODE:
            self.session = StreamingSession.for_camera(self.config, client, callbacks)
        else:
            self.send_error(f"unknown mode: {mode!r}")
            return

        self._enqueue(session_info)
        self.session.start()

    def _resolve_file(self, file_id: Any) -> Path | None:
        if not file_id:
            return None
        if file_id == "example":
            path = self.config.default_video
            return path if path is not None and path.is_file() else None
        path = self.uploads.get(str(file_id))
        return path if path is not None and path.is_file() else None

    def _send_display_frame(self, jpeg: bytes, timestamp: float, seq: int) -> None:
        header = DISPLAY_FRAME_HEADER.pack(
            DISPLAY_FRAME_TYPE, seq & 0xFFFFFFFF, timestamp
        )
        self.send_threadsafe(header + jpeg)

    def _on_end(self, reason: str, error: str | None) -> None:
        message: dict[str, Any] = {"type": "ended", "reason": reason}
        if error:
            message["error"] = error
        session = self.session
        if session is not None:
            message["final_state"] = session.organizer.snapshot()
            if session.video_stream is not None:
                message["dropped_display_frames"] = session.video_stream.dropped_frames
        self.send_threadsafe(message)


def _safe_float(value: Any, default: float) -> float:
    try:
        rendered = float(value)
    except (TypeError, ValueError):
        return default
    if rendered != rendered or rendered in (float("inf"), float("-inf")):
        return default
    return rendered


def create_app(config_path: str | Path = DEFAULT_CONFIG_PATH) -> FastAPI:
    config = load_config(config_path)
    uploads: dict[str, Path] = {}
    upload_dir = Path(tempfile.mkdtemp(prefix="goodvision-live-"))

    app = FastAPI(title="GoodVision Live Perception")

    @app.get("/")
    async def index() -> FileResponse:
        return FileResponse(INDEX_PATH, media_type="text/html")

    @app.get("/api/info")
    async def info() -> JSONResponse:
        example: dict[str, Any] | None = None
        if config.default_video is not None and config.default_video.is_file():
            try:
                example = _video_payload(probe_video(config.default_video))
                example["file_id"] = "example"
                example["filename"] = config.default_video.name
            except ValueError:
                example = None
        return JSONResponse(
            {
                "event_label": config.inference.event_label,
                "model": config.inference.model,
                "sample_fps": config.video.fps,
                "display_fps": config.display.fps,
                "window": {
                    "size_frames": config.sliding_window.size_frames,
                    "stride_frames": config.sliding_window.stride_frames,
                },
                "api_key_present": api_key_present(config.openrouter),
                "api_key_env": config.openrouter.api_key_env,
                "example": example,
            }
        )

    @app.post("/api/upload")
    async def upload(file: UploadFile) -> JSONResponse:
        suffix = Path(file.filename or "upload.mp4").suffix or ".mp4"
        file_id = uuid.uuid4().hex
        destination = upload_dir / f"{file_id}{suffix}"
        with destination.open("wb") as handle:
            while chunk := await file.read(UPLOAD_CHUNK_BYTES):
                handle.write(chunk)
        try:
            info = probe_video(destination)
        except ValueError as exc:
            destination.unlink(missing_ok=True)
            return JSONResponse({"error": str(exc)}, status_code=400)
        uploads[file_id] = destination
        payload = _video_payload(info)
        payload["file_id"] = file_id
        payload["filename"] = file.filename or destination.name
        return JSONResponse(payload)

    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket) -> None:
        await websocket.accept()
        socket_session = _SocketSession(websocket, config, uploads)
        sender = asyncio.create_task(socket_session.drain_outbound())
        try:
            while True:
                message = await websocket.receive()
                if message["type"] == "websocket.disconnect":
                    break
                text = message.get("text")
                if text is not None:
                    try:
                        payload = json.loads(text)
                    except json.JSONDecodeError:
                        socket_session.send_error("control messages must be JSON")
                        continue
                    if isinstance(payload, dict):
                        socket_session.handle_control(payload)
                    continue
                data = message.get("bytes")
                if data:
                    socket_session.handle_camera_frame(data)
        except (WebSocketDisconnect, RuntimeError):
            pass
        finally:
            socket_session.shutdown()
            try:
                await asyncio.wait_for(sender, timeout=2.0)
            except (TimeoutError, WebSocketDisconnect, RuntimeError):
                sender.cancel()

    return app


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--host", help="bind address (defaults to server.host)")
    parser.add_argument("--port", type=int, help="port (defaults to server.port)")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    config = load_config(args.config)
    app = create_app(config.path)
    uvicorn.run(
        app,
        host=args.host or config.server.host,
        port=args.port or config.server.port,
        log_level="info",
    )


if __name__ == "__main__":
    main()
