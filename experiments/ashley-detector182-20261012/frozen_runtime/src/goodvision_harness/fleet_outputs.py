"""Camera-scoped, read-only outputs with bounded JSON and unbuffered media.

Local streams are dispatched as ASGI, never through the buffering ASGITransport.
Peer streams use a separate pool so viewers cannot exhaust the control pool.
"""

from __future__ import annotations

import asyncio
import copy
import json
import logging
import re
import struct
from urllib.parse import quote

import httpx
from fastapi import Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from fastapi.routing import APIRoute
from starlette.routing import Match

from .privacy import public_safe_value

log = logging.getLogger(__name__)
JSON_LIMIT = 2 * 1024 * 1024
FRAME_LIMIT = 8 * 1024 * 1024
PREFIX = "/v1/fleet/cameras/{camera_id}"

# Fixed routes precede variable event IDs. This is an allowlist, not a URL proxy.
OUTPUT_PATHS = (
    "/status",
    "/results/latest",
    "/results/events",
    "/results/overall/latest",
    "/results/overall/events",
    *(
        f"/results/{feature}/{kind}"
        for feature in ("fight", "smoking", "fall", "weapon", "stranger")
        for kind in ("latest", "events")
    ),
    "/rich-labels/latest",
    "/faces/fast",
    "/faces/gallery",
    "/faces/strangers",
    "/faces/strangers/{stranger_id}/photo",
    "/scene",
    "/experimental/scene",
    "/scene/events",
    "/scene/history",
    "/scene/evidence",
    "/scene/evidence/{request_id}/{frame_sequence}.jpg",
    "/events",
    "/events/stream",
    "/events/{event_id}/video",
    "/events/{event_id}",
    "/video/display/metadata",
    "/video/display/frames",
    *(
        f"/video/{channel}{kind}"
        for channel in ("", "live/", "annotated/")
        for kind in ("latest.jpg", "stream.mjpeg")
    ),
)
_OUTPUT_RE = re.compile(
    "^(?:"
    + "|".join(
        re.sub(r"\\\{[^}]+\\\}", "[^/?#]+", re.escape("/v1" + path))
        for path in OUTPUT_PATHS
    )
    + r")(?:[?#].*)?$"
)
_FORWARD_HEADERS = {"range", "if-range", "last-event-id"}
_RESPONSE_HEADERS = {
    b"content-type",
    b"content-length",
    b"content-range",
    b"content-disposition",
    b"accept-ranges",
    b"etag",
    b"last-modified",
    b"retry-after",
    b"cache-control",
    b"x-accel-buffering",
    b"x-frame-sequence",
    b"x-media-timestamp",
    b"x-observed-at",
    b"x-analysis-message-id",
    b"x-next-analysis-message-id",
    b"x-annotation-mode",
    b"x-stream-delay",
    b"x-result-cursor",
    b"x-result-stream",
    b"x-event-id",
}


def output_urls(camera_id: str) -> dict[str, str]:
    """Publish literal URLs and templates, including paths added in future."""
    base = PREFIX.format(camera_id=camera_id)
    return {path.lstrip("/"): base + path for path in OUTPUT_PATHS}


def scoped_payload(payload, camera_id: str):
    def rewrite(value, key=""):
        if isinstance(value, dict):
            return {k: rewrite(v, k) for k, v in value.items()}
        if isinstance(value, list):
            return [rewrite(v, key) for v in value]
        if (
            isinstance(value, str)
            and (key in {"url", "href", "replacement"} or key.endswith("_url"))
            and _OUTPUT_RE.fullmatch(value)
        ):
            return PREFIX.format(camera_id=camera_id) + value[3:]
        return value

    result = rewrite(public_safe_value(payload))
    if isinstance(result, dict):
        result["camera_id"] = camera_id
    elif isinstance(result, list):
        result = [
            dict(item, camera_id=camera_id) if isinstance(item, dict) else item
            for item in result
        ]
    return result


def _json_bytes(payload, camera_id):
    return json.dumps(
        scoped_payload(payload, camera_id),
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    ).encode("utf-8")


def _display_bytes(body, camera_id):
    if len(body) < 8 or body[:4] != b"GVD1":
        raise ValueError("invalid display packet")
    length = struct.unpack(">I", body[4:8])[0]
    if length > 1024 * 1024 or length > len(body) - 8:
        raise ValueError("invalid display metadata")
    header = _json_bytes(json.loads(body[8 : 8 + length]), camera_id)
    return b"GVD1" + struct.pack(">I", len(header)) + header + body[8 + length :]


class _SSE:
    """Incremental event framing; cursors, comments and event names stay intact."""

    def __init__(self, camera_id):
        self.camera_id = camera_id
        self.pending = bytearray()

    def feed(self, chunk):
        self.pending.extend(chunk)
        while match := re.search(rb"\r?\n\r?\n", self.pending):
            if match.start() > JSON_LIMIT:
                raise ValueError("oversized event")
            event = bytes(self.pending[: match.start()])
            del self.pending[: match.end()]
            lines = event.decode("utf-8").splitlines()
            data = [
                line[5:].removeprefix(" ") for line in lines if line.startswith("data:")
            ]
            if data:
                payload = json.loads("\n".join(data))
                lines = [line for line in lines if not line.startswith("data:")]
                lines.append("data: " + _json_bytes(payload, self.camera_id).decode())
            yield ("\n".join(lines) + "\n\n").encode()
        if len(self.pending) > JSON_LIMIT:
            raise ValueError("oversized event")


class CameraOutputResponse(Response):
    def __init__(self, controller, index: int, native_path: str):
        super().__init__()
        self.controller, self.index, self.native_path = controller, index, native_path

    async def __call__(self, scope, receive, send):
        camera_id = f"camera{self.index:02d}"
        started = False
        start = None
        mode = "binary"
        buffered = bytearray()
        sse = _SSE(camera_id)

        async def emit(message):
            nonlocal started
            if message["type"] == "http.response.start":
                started = True
            await send(message)

        async def scoped_send(message):
            nonlocal start, mode
            if message["type"] == "http.response.start":
                headers = [
                    (k.lower(), v)
                    for k, v in message.get("headers", [])
                    if k.lower() in _RESPONSE_HEADERS
                ]
                content_type = dict(headers).get(b"content-type", b"").split(b";")[0]
                if content_type == b"application/json":
                    mode = "json"
                elif content_type == b"text/event-stream":
                    mode = "sse"
                elif content_type == b"application/vnd.goodvision.display":
                    mode = "display"
                elif content_type == b"image/jpeg":
                    mode = "finite"
                elif content_type in {
                    b"video/mp4",
                    b"multipart/x-mixed-replace",
                    b"multipart/byteranges",
                }:
                    mode = "binary"
                elif message["status"] >= 400:
                    mode = "finite"
                else:
                    raise ValueError("unexpected camera content type")
                if mode in {"json", "sse", "display"}:
                    headers = [(k, v) for k, v in headers if k != b"content-length"]
                headers = [(k, v) for k, v in headers if k != b"cache-control"]
                headers += [
                    (b"x-camera-id", camera_id.encode()),
                    (b"cache-control", b"no-store"),
                ]
                start = {**message, "headers": headers}
                if mode in {"sse", "binary"}:
                    await emit(start)
                return
            if message["type"] != "http.response.body":
                await emit(message)
                return
            chunk = message.get("body", b"")
            if mode in {"json", "finite", "display"}:
                limit = JSON_LIMIT if mode == "json" else FRAME_LIMIT
                if len(buffered) + len(chunk) > limit:
                    raise ValueError("oversized camera response")
                buffered.extend(chunk)
                if not message.get("more_body", False):
                    body = (
                        _json_bytes(json.loads(buffered), camera_id)
                        if mode == "json"
                        else bytes(buffered)
                    )
                    if mode == "display":
                        body = _display_bytes(body, camera_id)
                    if mode in {"json", "display"}:
                        start["headers"].append(
                            (b"content-length", str(len(body)).encode())
                        )
                    await emit(start)
                    await emit({**message, "body": body})
            elif mode == "sse":
                for event in sse.feed(chunk):
                    await emit(
                        {"type": "http.response.body", "body": event, "more_body": True}
                    )
                if not message.get("more_body", False):
                    # Incomplete SSE events have no dispatch delimiter: discard them.
                    await emit({"type": "http.response.body", "body": b""})
            else:
                await emit(message)

        try:
            if self.index == self.controller.camera_index:
                await self._local(scope, receive, scoped_send)
            else:
                await self._peer(scope, receive, scoped_send)
        except (httpx.HTTPError, TimeoutError, ValueError):
            if started:
                # Closing the stream lets the client reconnect with its own cursor.
                log.warning(
                    "Camera output interrupted: %s %s", camera_id, self.native_path
                )
                await send({"type": "http.response.body", "body": b""})
            else:
                await JSONResponse(
                    {"detail": "camera output unavailable", "camera_id": camera_id},
                    status_code=502,
                    headers={"X-Camera-Id": camera_id, "Cache-Control": "no-store"},
                )(scope, receive, send)

    async def _local(self, scope, receive, send):
        native_scope = {
            **scope,
            "path": self.native_path,
            "raw_path": quote(self.native_path, safe="/").encode(),
            "path_params": {},
        }
        # Runtime lookup also finds native routes installed after the fleet routes.
        for route in self.controller.application.router.routes:
            if isinstance(route, APIRoute):
                match, child = route.matches(native_scope)
                if match == Match.FULL:
                    native_scope.update(child)
                    await route.handle(native_scope, receive, send)
                    return
        await JSONResponse({"detail": "camera output not supported"}, status_code=404)(
            native_scope, receive, send
        )

    async def _peer(self, scope, receive, send):
        client = self.controller.output_client()
        headers = {
            k.decode(): v.decode()
            for k, v in scope["headers"]
            if k.decode().lower() in _FORWARD_HEADERS
        }
        headers["accept-encoding"] = "identity"
        url = httpx.URL(
            self.controller._url(self.index, quote(self.native_path, safe="/"))
        ).copy_with(query=scope.get("query_string", b""))
        request = client.build_request("GET", url, headers=headers)
        upstream = await asyncio.wait_for(client.send(request, stream=True), timeout=8)
        try:
            if 300 <= upstream.status_code < 400 or upstream.headers.get(
                "content-encoding"
            ):
                raise ValueError("unsupported upstream response")
            response = StreamingResponse(
                upstream.aiter_bytes(), status_code=upstream.status_code
            )
            response.raw_headers = [(k.lower(), v) for k, v in upstream.headers.raw]
            content_type = upstream.headers.get("content-type", "").split(";")[0]
            if content_type in {
                "text/event-stream",
                "multipart/x-mixed-replace",
                "video/mp4",
                "multipart/byteranges",
            }:
                await response(scope, receive, send)
            else:
                await asyncio.wait_for(response(scope, receive, send), timeout=10)
        finally:
            await asyncio.shield(upstream.aclose())


def install_output_routes(application, controller):
    async def camera_output(request: Request):
        camera_id = request.path_params["camera_id"]
        index = controller.resolve_camera(camera_id)
        controller._require_primary()
        base = PREFIX.format(camera_id=camera_id)
        suffix = request.scope["path"][len(base) :]
        if suffix.startswith("/video/"):
            controller.require_enabled(index)
        return CameraOutputResponse(controller, index, "/v1" + suffix)

    for suffix in OUTPUT_PATHS:
        application.add_api_route(
            PREFIX + suffix,
            camera_output,
            methods=["GET"],
            name="camera_output_" + suffix,
            tags=["Camera outputs"],
            response_class=Response,
            description="Camera-scoped output. Native query parameters, Range and "
            "Last-Event-ID apply. JSON/SSE include camera_id; all responses "
            "include X-Camera-Id. Discover URLs via GET /v1/fleet.",
        )

    original_openapi = application.openapi

    def camera_openapi():
        schema = original_openapi()
        for suffix in OUTPUT_PATHS:
            operation = schema["paths"][PREFIX + suffix]["get"]
            is_sse = suffix in {"/scene/events", "/events/stream"} or (
                suffix.startswith("/results/") and suffix.endswith("/events")
            )
            native = schema["paths"].get("/v1" + suffix, {}).get("get", {})
            parameters = copy.deepcopy(native.get("parameters", []))
            # Some compatibility paths are intentionally hidden from the native schema.
            for name in re.findall(r"{([^}]+)}", suffix):
                if not any(
                    p.get("in") == "path" and p.get("name") == name for p in parameters
                ):
                    parameters.append(
                        {
                            "name": name,
                            "in": "path",
                            "required": True,
                            "schema": {"type": "string"},
                        }
                    )
            parameters.insert(
                0,
                {
                    "name": "camera_id",
                    "in": "path",
                    "required": True,
                    "schema": {"type": "string"},
                    "example": "camera02",
                },
            )
            if is_sse:
                parameters.append(
                    {
                        "name": "Last-Event-ID",
                        "in": "header",
                        "schema": {"type": "string"},
                    }
                )
            if suffix == "/events/{event_id}/video":
                parameters.extend(
                    {"name": h, "in": "header", "schema": {"type": "string"}}
                    for h in ("Range", "If-Range")
                )
            operation["parameters"] = parameters
            if native.get("responses"):
                operation["responses"] = copy.deepcopy(native["responses"])
            media = None
            if is_sse:
                media = "text/event-stream"
            elif suffix.endswith(".mjpeg"):
                media = "multipart/x-mixed-replace"
            elif suffix.endswith(".jpg"):
                media = "image/jpeg"
            elif suffix == "/video/display/frames":
                media = "application/vnd.goodvision.display"
            if media:
                operation["responses"].setdefault(
                    "200", {"description": "Camera output"}
                )["content"] = {media: {"schema": {"type": "string"}}}
            operation["responses"].setdefault(
                "502",
                {
                    "description": "Camera output unavailable or invalid"
                },
            )
            for response in operation["responses"].values():
                response.setdefault("headers", {})["X-Camera-Id"] = {
                    "schema": {"type": "string"},
                    "description": "Owning camera",
                }
        return schema

    application.openapi = camera_openapi
