"""Exercise a fresh, isolated harness container using synthetic frames only.

Run inside the already-running container:
    python /tmp/delivery_smoke.py --isolated

The service must start with HARNESS_AUTOSTART=false, HARNESS_DRY_RUN=true,
RICH_LABEL_ENABLED=false, FACE_ALLOW_MODEL_DOWNLOAD=false, no webhook, and a
fresh temporary data directory. Prefer a container with --network none. With
--with-rich, enable rich labeling and connect only to an isolated deterministic
test worker on an internal Docker network. This check creates one temporary
camera-mode session and stops it before returning. It does not test real-model
accuracy, GPU execution, or create real stranger evidence photos.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import struct
import sys
import time
from urllib.parse import urlsplit

import cv2
import httpx
import numpy as np
import websockets

FEATURES = ("fight", "smoking", "fall", "weapon", "stranger")
RESULT_ROOTS = ("/v1/results", *(f"/v1/results/{item}" for item in FEATURES))
VIDEO_ROOTS = ("/v1/video", "/v1/video/live", "/v1/video/annotated")
PHOTO_LIST = "/v1/faces/strangers"
PHOTO_MISSING = "/v1/faces/strangers/stranger-00000000000000000000000000000000/photo"
TIMEOUT = 8.0
WIDTH, HEIGHT = 240, 160
AUDIT_BRANDING = False
PRIVATE_MARKERS = re.compile(
    r"openrouter|qwen|insightface|paddle|pp[-_]yolo|pp[-_]lcnet|picodet|"
    r"buffalo_l|aiclub|zimaocloud|onnxruntime|opencv|nvidia|cuda|cudnn|"
    r"huggingface|modelscope|fastapi|uvicorn|swagger|redoc|jsdelivr|"
    r"openai|anthropic|deepseek|ffmpeg",
    re.IGNORECASE,
)


def check_public(value: object) -> None:
    if AUDIT_BRANDING:
        require(not PRIVATE_MARKERS.search(str(value)), "private implementation detail in public output")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def passed(message: str) -> None:
    print(f"PASS  {message}", flush=True)


async def request(
    client: httpx.AsyncClient, path: str, expected_status: int = 200
) -> httpx.Response:
    response = await client.get(path)
    require(
        response.status_code == expected_status,
        f"{path}: expected HTTP {expected_status}, got {response.status_code}",
    )
    if AUDIT_BRANDING:
        require("server" not in response.headers, "server implementation header exposed")
        check_public(dict(response.headers))
        if "json" in response.headers.get("content-type", ""):
            check_public(response.text)
    return response


async def missing(client: httpx.AsyncClient, path: str, detail: str) -> None:
    response = await request(client, path, 404)
    require(response.json().get("detail") == detail, f"{path}: wrong 404 detail")


def check_stream_headers(response: httpx.Response, content_type: str) -> None:
    require(response.status_code == 200, "stream did not return HTTP 200")
    require(
        response.headers.get("content-type", "").startswith(content_type),
        f"stream Content-Type is not {content_type}",
    )
    require(response.headers.get("x-accel-buffering") == "no", "buffering header")
    if AUDIT_BRANDING:
        require("server" not in response.headers, "stream server header exposed")
        check_public(dict(response.headers))


def check_result(payload: dict, feature: str | None = None) -> None:
    check_public(json.dumps(payload, ensure_ascii=False))
    require(payload.get("type") == "analysis_result", "wrong result type")
    require(payload.get("analysis_kind") == "fixed_five_events", "analysis kind")
    require(bool(payload.get("message_id")), "missing analysis message ID")
    frame = payload.get("frame", {})
    require((frame.get("width"), frame.get("height")) == (WIDTH, HEIGHT), "frame size")
    judgments = payload.get("judgments", [])
    expected = [feature] if feature else list(FEATURES)
    require([item.get("event_label") for item in judgments] == expected, "event labels")
    objects = payload.get("objects", [])
    object_ids = {item["object_id"] for item in objects}
    selected_ids = set()
    for judgment in judgments:
        require(isinstance(judgment.get("detected"), bool), "detected must be boolean")
        require(0 <= judgment.get("confidence", -1) <= 1, "confidence out of bounds")
        ids = set(judgment.get("object_ids", []))
        require(ids <= object_ids, "judgment references a missing object")
        require(not judgment["detected"] or bool(ids), "positive event has no evidence")
        selected_ids.update(ids)
    if feature:
        require(object_ids == selected_ids, f"{feature}: unrelated objects leaked")
    for item in objects:
        require(
            item.get("attributes", {}).get("dry_run") is True,
            "non-synthetic object",
        )
        require(
            item.get("bbox", {}).get("coordinate_system") == "source_normalized_xyxy",
            "wrong bounding-box coordinate system",
        )


def check_jpeg(data: bytes) -> None:
    require(data.startswith(b"\xff\xd8") and data.endswith(b"\xff\xd9"), "invalid JPEG")
    decoded = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    require(decoded is not None, "JPEG decode failed")
    require(decoded.shape[:2] == (HEIGHT, WIDTH), "decoded JPEG dimensions")


async def check_sse(client: httpx.AsyncClient, root: str, feature: str | None) -> None:
    detected_features: set[str] = set()
    async with client.stream("GET", root + "/events") as response:
        check_stream_headers(response, "text/event-stream")
        if feature:
            require(
                response.headers.get("x-result-feature") == feature,
                "SSE feature header",
            )
        event_name = ""
        event_id = ""
        async for line in response.aiter_lines():
            if line.startswith("id: "):
                event_id = line[4:]
            elif line.startswith("event: "):
                event_name = line[7:]
            elif line.startswith("data: "):
                require(
                    event_name == "analysis_result" and ":" in event_id,
                    "SSE envelope",
                )
                payload = json.loads(line[6:])
                check_result(payload, feature)
                detected_features.update(
                    item["event_label"]
                    for item in payload["judgments"]
                    if item["detected"]
                )
                if feature or detected_features == set(FEATURES):
                    return
    raise RuntimeError(f"{root}/events: missing synthetic results")


async def check_mjpeg(client: httpx.AsyncClient, root: str) -> None:
    async with client.stream("GET", root + "/stream.mjpeg") as response:
        check_stream_headers(response, "multipart/x-mixed-replace")
        boundary = response.headers["content-type"].split("boundary=", 1)[1]
        data = bytearray()
        async for chunk in response.aiter_bytes():
            data.extend(chunk)
            require(len(data) < 2_000_000, "MJPEG first part exceeds size limit")
            if b"\r\n\r\n" not in data:
                continue
            header_bytes, jpeg_tail = bytes(data).split(b"\r\n\r\n", 1)
            lines = header_bytes.decode("ascii").split("\r\n")
            require(lines[0] == "--" + boundary, "MJPEG boundary mismatch")
            headers = dict(line.lower().split(": ", 1) for line in lines[1:])
            require(headers.get("content-type") == "image/jpeg", "MJPEG part type")
            size = int(headers["content-length"])
            if len(jpeg_tail) < size:
                continue
            check_jpeg(jpeg_tail[:size])
            require("x-frame-sequence" in headers, "MJPEG frame sequence")
            if root == "/v1/video":
                require("x-analysis-message-id" in headers, "MJPEG analysis linkage")
            if root.endswith("annotated"):
                require("x-annotation-mode" in headers, "MJPEG annotation metadata")
            return
    raise RuntimeError(f"{root}/stream.mjpeg: missing JPEG part")


async def recv_json(socket) -> dict:
    response = json.loads(await asyncio.wait_for(socket.recv(), timeout=TIMEOUT))
    check_public(json.dumps(response, ensure_ascii=False))
    require(response.get("type") != "error", "WebSocket returned an error")
    return response


async def send_frames(socket) -> None:
    # Colored geometric shapes contain no people or personally identifying data.
    for index in range(65):
        pixels = np.full((HEIGHT, WIDTH, 3), 32, dtype=np.uint8)
        x = 15 + index % 90
        cv2.rectangle(pixels, (x, 35), (x + 55, 125), (40, 160, 220), -1)
        cv2.circle(pixels, (180, 75), 24, (180, 90, 40), -1)
        encoded, jpeg = cv2.imencode(".jpg", pixels)
        require(encoded, "could not encode synthetic frame")
        await asyncio.wait_for(
            socket.send(struct.pack(">HH", WIDTH, HEIGHT) + jpeg.tobytes()),
            timeout=TIMEOUT,
        )
        await asyncio.sleep(0.1)


async def check_rich(client: httpx.AsyncClient, session_id: str) -> None:
    deadline = time.monotonic() + TIMEOUT
    while True:
        response = await client.get("/v1/rich-labels/latest")
        require(response.status_code in {200, 404}, "rich latest HTTP failure")
        payload = response.json() if response.status_code == 200 else {}
        people = [
            item for item in payload.get("objects", []) if item.get("label") == "person"
        ]
        status = (await request(client, "/v1/status")).json()
        rich = status.get("rich_labeling", {})
        if (
            payload.get("frame_sequence", -1) >= 55
            and people
            and all(item.get("attributes") for item in people)
            and not rich.get("in_flight")
            and not rich.get("pending")
            and rich.get("last_frame_sequence") == payload.get("frame_sequence")
        ):
            break
        require(time.monotonic() < deadline, "rich result/attributes did not arrive")
        await asyncio.sleep(0.2)

    require(response.headers.get("cache-control") == "no-store", "rich JSON caching")
    require(payload.get("stream_session_id") == session_id, "rich session mismatch")
    require(55 <= payload["frame_sequence"] < 65, "rich source frame sequence")
    require(payload.get("timestamp_seconds", -1) >= 0, "rich timestamp")
    objects = payload.get("objects", [])
    track_ids = [item.get("track_id") for item in objects]
    require(all(track_ids), "rich object has no track ID")
    require(len(set(track_ids)) == len(track_ids), "duplicate rich track IDs")
    require(
        {item.get("label") for item in objects} == {"person", "bottle"},
        "unexpected deterministic worker labels",
    )
    for item in objects:
        bbox = item.get("bbox", {})
        require(
            0 <= bbox.get("x1", -1) < bbox.get("x2", -1) <= 1
            and 0 <= bbox.get("y1", -1) < bbox.get("y2", -1) <= 1,
            "rich bounding box is not normalized",
        )
    for person in people:
        attributes = person["attributes"]
        require(
            attributes.get("synthetic_protocol_test") is True,
            "rich smoke requires the isolated deterministic test worker",
        )
        require(attributes.get("estimated_age_range") == "18-60", "rich age decode")
        require(
            attributes.get("estimated_gender") == "female-presenting",
            "rich appearance decode",
        )

    require(status.get("rich_labeling_enabled") is True, "rich labeling disabled")
    require(status.get("stream_session_id") == session_id, "active session changed")
    require(rich.get("stream_session_id") == session_id, "rich stats session mismatch")
    require(rich.get("succeeded", 0) >= 2, "too few successful rich HTTP requests")
    require(rich.get("last_error") is None, "rich worker reported an error")
    require(rich.get("failed", 0) == 0, "rich HTTP requests failed")
    require(
        rich.get("last_frame_sequence") == payload["frame_sequence"],
        "rich stats/latest frame mismatch",
    )
    passed("rich JSON: session/frame, tracked objects, async attributes and HTTP stats")
    passed("rich verification uses synthetic models; no GPU/model inference verified")


async def positive_checks(
    client: httpx.AsyncClient, socket, *, with_rich: bool = False
) -> None:
    await asyncio.wait_for(
        socket.send(
            json.dumps({
                "type": "start",
                "mode": "camera",
                "analysis_mode": "multi",
                "dry_run": True,
                "features": list(FEATURES),
            })
        ),
        timeout=TIMEOUT,
    )
    session = await recv_json(socket)
    require(session.get("type") == "session", "camera session did not start")
    require(session.get("dry_run") is True, "session is not dry-run")
    require(session.get("features") == list(FEATURES), "session features")
    status = (await request(client, "/v1/status")).json()
    require(not status.get("webhook_enabled"), "webhook must be disabled")
    await send_frames(socket)

    # Let final asynchronous inference finish before comparing exact-frame IDs.
    deadline = time.monotonic() + TIMEOUT
    while True:
        response = await request(client, "/v1/results/latest")
        result = response.json()
        if result.get("frame", {}).get("source_frame_index", -1) >= 55:
            break
        require(time.monotonic() < deadline, "synthetic frames did not reach analysis")
        await asyncio.sleep(0.2)
    await asyncio.sleep(0.5)

    if with_rich:
        await check_rich(client, session["stream_session_id"])

    for index, root in enumerate(RESULT_ROOTS):
        feature = None if index == 0 else FEATURES[index - 1]
        response = await request(client, root + "/latest")
        check_result(response.json(), feature)
        require(response.headers.get("cache-control") == "no-store", "JSON caching")
        require(int(response.headers.get("x-result-cursor", "0")) > 0, "result cursor")
        require(bool(response.headers.get("x-result-stream")), "result stream marker")
        if feature:
            require(
                response.headers.get("x-result-feature") == feature,
                "feature header",
            )
        await asyncio.wait_for(check_sse(client, root, feature), timeout=TIMEOUT)
    passed("6 latest JSON + 6 SSE; all five synthetic event types and filtered objects")

    for root in VIDEO_ROOTS:
        response = await request(client, root + "/latest.jpg")
        require(response.headers.get("content-type") == "image/jpeg", "snapshot type")
        require(response.headers.get("cache-control") == "no-store", "snapshot caching")
        require("x-frame-sequence" in response.headers, "snapshot frame sequence")
        check_jpeg(response.content)
        if root == "/v1/video":
            latest = (await request(client, "/v1/results/latest")).json()
            require(
                response.headers.get("x-analysis-message-id") == latest["message_id"],
                "evidence snapshot and latest JSON are not correlated",
            )
        if root.endswith("annotated"):
            require("x-annotation-mode" in response.headers, "annotation mode missing")
            require(
                float(response.headers.get("x-stream-delay", "-1")) >= 0,
                "stream delay",
            )
        await asyncio.wait_for(check_mjpeg(client, root), timeout=TIMEOUT)
    passed("3 JPEG snapshots + 3 MJPEG streams; valid frames and analysis linkage")
    require(
        (await request(client, PHOTO_LIST)).json() == {"count": 0, "strangers": []},
        "dry-run unexpectedly created stranger evidence",
    )
    passed(
        "dry-run creates no real stranger photos "
        "(photo route positive storage not tested)"
    )


async def run(base_url: str, *, with_rich: bool = False) -> None:
    async with httpx.AsyncClient(
        base_url=base_url, timeout=TIMEOUT, trust_env=False
    ) as client:
        await request(client, "/health/live")
        if AUDIT_BRANDING:
            for doc_path in ("/docs", "/redoc"):
                document = await request(client, doc_path)
                check_public(document.text)
                require(
                    not re.search(r'(?:src|href)\s*=\s*[\"\'](?:https?:)?//', document.text),
                    "API reference requests external assets",
                )
            for public_path in ("/", "/v1/ui/info", "/v1/deployment/settings"):
                public_response = await request(client, public_path)
                check_public(public_response.text)
            invalid = await request(client, "/v1/results/events?after=PaddleRuntimeError", 422)
            check_public(invalid.text)
            passed("customer surfaces: neutral UI/docs/errors/headers without external assets")
        status = (await request(client, "/v1/status")).json()
        require(status.get("pipeline_state") == "idle", "requires a fresh idle service")
        require(status.get("dry_run") is True, "service default must be dry-run")
        require(
            bool(status.get("rich_labeling_enabled")) == with_rich,
            "rich labeling must match the --with-rich option",
        )
        require(not status.get("webhook_enabled"), "webhook must be disabled")
        paths = (await request(client, "/openapi.json")).json()["paths"]
        required = [
            root + suffix
            for root in RESULT_ROOTS
            for suffix in ("/latest", "/events")
        ]
        required += [
            root + suffix
            for root in VIDEO_ROOTS
            for suffix in ("/latest.jpg", "/stream.mjpeg")
        ]
        required += [PHOTO_LIST, PHOTO_LIST + "/{stranger_id}/photo"]
        if with_rich:
            required.append("/v1/rich-labels/latest")
        for path in required:
            require("get" in paths.get(path, {}), f"missing OpenAPI GET route: {path}")
        passed(f"OpenAPI: all {len(required)} requested GET routes exist")

        if with_rich:
            await missing(client, "/v1/rich-labels/latest", "no rich labels yet")

        for root in RESULT_ROOTS:
            await missing(client, root + "/latest", "no analysis result yet")
        for root, detail in zip(
            VIDEO_ROOTS,
            (
                "no analyzed frame yet",
                "no live source frame yet",
                "no delayed annotated frame yet",
            ),
            strict=True,
        ):
            await missing(client, root + "/latest.jpg", detail)
        require(
            (await request(client, PHOTO_LIST)).json() == {"count": 0, "strangers": []},
            "requires an empty isolated stranger gallery",
        )
        await missing(client, PHOTO_MISSING, "stranger photo not found")
        passed(
            "idle: 9 result/snapshot 404 details; "
            "empty strangers 200; missing photo 404"
        )

        for root in RESULT_ROOTS + VIDEO_ROOTS:
            is_result = root in RESULT_ROOTS
            path = root + ("/events" if is_result else "/stream.mjpeg")
            media = "text/event-stream" if is_result else "multipart/x-mixed-replace"
            async with client.stream("GET", path) as response:
                check_stream_headers(response, media)
        passed("idle: all 6 SSE + 3 MJPEG response headers return HTTP 200")

        ws_url = base_url.replace("http://", "ws://", 1) + "/ws"
        async with websockets.connect(
            ws_url, open_timeout=TIMEOUT, close_timeout=3, max_size=1_000_000
        ) as socket:
            require((await recv_json(socket)).get("type") == "hello", "WebSocket hello")
            try:
                await positive_checks(client, socket, with_rich=with_rich)
            finally:
                await asyncio.wait_for(socket.send('{"type":"stop"}'), timeout=TIMEOUT)
                require(
                    (await recv_json(socket)).get("type") == "ended",
                    "session stop",
                )
        final_status = (await request(client, "/v1/status")).json()
        require(
            final_status.get("pipeline_state") == "idle",
            "service did not return to idle",
        )
        passed("synthetic session stopped; service returned to idle")


def main() -> int:
    global AUDIT_BRANDING
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8080")
    parser.add_argument("--audit-branding", action="store_true")
    parser.add_argument(
        "--with-rich",
        action="store_true",
        help="Also verify a connected isolated deterministic rich-label test worker.",
    )
    parser.add_argument(
        "--isolated",
        action="store_true",
        required=True,
        help="Confirm this is a disposable isolated container.",
    )
    args = parser.parse_args()
    AUDIT_BRANDING = args.audit_branding
    parsed = urlsplit(args.base_url)
    require(
        parsed.scheme == "http"
        and parsed.hostname in {"127.0.0.1", "localhost", "::1"}
        and not parsed.username
        and not parsed.password
        and not parsed.query
        and not parsed.fragment
        and parsed.path in {"", "/"},
        "only a plain local container HTTP endpoint is allowed",
    )
    started = time.monotonic()
    try:
        asyncio.run(
            asyncio.wait_for(
                run(args.base_url.rstrip("/"), with_rich=args.with_rich), timeout=110
            )
        )
    except Exception as exc:
        print(f"FAIL  {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        return 1
    print(
        f"PASS  delivery smoke completed in {time.monotonic() - started:.1f}s",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
