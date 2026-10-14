"""View-only detector transport and source-time freshness, without model calls."""

from __future__ import annotations

import base64
import json
import os
import shutil
import subprocess
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import httpx
import pytest

from tests.test_vlm_monitor import args, job, result
from tests.test_vlm_preview_tracking import encoded, scene
from tools.vlm_benchmark.monitor import LiveMonitor

BOX = {"track_id": "person-1", "bbox": {"x1": 0.1, "y1": 0.1, "x2": 0.3, "y2": 0.8}}


@pytest.fixture
def detector(monkeypatch):
    requests = []
    gate, entered = threading.Event(), threading.Event()
    gate.set()
    response = {"objects": [BOX]}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def do_POST(self):
            requests.append(
                (
                    self.path,
                    {key.lower(): value for key, value in self.headers.items()},
                    self.rfile.read(int(self.headers["Content-Length"])),
                )
            )
            entered.set()
            assert gate.wait(timeout=5)
            data = json.dumps(response).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    monkeypatch.setenv(
        "PREVIEW_DETECTOR_URL", f"http://127.0.0.1:{server.server_port}/"
    )
    monkeypatch.setenv("PREVIEW_DETECTOR_MODEL", "fake-offline")
    try:
        yield requests, response, gate, entered
    finally:
        gate.set()
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def payload(at=0, epoch=0, client=None):
    return {
        "client": client or str(uuid.uuid4()),
        "time": at,
        "epoch": epoch,
        "jpeg": base64.b64encode(encoded(scene(0)[0])).decode(),
    }


def monitor_for(tmp_path):
    monitor = LiveMonitor(tmp_path, [[encoded(scene(0)[0])] * 4], args())
    monitor.previews = [None]
    monitor.begin({**job(), "mode": "events"}, tmp_path)
    monitor.source_started(time.monotonic(), [0])
    return monitor


def test_preview_detector_url_uses_own_frame_not_latest_benchmark_verdict(
    tmp_path, detector
):
    requests, _, _, _ = detector
    monitor = monitor_for(tmp_path)
    try:
        alert = result(3, "events")
        alert["judgments"][0]["observed_detected"] = True
        monitor.result(0, alert)
        request = payload(0)
        response = monitor.track_preview("stage-one", 0, request)
        assert response == {
            "detector": True,
            "source_time": 0,
            "epoch": 0,
            "status": "tracking",
            "objects": [BOX],
        }
        path, headers, jpeg = requests[0]
        assert path == "/v1/tracks"
        assert headers["x-goodvision-detector-model"] == "fake-offline"
        assert headers["x-goodvision-frame-sequence"] == "1"
        assert jpeg == base64.b64decode(request["jpeg"])
        assert monitor.snapshot()["counters"]["events"]["complete"] == 1
        assert "event_state" not in response
        assert "event_state" not in response["objects"][0]
    finally:
        monitor.log.close()


def test_detector_loop_restarts_session_and_does_not_rewind_transport_clock(
    tmp_path, detector
):
    requests, _, _, _ = detector
    monitor = monitor_for(tmp_path)
    client = str(uuid.uuid4())
    try:
        for at, epoch in ((1.4, 0), (1.5, 0), (1.65, 1)):
            response = monitor.track_preview("stage-one", 0, payload(at, epoch, client))
            assert (response["source_time"], response["epoch"]) == (at, epoch)
        headers = [request[1] for request in requests]
        assert [h["x-goodvision-frame-sequence"] for h in headers] == ["1", "2", "1"]
        sessions = [h["x-goodvision-stream-session-id"] for h in headers]
        assert sessions[0] == sessions[1] and sessions[2] != sessions[1]
        times = [float(h["x-goodvision-timestamp-seconds"]) for h in headers]
        assert times == sorted(set(times))
        assert len(monitor._detector_seq) == 1
    finally:
        monitor.log.close()


def test_delayed_detector_response_is_empty_after_run_stops(tmp_path, detector):
    requests, _, gate, entered = detector
    monitor = monitor_for(tmp_path)
    gate.clear()
    replies = []
    thread = threading.Thread(
        target=lambda: replies.append(monitor.track_preview("stage-one", 0, payload()))
    )
    try:
        thread.start()
        assert entered.wait(timeout=2)
        monitor.draining()
        gate.set()
        thread.join(timeout=3)
        assert not thread.is_alive()
        assert replies[0]["status"] == "stopped" and not replies[0]["objects"]
        assert monitor.track_preview("stage-one", 0, payload())["status"] == "stopped"
        assert len(requests) == 1
    finally:
        gate.set()
        thread.join(timeout=3)
        monitor.log.close()


def test_delayed_detector_response_cannot_cross_stage_boundary(tmp_path, detector):
    _, _, gate, entered = detector
    monitor = monitor_for(tmp_path)
    gate.clear()
    replies = []
    thread = threading.Thread(
        target=lambda: replies.append(monitor.track_preview("stage-one", 0, payload()))
    )
    try:
        thread.start()
        assert entered.wait(timeout=2)
        next_stage = tmp_path / "next-stage"
        next_stage.mkdir()
        monitor.begin({**job(), "id": "stage-two", "mode": "events"}, next_stage)
        monitor.source_started(time.monotonic(), [0])
        gate.set()
        thread.join(timeout=3)
        assert not thread.is_alive()
        assert replies[0]["status"] == "stopped" and not replies[0]["objects"]
        fresh = monitor.track_preview("stage-two", 0, payload())
        assert fresh["status"] == "tracking" and fresh["objects"] == [BOX]
    finally:
        gate.set()
        thread.join(timeout=3)
        monitor.log.close()


def test_detector_timeout_returns_empty_with_explicit_provenance(
    tmp_path, detector, monkeypatch
):
    monitor = monitor_for(tmp_path)

    def timeout(*_, **__):
        raise TimeoutError("fake detector timeout")

    monkeypatch.setattr("tools.vlm_benchmark.monitor.urllib.request.urlopen", timeout)
    try:
        response = monitor.track_preview("stage-one", 0, payload(0.8))
        assert response == {
            "detector": True,
            "source_time": 0.8,
            "epoch": 0,
            "status": "detector_unavailable",
            "objects": [],
        }
    finally:
        monitor.log.close()


def test_detector_filters_malformed_boxes_and_failures(tmp_path, detector):
    _, response, _, _ = detector
    response["objects"] = [
        BOX,
        None,
        {"track_id": "nan", "bbox": {"x1": float("nan")}},
        {"track_id": 4, "bbox": BOX["bbox"]},
    ]
    monitor = monitor_for(tmp_path)
    try:
        assert monitor.track_preview("stage-one", 0, payload())["objects"] == [BOX]
        response["objects"] = None
        assert monitor.track_preview("stage-one", 0, payload())["objects"] == []
    finally:
        monitor.log.close()


def test_detector_preview_route_remains_view_only(tmp_path, detector):
    with LiveMonitor(tmp_path, [[encoded(scene(0)[0])] * 4], args()) as monitor:
        stage = tmp_path / "stage"
        stage.mkdir()
        monitor.begin({**job(), "mode": "events"}, stage)
        monitor.source_started(time.monotonic(), [0])
        with httpx.Client(base_url=monitor.url, trust_env=False) as client:
            response = client.post("tracking/stage-one/0", json=payload())
        assert response.status_code == 200
        assert response.json()["objects"] == [BOX]
        assert monitor.snapshot()["counters"]["events"]["complete"] == 0


def test_detector_browser_freshness_stop_and_loop_with_fake_clock():
    node = shutil.which("node")
    if not node:
        pytest.skip("Node.js is needed for the deterministic browser callback test")
    script = Path(__file__).with_name("vlm_preview_tracking.test.cjs")
    result = subprocess.run(
        [node, "--test", str(script)], capture_output=True, text=True, timeout=15
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.skipif(
    os.environ.get("VLM_MONITOR_BROWSER_TESTS") != "1",
    reason="opt-in installed Chromium integration",
)
def test_real_browser_keeps_slow_detector_evidence_and_clears_on_loop_stop(
    tmp_path, detector
):
    from fractions import Fraction
    from io import BytesIO

    import av
    from PIL import Image
    from playwright.sync_api import sync_playwright

    from tools.vlm_multistream_benchmark import DecodedFrames

    requests, _, gate, entered = detector
    video = tmp_path / "detector-source.mp4"
    images = [
        Image.new("RGB", (320, 180), (30 + n * 3, 50 + n * 2, 180 - n * 2))
        for n in range(48)
    ]
    with av.open(str(video), "w") as writer:
        stream = writer.add_stream("libx264", rate=15)
        stream.width, stream.height, stream.pix_fmt = 320, 180, "yuv420p"
        for n, image in enumerate(images):
            frame = av.VideoFrame.from_image(image)
            frame.pts, frame.time_base = n, Fraction(1, 15)
            for packet in stream.encode(frame):
                writer.mux(packet)
        for packet in stream.encode():
            writer.mux(packet)
    samples = []
    for n in range(8):
        output = BytesIO()
        images[n * 6].save(output, format="JPEG")
        samples.append(output.getvalue())
    frames = DecodedFrames(samples, str(video), [n * 0.4 for n in range(8)], 3.2)
    output = tmp_path / "monitor"
    output.mkdir()
    gate.clear()
    with LiveMonitor(output, [frames], args()) as monitor:
        stage = tmp_path / "stage"
        stage.mkdir()
        monitor.begin({**job(), "mode": "events", "duration": 8}, stage)
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(
                args=["--autoplay-policy=no-user-gesture-required"]
            )
            page = browser.new_page(viewport={"width": 1440, "height": 1000})
            errors = []
            page.on("pageerror", lambda error: errors.append(str(error)))
            page.goto(monitor.url)
            page.wait_for_function("window.monitorReady")
            monitor.source_started(time.monotonic(), [0])
            assert entered.wait(timeout=3)
            # Delay a valid response beyond the current-frame overlay limit,
            # then hold the following response to inspect this captured frame.
            page.wait_for_timeout(950)
            gate.set()
            gate.clear()
            page.wait_for_function(
                "window.monitorDiagnostics.tracking[0]?.lastStatus === 'evidence_only'"
            )
            diagnostic = page.evaluate("window.monitorDiagnostics.tracking[0]")
            assert diagnostic["evidenceFramesWithBoxes"] >= 1
            assert diagnostic["evidenceSamples"][0]["age"] >= 0.7
            assert page.locator(".detector-evidence").is_visible()
            assert "YOLO 证据" in page.locator(".detector-evidence-note").inner_text()
            assert page.locator(".tracking-overlay").evaluate(
                "canvas => canvas.getContext('2d')"
                ".getImageData(0,0,canvas.width,canvas.height)"
                ".data.every(v => v === 0)"
            )
            shown = page.locator(".detector-evidence canvas").evaluate(
                "canvas => Array.from(canvas.getContext('2d')"
                ".getImageData(200,40,1,1).data).slice(0,3)"
            )
            sent = (
                Image.open(BytesIO(requests[0][2])).convert("RGB").getpixel((200, 40))
            )
            assert all(abs(a - b) <= 5 for a, b in zip(shown, sent, strict=True))
            live = page.locator("video").evaluate(
                "video => {const c=document.createElement('canvas');"
                "c.width=320;c.height=180;const x=c.getContext('2d');"
                "x.drawImage(video,0,0,320,180);"
                "return Array.from(x.getImageData(200,40,1,1).data).slice(0,3);}"
            )
            assert max(abs(a - b) for a, b in zip(shown, live, strict=True)) > 20
            if folder := os.environ.get("VLM_TRACKING_REVIEW_OUTPUT"):
                destination = Path(folder)
                destination.mkdir(parents=True, exist_ok=True)
                page.screenshot(path=str(destination / "detector-slow-evidence.png"))
            # Seek near the end and let the real video loop naturally. The
            # old image disappears immediately and new evidence has epoch 1.
            page.locator("video").evaluate("video => {video.currentTime=2.8;}")
            page.wait_for_function(
                "document.querySelector('.detector-evidence').hidden"
            )
            gate.set()
            page.wait_for_function(
                "window.monitorDiagnostics.tracking[0]"
                "?.evidenceSamples.some(s => s.epoch >= 1)",
                timeout=5000,
            )
            assert "第 2 轮" in page.locator(".detector-evidence-note").inner_text()
            monitor.draining()
            monitor.finish(
                {
                    "lanes": {},
                    "input_timing_valid": True,
                    "all_drained": True,
                    "result_records_missing": 0,
                }
            )
            page.wait_for_function("window.finalPainted", timeout=5000)
            assert page.locator("video").evaluate("video => video.paused")
            assert not page.locator(".detector-evidence").is_visible()
            assert page.locator(".tracking-overlay").evaluate(
                "canvas => canvas.getContext('2d')"
                ".getImageData(0,0,canvas.width,canvas.height)"
                ".data.every(v => v === 0)"
            )
            assert monitor.snapshot()["counters"]["events"]["complete"] == 0
            assert not errors
            assert not page.evaluate("window.monitorDiagnostics.mediaErrors")
            browser.close()
