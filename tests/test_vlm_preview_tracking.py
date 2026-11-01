"""Known-identity motion, delayed anchors and fail-closed preview tracking."""

from __future__ import annotations

import base64
import os
import shutil
import subprocess
import uuid
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from tools.vlm_benchmark import preview_tracking
from tools.vlm_benchmark.preview_tracking import PreviewTracker, iou


def test_detector_preview_browser_logic():
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is required for detector preview regression tests")
    completed = subprocess.run(
        [node, "--test", str(Path(__file__).with_name("preview_detector.cjs"))],
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=30,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


def encoded(image):
    return cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 95])[1].tobytes()


def scene(step, *, crossing=False, occluded=False):
    image = np.full((180, 320), 32, np.uint8)
    boxes = []
    for person, x in enumerate((25 + 2 * step, 225 - 2 * step if crossing else 225)):
        box = (x, 35, x + 38, 135)
        texture = np.random.default_rng(person + 70).integers(
            40, 235, (100, 38), np.uint8
        )
        image[35:135, x : x + 38] = texture
        boxes.append(box)
    if occluded:
        image[25:145, 10:310] = 110
    return image, boxes


def anchor(image, boxes, at=0, name="seed"):
    return {
        "id": name,
        "time": at,
        "jpeg": encoded(image),
        "objects": [
            {
                "label": "person",
                "bbox": dict(
                    zip(
                        ("x1", "y1", "x2", "y2"),
                        (np.array(b) / (320, 180, 320, 180)).tolist(),
                        strict=True,
                    )
                ),
            }
            for b in boxes
        ],
    }


def as_pixels(obj):
    b = obj["bbox"]
    return np.array([b[k] for k in ("x1", "y1", "x2", "y2")]) * (320, 180, 320, 180)


def test_delayed_anchor_replays_original_frame_and_keeps_both_people(monkeypatch):
    # Verify optical-flow identity independently of the CI host's CPU speed.
    # A separate controlled-clock test below verifies the replay time budget.
    monkeypatch.setattr(
        preview_tracking, "time", SimpleNamespace(perf_counter=lambda: 0.0)
    )
    tracker = PreviewTracker()
    image, boxes = scene(0)
    seed = anchor(image, boxes)
    for n in range(46):
        frame, expected = scene(n)
        r = tracker.advance(frame, n / 15, anchor=seed if n >= 30 else None)
        if n >= 30:
            assert len(r["objects"]) == 2
            for index, obj in enumerate(r["objects"]):
                assert obj["track_id"] == f"seed:{index}"
                assert iou(as_pixels(obj), expected[index]) > 0.95
    assert r["anchor_age"] == 3


def test_delayed_anchor_hides_boxes_when_replay_exceeds_budget(monkeypatch):
    now = [0.0]
    monkeypatch.setattr(
        preview_tracking, "time", SimpleNamespace(perf_counter=lambda: now[0])
    )
    tracker = PreviewTracker()
    image, boxes = scene(0)
    seed = anchor(image, boxes)
    for n in range(31):
        tracker.advance(scene(n)[0], n / 15)
    step = tracker._step

    def slow_step(*args):
        now[0] += 0.13
        return step(*args)

    monkeypatch.setattr(tracker, "_step", slow_step)
    result = tracker.advance(scene(31)[0], 31 / 15, anchor=seed)
    assert result["objects"] == []
    assert result["status"] == "lost"
    assert result["replayed_frames"] == 1
    # A rejected delayed seed must stay hidden until a new localization arrives.
    assert tracker.advance(scene(32)[0], 32 / 15, anchor=seed)["objects"] == []
    fresh_image, fresh_boxes = scene(33)
    fresh = anchor(fresh_image, fresh_boxes, 33 / 15, "fresh")
    recovered = tracker.advance(fresh_image, 33 / 15, anchor=fresh)
    assert len(recovered["objects"]) == 2
    assert all(o["track_id"].startswith("fresh:") for o in recovered["objects"])


def test_crossing_drops_ambiguous_boxes_instead_of_swapping_identities():
    tracker = PreviewTracker()
    image, boxes = scene(0, crossing=True)
    seed = anchor(image, boxes)
    seen, hidden = 0, 0
    for n in range(70):
        frame, expected = scene(n, crossing=True)
        r = tracker.advance(frame, n / 15, anchor=seed)
        for obj in r["objects"]:
            person = int(obj["track_id"].split(":")[-1])
            assert iou(as_pixels(obj), expected[person]) > 0.85
            seen += 1
        hidden += not r["objects"]
    assert seen > 30  # Can't pass by never showing boxes.
    assert hidden > 10
    assert r["status"] == "lost"


def test_occlusion_and_wrong_appearance_hide_boxes_until_fresh_localization():
    tracker = PreviewTracker()
    image, boxes = scene(0)
    seed = anchor(image, boxes)
    assert len(tracker.advance(image, 0, anchor=seed)["objects"]) == 2
    frame, _ = scene(1, occluded=True)
    assert tracker.advance(frame, 1 / 15, anchor=seed)["objects"] == []
    frame, boxes = scene(2)
    assert tracker.advance(frame, 2 / 15, anchor=seed)["objects"] == []
    refreshed = anchor(frame, boxes, 2 / 15, "new-seed")
    r = tracker.advance(*[scene(3)[0], 3 / 15], anchor=refreshed)
    assert len(r["objects"]) == 2
    assert all(o["track_id"].startswith("new-seed:") for o in r["objects"])


@pytest.mark.parametrize("timestamp,epoch", [(1.0, 0), (0.0, 0), (0.1, 1)])
def test_seek_gap_or_loop_never_carries_boxes_forward(timestamp, epoch):
    image, boxes = scene(0)
    seed, tracker = anchor(image, boxes), PreviewTracker()
    tracker.advance(image, 0, anchor=seed)
    r = tracker.advance(scene(1)[0], timestamp, epoch=epoch, anchor=seed)
    assert r["objects"] == []
    assert r["reason"] == "source_discontinuity"


def test_expired_future_and_out_of_order_anchors_cannot_become_current():
    image, boxes = scene(0)
    tracker = PreviewTracker()
    assert tracker.advance(image, 10, anchor=anchor(image, boxes))["objects"] == []
    assert (
        tracker.advance(image, 10.1, anchor=anchor(image, boxes, 20, "future"))[
            "objects"
        ]
        == []
    )
    assert tracker.advance(image, 10.2, anchor=anchor(image, boxes, 10.2, "fresh"))[
        "objects"
    ]
    r = tracker.advance(image, 10.3, anchor=anchor(image, boxes, 9, "old"))
    assert r["anchor_id"] == "fresh"
    for n in range(130):
        r = tracker.advance(image, 10.4 + n / 15)
    assert not r["objects"]
    assert r["reason"] == "grounding_expired"
    assert r["history_frames"] <= 121


def test_monitor_tracking_is_separate_from_inference_and_failed_results(tmp_path):
    from http.client import HTTPConnection
    from urllib.parse import urlsplit

    import httpx

    from tests.test_vlm_monitor import args, job, result
    from tools.vlm_benchmark.monitor import LiveMonitor

    image, boxes = scene(0)
    with LiveMonitor(tmp_path, [[encoded(image)] * 4], args()) as monitor:
        stage = tmp_path / "stage"
        stage.mkdir()
        monitor.begin(job(), stage)
        report = result(0, "grounding")
        report["objects"] = anchor(image, boxes)["objects"]
        monitor.result(0, report)
        monitor.result(0, result(1, "grounding", error="timeout"))
        payload = {
            "client": str(uuid.uuid4()),
            "time": 0,
            "epoch": 0,
            "jpeg": base64.b64encode(encoded(image)).decode(),
        }
        with httpx.Client(base_url=monitor.url, trust_env=False) as client:
            response = client.post("tracking/stage-one/0", json=payload)
            assert response.status_code == 200
            assert len(response.json()["objects"]) == 2
            assert client.post("tracking/old-stage/0", json=payload).status_code == 400
            assert client.post("tracking/stage-one/99", json=payload).status_code == 400
            assert (
                client.post(
                    "tracking/stage-one/0", json={**payload, "time": -1}
                ).status_code
                == 400
            )
        # Reject oversized uploads from their headers, before reading the body.
        # Sending the body too can reset the HTTP/1.0 connection on Windows.
        target = urlsplit(monitor.url)
        connection = HTTPConnection(target.hostname, target.port, timeout=2)
        try:
            connection.putrequest("POST", target.path + "tracking/stage-one/0")
            connection.putheader("Content-Type", "application/json")
            connection.putheader("Content-Length", "200000")
            connection.endheaders()
            response = connection.getresponse()
            assert response.status == 400
            assert response.read() == b'{"error":"invalid tracking request"}'
        finally:
            connection.close()
        snapshot = monitor.snapshot()
        assert snapshot["cameras"][0]["tracking_anchor"]["tick"] == 0
        assert snapshot["counters"]["grounding"]["complete"] == 1


@pytest.mark.skipif(
    os.environ.get("VLM_MONITOR_BROWSER_TESTS") != "1",
    reason="opt-in installed Chromium integration",
)
@pytest.mark.parametrize("streams", [1, 12])
@pytest.mark.parametrize("tracking_query", [None, "off", "on"])
def test_live_tracking_browser_keeps_correct_boxes_and_bounded_playback(
    tmp_path, streams, tracking_query
):
    import json
    import os
    import time
    from fractions import Fraction
    from pathlib import Path

    import av
    from PIL import Image
    from playwright.sync_api import sync_playwright

    from tests.test_vlm_monitor import args, job, result
    from tools.vlm_benchmark.monitor import LiveMonitor
    from tools.vlm_multistream_benchmark import DecodedFrames

    tracking_enabled = tracking_query == "on"
    video = tmp_path / "moving.mp4"
    # No overlap here: every visible estimate can be checked against identity GT.
    with av.open(str(video), "w") as writer:
        out = writer.add_stream("libx264", rate=15)
        out.width, out.height, out.pix_fmt = 320, 180, "yuv420p"
        for n in range(90):
            image, _ = scene(n)
            frame = av.VideoFrame.from_image(Image.fromarray(image).convert("RGB"))
            frame.pts, frame.time_base = n, Fraction(1, 15)
            for packet in out.encode(frame):
                writer.mux(packet)
        for packet in out.encode():
            writer.mux(packet)
    samples = DecodedFrames(
        [encoded(scene(n * 6)[0]) for n in range(15)],
        str(video),
        [n * 0.4 for n in range(15)],
        6.0,
    )
    (tmp_path / "monitor").mkdir()
    with LiveMonitor(tmp_path / "monitor", [samples], args()) as monitor:
        location = tmp_path / "stage"
        location.mkdir()
        monitor.begin({**job(), "streams": streams, "duration": 5.5}, location)
        with sync_playwright() as p:
            browser = p.chromium.launch(
                args=["--autoplay-policy=no-user-gesture-required"]
            )
            page = browser.new_page(viewport={"width": 1920, "height": 1080})
            errors, tracking_requests = [], []
            page.on("pageerror", lambda e: errors.append(str(e)))
            page.on(
                "request",
                lambda request: tracking_requests.append(request.url)
                if "/tracking/" in request.url
                else None,
            )
            query = "" if tracking_query is None else f"?tracking={tracking_query}"
            page.goto(monitor.url + query)
            page.wait_for_function("window.monitorReady")
            monitor.source_started(time.monotonic(), [0] * streams)
            page.wait_for_timeout(1800)
            r = result(2, "grounding")
            r["objects"] = anchor(*scene(12))["objects"]
            for camera in range(streams):
                monitor.result(camera, r)
            # Keep publishing evidence-aligned localizations, as a running VLM
            # does. A single delayed seed may correctly exceed the replay budget
            # on a loaded host; requiring it to survive forever tests CPU speed,
            # not browser identity/display correctness. Keep the real budget and
            # the strict per-camera box/IoU assertions, and exercise recovery.
            recovery_anchors = set()
            for _ in range(3):
                page.wait_for_timeout(800)
                media_times = page.locator(".camera video").evaluate_all(
                    "videos => videos.map(video => video.currentTime)"
                )
                tick = min(len(samples) - 1, int(min(media_times) / 0.4))
                fresh = result(tick, "grounding")
                fresh["objects"] = anchor(*scene(tick * 6))["objects"]
                if tick > r["frame"]["source_frame_index"]:
                    recovery_anchors.add(f"{tick}:{fresh['message_id']}")
                for camera in range(streams):
                    monitor.result(camera, fresh)
            page.wait_for_timeout(600)
            d = page.evaluate("window.monitorDiagnostics")
            suffix = "" if tracking_enabled else f"-{tracking_query or 'default'}"
            if folder := os.environ.get("VLM_TRACKING_REVIEW_OUTPUT"):
                destination = Path(folder)
                destination.mkdir(parents=True, exist_ok=True)
                (destination / f"browser-{streams}{suffix}.json").write_text(
                    json.dumps(d, indent=2), encoding="utf-8"
                )
                page.screenshot(
                    path=str(destination / f"browser-{streams}{suffix}.png")
                )
            assert not errors and not d["mediaErrors"]
            assert d["trackingEnabled"] is tracking_enabled
            assert bool(tracking_requests) is tracking_enabled
            assert page.locator(".tracking-overlay, .tracking-note").count() == (
                streams * 2 if tracking_enabled else 0
            )
            if not tracking_enabled:
                assert monitor.preview_trackers == {}
            assert len(d["tracking"]) == streams
            for tracking in d["tracking"]:
                if not tracking_enabled:
                    assert tracking is None
                    continue
                assert tracking["errors"] == 0
                assert tracking["framesWithBoxes"] >= 5
                # Every lane must recover on published post-initial evidence.
                # A later source gap may correctly hide its boxes again; the
                # final seed need not survive the replay/response budgets.
                assert any(
                    sample["anchor"] in recovery_anchors and sample["objects"]
                    for sample in tracking["samples"]
                )
                for sample in tracking["samples"]:
                    _, expected = scene(round(sample["time"] * 15))
                    for obj in sample["objects"]:
                        person = int(obj["track_id"].split(":")[-1])
                        assert iou(as_pixels(obj), expected[person]) > 0.8
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
            if folder:
                page.screenshot(
                    path=str(destination / f"browser-{streams}{suffix}-final.png")
                )
            browser.close()


def test_person_exiting_side_does_not_leave_a_background_track():
    tracker = PreviewTracker()
    first, boxes = scene(0)
    seed = anchor(first, [boxes[0]])
    kept = 0
    for n in range(40):
        # Shift the entire textured subject off the left side of the image.
        image = np.full_like(first, 32)
        x = 25 - 2 * n
        width = min(38, x + 38)
        if width > 0:
            image[35:135, max(x, 0) : x + 38] = first[35:135, 25 + 38 - width : 25 + 38]
        r = tracker.advance(image, n / 15, anchor=seed)
        kept += bool(r["objects"])
        if x < 4:
            assert not r["objects"]
    assert kept > 5
    clipped = anchor(scene(0)[0], [(0, 0, 95, 120)], 3, "side-clipped")
    assert not tracker.advance(scene(0)[0], 3, anchor=clipped)["objects"]


@pytest.mark.parametrize(
    "evidence_size,preview_size",
    [
        ((405, 720), (404, 720)),  # 1080x1920: thumbnail vs even H.264 width.
        ((720, 375), (720, 374)),  # 1920x1000: the height is rounded instead.
        ((364, 720), (362, 720)),  # Near 1:2, rounding spans three tracker rows.
        ((720, 191), (720, 190)),  # One row exceeds 1% for a wide landscape.
    ],
)
def test_delayed_anchor_tolerates_proxy_dimension_rounding(evidence_size, preview_size):
    texture = np.random.default_rng(23).integers(20, 235, (720, 720), np.uint8)
    evidence = cv2.resize(texture, evidence_size, interpolation=cv2.INTER_AREA)
    preview = cv2.resize(texture, preview_size, interpolation=cv2.INTER_AREA)
    frame = preview_tracking.gray_frame(encoded(preview))
    seed = {
        "id": "rounded-proxy",
        "time": 0,
        "jpeg": encoded(evidence),
        "objects": [
            {
                "label": "person",
                "bbox": {"x1": 0.25, "y1": 0.2, "x2": 0.75, "y2": 0.8},
            }
        ],
    }
    tracker = PreviewTracker()
    tracker.advance(frame, 0)
    tracker.advance(frame, 1 / 15)
    result = tracker.advance(frame, 2 / 15, anchor=seed)
    assert result["status"] == "tracking"
    assert len(result["objects"]) == 1
    assert result["replayed_frames"] == 2
    assert result["objects"][0]["bbox"] == pytest.approx(
        seed["objects"][0]["bbox"], abs=0.005
    )
    assert tracker.advance(frame, 3 / 15, anchor=seed)["objects"]


def test_anchor_with_incompatible_aspect_ratio_stays_hidden():
    texture = np.random.default_rng(23).integers(20, 235, (240, 320), np.uint8)
    seed = anchor(texture, [(80, 36, 240, 144)])
    frame = cv2.resize(texture, (320, 180))
    result = PreviewTracker().advance(frame, 0, anchor=seed)
    assert result["objects"] == []
    assert result["status"] == "lost"

    tracker = PreviewTracker()
    seed = anchor(*scene(0))
    assert tracker.advance(scene(0)[0], 0, anchor=seed)["objects"]
    result = tracker.advance(texture, 1 / 15, anchor=seed)
    assert result["objects"] == []
    assert result["reason"] == "source_discontinuity"


@pytest.mark.skipif(
    os.environ.get("VLM_MONITOR_BROWSER_TESTS") != "1",
    reason="opt-in installed Chromium integration",
)
def test_portrait_source_tracks_through_real_thumbnail_and_browser_proxy(tmp_path):
    import time
    from fractions import Fraction

    import av
    from PIL import Image
    from playwright.sync_api import sync_playwright

    from tests.test_vlm_monitor import args, job, result
    from tools.vlm_benchmark.monitor import LiveMonitor
    from tools.vlm_multistream_benchmark import decode

    image = np.full((320, 180), 32, np.uint8)
    image[60:260, 40:140] = np.random.default_rng(23).integers(
        40, 235, (200, 100), np.uint8
    )
    source = tmp_path / "portrait.mp4"
    with av.open(str(source), "w") as writer:
        stream = writer.add_stream("libx264", rate=15)
        stream.width, stream.height, stream.pix_fmt = 1080, 1920, "yuv420p"
        stream.options = {"preset": "ultrafast", "crf": "18"}
        original = Image.fromarray(image).resize((1080, 1920)).convert("RGB")
        for n in range(45):
            frame = av.VideoFrame.from_image(original)
            frame.pts, frame.time_base = n, Fraction(1, 15)
            for packet in stream.encode(frame):
                writer.mux(packet)
        for packet in stream.encode():
            writer.mux(packet)
    frames = decode(source, 2.5, 720, 90)
    assert preview_tracking.gray_frame(frames[0]).shape == (569, 320)
    destination = tmp_path / "monitor"
    destination.mkdir()
    with LiveMonitor(destination, [frames], args()) as monitor:
        stage = tmp_path / "stage"
        stage.mkdir()
        monitor.begin({**job(), "duration": 2.5}, stage)
        with sync_playwright() as p:
            browser = p.chromium.launch(
                args=["--autoplay-policy=no-user-gesture-required"]
            )
            page = browser.new_page(viewport={"width": 1920, "height": 1080})
            page.goto(monitor.url + "?tracking=on")
            page.wait_for_function("window.monitorReady")
            assert page.locator("video").evaluate(
                "v => [v.videoWidth, v.videoHeight]"
            ) == [404, 720]
            report = result(0, "grounding")
            bbox = {"x1": 40 / 180, "y1": 60 / 320, "x2": 140 / 180, "y2": 260 / 320}
            report["objects"] = [{"label": "person", "bbox": bbox}]
            monitor.result(0, report)
            monitor.source_started(time.monotonic(), [0])
            page.wait_for_function(
                "window.monitorDiagnostics.tracking[0]?.framesWithBoxes >= 5",
                timeout=5000,
            )
            diagnostics = page.evaluate("window.monitorDiagnostics")
            assert not diagnostics["mediaErrors"]
            tracking = diagnostics["tracking"][0]
            assert tracking["errors"] == 0
            for sample in tracking["samples"]:
                for obj in sample["objects"]:
                    assert obj["bbox"] == pytest.approx(bbox, abs=0.01)
            browser.close()


@pytest.mark.skipif(
    os.environ.get("VLM_MONITOR_BROWSER_TESTS") != "1",
    reason="opt-in installed Chromium integration",
)
@pytest.mark.parametrize(
    "streams,source_size,viewport",
    [
        (12, (720, 190), {"width": 1920, "height": 1080}),
        (2, (180, 320), {"width": 700, "height": 1080}),
    ],
    ids=["wide-compact-multicam", "portrait-tall-pane"],
)
def test_tracking_boxes_align_with_rendered_video_pixels(
    tmp_path, streams, source_size, viewport
):
    from fractions import Fraction
    from io import BytesIO

    import av
    from PIL import Image
    from playwright.sync_api import sync_playwright

    from tests.test_vlm_monitor import args, job
    from tools.vlm_benchmark.monitor import LiveMonitor
    from tools.vlm_multistream_benchmark import DecodedFrames

    width, height = source_size
    image = np.full((height, width, 3), 32, np.uint8)
    # A visible rectangle gives an independent pixel reference on the native
    # video. Comparing normalized tracker boxes alone misses double letterboxing.
    image[height // 4 : 3 * height // 4, width // 4 : 3 * width // 4] = (240, 30, 30)
    source = tmp_path / "geometry.mp4"
    with av.open(str(source), "w") as writer:
        stream = writer.add_stream("libx264", rate=15)
        stream.width, stream.height, stream.pix_fmt = width, height, "yuv420p"
        for n in range(15):
            frame = av.VideoFrame.from_image(Image.fromarray(image))
            frame.pts, frame.time_base = n, Fraction(1, 15)
            for packet in stream.encode(frame):
                writer.mux(packet)
        for packet in stream.encode():
            writer.mux(packet)
    frames = DecodedFrames([encoded(image)] * 3, str(source), [0, 0.4, 0.8], 1.0)
    destination = tmp_path / "monitor"
    destination.mkdir()
    with LiveMonitor(destination, [frames], args()) as monitor:
        stage = tmp_path / "stage"
        stage.mkdir()
        monitor.begin({**job(), "streams": streams}, stage)
        with sync_playwright() as p:
            browser = p.chromium.launch()
            page = browser.new_page(viewport=viewport)
            page.goto(monitor.url + "?tracking=on")
            page.wait_for_function("window.monitorReady")
            screen = page.locator(".camera .screen").first
            before = np.array(Image.open(BytesIO(screen.screenshot())).convert("RGB"))
            red = (
                (before[:, :, 0] > 180)
                & (before[:, :, 1] < 80)
                & (before[:, :, 2] < 80)
            )
            red_y, red_x = np.nonzero(red)
            assert len(red_x) > 100
            expected = np.array([red_x.min(), red_y.min(), red_x.max(), red_y.max()])
            page.evaluate(
                """() => {
                  const video=document.querySelector('.camera video');
                  const canvas=document.querySelector('.camera .tracking-overlay');
                  const image=document.createElement('canvas');
                  image.width=320;
                  image.height=Math.max(8,Math.round(320*video.videoHeight/video.videoWidth));
                  image.getContext('2d').drawImage(video,0,0,image.width,image.height);
                  const tracker={
                    canvas,ctx:canvas.getContext('2d'),stats:{painted:0},lastPaint:0
                  };
                  window.PreviewTracking.prototype.paint.call(tracker,image,[{
                    track_id:'geometry:0',bbox:{x1:.25,y1:.25,x2:.75,y2:.75}
                  }]);
                }"""
            )
            after = np.array(Image.open(BytesIO(screen.screenshot())).convert("RGB"))
            # Require a visible cyan border, not just tiny antialiasing changes.
            cyan = after[:, :, 0] < 150
            cyan &= after[:, :, 1] > 130
            cyan &= after[:, :, 2] > 150
            # Exclude the label above the box; inspect the actual rendered border.
            cyan[: expected[1] + 3] = False
            cyan_y, cyan_x = np.nonzero(cyan)
            assert len(cyan_x) > 10
            actual = np.array([cyan_x.min(), cyan_x.max(), cyan_y.max()])
            assert actual == pytest.approx(expected[[0, 2, 3]], abs=2)
            browser.close()
