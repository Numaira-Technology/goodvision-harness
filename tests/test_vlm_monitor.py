"""Live input/result mapping and recording integrity, without paid model calls."""

import json
import os
import time
from types import SimpleNamespace

import httpx
import pytest
from test_vlm_independent import jpeg

from tools.vlm_benchmark import modes
from tools.vlm_benchmark.monitor import LiveMonitor, media_time
from tools.vlm_multistream_benchmark import DecodedFrames, planned_observations


@pytest.mark.parametrize(
    "problem",
    [
        "short",
        "no-source",
        "no-tail",
        "pending-evidence",
        "js-error",
        "media-error",
        "poll-stall",
        "playback-stall",
        "frozen-video",
    ],
)
def test_truncated_or_frozen_capture_is_not_deliverable(problem):
    from tools.vlm_benchmark.recording import recording_is_complete

    measured = {"duration_seconds": 10}
    d = {
        "sawPlaying": True,
        "finalPainted": True,
        "pendingEvidence": 0,
        "mediaErrors": [],
        "maxPollGapSeconds": 0.2,
        "maxPlaybackGapSeconds": 0.1,
        "videoFrames": [120] * 12,
    }
    errors = []
    assert recording_is_complete(measured, 10, d, errors)
    assert not recording_is_complete(measured, 10, d, errors, benchmark_completed=False)
    if problem == "short":
        measured["duration_seconds"] = 5
    elif problem == "no-source":
        d["sawPlaying"] = False
    elif problem == "no-tail":
        d["finalPainted"] = False
    elif problem == "pending-evidence":
        d["pendingEvidence"] = 1
    elif problem == "js-error":
        errors.append("fixture")
    elif problem == "media-error":
        d["mediaErrors"].append("decode failed")
    elif problem == "poll-stall":
        d["maxPollGapSeconds"] = 2.1
    elif problem == "playback-stall":
        d["maxPlaybackGapSeconds"] = 2.1
    else:
        d["videoFrames"][11] = 1
    assert not recording_is_complete(measured, 10, d, errors)


def test_input_pts_and_preview_keep_the_original_duration(tmp_path):
    from fractions import Fraction

    from PIL import Image

    from tools.vlm_benchmark.recording import inspect_video, make_preview
    from tools.vlm_multistream_benchmark import decode

    av = pytest.importorskip("av")
    source, preview = tmp_path / "offset.mp4", tmp_path / "preview.mp4"
    with av.open(str(source), "w") as writer:
        stream = writer.add_stream("libx264", rate=25)
        stream.width, stream.height, stream.pix_fmt = 96, 64, "yuv420p"
        for n in range(100):
            frame = av.VideoFrame.from_image(Image.new("RGB", (96, 64), (n, 0, 0)))
            frame.pts, frame.time_base = n + 125, Fraction(1, 25)
            for packet in stream.encode(frame):
                writer.mux(packet)
        for packet in stream.encode():
            writer.mux(packet)
    samples = decode(source, 2.5, 720, 90)
    assert len(samples) == 10
    assert samples.media_seconds == pytest.approx(
        [n * 0.4 for n in range(10)], abs=0.001
    )
    make_preview(source, preview)
    details = inspect_video(preview)
    assert details["duration_seconds"] == pytest.approx(4, abs=0.1)
    assert details["frames"] == 60  # 4 seconds at the advertised 15 FPS.


def args():
    return modes.parser().parse_args(["--monitor", "--streams", "1"])


def job():
    return {
        "id": "stage-one",
        "mode": "parallel",
        "phase": "measurement",
        "duration": 3,
        "repeat": 1,
        "streams": 1,
    }


def result(tick, lane="information", *, error=None, seconds=1):
    return {
        "message_id": f"{lane}-{tick}-{error}",
        "window_id": f"window-{tick}",
        "vlm_lane": lane,
        "frame": {"source_frame_index": tick, "timestamp_seconds": tick * 0.4},
        "state": "failed" if error else "completed",
        "analysis_complete": not error,
        "error_code": error,
        "timing": {"source_to_publish_seconds": seconds},
        "judgments": [{"event_label": "fight", "observed_detected": True}],
        "objects": [
            {"label": "person", "bbox": {"x1": 0.1, "y1": 0.1, "x2": 0.5, "y2": 0.9}}
        ],
    }


def test_variable_rate_preview_does_not_round_adjacent_frames_to_same_pts(tmp_path):
    from fractions import Fraction

    from PIL import Image

    from tools.vlm_benchmark.recording import inspect_video, make_preview

    av = pytest.importorskip("av")
    source, preview = tmp_path / "vfr.mp4", tmp_path / "preview.mp4"
    stamps = [0, 120, 140, 200, 280, 340, 400, 480, 540, 600, 680, 740]
    with av.open(str(source), "w") as writer:
        stream = writer.add_stream("libx264", rate=100)
        stream.width, stream.height, stream.pix_fmt = 96, 64, "yuv420p"
        for pts in stamps:
            frame = av.VideoFrame.from_image(Image.new("RGB", (96, 64), "gray"))
            frame.pts, frame.time_base = pts, Fraction(1, 1000)
            for packet in stream.encode(frame):
                writer.mux(packet)
        for packet in stream.encode():
            writer.mux(packet)
    make_preview(source, preview)
    assert inspect_video(preview)["frames"] == len(stamps)
    with av.open(str(preview)) as reader:
        assert [f.time for f in reader.decode(video=0)] == pytest.approx(
            [p / 1000 for p in stamps], abs=0.001
        )


def test_monitor_keeps_lanes_and_real_frame_mapping_without_backdating(tmp_path):
    frames = DecodedFrames([jpeg()] * 4, "unused.mp4", [0.04, 0.45, 0.83, 1.23], 1.6)
    monitor = LiveMonitor(tmp_path, [frames], args())
    monitor.previews = [None]
    monitor.begin(job(), tmp_path)
    started = time.monotonic()
    monitor.source_started(started, [0])
    monitor.sample(0, 5, True)
    r = result(5)
    monitor.result(0, r)
    monitor.result(0, r)  # Duplicate polling must not inflate completeness.
    monitor.result(0, result(2))  # Late old response cannot overwrite newer state.
    monitor.result(0, result(5, "grounding", seconds=2))
    snapshot = monitor.snapshot()
    camera = snapshot["cameras"][0]
    assert camera["lanes"]["events"]["tick"] == 5
    assert camera["lanes"]["events"]["media_seconds"] == 0.45  # Exact PTS in loop 2.
    assert camera["lanes"]["grounding"]["latency"] == 2
    assert camera["out_of_order"] == 1
    assert snapshot["counters"]["events"]["complete"] == 2
    assert snapshot["planned"] == planned_observations(3, 1.25)
    assert media_time(frames, 4, 2.5) == 0.04
    assert camera["lanes"]["events"]["received_elapsed"] >= 0
    monitor.finish({"lanes": {}})
    monitor.log.close()
    entries = [
        json.loads(line)
        for line in (tmp_path / "monitor.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert [e["kind"] for e in entries] == [
        "stage",
        "source_started",
        "sample",
        "result",
        "result",
        "result",
        "finish",
    ]
    assert entries[4]["stale"]


def test_failed_or_unmapped_result_never_fabricates_a_frame_or_valid_result(tmp_path):
    monitor = LiveMonitor(tmp_path, [[jpeg()] * 4], args())
    monitor.previews = [None]
    monitor.begin(job(), tmp_path)
    monitor.source_started(time.monotonic(), [0])
    bad = result(999, error="timeout")
    monitor.result(0, bad)
    snapshot = monitor.snapshot()
    assert snapshot["cameras"][0]["lanes"]["events"]["tick"] is None
    assert snapshot["counters"]["events"]["complete"] == 0
    monitor.draining()
    monitor.finish(error=RuntimeError("fixture"))
    assert monitor.snapshot()["state"] == "failed"
    monitor.log.close()


def test_monitor_serves_only_allowlisted_media_and_supports_ranges(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("OPENROUTER_API_KEY", "secret-must-not-be-served")
    frame = jpeg()
    with LiveMonitor(tmp_path, [[frame] * 4], args()) as monitor:
        stage = tmp_path / "stage"
        stage.mkdir()
        monitor.begin(job(), stage)
        media = tmp_path / "fixture.mp4"
        media.write_bytes(b"0123456789")
        monitor.previews = [media]
        with httpx.Client(base_url=monitor.url, trust_env=False) as client:
            assert client.get("../../../.env").status_code == 404
            assert client.get("../state").status_code == 404
            response = client.get("state")
            assert "secret-must-not-be-served" not in response.text
            assert response.json()["stage"] == "stage-one"
            assert (
                client.get("media/0", headers={"Range": "bytes=2-5"}).content == b"2345"
            )
            assert (
                client.get("media/0", headers={"Range": "bytes=20-"}).status_code == 416
            )
            assert client.head("media/0").headers["content-length"] == "10"
            assert client.get("frame/stage-one/0/7.jpg").content == frame
            assert client.get("frame/old-stage/0/1.jpg").status_code == 404
            assert client.get("frame/stage-one/1/1.jpg").status_code == 404
            assert client.get("frame/stage-one/0/99.jpg").status_code == 404


def test_recording_failure_cannot_be_marked_as_success(tmp_path):
    monitor = LiveMonitor(tmp_path, [[jpeg()] * 4], args())
    monitor.begin(job(), tmp_path)

    def fail(*_):
        raise RuntimeError("browser disconnected")

    monitor.recorder = SimpleNamespace(finish=fail)
    monitor.recording_active = True
    record = monitor.finish({"lanes": {}})
    assert not record["complete"] and record["state"] == "failed"
    monitor.log.close()


def test_late_encoder_completion_cannot_undo_an_interrupt(tmp_path):
    import queue
    import threading

    from tools.vlm_benchmark.recording import BrowserRecorder, save_recording

    finish_encoding = threading.Event()

    class InterruptQueue(queue.Queue):
        def put(self, command):
            name, _, future = command
            if name == "finish":

                def interrupted(**_):
                    raise KeyboardInterrupt

                future.result = interrupted
            super().put(command)

    recorder = BrowserRecorder.__new__(BrowserRecorder)
    recorder.commands = InterruptQueue()
    recorder.failure = None
    recorder.cancelled = threading.Event()
    recorder.folder = tmp_path

    def worker():
        _, _, future = recorder.commands.get()
        assert finish_encoding.wait(5)
        # Simulate the encoder already preparing a success payload when the
        # caller presses Ctrl-C; cleanup must preserve the interruption.
        save_recording(tmp_path / "recording.json", {"complete": True})
        future.set_result({"complete": True})
        name, _, future = recorder.commands.get()
        assert name == "close"
        future.set_result(None)

    recorder.thread = threading.Thread(target=worker, daemon=True)
    recorder.thread.start()
    with pytest.raises(KeyboardInterrupt):
        recorder.finish("stage", True)
    finish_encoding.set()
    recorder.close()
    saved = json.loads((tmp_path / "recording.json").read_text(encoding="utf-8"))
    assert saved["complete"] is False
    assert saved["state"] == "interrupted"


def test_partial_recording_stops_remaining_stages_and_stays_in_report(
    tmp_path, monkeypatch
):
    options = modes.parser().parse_args(
        [
            "--record",
            "--modes",
            "events",
            "--streams",
            "1",
            "--duration",
            "3",
            "--repeats",
            "1",
            "--output",
            str(tmp_path / "run"),
        ]
    )

    class Monitor:
        def __init__(self, *_):
            self.phase = None

        def __enter__(self):
            return self

        def __exit__(self, *_):
            pass

        def begin(self, job, location):
            self.phase = job["phase"]

        def finish(self, *_, **__):
            return {"complete": False} if self.phase == "measurement" else None

    calls = []

    def stage(*_args, **kwargs):
        assert kwargs["observer"]
        calls.append(True)
        return {
            "streams": 1,
            "repeat": 1,
            "all_drained": True,
            "input_timing_valid": True,
            "result_records_missing": 0,
            "lanes": {},
            "per_stream": [],
            "paired_windows": None,
            "nominal_requests_per_second": 1.25,
            "rate_limit_below_offered_load": False,
        }

    monkeypatch.setattr("tools.vlm_benchmark.monitor.LiveMonitor", Monitor)
    monkeypatch.setattr(modes, "run_stage", stage)
    with pytest.raises(RuntimeError, match="incomplete browser recording"):
        modes.run(options)
    doc = json.loads((options.output / "summary.json").read_text(encoding="utf-8"))
    assert doc["state"] == "failed"
    assert len(calls) == 3  # Preflight, warmup, then the incomplete measurement.
    assert doc["jobs"][-1]["recording"] == {"complete": False}
    assert "recording/viewport.mp4" not in (options.output / "report.html").read_text(
        encoding="utf-8"
    )


@pytest.mark.parametrize("during", ["begin", "running", "finalizing"])
@pytest.mark.parametrize("exception", [RuntimeError, KeyboardInterrupt])
def test_interrupted_stage_keeps_capture_and_no_broken_mp4_link(
    tmp_path, monkeypatch, during, exception
):
    options = modes.parser().parse_args(
        [
            "--record",
            "--modes",
            "events",
            "--streams",
            "1",
            "--duration",
            "3",
            "--repeats",
            "1",
            "--output",
            str(tmp_path / "run"),
        ]
    )
    calls = []

    class Monitor:
        def __init__(self, *_):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_):
            pass

        def begin(self, job, location):
            self.phase, self.location = job["phase"], location
            if self.phase == "measurement":
                (location / "recording").mkdir()
                (location / "recording/partial.webm").write_bytes(b"fixture")
                if during == "begin":
                    raise exception("fixture interrupted")

        def finish(self, *_, **kwargs):
            if self.phase == "measurement":
                if during == "finalizing" and not kwargs.get("error"):
                    raise exception("fixture interrupted")
                return {"complete": False, "state": "interrupted"}

    def stage(*_args, **kwargs):
        monitor = kwargs["observer"]
        calls.append(monitor.phase)
        if monitor.phase == "measurement" and during == "running":
            raise exception("fixture interrupted")
        return {
            "streams": 1,
            "repeat": 1,
            "all_drained": True,
            "input_timing_valid": True,
            "result_records_missing": 0,
            "lanes": {},
            "per_stream": [],
            "paired_windows": None,
            "nominal_requests_per_second": 1.25,
            "rate_limit_below_offered_load": False,
        }

    monkeypatch.setattr("tools.vlm_benchmark.monitor.LiveMonitor", Monitor)
    monkeypatch.setattr(modes, "run_stage", stage)
    with pytest.raises(exception, match="fixture interrupted"):
        modes.run(options)
    doc = json.loads((options.output / "summary.json").read_text(encoding="utf-8"))
    interrupted = doc["interrupted_jobs"][-1]
    assert interrupted["id"] == "1-streams-events-repeat-1"
    assert interrupted["recording"]["complete"] is False
    assert interrupted["recording"]["benchmark_completed"] is False
    assert len(doc["jobs"]) == 2  # Preflight + warmup; no made-up final statistics.
    html = (options.output / "report.html").read_text(encoding="utf-8")
    assert "recording/partial.webm" in html
    assert "recording/recording.json" in html
    assert "recording/viewport.mp4" not in html


@pytest.mark.skipif(
    os.environ.get("VLM_MONITOR_BROWSER_TESTS") != "1",
    reason="opt-in installed Chromium integration",
)
def test_browser_renders_actual_evidence_uncertainty_and_failure(tmp_path):
    from playwright.sync_api import sync_playwright

    with LiveMonitor(tmp_path, [[jpeg()] * 4], args()) as monitor:
        stage = tmp_path / "stage"
        stage.mkdir()
        monitor.begin(job(), stage)
        monitor.source_started(time.monotonic(), [0])
        monitor.sample(0, 2, True)
        with sync_playwright() as p:
            browser = p.chromium.launch()
            page = browser.new_page(viewport={"width": 1920, "height": 1080})
            page.goto(monitor.url)
            page.wait_for_function("window.monitorReady")
            assert page.locator(".chip.positive").count() == 0
            r = result(2, "grounding")
            r["judgments"].append({"event_label": "fall", "observed_detected": None})
            monitor.result(0, r)
            page.wait_for_function(
                "document.querySelectorAll('.chip.positive').length === 1 && "
                "document.querySelector('.empty').hidden"
            )
            assert page.locator(".chip.uncertain").count() == 1
            assert "目标框 1" in page.locator(".evidence-note").inner_text()
            # The normalized box is painted only onto its matching evidence canvas.
            green_pixels = page.locator("canvas").evaluate(
                """e => {
                  const d=e.getContext('2d').getImageData(0,0,640,360).data;
                  let n=0;
                  for(let i=0;i<d.length;i+=4)
                    if(d[i]<100 && d[i+1]>200 && d[i+2]>120)n++;
                  return n;
                }"""
            )
            assert green_pixels > 100
            monitor.result(0, result(1, "grounding"))
            page.wait_for_function(
                "document.querySelector('.camera-foot').textContent"
                ".includes('保留旧结果 1')"
            )
            assert "帧 2" in page.locator(".evidence-note").inner_text()
            monitor.result(0, result(3, "grounding", error="timeout"))
            page.wait_for_function(
                "document.querySelectorAll('.chip.failed').length === 6"
            )
            assert page.locator(".chip.positive").count() == 0
            monitor.draining()
            monitor.finish(
                {
                    "lanes": {},
                    "input_timing_valid": True,
                    "all_drained": True,
                    "result_records_missing": 0,
                }
            )
            page.wait_for_function("window.finalPainted")
            assert page.locator("#summary").is_visible()
            browser.close()


@pytest.mark.skipif(
    os.environ.get("VLM_MONITOR_BROWSER_TESTS") != "1",
    reason="opt-in installed Chromium integration",
)
def test_final_screen_waits_for_the_last_evidence_image(tmp_path):
    from playwright.sync_api import sync_playwright

    with LiveMonitor(tmp_path, [[jpeg()] * 4], args()) as monitor:
        stage = tmp_path / "stage"
        stage.mkdir()
        monitor.begin(job(), stage)
        monitor.source_started(time.monotonic(), [0])
        with sync_playwright() as p:
            browser = p.chromium.launch()
            page = browser.new_page(viewport={"width": 1920, "height": 1080})
            pending = []
            page.route("**/frame/**", lambda route: pending.append(route))
            page.goto(monitor.url)
            page.wait_for_function("window.monitorReady")
            monitor.result(0, result(2))
            page.wait_for_function(
                "document.querySelector('.empty').textContent.includes('正在加载')"
            )
            monitor.finish(
                {
                    "lanes": {},
                    "input_timing_valid": True,
                    "all_drained": True,
                    "result_records_missing": 0,
                }
            )
            page.wait_for_function("!document.querySelector('#summary').hidden")
            page.wait_for_timeout(350)
            assert page.evaluate("window.finalPainted") is False
            for route in pending:
                route.continue_()
            page.wait_for_function("window.finalPainted")
            assert page.locator(".empty").is_hidden()
            browser.close()


@pytest.mark.skipif(
    os.environ.get("VLM_MONITOR_BROWSER_TESTS") != "1",
    reason="opt-in installed Chromium integration",
)
def test_buffer_starvation_counts_as_a_playback_stall(tmp_path):
    from fractions import Fraction

    from PIL import Image
    from playwright.sync_api import sync_playwright

    av = pytest.importorskip("av")
    video = tmp_path / "video.mp4"
    with av.open(str(video), "w") as writer:
        stream = writer.add_stream("libx264", rate=15)
        stream.width, stream.height, stream.pix_fmt = 96, 64, "yuv420p"
        for n in range(90):
            f = av.VideoFrame.from_image(Image.new("RGB", (96, 64), (n, 0, 0)))
            f.pts, f.time_base = n, Fraction(1, 15)
            for packet in stream.encode(f):
                writer.mux(packet)
        for packet in stream.encode():
            writer.mux(packet)
    with LiveMonitor(tmp_path, [[jpeg()] * 30], args()) as monitor:
        stage = tmp_path / "stage"
        stage.mkdir()
        monitor.previews = [video]
        monitor.begin({**job(), "duration": 8}, stage)
        with sync_playwright() as p:
            browser = p.chromium.launch(
                args=["--autoplay-policy=no-user-gesture-required"]
            )
            page = browser.new_page(viewport={"width": 1920, "height": 1080})
            page.goto(monitor.url)
            page.wait_for_function("window.monitorReady")
            monitor.source_started(time.monotonic(), [0])
            page.wait_for_function("window.monitorDiagnostics.videoFrames[0] > 2")
            page.locator("video").evaluate("""v => {
                v.pause();
                Object.defineProperty(v, 'readyState', {get: () => 1});
            }""")
            page.wait_for_timeout(2300)
            assert page.evaluate("window.monitorDiagnostics.maxPlaybackGapSeconds") >= 2
            browser.close()
