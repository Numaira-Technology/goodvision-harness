"""Streaming protocol regressions using CPU fixtures and zero provider traffic."""

import http.client
import io
import json
import subprocess
import sys
import threading
import time
import urllib.error
from contextlib import nullcontext
from dataclasses import dataclass
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
from PIL import Image

from goodvision_harness import video_to_stream
from goodvision_harness.decision3_runtime import Decision3Runtime
from tools.event_benchmark import backends, realtime, run_eval


@pytest.fixture(autouse=True)
def no_provider_traffic(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("offline tests must never call a provider")

    monkeypatch.setattr(backends.urllib.request, "urlopen", forbidden)


def window(**extra):
    return {
        "source_id": "offline-source",
        "request_id": "window-1",
        "source_times": [0.0, 1.0],
        "images": [b"jpeg-one", b"jpeg-two"],
        **extra,
    }


def remote(kind="language", questions=None):
    model, _ = backends.CANDIDATES["qwen" if kind == "language" else kind]
    return backends.RemoteBackend(
        model,
        kind,
        questions or {"fight": "Fight?"},
        endpoint="http://offline.invalid/predict",
    )


def reply(monkeypatch, payload, *, status=200):
    encoded = json.dumps(payload).encode()
    response = SimpleNamespace(status=status, read=lambda _: encoded)
    monkeypatch.setattr(
        backends.urllib.request, "urlopen", lambda *a, **k: nullcontext(response)
    )


def test_language_partial_answers_keep_semantic_unknown_separate(monkeypatch):
    backend = remote(
        questions={"fight": "Fight?", "fall": "Fall?", "smoking": "Smoke?"}
    )
    reply(
        monkeypatch,
        {
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {"content": '{"fight":false,"fall":null}'},
                }
            ]
        },
    )
    predictions = backend.predict(window())["predictions"]
    assert predictions["fight"]["answer"] is False
    assert predictions["fall"]["answer"] is None
    assert predictions["fall"]["format_state"] == "valid"
    assert predictions["smoking"]["format_state"] == "invalid"


def native_answer():
    return {
        "type": "choice",
        "choice": backends.OPTIONS[0],
        "probabilities": dict(zip(backends.OPTIONS, [1.0, 0.0, 0.0], strict=True)),
    }


@pytest.mark.parametrize(
    "field,value",
    [
        ("request_id", "other-window"),
        ("request_id", None),
        ("model", "wrong-model"),
        ("revision", "wrong-revision"),
    ],
)
def test_native_responses_must_identify_the_requested_window_and_model(
    monkeypatch, field, value
):
    backend = remote("decision3")
    payload = {
        "request_id": "window-1",
        "model": backend.model,
        "revision": backends.REVISIONS[backend.model],
        "raw": {"answers": {"fight": native_answer()}},
    }
    payload[field] = value
    reply(monkeypatch, payload)
    result = backend.predict(window())
    assert result["predictions"]["fight"]["request_state"] == "response_mismatch"
    assert result["predictions"]["fight"]["answer"] is None


def test_native_http_error_preserves_valid_sibling_answers(monkeypatch):
    backend = remote("decision3", {"fight": "Fight?", "fall": "Fall?"})
    payload = {
        "request_id": "window-1",
        "error": "invalid_response",
        "raw": {
            "answers": {"fight": native_answer(), "fall": {"error": "inference_failed"}}
        },
    }
    error = urllib.error.HTTPError(
        backend.endpoint,
        500,
        "partial inference",
        {},
        io.BytesIO(json.dumps(payload).encode()),
    )
    calls = []

    def send(*args, **kwargs):
        calls.append(args)
        raise error

    monkeypatch.setattr(backends.urllib.request, "urlopen", send)
    result = backend.predict(window())
    assert len(calls) == 1
    assert result["raw"] == payload
    assert result["predictions"]["fight"]["answer"] is True
    assert result["predictions"]["fall"]["format_state"] == "invalid"


@pytest.mark.parametrize("status", [401, 402, 403])
def test_provider_rejection_stops_later_requests_even_before_body_read(
    monkeypatch, status
):
    backend = remote()
    nested, calls = [], []

    class ErrorBody:
        def read(self, _):
            nested.append(backend.predict(window(request_id="window-2")))
            return b"denied"

        def close(self):
            pass

    def send(*args, **kwargs):
        calls.append(args)
        if len(calls) > 1:
            raise OSError("unexpected second transmission")
        raise urllib.error.HTTPError(
            backend.endpoint, status, "denied", {}, ErrorBody()
        )

    monkeypatch.setattr(backends.urllib.request, "urlopen", send)
    result = backend.predict(window())
    assert len(calls) == 1
    assert result["shared_error"]
    assert nested[0]["predictions"]["fight"]["request_state"] == "not_sent"


@pytest.mark.parametrize("retry_after,delay", [("120", 120), ("5", 60), ("nan", 60)])
def test_429_records_immediately_without_retry_and_cools_subsequent_work(
    monkeypatch, retry_after, delay
):
    backend, now, calls, waits = remote(), [100.0], [], []
    monkeypatch.setattr(backends, "time", SimpleNamespace(monotonic=lambda: now[0]))

    def send(*args, **kwargs):
        calls.append(args)
        raise urllib.error.HTTPError(
            backend.endpoint,
            429,
            "rate limited",
            {"Retry-After": retry_after},
            io.BytesIO(b"busy"),
        )

    def wait(seconds):
        waits.append(seconds)
        now[0] += seconds

    monkeypatch.setattr(backends.urllib.request, "urlopen", send)
    monkeypatch.setattr(backend._cooldown, "wait", wait)
    result = backend.predict(window())
    assert result["raw"] == "busy"
    assert len(calls) == 1
    assert not waits  # The completed HTTP response must be persisted before cooling.
    assert backend._cooldown_until == 100.0 + delay
    expired = backend.predict(window(request_deadline_monotonic=101.0))
    assert waits
    assert len(calls) == 1
    assert expired["expired_before_request"]


@pytest.mark.parametrize(
    "error",
    [
        http.client.IncompleteRead(b"partial"),
        http.client.BadStatusLine("broken"),
        TimeoutError("no response"),
    ],
)
def test_uncertain_transport_outcomes_are_not_retried(monkeypatch, error):
    backend, calls = remote(), []

    def send(*args, **kwargs):
        calls.append(args)
        raise error

    monkeypatch.setattr(backends.urllib.request, "urlopen", send)
    result = backend.predict(window())
    assert len(calls) == 1
    assert result["predictions"]["fight"]["request_state"] == "outcome_unknown"


def test_gpu_lock_expiry_never_starts_image_processing():
    backend = object.__new__(backends.ContrastiveBackend)
    backend.questions = {"fight": "Fight?"}
    backend.lock = threading.Lock()
    backend.torch = SimpleNamespace(inference_mode=nullcontext)
    result = backend.predict(window(request_deadline_monotonic=time.monotonic() - 1))
    assert result["expired_before_request"] is True
    assert result["predictions"]["fight"]["request_state"] == "not_sent"


def fake_video(monkeypatch, *, fps=10, frames=3):
    class Capture:
        def __init__(self, _):
            self.index = 0

        def isOpened(self):
            return True

        def read(self):
            self.index += 1
            if self.index > frames:
                return False, None
            return True, np.full((4, 4, 3), self.index, dtype=np.uint8)

        def get(self, _):
            return (self.index - 1) * 1000 / fps

        def release(self):
            pass

    monkeypatch.setattr(cv2, "VideoCapture", Capture)
    monkeypatch.setattr(
        video_to_stream,
        "probe_video",
        lambda _: SimpleNamespace(source_fps=fps, frame_count=frames),
    )


def test_oversampling_uses_distinct_frames_and_counts_only_possible_windows(
    tmp_path, monkeypatch
):
    fake_video(monkeypatch)
    windows = []

    def predict(payload, **kwargs):
        windows.append(payload)
        return {"predictions": {"fight": backends._result(False)}}

    backend = SimpleNamespace(metadata={}, predict=predict)
    sample = {
        "number": 1,
        "filename": "clip.mp4",
        "event": "fight",
        "proposed_answer": "no",
    }
    result = realtime.run_streams(
        [sample],
        {"clip.mp4": tmp_path / "clip.mp4"},
        {"fight": "Fight?"},
        backend,
        tmp_path / "run.json",
        slots=1,
        workers=1,
        interval=0.02,
        window_frames=2,
        frame_width=4,
    )
    assert [row["frame_indices"] for row in windows] == [[0, 1], [1, 2]]
    assert all(row["sampling_fps"] == 10 for row in windows)
    assert result["primary_windows"]["planned_windows"] == 2
    assert result["primary_windows"]["unformed_windows"] == 0
    assert result["per_video"][0]["evidence_complete"]
    assert result["overall"]["correct"] == 1


def test_single_frame_source_keeps_its_original_display_duration(tmp_path, monkeypatch):
    fake_video(monkeypatch, frames=1)
    backend = SimpleNamespace(
        metadata={},
        predict=lambda *a, **k: pytest.fail("one frame cannot form a two-frame window"),
    )
    sample = {
        "number": 1,
        "filename": "clip.mp4",
        "event": "fight",
        "proposed_answer": "no",
    }
    result = realtime.run_streams(
        [sample],
        {"clip.mp4": tmp_path / "clip.mp4"},
        {"fight": "Fight?"},
        backend,
        tmp_path / "run.json",
        slots=1,
        workers=1,
        interval=0.1,
        frame_width=4,
    )
    state = result["streams"][0]
    assert state["input_end_monotonic"] - state["input_start_monotonic"] >= 0.09
    assert result["execution_complete"]
    assert result["per_video"][0]["pred"] == "unknown"


def test_corrupt_source_is_retired_from_filler_rotation(tmp_path, monkeypatch):
    fake_video(monkeypatch)
    original_capture = cv2.VideoCapture

    def capture(path):
        instance = original_capture(path)
        if path.endswith("corrupt.mp4"):
            original_read = instance.read
            instance.read = lambda: (
                original_read() if instance.index == 0 else (False, None)
            )
        else:
            original_read = instance.read

            def read():
                if instance.index == 0:
                    time.sleep(0.05)
                return original_read()

            instance.read = read
        return instance

    monkeypatch.setattr(cv2, "VideoCapture", capture)
    samples = [
        {"number": i, "filename": name, "event": "fight", "proposed_answer": "no"}
        for i, name in enumerate(["corrupt.mp4", "valid.mp4"])
    ]
    backend = SimpleNamespace(
        metadata={},
        predict=lambda *a, **k: {"predictions": {"fight": backends._result(False)}},
    )
    result = realtime.run_streams(
        samples,
        {s["filename"]: tmp_path / s["filename"] for s in samples},
        {"fight": "Fight?"},
        backend,
        tmp_path / "run.json",
        slots=2,
        workers=2,
        interval=0.1,
        frame_width=4,
    )
    corrupt = [
        state for state in result["streams"] if state["filename"] == "corrupt.mp4"
    ]
    assert len(corrupt) <= 2  # At most one already-started filler can race the primary.
    assert all(state["source_error"] for state in corrupt)
    assert not result["execution_complete"]


def test_replacement_drains_prior_source_independently_and_uses_unique_ids(
    tmp_path, monkeypatch
):
    fake_video(monkeypatch)
    original_capture = cv2.VideoCapture
    next_started = threading.Event()
    windows = []

    def capture(path):
        if path.endswith("second.mp4"):
            next_started.set()
        return original_capture(path)

    def predict(payload, **kwargs):
        windows.append(payload)
        if payload["filename"] == "first.mp4":
            assert next_started.wait(2), "input replacement waited for prior inference"
        return {"predictions": {"fight": backends._result(True)}}

    monkeypatch.setattr(cv2, "VideoCapture", capture)
    samples = [
        {"number": i, "filename": name, "event": "fight", "proposed_answer": "yes"}
        for i, name in enumerate(["first.mp4", "second.mp4"])
    ]
    result = realtime.run_streams(
        samples,
        {s["filename"]: tmp_path / s["filename"] for s in samples},
        {"fight": "Fight?"},
        SimpleNamespace(metadata={}, predict=predict),
        tmp_path / "run.json",
        slots=1,
        workers=1,
        interval=0.1,
        frame_width=4,
    )
    assert result["execution_complete"]
    assert len({row["request_id"] for row in windows}) == len(windows)
    assert len({row["source_id"] for row in windows}) == 2
    assert result["overall"]["correct"] == 2
    summary = result["primary_windows"]
    assert summary["submitted"] == summary["completed"]
    assert summary["opportunities"] == summary["submitted"] + summary["busy_missed"]


def test_queued_expiration_is_unsent_and_never_scored_negative(tmp_path, monkeypatch):
    fake_video(monkeypatch, frames=2)
    backend = SimpleNamespace(
        metadata={},
        wait_cooldown=lambda stop: time.sleep(0.02),
        predict=lambda *a, **k: pytest.fail("expired windows must not reach inference"),
    )
    sample = {
        "number": 1,
        "filename": "clip.mp4",
        "event": "fight",
        "proposed_answer": "no",
    }
    result = realtime.run_streams(
        [sample],
        {"clip.mp4": tmp_path / "clip.mp4"},
        {"fight": "Fight?"},
        backend,
        tmp_path / "run.json",
        slots=1,
        workers=1,
        interval=0.1,
        frame_width=4,
        queue_age=0.01,
    )
    assert result["primary_windows"]["queue_expired"] == 1
    assert result["primary_windows"]["completed"] == 0
    assert result["per_video"][0]["pred"] == "unknown"
    assert not result["per_video"][0]["evidence_complete"]
    assert result["overall"]["correct"] == 0
    assert result["overall"]["unknown"] == 1
    saved = json.loads((tmp_path / "run-windows.jsonl").read_text())
    assert saved["status"] == "expired_before_request"
    assert saved["request_started_monotonic"] is None


def test_question_accounting_separates_partial_invalid_and_unknown_results(
    tmp_path, monkeypatch
):
    fake_video(monkeypatch, frames=2)
    questions = {label: label + "?" for label in ("fight", "fall", "smoking", "weapon")}
    predictions = {
        "fight": backends._result(True),
        "fall": backends._result(False),
        "smoking": backends._result(None),
        "weapon": backends._result(format_state="invalid"),
    }
    backend = SimpleNamespace(
        metadata={}, predict=lambda *a, **k: {"predictions": predictions}
    )
    samples = [
        {
            "number": i,
            "filename": "clip.mp4",
            "event": label,
            "proposed_answer": "yes" if label == "fight" else "no",
        }
        for i, label in enumerate(questions)
    ]
    result = realtime.run_streams(
        samples,
        {"clip.mp4": tmp_path / "clip.mp4"},
        questions,
        backend,
        tmp_path / "run.json",
        slots=1,
        workers=1,
        interval=0.1,
        frame_width=4,
    )
    assert result["overall"]["total"] == 4
    assert result["overall"]["correct"] == result["overall"]["unknown"] == 2
    evidence = result["primary_event_evidence"]
    assert evidence["smoking"]["unknown"] == 1
    assert evidence["smoking"]["format_invalid"] == 0
    assert evidence["weapon"]["unknown"] == 0
    assert evidence["weapon"]["format_invalid"] == 1


@pytest.mark.parametrize(
    "invocation",
    [["-m", "tools.event_benchmark.run_eval"], ["tools/event_benchmark/run_eval.py"]],
)
def test_cli_help_supports_script_and_module_invocation(invocation):
    result = subprocess.run(
        [sys.executable, *invocation, "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "streaming-v3" in result.stdout


def test_decision3_binds_actual_sampling_fps_without_resampling():
    @dataclass
    class Video:
        frames: object
        fps: float = 2.0

    runtime = object.__new__(Decision3Runtime)
    runtime.shared_window_state = False
    captured = []

    def prepare_native(*args):
        captured.extend(args[-1])
        return SimpleNamespace(media={})

    runtime.native = SimpleNamespace(
        load_videos=lambda arrays: [Video(a) for a in arrays],
        _prepare_videos=prepare_native,
    )
    template = SimpleNamespace(rows=[], bind=lambda *args: ({}, {}, {}, {}, (), 0))
    payloads = []
    for color in ("black", "white"):
        stream = io.BytesIO()
        Image.new("RGB", (4, 4), color).save(stream, format="JPEG")
        payloads.append(stream.getvalue())
    prepared = runtime.prepare(
        payloads, 0.25, question_template=template, frame_times=[0.0, 4.0]
    )
    assert captured[0].fps == 0.25
    assert captured[0].frames is prepared.frames
    assert prepared.frames.shape[0] == 2


def test_streaming_inputs_preserve_missing_media_as_unavailable(tmp_path):
    samples = tmp_path / "samples.json"
    samples.write_text(
        json.dumps(
            {
                "samples": [
                    {
                        "number": 1,
                        "filename": "missing.mp4",
                        "event": "fight",
                        "question": "Fight?",
                        "proposed_answer": "unknown",
                    }
                ]
            }
        )
    )
    selected, media, source, _ = run_eval.streaming_inputs(
        samples, tmp_path / "missing.json", [tmp_path], ["fight"]
    )
    assert len(selected) == 1 and media == {}
    assert source["read_error"]


@pytest.mark.parametrize("duration", [0, -1, float("nan"), float("inf"), True, "1"])
def test_direct_runner_rejects_invalid_load_horizon_before_creating_output(
    tmp_path, duration
):
    with pytest.raises(ValueError, match="load_seconds"):
        realtime.run_streams(
            [],
            {},
            {},
            SimpleNamespace(metadata={}),
            tmp_path / "run.json",
            load_seconds=duration,
        )
    assert not (tmp_path / "run-windows.jsonl").exists()


def test_load_horizon_preserves_primary_eof_even_after_deadline(tmp_path, monkeypatch):
    fake_video(monkeypatch, fps=10, frames=3)
    samples = [
        {"number": i, "filename": name, "event": "fight", "proposed_answer": "no"}
        for i, name in enumerate(["first.mp4", "second.mp4"])
    ]
    result = realtime.run_streams(
        samples,
        {row["filename"]: tmp_path / row["filename"] for row in samples},
        {"fight": "Fight?"},
        SimpleNamespace(
            metadata={},
            predict=lambda *a, **k: {"predictions": {"fight": backends._result(False)}},
        ),
        tmp_path / "run.json",
        slots=1,
        workers=1,
        interval=0.1,
        frame_width=4,
        load_seconds=0.05,
    )
    assert result["execution_complete"]
    assert result["overall"]["total"] == result["overall"]["correct"] == 2
    assert len(result["streams"]) == 2
    assert all(row["eof_seen"] and row["raw_frames"] == 3 for row in result["streams"])
    coverage = result["input_coverage"]
    assert coverage["input_end_monotonic"] > coverage["load_deadline_monotonic"]
    assert coverage["measurement_seconds"] >= 0.59


def test_sustained_filler_counts_actual_ingress_and_drains_after_horizon(
    tmp_path, monkeypatch
):
    fake_video(monkeypatch, fps=10, frames=3)
    calls = []

    def predict(payload, **kwargs):
        calls.append(payload)
        if not payload["quality_reference"]:
            time.sleep(0.2)  # Accepted work must drain beyond the input deadline.
        return {
            "predictions": {
                "fight": backends._result(not payload["quality_reference"]),
                "bag_knife": backends._result(
                    request_state="not_sent",
                    format_state="unavailable",
                ),
            }
        }

    questions = {"fight": "Fight?", "bag_knife": "Knife?"}
    samples = [
        {"number": i, "filename": "clip.mp4", "event": label, "proposed_answer": "no"}
        for i, label in enumerate(questions)
    ]
    result = realtime.run_streams(
        samples,
        {"clip.mp4": tmp_path / "clip.mp4"},
        questions,
        SimpleNamespace(metadata={}, predict=predict),
        tmp_path / "run.json",
        slots=1,
        workers=1,
        interval=0.1,
        frame_width=4,
        load_seconds=0.45,
    )
    primary, *fillers = result["streams"]
    assert primary["quality_reference"] and primary["eof_seen"]
    assert fillers and all(not row["quality_reference"] for row in fillers)
    assert result["execution_complete"]
    assert result["overall"]["total"] == 2
    assert result["overall"]["correct"] == result["overall"]["unknown"] == 1
    assert result["primary_event_evidence"]["fight"]["yes"] == 0
    assert result["per_video"][1]["evidence"]["not_sent"] > 0
    assert all(row["submitted"] == row["completed"] for row in result["streams"])
    assert any(row["drained_monotonic"] > row["input_end_monotonic"] for row in fillers)
    cutoff = fillers[-1]
    assert cutoff["input_end_reason"] == "load_duration_reached"
    assert cutoff["planned_windows"] == max(0, cutoff["raw_frames"] - 1)
    assert cutoff["unformed_windows"] == 0
    coverage = result["input_coverage"]
    assert coverage["load_deadline_monotonic"] - coverage[
        "input_start_monotonic"
    ] == pytest.approx(0.45)
    assert coverage["measurement_seconds"] >= 0.44
    assert sum(coverage["seconds_by_active_inputs"].values()) == pytest.approx(
        coverage["measurement_seconds"]
    )
    assert result["finished_monotonic"] > coverage["input_end_monotonic"]
    records = [
        json.loads(line)
        for line in (tmp_path / "run-windows.jsonl").read_text().splitlines()
    ]
    assert any(not row["quality_reference"] for row in records)


@pytest.mark.parametrize("fail", [False, True])
def test_frame_observer_shares_encoded_evidence_and_isolates_first_exception(
    tmp_path, monkeypatch, fail
):
    fake_video(monkeypatch, fps=10, frames=3)
    observed, windows = [], []

    def observer(metadata, jpeg):
        observed.append((dict(metadata), jpeg))
        metadata["source_id"] = "callback mutation must remain local"
        if fail:
            raise RuntimeError("recording failed")

    def predict(payload, **kwargs):
        windows.append(payload)
        return {"predictions": {"fight": backends._result(False)}}

    samples = [
        {"number": i, "filename": name, "event": "fight", "proposed_answer": "no"}
        for i, name in enumerate(["first.mp4", "second.mp4"])
    ]
    result = realtime.run_streams(
        samples,
        {row["filename"]: tmp_path / row["filename"] for row in samples},
        {"fight": "Fight?"},
        SimpleNamespace(metadata={}, predict=predict),
        tmp_path / "run.json",
        slots=2,
        workers=2,
        interval=0.1,
        frame_width=4,
        frame_observer=observer,
    )
    assert result["execution_complete"]
    assert result["overall"]["total"] == result["overall"]["correct"] == 2
    assert not result["shared_errors"]
    assert all(row["source_error"] is None for row in result["streams"])
    assert all(
        row["source_id"] != "callback mutation must remain local" for row in windows
    )
    if fail:
        assert len(observed) == 1
        assert result["frame_observer_error"] == "RuntimeError: recording failed"
    else:
        assert result["frame_observer_error"] is None
        assert len(observed) == sum(row["sampled_frames"] for row in result["streams"])
        for payload in windows:
            for index, jpeg in zip(
                payload["frame_indices"], payload["images"], strict=True
            ):
                matches = [
                    (metadata, data)
                    for metadata, data in observed
                    if metadata["source_id"] == payload["source_id"]
                    and metadata["frame_index"] == index
                ]
                assert len(matches) == 1
                metadata, data = matches[0]
                assert data == jpeg
                assert metadata["filename"] == payload["filename"]
                assert metadata["source_time"] == pytest.approx(index / 10)
                assert metadata["source_time_basis"] == "decoder_pts"
                assert Image.open(io.BytesIO(data)).size == (
                    metadata["width"],
                    metadata["height"],
                )
