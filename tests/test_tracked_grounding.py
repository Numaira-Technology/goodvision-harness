"""Offline contracts and complete HTTP/process path; no GPU or paid APIs."""

import asyncio
import copy
import json
import threading
import time
from dataclasses import replace
from functools import partial
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import httpx
import pytest
from benchmark_runtime_fixtures import BOX, config, job, jpeg, report

from goodvision_harness.model_api import ModelAPISettings
from tools.benchmark_runtime.scene_model import _ResponseValidationError
from tools.benchmark_runtime.vlm_event_policy import event_policy_text
from tools.benchmark_runtime.vlm_independent import IndependentEventState
from tools.tracked_grounding.detector import DetectorBroker, validate_snapshot
from tools.tracked_grounding.inspector import (
    TrackedPersonInspector,
    inspector,
    parse_verdicts,
)
from tools.tracked_grounding.suite import LABELS, arguments, main, plan
from tools.vlm_benchmark import modes
from tools.vlm_multistream_benchmark import run_stage


def task(session="camera-one"):
    return job(replace(config(session), labels=LABELS), 0.8, 1)


def answer(ids=("P0001",)):
    return {
        "v": 1,
        "e": {
            label: [
                {
                    "p": [] if label == "ground_litter" else list(ids),
                    "d": False,
                    "c": 0.8,
                }
            ]
            for label in LABELS
        },
    }


def snapshot(session="camera-one", seq=1, timestamp=0.4):
    return {
        "stream_session_id": session,
        "frame_sequence": seq,
        "timestamp_seconds": timestamp,
        "objects": [
            {
                "track_id": "P0001",
                "label": "person",
                "bbox": {k: getattr(BOX, k) for k in ("x1", "y1", "x2", "y2")},
                "observation_state": "observed",
                "last_observed_frame_sequence": seq,
                "last_observed_timestamp_seconds": timestamp,
            }
        ],
    }


def test_plan_is_offline_and_preserves_load_policy(monkeypatch, capsys):
    def forbidden(*a, **kw):
        pytest.fail("planning accessed a model or credential")

    monkeypatch.setattr("urllib.request.urlopen", forbidden)
    monkeypatch.setattr("tools.tracked_grounding.suite.read_api_key", forbidden)
    assert main([]) == 0
    p = json.loads(capsys.readouterr().out)
    assert not p["execute"]
    assert (p["streams"], p["repeats"], p["observation_fps"]) == (12, 2, 1.25)
    assert p["planned_windows_per_branch_per_round"] == 828
    assert p["formal_vlm_requests"] == 6624
    assert plan(baseline=True)["event_policy"] == p["event_policy"]
    args = arguments()
    json.dumps({k: v for k, v in vars(args).items() if k not in {"output", "video"}})


@pytest.mark.parametrize("state", ["ready", "cpu", "wrong-model"])
def test_detector_readiness_requires_yolo26s_gpu(monkeypatch, tmp_path, state):
    client = httpx.Client

    def response(request):
        assert request.url.path == "/health/tracking"
        return httpx.Response(
            200,
            json={
                "ready": True,
                "gpu_ready": state != "cpu",
                "model": "yolo26-s" if state != "wrong-model" else "yolov8-s",
            },
        )

    monkeypatch.setattr(
        httpx,
        "Client",
        lambda **kwargs: client(transport=httpx.MockTransport(response), **kwargs),
    )
    broker = DetectorBroker("http://detector.test", tmp_path)
    if state == "ready":
        assert broker.check_ready() == {"model": "yolo26-s", "gpu_ready": True}
    else:
        with pytest.raises(ValueError, match="YOLO26-S"):
            broker.check_ready()


def test_broker_processes_and_closes_without_python311_runner(tmp_path, monkeypatch):
    monkeypatch.delattr(asyncio, "Runner", raising=False)
    with DetectorBroker(
        "http://detector.test", tmp_path, transport=lambda row, image: []
    ) as broker:
        broker.submit(0, "camera", 1, 0.4, jpeg())
        rows = broker.lookup("camera", [1])
        assert rows[0]["state"] == "ready"
        assert broker.drain()["ready"] == 1
    assert not broker.thread.is_alive()
    assert broker.loop.is_closed()


@pytest.mark.parametrize(
    "problem",
    [
        "unknown-id",
        "missing-person",
        "missing-label",
        "duplicate-person",
        "litter-person",
        "coordinate-output",
        "nonfinite",
        "bool-confidence",
        "duplicate-unassigned",
        "wrong-version",
        "missing-field",
    ],
)
def test_incomplete_or_fabricated_person_results_rejected(problem):
    raw = answer()
    vote = raw["e"]["smoking"][0]
    if problem == "unknown-id":
        vote["p"] = ["P0002"]
    elif problem == "missing-person":
        vote["p"] = []
    elif problem == "missing-label":
        del raw["e"]["fall"]
    elif problem == "duplicate-person":
        raw["e"]["smoking"].append(copy.deepcopy(vote))
    elif problem == "litter-person":
        raw["e"]["ground_litter"][0]["p"] = ["P0001"]
    elif problem == "coordinate-output":
        raw["s"] = [{"i": "P0001", "b": [0, 0, 1, 1]}]
    elif problem == "nonfinite":
        vote["c"] = float("nan")
    elif problem == "bool-confidence":
        vote["c"] = True
    elif problem == "duplicate-unassigned":
        raw["e"]["ground_litter"] *= 2
    elif problem == "wrong-version":
        raw["v"] = True
    else:
        del vote["d"]
    with pytest.raises(_ResponseValidationError):
        parse_verdicts(raw, task(), {"P0001": BOX})


def test_scene_events_without_detected_people_remain_explicit():
    raw = answer(())
    raw["e"]["smoking"][0]["d"] = None
    raw["e"]["ground_litter"][0]["d"] = True
    result = parse_verdicts(raw, task(), {})
    assert len(result) == 6
    assert result[1].detected is None
    assert result[-1].detected is True and result[-1].subject_ids == ()


def test_ids_persist_within_camera_but_do_not_merge_between_cameras():
    cfg = replace(config("a"), labels=LABELS)
    verdicts = parse_verdicts(answer(), task("a"), {"P0001": BOX})
    state = IndependentEventState(cfg)
    r1 = report(state, 1, 2, verdicts)
    r2 = report(state, 2, 4, verdicts)
    r3 = report(
        IndependentEventState(replace(cfg, stream_session_id="b")),
        1,
        2,
        parse_verdicts(answer(), task("b"), {"P0001": BOX}),
    )
    assert r1["objects"][0]["object_id"] == r2["objects"][0]["object_id"]
    assert r1["objects"][0]["object_id"] != r3["objects"][0]["object_id"]
    assert r1["subject_id_scope"] == "stream_session"
    assert r1["objects"][0]["attributes"]["track_id"] == "P0001"
    assert r1["observations"][-1]["subject_ids"] == []


@pytest.mark.parametrize(
    "problem", ["session", "sequence", "timestamp", "nan", "bbox", "duplicate", "stale"]
)
def test_detector_metadata_and_geometry_fail_closed(problem):
    raw = snapshot()
    if problem == "session":
        raw["stream_session_id"] = "other-camera"
    elif problem == "sequence":
        raw["frame_sequence"] += 1
    elif problem == "timestamp":
        raw["timestamp_seconds"] += 0.1
    elif problem == "nan":
        raw["objects"][0]["bbox"]["x1"] = float("nan")
    elif problem == "bbox":
        raw["objects"][0]["bbox"]["x2"] = -0.1
    elif problem == "duplicate":
        raw["objects"] *= 2
    else:
        raw["objects"][0]["last_observed_frame_sequence"] = 0
    with pytest.raises(ValueError):
        validate_snapshot(raw, "camera-one", 1, 0.4)


def test_predictions_are_not_visual_evidence():
    raw = snapshot()
    raw["objects"][0]["observation_state"] = "predicted"
    assert validate_snapshot(raw, "camera-one", 1, 0.4) == []


def test_broker_reuses_ordered_frames_and_qwen_gets_original_context(tmp_path):
    calls = []

    def detect(row, data):
        calls.append((row["stream_session_id"], row["frame_sequence"]))
        return [{"track_id": "P0001", "bbox": BOX.as_dict()}]

    inspection = task()
    with DetectorBroker("http://localhost:1", tmp_path, transport=detect) as broker:
        for frame in inspection.frames:
            broker.submit(
                0,
                inspection.stream_session_id,
                frame.frame_sequence,
                frame.timestamp_seconds,
                frame.jpeg_bytes,
            )
        client = TrackedPersonInspector(
            ModelAPISettings(provider="local"),
            model="fixture",
            broker_url=broker.endpoint,
        )
        try:
            messages = client.build_messages(inspection)
            client.build_messages(inspection)  # Parallel/overlapping consumers reuse.
        finally:
            client.close()
        assert len(calls) == 2
        assert messages[0]["content"].startswith(event_policy_text(LABELS, {}))
        content = messages[1]["content"]
        assert len([p for p in content if p["type"] == "image_url"]) == 2
        metadata = [json.loads(p["text"]) for p in content[1:] if p["type"] == "text"]
        assert metadata[-1]["subject_boxes"][0]["subject_id"] == "P0001"
        assert metadata[-1]["subject_boxes"][0]["bbox"] == [0.1, 0.1, 0.5, 0.9]
        assert inspection.subject_boxes == ()  # Caller evidence is immutable.
        assert client.detector_timing["detector_failed"] == 0
        with pytest.raises(ValueError, match="increase"):
            broker.submit(0, inspection.stream_session_id, 1, 0.4, jpeg())


def test_detector_failure_never_invokes_qwen(tmp_path, monkeypatch):
    def failed(*_):
        raise OSError("fixture")

    def forbidden(*a, **kw):
        pytest.fail("Qwen was invoked after missing detector evidence")

    inspection = task()
    with DetectorBroker("http://localhost:1", tmp_path, transport=failed) as broker:
        for frame in inspection.frames:
            broker.submit(
                0,
                inspection.stream_session_id,
                frame.frame_sequence,
                frame.timestamp_seconds,
                frame.jpeg_bytes,
            )
        client = TrackedPersonInspector(
            ModelAPISettings(provider="local"),
            model="fixture",
            broker_url=broker.endpoint,
        )
        monkeypatch.setattr(client, "_request", forbidden)
        with pytest.raises(ValueError, match="unavailable"):
            client.inspect(inspection)
        assert client.last_http_timing["http_request_attempted"] is False
        assert client.last_inspection_timing["detector_failed"] == 1
        client.close()


def test_bounded_detector_queue_does_not_block_source_or_roll_back_ui(tmp_path):
    entered, release = threading.Event(), threading.Event()

    def slow(row, data):
        entered.set()
        assert release.wait(2)
        return []

    with DetectorBroker(
        "http://localhost:1", tmp_path, transport=slow, capacity=1
    ) as broker:
        broker.submit(0, "camera-a", 0, 0, jpeg())
        assert entered.wait(1)
        broker.submit(0, "camera-a", 1, 0.4, jpeg())
        started = time.monotonic()
        broker.submit(0, "camera-a", 2, 0.8, jpeg())
        assert time.monotonic() - started < 0.2
        assert broker.counts["queue_full"] == 1
        release.set()
        rows = broker.lookup("camera-a", [0, 1, 2])
        assert [r["state"] for r in rows] == ["ready", "ready", "queue_full"]
        assert broker.latest[0]["frame_sequence"] == 2
        assert broker.latest[0]["state"] == "queue_full"


@pytest.mark.parametrize("low_rate", [False, True])
def test_report_never_excludes_missing_or_late_windows(low_rate):
    from tools.benchmark_runtime.vlm_event_policy import event_policy_snapshot
    from tools.tracked_grounding.report import summarize_job

    def record(wid, seconds, *, complete=True, detector_failed=0):
        return {
            "camera": 1,
            "report": {
                "window_id": wid,
                "vlm_lane": "grounding",
                "state": "completed" if complete else "failed",
                "analysis_complete": complete,
                "event_policy": event_policy_snapshot(LABELS),
                "evidence": [
                    {"frame_sequence": 0, "timestamp_seconds": 0},
                    {"frame_sequence": 1, "timestamp_seconds": 0.4},
                ],
                "judgments": [
                    {"event_label": lab, "observed_detected": False} for lab in LABELS
                ],
                "timing": {
                    "source_to_publish_seconds": seconds,
                    "http_request_attempted": not detector_failed,
                    "inspection_timing": {"detector_failed": detector_failed},
                },
            },
        }

    raw = [
        record("w1", 2),
        record("w2", 4),
        record("w3", 1, complete=False, detector_failed=1),
    ]
    reference = {
        "lookup": {(0, 1): "w"},
        "refs": {"w": {"judgments": {lab: {"detected": False} for lab in LABELS}}},
    }
    trial = {
        "id": "test",
        "mode": "compact",
        "repeat": 1,
        "detector_counts": {},
        "recording": {"complete": True},
        "result": {
            "all_drained": True,
            "input_timing_valid": True,
            "result_records_missing": 0,
            "result_retention": {"valid": True},
            "lanes": {
                "grounding": {"http_status_counts": {}, "request_phase_metrics": {}}
            },
        },
    }
    if low_rate:
        trial["result"].update(observation_fps=0.25, source_sample_fps=2.5)
    expected = 168 if low_rate else 828
    metrics = summarize_job(trial, raw, reference)["lanes"]["grounding"]
    assert metrics["complete_fraction"] == 2 / expected
    assert metrics["timely"]["on_time_coverage"] == 1 / expected
    assert not metrics["timely"]["all_cameras_pass"]
    assert metrics["timely"]["worst_camera_coverage"] == 0
    assert metrics["detector_failed_windows"] == 1
    assert metrics["astra"]["binary_comparable"] == 12
    with pytest.raises(ValueError, match="duplicate"):
        summarize_job(trial, [*raw, raw[0]], reference)


@pytest.fixture
def fake_models():
    calls = {"detector": [], "model": []}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def do_HEAD(self):
            self.send_response(200)
            self.end_headers()

        def do_POST(self):
            data = self.rfile.read(int(self.headers["Content-Length"]))
            if self.path == "/v1/tracks":
                seq = int(self.headers["x-goodvision-frame-sequence"])
                timestamp = float(self.headers["x-goodvision-timestamp-seconds"])
                session = self.headers["x-goodvision-stream-session-id"]
                assert self.headers["x-goodvision-detector-model"] == "yolo26-s"
                calls["detector"].append((session, seq))
                time.sleep(0.002)
                raw = snapshot(session, seq, timestamp)
            else:
                payload = json.loads(data)
                calls["model"].append(payload)
                tracked = "Allowed last-frame IDs:" in payload["messages"][0]["content"]
                reply = answer() if tracked else dict.fromkeys(LABELS, False)
                raw = {
                    "id": "gen-local-fixture",
                    "model": payload["model"],
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": json.dumps(reply),
                            },
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"prompt_tokens": 100, "completion_tokens": 50},
                }
            encoded = json.dumps(raw).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}", calls
    server.shutdown()
    server.server_close()
    thread.join()


@pytest.mark.parametrize("mode", ["events", "compact", "parallel"])
@pytest.mark.parametrize("load_profile", ["historical", "four-second"])
def test_twelve_sources_through_http_broker_and_spawned_workers(
    tmp_path, fake_models, monkeypatch, mode, load_profile
):
    url, calls = fake_models
    args = arguments(load_profile=load_profile)
    args.real = True
    low_rate = load_profile == "four-second"
    args.duration, args.output_mode, args.deadline = 5.6 if low_rate else 1.6, mode, 3
    frame_count = 14 if low_rate else 4
    # This checks offline routing/accounting, not the host journal throughput.
    # Keep production source freshness and the real shared gate unchanged.
    args.queue_timeout = 3
    args.admission_storage = (
        "workspace"  # Portable offline fixture, not formal profile.
    )
    args.backend = "local"
    base = modes.configure(arguments())
    base = replace(
        base,
        aiclub=ModelAPISettings(provider="local", endpoint=url + "/chat/completions"),
    )
    monkeypatch.delenv("LOCAL_VLM_API_KEY", raising=False)
    with DetectorBroker(url, tmp_path) as broker:
        result = run_stage(
            args,
            base,
            12,
            [[jpeg()] * frame_count],
            tmp_path,
            1,
            inspector_factory=partial(inspector, broker_url=broker.endpoint),
            source_hook=broker.submit,
        )
        # Model windows can finish before the detector-only source tail.
        detector_counts = broker.drain()
    assert result["input_timing_valid"]
    assert result["all_drained"] and result["result_records_missing"] == 0
    assert result["result_retention"]["valid"]
    assert detector_counts == {
        "submitted": frame_count * 12,
        "ready": frame_count * 12,
    }
    assert (
        len(calls["detector"]) == frame_count * 12
    )  # Same 2.5 FPS detector stream, independent of VLM cadence or lane count.
    assert len(set(calls["detector"])) == frame_count * 12
    for lane in result["lanes"].values():
        expected = 24 if low_rate else 12
        assert lane["expected_observations"] == expected
        assert lane["complete_valid_results"] == expected, lane["errors"]
    if low_rate:
        assert result["source_sample_fps"] == 2.5
        assert result["nominal_requests_per_second"] == (6 if mode == "parallel" else 3)
    rows = [
        json.loads(s)["report"]
        for s in (tmp_path / "streams-12-repeat-1/reports.jsonl")
        .read_text()
        .splitlines()
    ]
    if low_rate:
        assert {r["evidence"][-1]["frame_sequence"] for r in rows} == {2, 12}
        assert {r["evidence"][-1]["timestamp_seconds"] for r in rows} == {0.8, 4.8}
    for row in rows:
        if row["vlm_lane"] == "grounding":
            assert row["subject_id_scope"] == "stream_session"
            assert row["objects"][0]["attributes"]["track_id"] == "P0001"
            assert (
                row["timing"]["source_to_publish_seconds"]
                >= row["timing"]["inspection_timing"]["detector_wait_seconds"]
            )
