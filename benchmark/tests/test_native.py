"""CPU verification of real native scene/capture and frozen provenance seams."""

import io
import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bench.native import (
    ROOT,
    ModelCallBudget,
    deterministic_answer,
    evaluate_frames,
    verify_frozen_source,
)
from bench.schema import Question

from goodvision_harness.contracts import NormalizedBBox
from goodvision_harness.detector_native_session import DetectorNativeSessionAdapter
from goodvision_harness.native_event_audit import NativeAudit, redact_images
from goodvision_harness.rich_labeling import (
    RichLabelObject,
    RichLabelProtocolError,
    RichLabelSnapshot,
)
from goodvision_harness.scene_runtime import TemporalVerdict
from goodvision_harness.video_tracking import VideoFrame


def jpeg():
    sink = io.BytesIO()
    Image.new("RGB", (20, 20)).save(sink, format="JPEG")
    return sink.getvalue()


class Tracker:
    def __init__(self, empty_at=()):
        self.empty_at = set(empty_at)
        self.closed = False
        self.last = None

    def open_session(self, session):
        self.session = session

    def append(self, frames, *, stream_session_id, is_last_batch=False):
        snapshots = []
        for frame in frames:
            self.last = frame
            objects = (
                ()
                if frame.frame_sequence in self.empty_at
                else (
                    RichLabelObject(
                        "native1",
                        "person",
                        NormalizedBBox(0.1, 0.1, 0.5, 0.9),
                        observation_state="observed",
                        last_observed_frame_sequence=frame.frame_sequence,
                        last_observed_timestamp_seconds=frame.timestamp_seconds,
                        attributes={
                            "native_track": {
                                "backend": "detector_stream",
                                "epoch": "epoch",
                                "object_id": "native1",
                            }
                        },
                    ),
                )
            )
            snapshots.append(
                RichLabelSnapshot(
                    stream_session_id,
                    frame.frame_sequence,
                    frame.timestamp_seconds,
                    objects,
                )
            )
        return SimpleNamespace(
            frames=tuple(snapshots),
            backend="detector_stream",
            predictor_epoch="epoch",
            completed_frame_sequence=self.last.frame_sequence,
            completed_timestamp_seconds=self.last.timestamp_seconds,
            detector_model="yolo26-s",
            detector_provenance=tuple(
                {
                    "detector_model": "yolo26-s",
                    "weights_sha256": "a" * 64,
                    "actual_parameter_device": "cuda:0",
                }
                for _ in snapshots
            ),
            predicted_frames=tuple({"objects": []} for _ in snapshots),
        )

    def close(self):
        self.closed = True


class Inspector:
    def __init__(self):
        self.closed = False
        self.seen = []

    def inspect(self, job):
        self.seen.append(job)
        subjects = tuple(sorted({subject for _, subject, _ in job.subject_boxes}))
        return tuple(
            TemporalVerdict(label, subjects, True) for label in job.request.event_labels
        )

    def close(self):
        self.closed = True


def frames(count=8):
    return [VideoFrame(index, index * 0.5, jpeg()) for index in range(count)]


def test_native_geometry_keeps_empty_frames_and_distinct_persistent_identity(tmp_path):
    tracker = Tracker(empty_at=(1,))
    result = evaluate_frames(
        frames(4),
        tracker,
        video_id="clip",
        output=tmp_path,
        metadata={"runtime_profile": "current-main", "width": 20, "height": 20},
    )
    assert result["complete"] and tracker.closed
    assert [item["frame"] for item in result["frames"]] == [0, 1, 2, 3]
    assert result["frames"][1]["subjects"] == []
    ids = [f["subjects"][0]["id"] for f in result["frames"] if f["subjects"]]
    assert len(set(ids)) == 1 and ids[0] != "native1"
    question = Question.from_json(
        {
            "qa_id": "q",
            "video_id": "clip",
            "question": "count",
            "capability": "counting",
        }
    )
    assert deterministic_answer(question, result) == "1"
    journal = [
        json.loads(line)
        for line in (tmp_path / "records.jsonl").read_text().splitlines()
    ]
    assert [row["kind"] for row in journal][-2:] == ["tracking_finished", "final_scene"]
    assert journal[0]["final_llm"] is False


def test_native_events_use_current_scene_confirmation_and_drain(tmp_path):
    inspector = Inspector()
    result = evaluate_frames(
        frames(10),
        Tracker(),
        video_id="clip",
        output=tmp_path,
        metadata={"runtime_profile": "current-main"},
        labels=("fight",),
        inspector=inspector,
        observation_fps=1,
    )
    assert inspector.closed and inspector.seen
    assert any(event["confirmed_since"] is not None for event in result["events"])
    question = Question.from_json(
        {
            "qa_id": "q",
            "video_id": "clip",
            "question": "conflict?",
            "capability": "event_recognition",
            "question_form": "event_presence_binary",
            "target_label": "physical_conflict",
        }
    )
    assert deterministic_answer(question, result) == "yes"
    assert all(not hasattr(job, "questions") for job in inspector.seen)


def test_regressed_source_does_not_emit_complete_capture(tmp_path):
    tracker = Tracker()
    with pytest.raises(ValueError, match="strictly increase"):
        evaluate_frames(
            [VideoFrame(0, 0, jpeg()), VideoFrame(1, 0, jpeg())],
            tracker,
            video_id="clip",
            output=tmp_path,
            metadata={},
        )
    assert tracker.closed and not (tmp_path / "capture.json").exists()
    assert (
        json.loads((tmp_path / "records.jsonl").read_text().splitlines()[-1])["kind"]
        == "evaluation_failed"
    )


def response(request, *, device="cuda:0", model="yolo26-s", weight="a" * 64):
    sequence = int(request.headers["X-GoodVision-Frame-Sequence"])
    at = float(request.headers["X-GoodVision-Timestamp-Seconds"])
    sid = request.headers["X-GoodVision-Stream-Session-Id"]
    raw = RichLabelSnapshot(
        sid,
        sequence,
        at,
        (
            RichLabelObject(
                "1",
                "person",
                NormalizedBBox(0.1, 0.1, 0.4, 0.8),
                observation_state="observed",
                last_observed_timestamp_seconds=at,
                last_observed_frame_sequence=sequence,
            ),
        ),
    ).as_dict()
    raw["detector_model"] = "yolo26-s"
    raw["detector_provenance"] = {
        "detector_model": model,
        "weights_sha256": weight,
        "actual_parameter_device": device,
    }
    return httpx.Response(200, json=raw)


def test_detector_adapter_preserves_model_weights_cuda_and_source_epoch():
    requests = []

    def handler(request):
        requests.append(request)
        return response(request)

    adapter = DetectorNativeSessionAdapter(
        "http://test", model="yolo26-s", transport=httpx.MockTransport(handler)
    )
    session = adapter.open_session("s")
    result = adapter.append((VideoFrame(1, 1, jpeg()),), stream_session_id="s")
    assert result.backend == "detector_stream"
    assert (
        result.frames[0].objects[0].attributes["native_track"]["epoch"]
        == session.predictor_epoch
    )
    assert result.detector_provenance[0]["weights_sha256"] == "a" * 64
    assert requests[0].headers["X-GoodVision-Detector-Model"] == "yolo26-s"
    assert (
        adapter.append((), stream_session_id="s", is_last_batch=True).state == "closed"
    )
    adapter.close()


@pytest.mark.parametrize(
    "bad", [{"device": "cpu"}, {"model": "rfdetr-l"}, {"weight": "invalid"}]
)
def test_detector_invalid_provenance_fails_closed(bad):
    adapter = DetectorNativeSessionAdapter(
        "http://test",
        model="yolo26-s",
        transport=httpx.MockTransport(lambda request: response(request, **bad)),
    )
    adapter.open_session("s")
    with pytest.raises(RichLabelProtocolError, match="provenance"):
        adapter.append((VideoFrame(1, 1, jpeg()),), stream_session_id="s")
    adapter.close()


def test_audit_never_overwrites_and_failure_blocks_further_records(tmp_path):
    audit = NativeAudit(tmp_path / "audit.jsonl")
    audit.record("test", value=1)
    with pytest.raises(FileExistsError):
        NativeAudit(audit.path)
    audit._handle.close()
    with pytest.raises(ValueError):
        audit.record("test", value=2)
    with pytest.raises(RuntimeError, match="unavailable"):
        audit.record("test", value=3)
    assert len(audit.rows) == 1


def test_image_audit_hashes_inputs_and_call_budget_prevents_additional_requests():
    redacted = redact_images({"content": [{"url": "data:image/jpeg;base64,AA=="}]})
    assert "data:image" not in json.dumps(redacted)
    assert len(redacted["content"][0]["url"]["image_data_url_sha256"]) == 64
    budget = ModelCallBudget(1)
    budget.reserve()
    with pytest.raises(RuntimeError, match="budget"):
        budget.reserve()
    assert budget.calls == 1


def test_exact_archive_matches_all_frozen_and_publication_hashes():
    commit = verify_frozen_source()
    assert len(commit) == 40 and commit.startswith("d167188")


def test_frozen51_loads_in_separate_process_and_does_not_replace_main_labels():
    program = """
import sys

sys.path.insert(0, "benchmark")
from bench.native import select_runtime, available_events

select_runtime("frozen51")
from goodvision_harness.scene_runtime import SceneRuntime

labels = available_events("frozen51")
SceneRuntime("s", "research", event_labels=(labels[-1],), tracking_mode="batch")
print(len(labels))
"""
    result = subprocess.run(
        [sys.executable, "-c", program], cwd=ROOT, text=True, capture_output=True
    )
    assert result.returncode == 0, result.stderr
    assert int(result.stdout.strip()) >= 51
    from goodvision_harness.grounding import SAFETY_EVENT_LABELS

    assert "person_texts_on_phone" not in SAFETY_EVENT_LABELS


@pytest.mark.parametrize("profile", ["current-main", "frozen51"])
def test_checkout_cli_replays_real_file_into_mock_detector_for_both_profiles(
    tmp_path, profile
):
    """Actual decode/HTTP/CLI/extraction path, without a GPU or model API."""
    from fractions import Fraction
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    from threading import Thread

    import av
    import numpy as np

    video = tmp_path / "clip.mp4"
    with av.open(str(video), "w") as container:
        stream = container.add_stream("mpeg4", rate=10)
        stream.width = 20
        stream.height = 20
        stream.pix_fmt = "yuv420p"
        for index in range(12):
            frame = av.VideoFrame.from_ndarray(
                np.zeros((20, 20, 3), dtype=np.uint8), format="rgb24"
            )
            frame.pts = index
            frame.time_base = Fraction(1, 10)
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            self.rfile.read(int(self.headers["content-length"]))
            requests.append(self.path)
            request = httpx.Request("POST", "http://test", headers=dict(self.headers))
            payload = response(request).content
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    questions = tmp_path / "questions.jsonl"
    questions.write_text(
        json.dumps(
            {
                "qa_id": "q",
                "video_id": "v",
                "question": "count",
                "capability": "counting",
            }
        )
        + "\n"
    )
    media = tmp_path / "media.jsonl"
    media.write_text(json.dumps({"video_id": "v", "path": "clip.mp4"}) + "\n")
    arguments = [
        sys.executable,
        "benchmark/run_native.py",
        "--runtime-profile",
        profile,
        "--questions",
        str(questions),
        "--media",
        str(media),
        "--media-root",
        str(tmp_path),
        "--output-dir",
        str(tmp_path / "result"),
        "--tracking-endpoint",
        f"http://127.0.0.1:{server.server_port}",
        "--geometry-only",
        "--max-model-calls",
        "0",
    ]
    try:
        plan = subprocess.run(
            [*arguments, "--plan-only"], cwd=ROOT, capture_output=True, text=True
        )
        assert plan.returncode == 0, plan.stderr
        assert not requests and not (tmp_path / "result").exists()
        run = subprocess.run(arguments, cwd=ROOT, capture_output=True, text=True)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
    assert run.returncode == 0, run.stderr + run.stdout
    assert requests and all(path.startswith("/v1/tracks") for path in requests)
    predictions = [
        json.loads(line)
        for line in (tmp_path / "result/predictions.jsonl").read_text().splitlines()
    ]
    assert predictions == [{"qa_id": "q", "prediction": "1"}]
    capture = json.loads((tmp_path / "result/questions/q/capture.json").read_text())
    assert capture["complete"] and capture["width"] == 20 and capture["height"] == 20
    assert capture["provenance"]["runtime_profile"] == profile
    records = [
        json.loads(line)
        for line in (tmp_path / "result/questions/q/records.jsonl")
        .read_text()
        .splitlines()
    ]
    eof = next(row for row in records if row["kind"] == "source_eof")
    assert eof["decoded_frames"] == 12 and eof["dropped_due_to_lag"] == 0
    assert not any(row["kind"].startswith("model_") for row in records)


def test_frozen51_native_observations_confirm_an_extended_event(tmp_path):
    """Frozen profile executes native geometry, window validation and lifecycle."""
    program = r"""
import sys

sys.path.insert(0, "benchmark")
from bench.native import select_runtime, evaluate_frames

select_runtime("frozen51")
from types import SimpleNamespace
from goodvision_harness.contracts import NormalizedBBox
from goodvision_harness.rich_labeling import RichLabelObject, RichLabelSnapshot
from goodvision_harness.scene_runtime import TemporalVerdict
from goodvision_harness.video_tracking import VideoFrame


class Tracker:
    def open_session(self, s):
        self.s = s

    def close(self):
        pass

    def append(self, frames, *, stream_session_id, is_last_batch=False):
        rows = []
        for f in frames:
            self.last = f
            obj = RichLabelObject(
                "a",
                "person",
                NormalizedBBox(0.1, 0.1, 0.5, 0.9),
                attributes={
                    "native_track": {
                        "backend": "detector_stream",
                        "epoch": "e",
                        "object_id": "a",
                    }
                },
                observation_state="observed",
                last_observed_timestamp_seconds=f.timestamp_seconds,
                last_observed_frame_sequence=f.frame_sequence,
            )
            rows.append(
                RichLabelSnapshot(self.s, f.frame_sequence, f.timestamp_seconds, (obj,))
            )
        return SimpleNamespace(
            frames=tuple(rows),
            backend="detector_stream",
            predictor_epoch="e",
            completed_frame_sequence=self.last.frame_sequence,
            completed_timestamp_seconds=self.last.timestamp_seconds,
            detector_model="yolo26-s",
            detector_provenance=tuple({"weights_sha256": "a" * 64} for r in rows),
            predicted_frames=tuple({"objects": []} for r in rows),
        )


class Inspector:
    def close(self):
        pass

    def inspect(self, job):
        ids = tuple(sorted({s for _, s, _ in job.subject_boxes}))
        return (TemporalVerdict("person_texts_on_phone", ids, True),)


result = evaluate_frames(
    [VideoFrame(i, i * 0.5, b"jpeg") for i in range(10)],
    Tracker(),
    video_id="v",
    output=sys.argv[1],
    metadata={"runtime_profile": "frozen51"},
    labels=("person_texts_on_phone",),
    inspector=Inspector(),
    observation_fps=1,
)
assert result["complete"]
assert any(
    e["event_label"] == "person_texts_on_phone" and e["confirmed_since"] is not None
    for e in result["events"]
)
"""
    result = subprocess.run(
        [sys.executable, "-c", program, str(tmp_path)],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr


def test_omitted_unconfirmed_and_explicit_negative_events_have_distinct_answers():
    question = Question.from_json(
        {
            "qa_id": "q",
            "video_id": "clip",
            "question": "fight?",
            "capability": "event_recognition",
            "target_label": "physical_conflict",
            "response_options": ["yes", "no"],
            "answer_type": "single_choice",
        }
    )
    base = {"video_id": "clip", "complete": True, "events": [], "observations": []}
    assert deterministic_answer(question, base) == ""
    observed = {
        "accepted": True,
        "requested_event_labels": ["fight"],
        "allowed_subject_ids": ["a"],
        "completion_counts": {"accepted_verdicts": 1},
        "verdicts": [{"event_label": "fight", "subject_ids": ["a"], "detected": False}],
    }
    assert deterministic_answer(question, {**base, "observations": [observed]}) == "no"
    assert (
        deterministic_answer(
            question, {**base, "observations": [{**observed, "verdicts": []}]}
        )
        == ""
    )
    assert (
        deterministic_answer(
            question,
            {
                **base,
                "events": [
                    {
                        "event_label": "fight",
                        "phase": "suspected",
                        "confirmed_since": None,
                    }
                ],
                "observations": [observed],
            },
        )
        == ""
    )


@pytest.mark.parametrize("count", [3, 20])
def test_native_reader_respects_startup_and_short_final_batches(tmp_path, count):
    class StartupTracker(Tracker):
        def __init__(self):
            super().__init__()
            self.appends = []

        def open_session(self, session):
            super().open_session(session)
            return SimpleNamespace(
                limits={"startup_frames": 16, "max_append_frames": 16}
            )

        def append(self, frames, *, stream_session_id, is_last_batch=False):
            if not self.appends and not is_last_batch:
                assert len(frames) >= 16
            self.appends.append((len(frames), is_last_batch))
            return super().append(
                frames, stream_session_id=stream_session_id, is_last_batch=is_last_batch
            )

    tracker = StartupTracker()
    result = evaluate_frames(
        frames(count), tracker, video_id="v", output=tmp_path, metadata={}
    )
    assert len(result["frames"]) == count
    journal = [
        json.loads(line)
        for line in (tmp_path / "records.jsonl").read_text().splitlines()
    ]
    assert (
        next(row for row in journal if row["kind"] == "source_eof")["sampled_frames"]
        == count
    )
    assert tracker.appends == ([(3, True)] if count == 3 else [(16, False), (4, True)])


def negative_window(labels):
    return {
        "accepted": True,
        "requested_event_labels": labels,
        "allowed_subject_ids": ["a"],
        "completion_counts": {"accepted_verdicts": len(labels)},
        "verdicts": [
            {"event_label": label, "subject_ids": ["a"], "detected": False}
            for label in labels
        ],
    }


def test_native_combo_none_and_public_alias_options_are_preserved_exactly():
    base = {
        "video_id": "clip",
        "complete": True,
        "events": [],
        "observations": [negative_window(["fight", "smoking"])],
    }
    combo = Question.from_json(
        {
            "qa_id": "q",
            "video_id": "clip",
            "question": "which combination?",
            "capability": "event_recognition",
            "answer_type": "single_choice",
            "public_context": {
                "response_format": "event_presence_combination",
                "event_labels": ["physical_conflict", "smoking"],
            },
            "response_options": ["A: present; B: absent", "A: absent; B: absent"],
        }
    )
    assert deterministic_answer(combo, base) == "A: absent; B: absent"
    choice = Question.from_json(
        {
            "qa_id": "q",
            "video_id": "clip",
            "question": "which?",
            "capability": "event_recognition",
            "answer_type": "single_choice",
            "response_options": [
                "physical_conflict",
                "smoking",
                "none of these events or actions",
            ],
        }
    )
    assert deterministic_answer(choice, base) == "none of these events or actions"
    positive = {
        **base,
        "events": [
            {"event_label": "fight", "confirmed_since": 1, "phase": "confirmed"}
        ],
    }
    assert deterministic_answer(choice, positive) == "physical_conflict"
    assert deterministic_answer(combo, positive) == "A: present; B: absent"
    assert deterministic_answer(combo, {**base, "observations": []}) == ""


def test_offloaded_image_payloads_are_hashed_without_mutating_inference_input():
    original = {"frames": [{"jpeg_base64": "AA==", "nested": {"image_base64": "BB=="}}]}
    result = redact_images(original)
    assert original["frames"][0]["jpeg_base64"] == "AA=="
    assert result["frames"][0]["jpeg_base64"]["base64_characters"] == 4
    assert "AA==" not in json.dumps(result) and "BB==" not in json.dumps(result)


def test_single_action_does_not_choose_over_an_unconfirmed_positive_competitor():
    question = Question.from_json(
        {
            "qa_id": "q",
            "video_id": "clip",
            "question": "which?",
            "capability": "event_recognition",
            "answer_type": "single_choice",
            "response_options": ["physical_conflict", "smoking"],
        }
    )
    window = negative_window(["fight", "smoking"])
    window["verdicts"][1]["detected"] = True
    capture = {
        "complete": True,
        "events": [
            {"event_label": "fight", "confirmed_since": 1, "phase": "confirmed"}
        ],
        "observations": [window],
    }
    assert deterministic_answer(question, capture) == ""


def test_native_cli_forwards_tracking_token_without_recording_it(tmp_path, monkeypatch):
    import bench.native as native

    import goodvision_harness.config as configuration
    import goodvision_harness.perception as perception

    base = configuration.load_config(ROOT / "config.yaml")
    monkeypatch.setattr(
        configuration,
        "load_config",
        lambda path: replace(
            base, rich_labeling=replace(base.rich_labeling, backend="sam31_stream")
        ),
    )
    secret = "mock-private-tracking-token"
    monkeypatch.setenv("NATIVE_TEST_TOKEN", secret)
    received = []
    monkeypatch.setattr(
        perception,
        "create_tracking_client",
        lambda rich, **options: received.append(options) or object(),
    )

    def fake_evaluation(*args, **kwargs):
        kwargs["metadata"]["_audit"].close()
        return {"complete": True, "video_id": "v", "frames": [], "events": []}

    monkeypatch.setattr(native, "evaluate_frames", fake_evaluation)
    (tmp_path / "clip.mp4").write_bytes(b"unread mock media")
    questions = tmp_path / "questions.jsonl"
    media = tmp_path / "media.jsonl"
    questions.write_text(
        json.dumps(
            {
                "qa_id": "q",
                "video_id": "v",
                "question": "count",
                "capability": "counting",
            }
        )
        + "\n"
    )
    media.write_text(json.dumps({"video_id": "v", "path": "clip.mp4"}) + "\n")
    result = native.main(
        [
            "--questions",
            str(questions),
            "--media",
            str(media),
            "--media-root",
            str(tmp_path),
            "--output-dir",
            str(tmp_path / "result"),
            "--backend",
            "sam",
            "--tracking-endpoint",
            "https://test/v1/contour/sessions",
            "--tracking-key-env",
            "NATIVE_TEST_TOKEN",
            "--geometry-only",
            "--max-model-calls",
            "0",
        ]
    )
    assert result == 0 and received == [{"api_key": secret}]
    assert all(
        secret not in path.read_text()
        for path in (tmp_path / "result").rglob("*.json*")
    )
