"""Real local CPU face inference + synthetic remote blocking, without a server.

Repeat a supplied local portrait to test scheduling and stranger confirmation.
This is not a motion benchmark, real-LLM accuracy test, or an HTTP/SSE test.
Requires installed local face dependencies and cached models; no downloads or
external model calls occur. Both model sessions must use CPUExecutionProvider.
"""

# A borrowed Python environment may have another checkout installed editable.
# Resolve this repository before importing the application under test.
# ruff: noqa: E402

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import sys
import tempfile
import threading
import time
from dataclasses import replace
from pathlib import Path

WORKTREE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKTREE / "src"))

import cv2
import numpy as np
import onnxruntime

from goodvision_harness.config import load_config
from goodvision_harness.contracts import NormalizedBBox
from goodvision_harness.face_recognition import (
    FaceRecognitionRuntime,
    FaceRecognitionSession,
    FaceRecognizerConfig,
)
from goodvision_harness.grounding import (
    GroundedEventClassification,
    GroundingResponse,
    ModelObject,
    MultiGroundedClassification,
    SelectedFeaturesGroundingClient,
)
from goodvision_harness.pipeline import SessionCallbacks, StreamingSession


def wait_until(predicate, timeout=10):
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() >= deadline:
            raise TimeoutError("bounded validation wait expired")
        threading.Event().wait(0.01)


def encode(image):
    ok, data = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 95])
    if not ok:
        raise RuntimeError("could not encode in-memory test frame")
    return data.tobytes()


class ControlledSafety:
    def __init__(self, settings, phase_started, gate=None):
        self.settings = settings
        self.phase_started = phase_started
        self.gate = gate
        self.calls = []
        self.active = 0
        self.max_active = 0
        self.lock = threading.Lock()

    def classify_window(self, window):
        started = time.monotonic()
        row = {
            "source_frame_index": window.newest_frame.source_frame_index,
            "started_seconds": started - self.phase_started,
            "input_width": window.newest_frame.width,
            "input_height": window.newest_frame.height,
        }
        with self.lock:
            self.calls.append(row)
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        try:
            if self.gate is not None and not self.gate.wait(20):
                raise TimeoutError("synthetic LLM hard wait limit reached")
            positive = window.newest_frame.source_frame_index % 2 == 0
            objects = (
                (ModelObject("cigarette", NormalizedBBox(0.2, 0.2, 0.3, 0.3)),)
                if positive
                else ()
            )
            row["completed_seconds"] = time.monotonic() - self.phase_started
            row["detected"] = positive
            return GroundingResponse(
                window,
                MultiGroundedClassification(
                    (
                        GroundedEventClassification(
                            "smoking",
                            positive,
                            0.9,
                            "Synthetic scheduler control; not an image verdict.",
                            objects,
                        ),
                    )
                ),
                time.monotonic() - started,
            )
        finally:
            with self.lock:
                self.active -= 1


class MeasuredFace:
    def __init__(self, runtime, phase_started):
        self.session = FaceRecognitionSession(runtime)
        self.settings = self.session.settings
        self.phase_started = phase_started
        self.calls = []
        self.active = 0
        self.max_active = 0
        self.lock = threading.Lock()

    def bind_person_registry(self, registry):
        self.session.bind_person_registry(registry)

    def classify_window(self, window):
        started = time.monotonic()
        row = {
            "source_frame_index": window.newest_frame.source_frame_index,
            "source_timestamp_seconds": window.end_timestamp,
            "started_seconds": started - self.phase_started,
            "input_width": window.newest_frame.width,
            "input_height": window.newest_frame.height,
        }
        with self.lock:
            self.calls.append(row)
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        try:
            response = self.session.classify_window(window)
            row.update(
                completed_seconds=time.monotonic() - self.phase_started,
                latency_seconds=response.latency_seconds,
                detected=(
                    response.classification.detected
                    if response.classification is not None
                    else None
                ),
                diagnostics=response.face_diagnostics,
                error=response.error,
            )
            return response
        finally:
            with self.lock:
                self.active -= 1


class Capture:
    def __init__(self, started):
        self.started = started
        self.results = []
        self.face_completions = []
        self.latest_state = {}
        self.end = None
        self.lock = threading.Lock()

    def on_result(self, result):
        row = {
            "published_seconds": time.monotonic() - self.started,
            "sequence": result.sequence,
            "source_frame_index": result.frame.source_frame_index,
            "source_timestamp_seconds": result.frame.timestamp_seconds,
            "missing_event_labels": list(result.missing_event_labels),
            "judgments": [
                {"event_label": j.event_label, "detected": j.detected}
                for j in result.judgments
            ],
        }
        with self.lock:
            self.results.append(row)

    def on_event(self, event):
        if event.get("branch") == "stranger" and event.get("diagnostics"):
            with self.lock:
                self.face_completions.append(event["diagnostics"])

    def on_state(self, state):
        with self.lock:
            self.latest_state = state

    def on_end(self, reason, error):
        self.end = {"reason": reason, "error": error}

    def feature(self, label):
        with self.lock:
            return [
                {**row, "detected": judgment["detected"]}
                for row in self.results
                for judgment in row["judgments"]
                if judgment["event_label"] == label
            ]


def make_session(runtime, gate=None):
    started = time.monotonic()
    config = load_config(WORKTREE / "config.yaml")
    config = replace(
        config,
        runtime=replace(config.runtime, max_concurrent_requests=6),
        sliding_window=replace(
            config.sliding_window,
            size_frames=1,
            stride_frames=1,
            include_partial_final_window=False,
        ),
        video=replace(config.video, fps=2.5),
    )
    safety = ControlledSafety(config.inference, started, gate)
    face = MeasuredFace(runtime, started)
    client = SelectedFeaturesGroundingClient(
        safety,
        face,
        safety_event_labels=("smoking",),
        face_live_scheduler="single_flight",
    )
    capture = Capture(started)
    session = StreamingSession.for_camera(
        config,
        client,
        SessionCallbacks(
            on_analysis_result=capture.on_result,
            on_event=capture.on_event,
            on_state=capture.on_state,
            on_end=capture.on_end,
        ),
        source_id="offline-local-portrait-scheduling-control",
    )
    session.start()
    return session, safety, face, capture, started


def push(session, image, index, elapsed):
    height, width = image.shape[:2]
    accepted = session.push_camera_frame(
        encode(image),
        width=width,
        height=height,
        source_frame_index=index,
        timestamp_seconds=elapsed,
    )
    if not accepted:
        raise RuntimeError("owned camera session refused a validation frame")


def cleanup(session, gate=None):
    if gate is not None:
        gate.set()
    try:
        wait_until(lambda: not session._has_pending(), timeout=10)
    finally:
        session.stop()
        session.join(timeout=5)
    if session.running:
        raise RuntimeError("owned session did not stop within watchdog")


def blocked_llm_phase(runtime, portrait):
    gate = threading.Event()
    session, safety, face, capture, started = make_session(runtime, gate)
    auto_release = threading.Event()
    timer = threading.Timer(20, lambda: (auto_release.set(), gate.set()))
    timer.daemon = True
    timer.start()
    supplied = []
    report = {}
    try:
        blank = np.zeros(portrait.shape, dtype=np.uint8)
        # Populate all six actual safety-pool slots before any portrait is fed.
        for index in range(6):
            threading.Event().wait(max(0, started + index * 0.4 - time.monotonic()))
            push(session, blank, index, index * 0.4)
            supplied.append({"index": index, "input": "blank"})
        wait_until(lambda: safety.active == 6, timeout=5)
        all_llm_slots_occupied = time.monotonic() - started
        for index in range(6, 31):
            threading.Event().wait(max(0, started + index * 0.4 - time.monotonic()))
            push(session, portrait, index, index * 0.4)
            supplied.append({"index": index, "input": "repeated_local_portrait"})
            if any(row["detected"] for row in capture.feature("stranger")):
                break
        wait_until(
            lambda: any(row["detected"] for row in capture.feature("stranger")),
            timeout=5,
        )
        positive = next(row for row in capture.feature("stranger") if row["detected"])
        before_release = {
            "safety_active": safety.active,
            "safety_public_results": len(capture.feature("smoking")),
            "auto_release_triggered": auto_release.is_set(),
            "gate_set": gate.is_set(),
            "branch_counts": {
                branch: state["counts"]
                for branch, state in capture.latest_state["branches"].items()
            },
        }
        released_seconds = time.monotonic() - started
        gate.set()
        report.update(
            all_six_llm_slots_occupied_seconds=all_llm_slots_occupied,
            first_stranger_positive=positive,
            release_seconds=released_seconds,
            state_before_release=before_release,
            positive_published_before_release=(
                positive["published_seconds"] < released_seconds
                and before_release["safety_active"] == 6
                and not before_release["gate_set"]
                and not before_release["auto_release_triggered"]
            ),
        )
    finally:
        timer.cancel()
        cleanup(session, gate)
    report.update(
        supplied_frames=supplied,
        face_calls=face.calls,
        safety_calls=safety.calls,
        published_results=capture.results,
        pipeline_face_completions=capture.face_completions,
        max_face_concurrency=face.max_active,
        max_safety_concurrency=safety.max_active,
        session_end=capture.end,
        session_stopped=not session.running,
        phase_elapsed_seconds=time.monotonic() - started,
    )
    reliable_portrait_observations = [
        row
        for row in face.calls
        if row["source_frame_index"] >= 6
        and row.get("diagnostics", {}).get("raw_status_counts", {}).get("unknown", 0)
    ]
    report["reliable_portrait_observations_after_llm_saturation"] = len(
        reliable_portrait_observations
    )
    assert report["positive_published_before_release"], report["state_before_release"]
    assert before_release["safety_public_results"] == 0
    assert len(reliable_portrait_observations) >= 3
    assert all(
        row["started_seconds"] > all_llm_slots_occupied
        for row in reliable_portrait_observations
    )
    assert face.max_active == 1 and safety.max_active == 6
    return report


def low_resolution_phase(runtime, portrait):
    session, safety, face, capture, started = make_session(runtime)
    variants = []
    try:
        for index, long_side in enumerate((160, 96, 64, 32, None)):
            if long_side is None:
                image = np.zeros((96, 96, 3), dtype=np.uint8)
            else:
                height, width = portrait.shape[:2]
                scale = long_side / max(height, width)
                image = cv2.resize(
                    portrait,
                    (round(width * scale), round(height * scale)),
                    interpolation=cv2.INTER_AREA,
                )
            input_started = time.monotonic() - started
            push(session, image, index, index * 0.4)
            wait_until(
                lambda index=index: any(
                    row["source_frame_index"] == index
                    for row in capture.feature("smoking")
                )
            )
            wait_until(
                lambda index=index: any(
                    row["source_frame_index"] == index
                    for row in capture.feature("stranger")
                )
            )
            local = next(
                row for row in face.calls if row["source_frame_index"] == index
            )
            remote = next(
                row
                for row in capture.feature("smoking")
                if row["source_frame_index"] == index
            )
            variants.append(
                {
                    "variant": "blank"
                    if long_side is None
                    else f"long_side_{long_side}",
                    "input_width": image.shape[1],
                    "input_height": image.shape[0],
                    "smoking_detected_synthetic": remote["detected"],
                    "safety_public_latency_seconds": remote["published_seconds"]
                    - input_started,
                    "safety_published_before_face_completed": (
                        remote["published_seconds"] < local["completed_seconds"]
                    ),
                    "face_latency_seconds": local["latency_seconds"],
                    "face_diagnostics": local["diagnostics"],
                }
            )
            wait_until(lambda: not session._has_pending())
    finally:
        cleanup(session)
    assert [row["smoking_detected_synthetic"] for row in variants] == [
        True,
        False,
        True,
        False,
        True,
    ]
    assert all(row["safety_published_before_face_completed"] for row in variants)
    assert variants[-1]["face_diagnostics"]["state"] == "no_faces"
    assert any(
        row["face_diagnostics"]["raw_status_counts"]["unusable"] > 0
        for row in variants[:-1]
    )
    return {
        "variants": variants,
        "face_calls": face.calls,
        "safety_calls": safety.calls,
        "published_results": capture.results,
        "max_face_concurrency": face.max_active,
        "session_stopped": not session.running,
        "session_end": capture.end,
        "phase_elapsed_seconds": time.monotonic() - started,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--image",
        type=Path,
        required=True,
        help="Local portrait with a clear face large enough for stranger confirmation",
    )
    parser.add_argument(
        "--model-root",
        type=Path,
        required=True,
        help="Local InsightFace root containing models/buffalo_l; downloads disabled",
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        required=True,
        help="Directory for report.json and the disposable empty gallery",
    )
    args = parser.parse_args()
    args.image = args.image.expanduser().resolve()
    args.model_root = args.model_root.expanduser().resolve()
    args.output_directory = args.output_directory.expanduser().resolve()
    output_path = args.output_directory / "report.json"
    if output_path.exists():
        raise ValueError("refusing to overwrite existing evidence report")
    if not args.image.is_file():
        raise ValueError("--image must be an existing local image file")
    if not (args.model_root / "models/buffalo_l").is_dir():
        raise ValueError("--model-root must contain cached models/buffalo_l")
    image = cv2.imdecode(
        np.frombuffer(args.image.read_bytes(), np.uint8), cv2.IMREAD_COLOR
    )
    if image is None:
        raise ValueError("local input could not be decoded")
    started = time.monotonic()
    args.output_directory.mkdir(parents=True, exist_ok=True)
    model_files_before = {
        p.name: [p.stat().st_size, p.stat().st_mtime_ns]
        for p in (args.model_root / "models/buffalo_l").glob("*.onnx")
    }
    with tempfile.TemporaryDirectory(
        prefix="empty-gallery-", dir=args.output_directory
    ) as tmp:
        gallery_dir = Path(tmp)
        runtime = FaceRecognitionRuntime(
            FaceRecognizerConfig(
                model_root=args.model_root,
                allow_model_download=False,
                provider="CPUExecutionProvider",
                det_size=640,
                gallery_path=gallery_dir / "gallery.json",
            )
        )
        warmup_seconds = runtime.warmup()
        actual_providers = {
            name: model.session.get_providers()
            for name, model in runtime._model.models.items()
            if name in ("detection", "recognition")
        }
        if set(actual_providers) != {"detection", "recognition"} or not all(
            value == ["CPUExecutionProvider"] for value in actual_providers.values()
        ):
            raise RuntimeError("both face model sessions must use CPUExecutionProvider")
        blocked = blocked_llm_phase(runtime, image)
        low_res = low_resolution_phase(runtime, image)
        gallery_files = [
            p.relative_to(gallery_dir).as_posix() for p in gallery_dir.rglob("*")
        ]
        assert not gallery_files
        assert not runtime.list_people()
    model_files_after = {
        p.name: [p.stat().st_size, p.stat().st_mtime_ns]
        for p in (args.model_root / "models/buffalo_l").glob("*.onnx")
    }

    def latency_summary(rows):
        latencies_ms = [row["latency_seconds"] * 1000 for row in rows]
        return {
            "count": len(latencies_ms),
            "p50_ms": round(float(np.percentile(latencies_ms, 50)), 3),
            "p95_ms": round(float(np.percentile(latencies_ms, 95)), 3),
            "max_ms": round(max(latencies_ms), 3),
        }

    first_positive = blocked["first_stranger_positive"]
    portrait_calls = [
        row for row in blocked["face_calls"] if row["source_frame_index"] >= 6
    ]
    report = {
        "scope": {
            "actual_components": [
                "local CPU InsightFace detection and recognition",
                "FaceRecognitionSession and PersonRegistry",
                "SelectedFeaturesGroundingClient independent branch APIs",
                "StreamingSession.start/push_camera_frame/public analysis callback",
            ],
            "synthetic_components": [
                "Repeated supplied still portrait; no camera motion or angle changes",
                "Fake smoking judgments and bounded LLM request blocking",
            ],
            "not_tested": [
                "real LLM quality or latency",
                "HTTP/SSE transport",
                "existing services",
            ],
            "remote_calls": 0,
            "archive_enabled": False,
            "nominal_input_fps": 2.5,
            "safety_capacity": 6,
            "face_scheduler": "single_flight",
            "gate_watchdog_seconds": 20,
        },
        "source": {
            "path": str(args.image),
            "sha256": hashlib.sha256(args.image.read_bytes()).hexdigest(),
            "width": image.shape[1],
            "height": image.shape[0],
        },
        "environment": {
            "python": sys.executable,
            "worktree": str(WORKTREE),
            "insightface_version": importlib.metadata.version("insightface"),
            "onnxruntime_version": onnxruntime.__version__,
            "actual_model_session_providers": actual_providers,
            "model_download_allowed": False,
            "model_files_unchanged_size_mtime": model_files_before == model_files_after,
            "warmup_seconds": warmup_seconds,
        },
        "blocked_llm": blocked,
        "low_resolution": low_res,
        "persistence": {
            "temporary_empty_gallery_files": gallery_files,
            "temporary_gallery_directory_removed": not gallery_dir.exists(),
            "images_written": 0,
            "media_or_embeddings_in_report": False,
            "existing_gallery_modified": False,
        },
        "summary": {
            "passed": True,
            "positive_published_before_gate_release_ms": round(
                (blocked["release_seconds"] - first_positive["published_seconds"])
                * 1000,
                3,
            ),
            "portrait_inference_start_to_positive_seconds": round(
                first_positive["published_seconds"]
                - min(row["started_seconds"] for row in portrait_calls),
                3,
            ),
            "latency_percentiles": {
                "method": (
                    "NumPy linear interpolation; small samples, not a production SLA"
                ),
                "blocked_all_faces": latency_summary(blocked["face_calls"]),
                "blocked_portrait_faces": latency_summary(portrait_calls),
                "observations_forming_first_confirmation": latency_summary(
                    [
                        row
                        for row in portrait_calls
                        if row["source_frame_index"]
                        <= first_positive["source_frame_index"]
                    ]
                ),
                "low_resolution_faces": latency_summary(low_res["face_calls"]),
            },
            "elapsed_seconds": time.monotonic() - started,
        },
    }
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({"output": str(output_path), **report["summary"]}))


if __name__ == "__main__":
    main()
