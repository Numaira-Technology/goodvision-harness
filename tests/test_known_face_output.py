from __future__ import annotations

import json
from concurrent.futures import Future
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import cv2
import jsonschema
import numpy as np
import pytest
from fastapi.testclient import TestClient

import goodvision_harness.face_recognition as face_module
import goodvision_harness.overlay as overlay_module
import goodvision_harness.service as service_module
from goodvision_harness.config import load_config
from goodvision_harness.contracts import AnalysisResult
from goodvision_harness.delayed_overlay import OverlaySelection
from goodvision_harness.face_recognition import FaceRecognitionRuntime
from goodvision_harness.grounding import (
    GroundedEventClassification,
    GroundingResponse,
    MultiGroundedClassification,
    SelectedFeaturesGroundingClient,
)
from goodvision_harness.pipeline import SessionCallbacks, StreamingSession
from goodvision_harness.sliding_window import FrameWindow, SampledFrame

ROOT = Path(__file__).resolve().parents[1]


def _face(left: int, embedding: tuple[float, ...]) -> SimpleNamespace:
    return SimpleNamespace(
        bbox=np.asarray((left, 80, left + 80, 200), dtype=np.float32),
        embedding=np.asarray(embedding, dtype=np.float32),
        det_score=0.98,
    )


@pytest.mark.parametrize("mode", ["face", "stranger", "multi"])
@pytest.mark.parametrize("include_unknown", [False, True])
def test_enrolled_ids_survive_pipeline_http_and_annotated_video(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    mode: str, include_unknown: bool,
) -> None:
    monkeypatch.setenv("HARNESS_AUTOSTART", "false")
    first = _face(10, (1.0, 0.0, 0.0))
    second = _face(110, (0.0, 1.0, 0.0))
    observed = [first, second]
    if include_unknown:
        observed.append(_face(210, (0.0, 0.0, 1.0)))
    responses = iter(([first], [second], observed))
    monkeypatch.setattr(
        face_module, "_load_face_analysis",
        lambda _config: SimpleNamespace(get=lambda _image: next(responses)),
    )
    config = load_config(ROOT / "config.yaml")
    face = FaceRecognitionRuntime(replace(
        config.face, gallery_path=tmp_path / "faces.json", threshold=0.8,
    ))
    image = np.zeros((240, 320, 3), dtype=np.uint8)
    ok, encoded = cv2.imencode(".jpg", image)
    assert ok
    jpeg = encoded.tobytes()
    # Identical names must not collapse two distinct enrolled identities.
    face.enroll("employee-007", "Alex", jpeg)
    face.enroll("staff-008", "Alex", jpeg)
    window = FrameWindow(index=0, frames=(SampledFrame(
        sample_index=0, source_frame_index=0, timestamp_seconds=0.0,
        jpeg_bytes=jpeg, width=320, height=240,
    ),))

    class ClearSafetyClient:
        settings = config.inference

        def classify_window(self, window: FrameWindow) -> GroundingResponse:
            return GroundingResponse(
                window=window, latency_seconds=0.01,
                classification=MultiGroundedClassification(judgments=(
                    GroundedEventClassification(
                        "smoking", False, 0.99, "No smoking.",
                    ),
                )),
            )

    client = face if mode == "face" else SelectedFeaturesGroundingClient(
        ClearSafetyClient() if mode == "multi" else None,
        face, safety_event_labels=("smoking",) if mode == "multi" else (),
        include_stranger=True,
    )
    emitted: list[AnalysisResult] = []
    session = StreamingSession.for_camera(
        config, client, SessionCallbacks(on_analysis_result=emitted.append),
        source_id="known-face-test",
    )
    future: Future[Any] = Future()
    future.set_result(client.classify_window(window))
    session._on_window_done(future, window)
    assert len(emitted) == 1
    analysis = emitted[0]
    payload = analysis.as_dict()
    schema = json.loads(
        (ROOT / "schemas/analysis-result-v2.schema.json").read_text(encoding="utf-8")
    )
    jsonschema.validate(payload, schema)
    known = [item for item in payload["objects"] if item["label"] == "known_person"]
    assert {item["attributes"]["person_id"] for item in known} == {
        "employee-007", "staff-008",
    }
    assert all(item["attributes"]["name"] == "Alex" for item in known)
    assert all(item["attributes"]["similarity"] == pytest.approx(1.0) for item in known)
    assert all(item["attributes"]["threshold"] == 0.8 for item in known)
    stranger = next(
        item for item in payload["judgments"]
        if item["event_label"] in {"stranger", "unknown_person"}
    )
    assert stranger["detected"] is include_unknown
    assert stranger["description"].startswith("最新帧人脸识别结果为2名已知人员")
    assert " / Face recognition on the newest frame: 2 known" in stranger["description"]
    assert len(stranger["object_ids"]) == int(include_unknown)
    assert not {item["object_id"] for item in known} & set(stranger["object_ids"])
    if not include_unknown:
        assert payload["description"] == "无异常 / No anomalies detected"
    for item in payload["objects"]:
        if item["attributes"]["status"] != "known":
            assert "person_id" not in item["attributes"]
            assert "name" not in item["attributes"]
        assert "embedding" not in item["attributes"]

    runtime = service_module.HarnessRuntime(config, face_runtime=face)
    runtime.results.publish(payload)
    with TestClient(service_module.create_app(lambda: runtime)) as http:
        assert http.get("/v1/results/latest").json() == payload
        selected = http.get("/v1/results/stranger/latest")
        assert selected.status_code == 200
        selected_payload = selected.json()
        assert selected_payload["objects"] == payload["objects"]
        assert selected_payload["judgments"][0]["object_ids"] == stranger["object_ids"]
        assert selected_payload["judgments"][0]["detected"] is include_unknown
        if mode == "multi":
            smoking = http.get("/v1/results/smoking/latest").json()
            assert smoking["objects"] == []

        labels: list[str] = []
        original_text = overlay_module._text_with_background

        def capture_text(image: Any, text: str, *args: Any, **kwargs: Any) -> None:
            labels.append(text)
            original_text(image, text, *args, **kwargs)

        monkeypatch.setattr(overlay_module, "_text_with_background", capture_text)
        rendered_frames = (
            service_module._render_analysis_jpeg(jpeg, analysis, quality=95),
            runtime._render_fused_label_jpeg(
                jpeg, OverlaySelection(analysis, "exact"), 0.0, 0,
            ),
        )
        for rendered in rendered_frames:
            pixels = cv2.imdecode(np.frombuffer(rendered, np.uint8), cv2.IMREAD_COLOR)
            assert pixels is not None
            for left in (10, 110):
                blue, green, red = (int(value) for value in pixels[150, left])
                assert green > 130 and blue < 110 and red < 110
            if include_unknown:
                blue, green, red = (int(value) for value in pixels[198, 210])
                assert red > 150 and blue < 110 and green < 110
        expected_labels = (
            ("ID: employee-007 | Alex", "ID: staff-008 | Alex")
            if mode == "face" else ("employee-007", "staff-008")
        )
        for expected in expected_labels:
            assert sum(expected in text for text in labels) == 2
