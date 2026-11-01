"""Local evaluation accounting and recoverable real-model formatting mistakes."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from goodvision_harness.grounding import parse_grounded_classification

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "low_resolution_model_eval",
    ROOT / "tools/low_resolution_model_eval.py",
)
assert SPEC and SPEC.loader
evaluation = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(evaluation)


def test_resize_reduces_source_information_without_upscaling(tmp_path):
    source = tmp_path / "image.png"
    image = np.zeros((180, 320, 3), np.uint8)
    image[::2, ::2] = 255
    assert cv2.imwrite(str(source), image)
    jpeg, width, height, original, _, _ = evaluation.prepare_frame(source, 0, 32, 75)
    assert (width, height, original) == (32, 18, (320, 180))
    decoded = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
    assert decoded.shape[:2] == (18, 32)
    assert evaluation.prepare_frame(source, 0, 720, 75)[1:3] == (320, 180)


def test_failed_response_is_not_a_negative_or_a_true_negative():
    record = {
        "long_side": 32,
        "error": "invalid_response",
        "actors": [],
        "expected": {"smoking": False},
        "judgments": [],
        "latency_seconds": 1,
    }
    summary = evaluation.summarize([record])["32"]
    assert summary["valid"] == 0
    assert summary["labels"]["smoking"]["missing"] == 1
    assert summary["labels"]["smoking"]["tn"] == 0


@pytest.mark.parametrize("detected", [True, False])
def test_reversed_translations_preserve_real_judgment_and_target(detected):
    objects = [{"label": "person", "bbox_2d": [100, 100, 200, 400]}] if detected else []
    result = parse_grounded_classification(
        json.dumps(
            {
                "detected": detected,
                "confidence": 0.91,
                "description": "The observed evidence. / 当前可见证据。",
                "objects": objects,
            }
        ),
        require_bilingual=True,
    )
    assert result.detected is detected
    assert result.description == "当前可见证据。 / The observed evidence."
    assert len(result.objects) == len(objects)


@pytest.mark.parametrize(
    "text", ["Only English", "只有中文。", " / ", "English / 中文 / more"]
)
def test_missing_or_ambiguous_translation_remains_invalid(text):
    with pytest.raises(ValueError):
        parse_grounded_classification(
            json.dumps(
                {
                    "detected": False,
                    "confidence": 0.9,
                    "description": text,
                    "objects": [],
                }
            ),
            require_bilingual=True,
        )


@pytest.mark.parametrize("include_actors", [False, True])
def test_actor_cost_control_changes_only_optional_text(monkeypatch, include_actors):
    received = []

    def completion(_self, **kwargs):
        received.append(kwargs)
        return {"choices": [{"message": {"content": "{}"}}]}

    monkeypatch.setattr(evaluation.ModelAPIClient, "chat_completion", completion)
    settings = evaluation.load_config(ROOT / "config.yaml").model_api
    monkeypatch.setenv(settings.api_key_env, "unit-test-key")
    transport = evaluation.MeasuredTransport(settings)
    transport.local.include_actors = include_actors
    original = [
        {"role": "system", "content": "Keep current event policy."},
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Event contract\n" + evaluation.ACTOR_ASSOCIATION_RULE,
                },
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/jpeg;base64,AAAA"},
                },
            ],
        },
    ]
    transport.chat_completion(model="test-model", messages=original, max_tokens=1200)
    sent = received[0]
    assert sent["max_tokens"] == 1200
    assert sent["messages"][0] == original[0]
    assert sent["messages"][1]["content"][1] == original[1]["content"][1]
    assert (
        evaluation.ACTOR_ASSOCIATION_RULE in sent["messages"][1]["content"][0]["text"]
    ) == include_actors
    assert evaluation.ACTOR_ASSOCIATION_RULE in original[1]["content"][0]["text"]


@pytest.mark.parametrize(
    "override,expected", [(None, "high"), ("standard", "standard")],
)
def test_cli_records_service_sensitivity_and_legacy_entrypoint(
    monkeypatch, tmp_path, override, expected,
):
    manifest, output = tmp_path / "manifest.json", tmp_path / "report.json"
    manifest.write_text(json.dumps({"clips": []}))
    seen = []

    def client(*args, **kwargs):
        seen.append(kwargs["event_sensitivities"])
        return object()  # Empty job list must not call any model service.

    monkeypatch.setattr(evaluation, "AIClubMultiGroundingClient", client)
    monkeypatch.setattr(evaluation, "MeasuredTransport", lambda *args: object())
    args = ["evaluation", "--manifest", str(manifest), "--output", str(output)]
    if override:
        args += ["--event-sensitivity", "smoking=" + override]
    monkeypatch.setattr(evaluation.sys, "argv", args)
    assert evaluation.main() == 0
    metadata = json.loads(output.with_suffix(".run.json").read_text())
    assert seen[0]["smoking"] == expected
    assert metadata["event_sensitivities"] == seen[0]
    assert metadata["entrypoint"] == "legacy_single_frame"
