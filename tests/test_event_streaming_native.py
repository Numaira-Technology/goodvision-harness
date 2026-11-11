"""Native streaming contracts with in-process services and no model weights."""

import io
import json
import urllib.error
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from PIL import Image
from test_event_streaming import fake_video, reply, window

from goodvision_harness import neural_service
from goodvision_harness.videojev_core import OPTIONS as CORE_OPTIONS
from tools.event_benchmark import backends, realtime


@pytest.fixture(autouse=True)
def no_provider_traffic(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("offline tests must never call a provider")

    monkeypatch.setattr(backends.urllib.request, "urlopen", forbidden)


def native(kind="neural", questions=None):
    model, _ = backends.CANDIDATES["clip-neural" if kind == "neural" else kind]
    return backends.RemoteBackend(
        model,
        kind,
        questions or {"fight": "Fight?"},
        endpoint="http://offline.invalid/predict",
    )


def choice(backend, label, index=1, **extra):
    return {
        "id": backend._wire_id(label),
        "task": "choice",
        "answer_index": index,
        "answer": backend.options[index],
        "probabilities": [float(i == index) for i in range(3)],
        **extra,
    }


def response(backend, rows):
    return {
        "request_id": "window-1",
        "model": backend.model,
        **(
            {"revision": backends.REVISIONS[backend.model]}
            if backend.backend != "neural"
            else {}
        ),
        "raw": rows,
    }


def asgi_transport(monkeypatch, client):
    calls = []

    def send(request, **kwargs):
        body = json.loads(request.data)
        calls.append(body)
        result = client.post("/predict", json=body)
        if result.status_code >= 400:
            raise urllib.error.HTTPError(
                request.full_url,
                result.status_code,
                "in-process service error",
                result.headers,
                io.BytesIO(result.content),
            )
        return nullcontext(
            SimpleNamespace(status=result.status_code, read=lambda _: result.content)
        )

    monkeypatch.setattr(backends.urllib.request, "urlopen", send)
    return calls


def jpeg(color):
    output = io.BytesIO()
    Image.new("RGB", (8, 8), color).save(output, format="JPEG")
    return output.getvalue()


def test_neural_body_uses_fixed_product_contract_and_deduplicates_aliases():
    backend = native(
        questions={
            "fight": "Custom text is not executed",
            "scene/fight": "Alias",
            "ppe_violation": "PPE?",
            "bag_knife": "Knife?",
        }
    )
    body = backend._body(window())
    assert body["frame_times"] == [0, 1]
    assert len(body["images"]) == 2
    assert body["question_template"] == [
        {"id": "scene/" + label, "task": "choice", "options": CORE_OPTIONS}
        for label in ("fight", "ppe_violation")
    ]
    assert backend.metadata["unsupported_events"] == ["bag_knife"]
    assert backend.metadata["scene_prompts"]["fight"] == neural_service.PROMPTS["fight"]


@pytest.mark.parametrize("kind", ["videojev", "neural"])
@pytest.mark.parametrize("detected", [True, False, None])
def test_ppe_uses_derived_state_even_when_diagnostic_choice_is_invalid(
    monkeypatch, kind, detected
):
    backend = native(kind, {"ppe_violation": "PPE?", "fight": "Fight?"})
    rows = [
        {
            "id": "scene/ppe_violation",
            "probabilities": "invalid diagnostic",
            "derived": {"detected": detected},
        },
        choice(backend, "fight"),
    ]
    reply(monkeypatch, response(backend, [rows]))
    result = backend.predict(window())["predictions"]
    assert result["ppe_violation"] == backends._result(detected)
    assert result["fight"]["answer"] is False
    assert result["fight"]["format_state"] == "valid"


@pytest.mark.parametrize("kind", ["videojev", "neural"])
@pytest.mark.parametrize("derived", [None, {}, {"detected": 0}, {"detected": "false"}])
def test_invalid_ppe_never_falls_back_to_native_choice_or_discards_siblings(
    monkeypatch, kind, derived
):
    backend = native(kind, {"ppe_violation": "PPE?", "fight": "Fight?"})
    rows = [
        choice(backend, "ppe_violation", derived=derived),
        choice(backend, "fight"),
    ]
    reply(monkeypatch, response(backend, rows))
    result = backend.predict(window())["predictions"]
    assert result["ppe_violation"]["answer"] is None
    assert result["ppe_violation"]["format_state"] == "invalid"
    assert result["fight"]["answer"] is False


@pytest.mark.parametrize("count", [0, 2])
def test_missing_or_duplicate_ppe_rows_are_invalid(monkeypatch, count):
    backend = native(questions={"ppe_violation": "PPE?"})
    rows = [choice(backend, "ppe_violation", derived={"detected": False})] * count
    reply(monkeypatch, response(backend, rows))
    result = backend.predict(window())["predictions"]["ppe_violation"]
    assert result["answer"] is None
    assert result["format_state"] == "invalid"


@pytest.mark.parametrize("field", ["request_id", "model"])
def test_neural_identity_mismatch_preserves_unsupported_status(monkeypatch, field):
    backend = native(questions={"fight": "Fight?", "bag_knife": "Knife?"})
    payload = response(backend, [choice(backend, "fight")])
    payload[field] = "wrong"
    reply(monkeypatch, payload)
    result = backend.predict(window())["predictions"]
    assert result["fight"]["request_state"] == "response_mismatch"
    assert result["bag_knife"]["request_state"] == "not_sent"


def test_all_unsupported_events_never_transmit():
    result = native(questions={"bag_knife": "Knife?"}).predict(window())
    assert result["raw"] is None
    assert result["predictions"]["bag_knife"]["request_state"] == "not_sent"


def test_neural_roundtrip_keeps_unsupported_event_in_denominator(tmp_path, monkeypatch):
    fake_video(monkeypatch, fps=5, frames=2)
    questions = {"fight": "Fight?", "bag_knife": "Knife?"}
    backend = native(questions=questions)
    seen = []

    def scores(image, labels):
        seen.append((image.getpixel((0, 0)), labels))
        return [[0.1, 0.9]]

    samples = [
        {"number": i, "filename": "clip.mp4", "event": label, "proposed_answer": "no"}
        for i, label in enumerate(questions)
    ]
    with TestClient(
        neural_service.create_app(SimpleNamespace(scores=scores))
    ) as client:
        calls = asgi_transport(monkeypatch, client)
        result = realtime.run_streams(
            samples,
            {"clip.mp4": tmp_path / "clip.mp4"},
            questions,
            backend,
            tmp_path / "run.json",
            slots=1,
            workers=1,
            interval=0.2,
            frame_width=4,
        )
    assert len(calls) == 1
    assert seen == [((2, 2, 2), ["fight"])]  # Product CLIP uses the last frame.
    assert result["overall"]["total"] == 2
    assert result["overall"]["correct"] == 1
    assert result["overall"]["unknown"] == 1
    unsupported = result["per_video"][1]
    assert unsupported["pred"] == "unknown"
    assert unsupported["evidence"]["not_sent"] == 1
    assert not unsupported["evidence_complete"]


@pytest.mark.parametrize(
    "outcome", [True, False, None, "local_failure", "cuda_failure"]
)
def test_neural_ppe_service_integration_keeps_frames_and_partial_evidence(
    monkeypatch, outcome
):
    ppe = pytest.importorskip("goodvision_harness.ppe", reason="requires merged PR #77")
    backend = native(
        questions={"ppe_violation": "PPE?", "fight": "Fight?", "bag_knife": "Knife?"}
    )
    images, seen = [jpeg("red"), jpeg("blue")], []

    class PPE:
        def __init__(self, encoder):
            pass

        def assess(self, sources, times):
            seen.append((sources, times))
            if outcome == "local_failure":
                raise ValueError("one detector failed")
            if outcome == "cuda_failure":
                error = RuntimeError("CUDA error: illegal memory access")
                error.ppe_observation = {"detected": True, "frames": []}
                raise error
            return {"detected": outcome, "frames": []}

    monkeypatch.setattr(ppe, "ClipPPE", PPE)
    encoder = SimpleNamespace(scores=lambda image, labels: [[0.1, 0.9]] * len(labels))
    with TestClient(neural_service.create_app(encoder)) as client:
        calls = asgi_transport(monkeypatch, client)
        result = backend.predict(window(images=images))
    expected = (
        True
        if outcome == "cuda_failure"
        else None
        if outcome == "local_failure"
        else outcome
    )
    assert seen == [(images, [0, 1])]
    assert len(calls) == 1
    assert result["predictions"]["ppe_violation"] == backends._result(expected)
    assert result["predictions"]["fight"]["answer"] is False
    assert result["predictions"]["bag_knife"]["request_state"] == "not_sent"
    assert (result["shared_error"] == "cuda_shared_failure") == (
        outcome == "cuda_failure"
    )
    assert "ppe_seconds" in result["timings"]
