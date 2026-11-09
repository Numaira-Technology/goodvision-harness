"""Exercise the neural HTTP contract without weights, GPU or remote calls."""

import asyncio
import base64
import json
import sys
from dataclasses import replace
from io import BytesIO
from threading import Event
from types import SimpleNamespace

import httpx
import pytest
from fastapi.testclient import TestClient
from test_vlm_independent import config, job

from goodvision_harness.model_api import ModelAPISettings
from goodvision_harness.neural_model import NeuralInspector
from goodvision_harness.neural_service import MODEL, create_app
from goodvision_harness.vlm_independent import _inspector


@pytest.mark.parametrize("failure", [False, True])
def test_entrypoint_requires_inference_warmup_before_listening(monkeypatch, failure):
    import uvicorn
    from PIL import Image

    from goodvision_harness import neural_service

    events = []
    encoder = object.__new__(neural_service.Encoder)

    def scores(image, labels):
        assert isinstance(image, Image.Image)
        assert labels == list(neural_service.PROMPTS)
        events.append("inference")
        if failure:
            raise RuntimeError("CUDA kernel unavailable")
        return [[0.5, 0.5] for _ in labels]

    monkeypatch.setattr(encoder, "scores", scores)
    monkeypatch.setattr(neural_service, "Encoder", lambda cache: encoder)
    monkeypatch.setattr(sys, "argv", ["neural_service", "--cache-dir", "unused"])
    monkeypatch.setattr(uvicorn, "run", lambda *a, **k: events.append("listen"))
    if failure:
        with pytest.raises(RuntimeError, match="CUDA kernel"):
            neural_service.main()
        assert events == ["inference"]
    else:
        neural_service.main()
        assert events == ["inference", "listen"]


@pytest.mark.parametrize("pairs", [[], [[float("nan"), 0]] * 6, [[0.8, 0.8]] * 6])
def test_warmup_rejects_invalid_model_output(monkeypatch, pairs):
    from goodvision_harness.neural_service import Encoder as ModelEncoder

    encoder = object.__new__(ModelEncoder)
    monkeypatch.setattr(encoder, "scores", lambda *a: pairs)
    with pytest.raises(ValueError, match="invalid model scores"):
        encoder.warmup()


def client_and_request():
    cfg = replace(
        config(),
        model=MODEL,
        model_settings=ModelAPISettings(
            provider="local", endpoint="http://localhost:8094"
        ),
        output_mode="events",
    )
    inspector = _inspector(cfg)
    assert isinstance(inspector, NeuralInspector)
    body, expected = inspector._payload(job(cfg, 1, 2))
    return inspector, json.loads(body), expected


class Encoder:
    def scores(self, images, labels):
        assert all(image.mode == "RGB" for image in images)
        assert labels == ["smoking", "fall"]
        return [[0.9, 0.1], [0.2, 0.8]]


def test_window_pairs_keep_each_events_strongest_frame():
    from goodvision_harness.neural_service import _window_pairs

    assert _window_pairs([
        [[0.9, 0.1], [0.2, 0.8]],
        [[0.3, 0.7], [0.8, 0.2]],
    ]) == [[0.9, 0.1], [0.8, 0.2]]
    assert _window_pairs([[[0.4, 0.6]]]) == [[0.4, 0.6]]


def test_window_pairs_do_not_hide_an_invalid_earlier_frame():
    from goodvision_harness.neural_service import _valid_pair, _window_pairs

    pair = _window_pairs([[[float("nan"), 0]], [[0.9, 0.1]]])[0]
    assert not _valid_pair(pair)


def test_encoder_uses_logits_from_both_frames():
    torch = pytest.importorskip("torch")
    from goodvision_harness.neural_service import Encoder as ModelEncoder

    encoder = object.__new__(ModelEncoder)
    encoder.torch = torch
    encoder.processor = lambda **kwargs: {}
    encoder.model = lambda **kwargs: SimpleNamespace(
        logits_per_image=torch.tensor([[2.0, 0.0, 0.0, 2.0], [0.0, 2.0, 3.0, 0.0]])
    )
    pairs = encoder.scores([object(), object()], ["smoking", "fall"])
    assert pairs[0] == pytest.approx([0.88079708, 0.11920292])
    assert pairs[1] == pytest.approx([0.95257413, 0.04742587])


def test_real_http_payload_decodes_scene_verdicts_and_sends_template_each_time():
    inspector, payload, expected = client_and_request()
    with TestClient(create_app(Encoder())) as client:
        assert client.get("/health").json() == {"ready": True, "model": MODEL}
        response = client.post("/predict", json=payload)
    assert response.status_code == 200
    result = inspector._decode(response.json(), expected, ("smoking", "fall"))
    assert [v.detected for v in result] == [True, False]
    assert all(v.confidence is None for v in result)
    inspector._template_registered = True
    again, _ = inspector._payload(job(inspector.config, 2, 3))
    assert "question_template" in json.loads(again)
    with pytest.raises(ValueError, match="mismatch"):
        inspector._decode(
            {**response.json(), "model": "wrong"}, expected, ("smoking", "fall")
        )


@pytest.mark.parametrize(
    "values,detected",
    [([255, 0], True), ([0, 255], True), ([0, 0], False), ([0, 255, 0, 0], True)],
)
def test_window_detects_evidence_in_either_input_frame(values, detected):
    from PIL import Image

    inspector, payload, expected = client_and_request()
    payload["images"] = []
    for value in values:
        encoded = BytesIO()
        Image.new("RGB", (8, 8), (value, value, value)).save(encoded, "JPEG")
        payload["images"].append(base64.b64encode(encoded.getvalue()).decode())

    class FrameEncoder:
        def scores(self, images, labels):
            frames = images if isinstance(images, list) else [images]
            present = any(image.getpixel((0, 0))[0] > 200 for image in frames)
            return [[0.9, 0.1] if present else [0.1, 0.9] for _ in labels]

    with TestClient(create_app(FrameEncoder())) as client:
        response = client.post("/predict", json=payload)
    assert response.status_code == 200
    verdicts = inspector._decode(response.json(), expected, ("smoking", "fall"))
    assert [verdict.detected for verdict in verdicts] == [detected, detected]


@pytest.mark.parametrize(
    "mutation",
    [
        lambda p: p.update(question_template=[None]),
        lambda p: p["question_template"][0].update(id=1),
        lambda p: p["question_template"][0].update(id="unknown"),
        lambda p: p["question_template"][0].update(options="abc"),
        lambda p: p["question_template"][0].update(options=["yes", "no", "maybe"]),
        lambda p: p.update(images=[]),
        lambda p: p.update(images=[base64.b64encode(b"not an image").decode()]),
        lambda p: p.update(
            images=[base64.b64encode(b"bad first frame").decode(), p["images"][-1]]
        ),
        lambda p: p.update(request_id=1),
    ],
)
def test_malformed_requests_fail_without_model_calls(mutation):
    _, payload, _ = client_and_request()
    mutation(payload)

    class Unused:
        def scores(self, *args):
            pytest.fail("invalid request reached the model")

    with TestClient(create_app(Unused())) as client:
        assert client.post("/predict", json=payload).status_code == 400


def test_invalid_model_scores_do_not_publish_negative_detection():
    _, payload, _ = client_and_request()

    class Invalid:
        def scores(self, *args):
            return [[float("nan"), 0], [0.2, 0.8]]

    with TestClient(create_app(Invalid())) as client:
        assert client.post("/predict", json=payload).status_code == 503


def test_inspector_performs_real_asgi_inference_with_exact_predict_endpoint():
    inspector, _, _ = client_and_request()
    inspector.settings = replace(
        inspector.settings, endpoint="http://model.test/predict"
    )
    inspector._loop = asyncio.new_event_loop()
    inspector._http = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=create_app(Encoder()))
    )
    try:
        verdicts = inspector.inspect(job(inspector.config, 1, 2))
        assert [v.detected for v in verdicts] == [True, False]
        assert inspector.last_http_timing["http_status"] == 200
    finally:
        inspector.close()


def test_base_cuda_failure_closes_shared_admission():
    _, payload, _ = client_and_request()
    calls = []

    class Failed:
        def scores(self, *args):
            calls.append(True)
            raise RuntimeError("CUDA error: illegal memory access")

    with TestClient(create_app(Failed()), raise_server_exceptions=False) as client:
        response = client.post("/predict", json=payload)
        assert response.status_code == 503
        assert response.json()["error"] == "cuda_shared_failure"
        assert client.get("/health").json()["ready"] is False
        assert client.post("/predict", json=payload).status_code == 503
    assert calls == [True]


def test_ppe_is_derived_from_both_frames_and_local_failures_keep_other_labels(
    monkeypatch,
):
    from goodvision_harness import ppe

    inspector, payload, expected = client_and_request()
    payload["question_template"][1]["id"] = "scene/ppe_violation"
    expected[1]["id"] = "scene/ppe_violation"
    observed = []

    class Scores:
        def scores(self, image, labels):
            return [[0.9, 0.1], [float("nan"), 0.1]]

    class PPE:
        def __init__(self, encoder):
            pass

        def assess(self, sources, times):
            observed.append((sources, times))
            raise ValueError("local crop failure")

    monkeypatch.setattr(ppe, "ClipPPE", PPE)
    with TestClient(create_app(Scores())) as client:
        response = client.post("/predict", json=payload)
        assert response.status_code == 200
        assert client.get("/health").json()["ready"] is True
    raw = response.json()["raw"]
    assert raw[1]["pair_scores"] == ["nan", 0.1]
    assert raw[1]["derived"]["detected"] is None
    assert observed == [
        ([base64.b64decode(x) for x in payload["images"]], payload["frame_times"])
    ]
    assert [
        v.detected
        for v in inspector._decode(
            response.json(),
            expected,
            ("smoking", "ppe_violation"),
        )
    ] == [True, None]


def test_cancelled_request_keeps_gpu_lock_until_worker_finishes():
    _, payload, _ = client_and_request()
    entered, release = Event(), Event()

    class Blocking(Encoder):
        def scores(self, *args):
            entered.set()
            assert release.wait(5)
            return super().scores(*args)

    async def run():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=create_app(Blocking())),
            base_url="http://offline.test",
        ) as client:
            first = asyncio.create_task(client.post("/predict", json=payload))
            try:
                assert await asyncio.to_thread(entered.wait, 3)
                first.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await first
                assert (await client.post("/predict", json=payload)).status_code == 429
            finally:
                release.set()

    asyncio.run(run())
