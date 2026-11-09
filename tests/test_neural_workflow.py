"""Exercise the neural HTTP contract without weights, GPU or remote calls."""

import asyncio
import base64
import json
import sys
from dataclasses import replace
from threading import Event

import httpx
import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image, ImageDraw
from test_vlm_independent import config, job

from goodvision_harness.model_api import ModelAPISettings
from goodvision_harness.neural_model import NeuralInspector
from goodvision_harness.neural_service import MODEL, create_app
from goodvision_harness.vlm_independent import _inspector


def scene_processor_input(image):
    """Stop at the real encoder's processor boundary, without loading a model."""
    from goodvision_harness.neural_service import Encoder as ModelEncoder

    class ReachedProcessor(Exception):
        pass

    observed = []

    def processor(*, images, **kwargs):
        observed.append(images)
        assert kwargs == {"return_tensors": "pt"}
        raise ReachedProcessor

    encoder = object.__new__(ModelEncoder)
    encoder.torch = None
    encoder.processor = processor
    encoder._text_features = {"smoking": object()}  # Text-cache math has separate tests.
    with pytest.raises(ReachedProcessor):
        encoder.scores(image, ["smoking"])
    return observed[0]


@pytest.mark.parametrize("size,bounds", [
    ((640, 360), (0, 49, 224, 175)),
    ((360, 640), (49, 0, 175, 224)),
    ((641, 359), (0, 50, 224, 175)),
    ((80, 45), (0, 49, 224, 175)),
    ((224, 224), (0, 0, 224, 224)),
    ((448, 448), (0, 0, 224, 224)),
])
def test_scene_encoder_preserves_edges_and_proportions(size, bounds):
    width, height = size
    image = Image.new("RGB", size)
    draw = ImageDraw.Draw(image)
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]
    for (x, y), color in zip(
        [(0, 0), (7 * width // 8, 0),
         (0, 7 * height // 8), (7 * width // 8, 7 * height // 8)],
        colors, strict=True,
    ):
        draw.rectangle((x, y, x + width // 8, y + height // 8), fill=color)
    radius = min(size) // 8
    draw.rectangle(
        (width // 2 - radius, height // 2 - radius,
         width // 2 + radius, height // 2 + radius), fill="white",
    )
    original = image.tobytes()

    processed = scene_processor_input(image)

    assert processed.mode == "RGB"
    assert processed.size == (224, 224)
    assert image.size == size and image.tobytes() == original
    pixels = np.asarray(processed)
    content = np.any(pixels != (123, 117, 104), axis=-1)
    ys, xs = np.nonzero(content)
    assert (xs.min(), ys.min(), xs.max() + 1, ys.max() + 1) == bounds
    left, top, right, bottom = bounds
    for (x, y), color in zip(
        [(left + 2, top + 2), (right - 3, top + 2),
         (left + 2, bottom - 3), (right - 3, bottom - 3)],
        colors, strict=True,
    ):
        assert processed.getpixel((x, y)) == color
    # A square landmark must remain square, including on portrait inputs.
    ys, xs = np.nonzero(np.all(pixels > 245, axis=-1))
    assert abs((xs.max() - xs.min()) - (ys.max() - ys.min())) <= 1


def test_scene_padding_keeps_the_manually_selected_stereo_view():
    from goodvision_harness.frame_crop import FrameCropSelector
    from goodvision_harness.video_to_stream import LiveFrame

    stereo = np.zeros((360, 1280, 3), dtype=np.uint8)
    stereo[:, :320] = (0, 0, 255)  # Left edge of the chosen view, in BGR.
    stereo[:, 320:640] = (0, 255, 0)  # Right edge of the same view.
    stereo[:, 640:] = (255, 0, 0)  # Other camera; must not reach CLIP.
    frame, = FrameCropSelector("left_half").frames(
        iter([LiveFrame(7, 1.5, stereo)]), stopped=lambda: False,
    )
    processed = scene_processor_input(Image.fromarray(frame.bgr[:, :, ::-1]))

    assert (frame.index, frame.timestamp_seconds) == (7, 1.5)
    assert processed.size == (224, 224)
    assert processed.getpixel((0, 112)) == (255, 0, 0)
    assert processed.getpixel((223, 112)) == (0, 255, 0)
    assert not np.any(np.all(np.asarray(processed) == (0, 0, 255), axis=-1))


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
    def scores(self, image, labels):
        assert image.mode == "RGB"
        assert labels == ["smoking", "fall"]
        return [[0.9, 0.1], [0.2, 0.8]]


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
    "mutation",
    [
        lambda p: p.update(question_template=[None]),
        lambda p: p["question_template"][0].update(id=1),
        lambda p: p["question_template"][0].update(id="unknown"),
        lambda p: p["question_template"][0].update(options="abc"),
        lambda p: p["question_template"][0].update(options=["yes", "no", "maybe"]),
        lambda p: p.update(images=[]),
        lambda p: p.update(images=[base64.b64encode(b"not an image").decode()]),
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
