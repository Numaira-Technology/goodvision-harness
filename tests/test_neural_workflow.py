"""Exercise the neural HTTP contract without weights, GPU or remote calls."""

import asyncio
import base64
import json
import sys
from concurrent.futures import CancelledError
from dataclasses import replace
from io import BytesIO
from threading import Event, Thread
from types import SimpleNamespace

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
        observed.extend(images)
        assert kwargs == {"return_tensors": "pt"}
        raise ReachedProcessor

    encoder = object.__new__(ModelEncoder)
    encoder.torch = None
    encoder.processor = processor
    with pytest.raises(ReachedProcessor):
        encoder.scores(image, ["smoking"])
    return observed if isinstance(image, list) else observed[0]


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


def test_scene_encoder_pads_every_frame_in_a_mixed_window():
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]
    sizes = [(640, 360), (360, 640), (224, 224), (80, 45)]
    images = [Image.new("RGB", size, color)
              for size, color in zip(sizes, colors, strict=True)]
    processed = scene_processor_input(images)

    assert len(processed) == len(images)
    for image, color in zip(processed, colors, strict=True):
        assert image.mode == "RGB"
        assert image.size == (224, 224)
        assert image.getpixel((112, 112)) == color
    assert processed[0].getpixel((112, 0)) == (123, 117, 104)
    assert processed[1].getpixel((0, 112)) == (123, 117, 104)
    assert [image.size for image in images] == sizes


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
    def prepare(self, images):
        return images

    def score_batch(self, windows, labels):
        return [self.scores(frames, requested)
                for frames, requested in zip(windows, labels, strict=True)]

    def scores(self, images, labels):
        assert all(image.mode == "RGB" for image in images)
        assert labels == ["smoking", "fall"]
        return [[0.9, 0.1], [0.2, 0.8]]


def test_ready_requests_keep_frame_groups_and_cancel_only_queued_work():
    from goodvision_harness.neural_service import _BatchWorker

    observed = []

    def score(windows, labels):
        observed.append((windows, labels))
        return [[(frames[0], label) for label in requested]
                for frames, requested in zip(windows, labels, strict=True)]

    worker = _BatchWorker(score, 4)
    def finish(scores, error, timings):
        return scores, error, timings

    first = worker.submit([1, 2], ['fall', 'smoking'], finish)
    cancelled = worker.submit([99], ['fight'], finish)
    last = worker.submit([3, 4, 5, 6], ['weapon'], finish)
    assert cancelled.cancel()
    worker.thread.start()
    try:
        a, error_a, timing_a = first.result(3)
        b, error_b, timing_b = last.result(3)
        assert a == [(1, 'fall'), (1, 'smoking')]
        assert b == [(3, 'weapon')]
        assert error_a is error_b is None
        assert timing_a['batch_size'] == timing_b['batch_size'] == 2
        assert observed == [([[1, 2], [3, 4, 5, 6]],
                             [['fall', 'smoking'], ['weapon']])]
        with pytest.raises(CancelledError):
            cancelled.result()
        single = worker.submit([7], ['fight'], finish)
        assert single.result(3)[0] == [(7, 'fight')]
    finally:
        worker.pending.put(None)
        worker.thread.join(3)
    assert not worker.thread.is_alive()


def test_missing_batch_member_does_not_leave_later_requests_waiting():
    from goodvision_harness.neural_service import _BatchWorker

    worker = _BatchWorker(lambda windows, labels: [['saved result']], 4)
    def finish(scores, error, timings):
        return scores

    first = worker.submit([1], ['fall'], finish)
    missing = worker.submit([2], ['fall'], finish)
    worker.thread.start()
    try:
        assert first.result(3) == ['saved result']
        with pytest.raises(IndexError):
            missing.result(3)
        later = worker.submit([3], ['fall'], finish)
        assert later.result(3) == ['saved result']
    finally:
        worker.pending.put(None)
        worker.thread.join(3)


@pytest.mark.parametrize("cap", [1, 2, 3, 4])
def test_ready_requests_respect_batch_cap_without_waiting_to_fill(cap):
    from goodvision_harness.neural_service import _BatchWorker

    batches = []

    def score(windows, labels):
        batches.append(windows)
        return windows

    def finish(scores, error, timings):
        assert error is None
        return scores

    worker = _BatchWorker(score, cap)
    futures = [worker.submit([i], ["fall"], finish) for i in range(9)]
    worker.thread.start()
    try:
        assert [future.result(3) for future in futures] == [[i] for i in range(9)]
        assert batches == [
            [[i] for i in range(start, min(start + cap, 9))]
            for start in range(0, 9, cap)
        ]
    finally:
        worker.pending.put(None)
        worker.thread.join(3)
    assert not worker.thread.is_alive()


@pytest.fixture
def submitted_requests(monkeypatch):
    """Observe admission after async preprocessing, without timing sleeps."""
    from goodvision_harness.neural_service import _BatchWorker

    submitted = asyncio.Queue()
    original_submit = _BatchWorker.submit

    def submit(self, *args):
        future = original_submit(self, *args)
        submitted.put_nowait(future)
        return future

    monkeypatch.setattr(_BatchWorker, "submit", submit)
    return submitted


def test_cancelled_queued_http_request_never_reaches_encoder(submitted_requests):
    _, payload, _ = client_and_request()
    entered, release = Event(), Event()
    batch_sizes = []

    class Blocking(Encoder):
        def score_batch(self, windows, labels):
            batch_sizes.append(len(windows))
            if len(batch_sizes) == 1:
                entered.set()
                assert release.wait(5)
            return super().score_batch(windows, labels)

    async def run():
        app = create_app(Blocking())
        async with app.router.lifespan_context(app), httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://offline.test"
        ) as client:
            running = asyncio.create_task(client.post("/predict", json=payload))
            await asyncio.wait_for(submitted_requests.get(), 3)
            try:
                assert await asyncio.to_thread(entered.wait, 3)
                cancelled = asyncio.create_task(client.post("/predict", json=payload))
                queued = await asyncio.wait_for(submitted_requests.get(), 3)
                cancelled.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await cancelled
                assert queued.cancelled()
                later = asyncio.create_task(client.post("/predict", json=payload))
                await asyncio.wait_for(submitted_requests.get(), 3)
                release.set()
                responses = await asyncio.wait_for(asyncio.gather(running, later), 3)
                assert [response.status_code for response in responses] == [200, 200]
            finally:
                release.set()

    asyncio.run(run())
    assert batch_sizes == [1, 1]


def test_batched_ppe_cuda_failure_preserves_scores_and_closes_queued_admission(
    monkeypatch, submitted_requests,
):
    from goodvision_harness import ppe

    _, payload, _ = client_and_request()
    ppe_payload = json.loads(json.dumps(payload))
    ppe_payload["question_template"][1]["id"] = "scene/ppe_violation"
    entered, release = Event(), Event()
    batch_sizes, ppe_calls = [], []
    partial = {"detected": True, "frames": [{"detected": True}]}

    class Blocking(Encoder):
        def score_batch(self, windows, labels):
            batch_sizes.append(len(windows))
            if len(batch_sizes) == 1:
                entered.set()
                assert release.wait(5)
            return [[[0.9, 0.1] for _ in requested] for requested in labels]

    class PPE:
        def __init__(self, encoder):
            pass

        def assess(self, sources, times):
            ppe_calls.append(True)
            error = RuntimeError("CUDA error: illegal memory access")
            error.ppe_observation = partial
            raise error

    monkeypatch.setattr(ppe, "ClipPPE", PPE)

    async def run():
        app = create_app(Blocking(), max_batch_size=3)
        async with app.router.lifespan_context(app), httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://offline.test"
        ) as client:
            first = asyncio.create_task(client.post("/predict", json=payload))
            await asyncio.wait_for(submitted_requests.get(), 3)
            try:
                assert await asyncio.to_thread(entered.wait, 3)
                requests = []
                for body in [ppe_payload, ppe_payload, payload, payload]:
                    requests.append(
                        asyncio.create_task(client.post("/predict", json=body))
                    )
                    # Preprocessing can reorder arrivals; wait for each admission.
                    await asyncio.wait_for(submitted_requests.get(), 3)
                release.set()
                responses = await asyncio.wait_for(
                    asyncio.gather(first, *requests), 3
                )
                assert [r.status_code for r in responses] == [200, 503, 503, 200, 503]
                for response in responses[1:4]:
                    row = response.json()["raw"][0]
                    assert row["id"] == "scene/smoking"
                    assert row["answer_index"] == 0
                assert responses[1].json()["raw"][1]["derived"] == partial
                assert responses[2].json()["raw"][1]["derived"]["detected"] is None
                assert (await client.get("/health")).json()["ready"] is False
                assert (await client.post("/predict", json=payload)).status_code == 503
            finally:
                release.set()

    asyncio.run(run())
    assert batch_sizes == [1, 3]
    assert ppe_calls == [True]


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


def test_encoder_keeps_mixed_window_frames_and_reuses_fixed_text(monkeypatch):
    torch = pytest.importorskip("torch")
    from goodvision_harness.neural_service import Encoder as ModelEncoder

    original_to = torch.Tensor.to

    def cpu_transfer(value, *args, **kwargs):
        return value if args == ('cuda',) else original_to(value, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, 'to', cpu_transfer)

    encoder = object.__new__(ModelEncoder)
    encoder.torch = torch
    encoder._text_features = None
    from goodvision_harness.neural_service import PROMPTS

    encoder.prompts = PROMPTS
    encoder.thresholds = None
    text_calls = []

    def text_model(**kwargs):
        text_calls.append(True)
        return SimpleNamespace(pooler_output=torch.eye(12))

    encoder.processor = lambda **kwargs: {}
    encoder.model = SimpleNamespace(
        text_model=text_model,
        text_projection=lambda value: value,
        vision_model=lambda **kwargs: SimpleNamespace(
            pooler_output=torch.eye(12)[kwargs['pixel_values']]),
        visual_projection=lambda value: value,
        logit_scale=torch.tensor(2.0).log(),
    )
    pairs = encoder.score_batch(
        [torch.tensor([2, 4]), torch.tensor([6]), torch.tensor([0, 0, 0, 0])],
        [['smoking', 'fall'], ['weapon', 'fall'], ['fight']],
    )
    assert pairs[0][0] == pytest.approx([0.88079708, 0.11920292])
    assert pairs[0][1] == pytest.approx([0.88079708, 0.11920292])
    assert pairs[1][0] == pytest.approx([0.88079708, 0.11920292])
    assert pairs[1][1] == pytest.approx([0.5, 0.5])
    assert pairs[2][0] == pytest.approx([0.88079708, 0.11920292])
    encoder.score_batch([torch.tensor([2])], [['smoking']])
    assert text_calls == [True]


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


def test_app_can_start_a_new_worker_after_a_completed_lifespan():
    _, payload, _ = client_and_request()
    app = create_app(Encoder())
    for _ in range(2):
        with TestClient(app) as client:
            assert client.post("/predict", json=payload).status_code == 200


def test_stopped_app_reports_unready_and_rejects_admission():
    _, payload, _ = client_and_request()
    app = create_app(Encoder())

    async def run():
        async with app.router.lifespan_context(app):
            pass
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://offline.test"
        ) as client:
            assert (await client.get("/health")).json()["ready"] is False
            response = await asyncio.wait_for(client.post("/predict", json=payload), 3)
            assert response.status_code == 503

    asyncio.run(run())


def test_shutdown_cancels_queued_work_and_waits_for_running_batch(monkeypatch):
    from goodvision_harness import neural_service

    entered, release, queued_cancelled, shutdown_done = (
        Event(), Event(), Event(), Event()
    )
    calls, errors = [], []

    def score(windows, labels):
        calls.append(windows)
        entered.set()
        assert release.wait(5)
        return windows

    def finish(scores, error, timings):
        assert error is None
        return scores

    worker = neural_service._BatchWorker(score, 4)
    monkeypatch.setattr(neural_service, "_BatchWorker", lambda *args: worker)
    client = TestClient(create_app(Encoder()))
    client.__enter__()

    def shutdown():
        try:
            client.__exit__(None, None, None)
        except Exception as error:
            errors.append(error)
        finally:
            shutdown_done.set()

    closing = Thread(target=shutdown)
    try:
        running = worker.submit([1], ["fall"], finish)
        assert entered.wait(3)
        queued = worker.submit([2], ["fall"], finish)
        queued.add_done_callback(lambda future: queued_cancelled.set())
        closing.start()
        assert queued_cancelled.wait(3)
        assert queued.cancelled()
        assert not shutdown_done.wait(0.05)
        assert worker.thread.is_alive()
        release.set()
        assert shutdown_done.wait(3)
        assert running.result(3) == [1]
        assert calls == [[[1]]]
        assert not worker.thread.is_alive()
        assert not errors
    finally:
        release.set()
        if closing.ident is None:
            closing.start()
        closing.join(3)


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

    class FrameEncoder(Encoder):
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

    class Invalid(Encoder):
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
    app = create_app(Encoder())
    with TestClient(app):
        inspector._http = httpx.AsyncClient(transport=httpx.ASGITransport(app=app))
        try:
            verdicts = inspector.inspect(job(inspector.config, 1, 2))
            assert [v.detected for v in verdicts] == [True, False]
            assert inspector.last_http_timing["http_status"] == 200
            assert inspector.last_http_timing['neural_batch_size'] == 1
            assert inspector.last_http_timing['neural_model_pipeline_seconds'] >= 0
        finally:
            inspector.close()


def test_base_cuda_failure_closes_shared_admission():
    _, payload, _ = client_and_request()
    calls = []

    class Failed(Encoder):
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

    class Scores(Encoder):
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


def test_cancelled_request_keeps_gpu_serial_and_next_request_completes():
    _, payload, _ = client_and_request()
    entered, release, prepared = Event(), Event(), Event()
    calls = []

    class Blocking(Encoder):
        def prepare(self, images):
            if entered.is_set():
                prepared.set()
            return images

        def scores(self, *args):
            calls.append(True)
            entered.set()
            assert release.wait(5)
            return super().scores(*args)

    async def run():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://offline.test",
        ) as client:
            first = asyncio.create_task(client.post("/predict", json=payload))
            try:
                assert await asyncio.to_thread(entered.wait, 3)
                first.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await first
                second = asyncio.create_task(client.post("/predict", json=payload))
                assert await asyncio.to_thread(prepared.wait, 3)
                assert calls == [True]
                release.set()
                assert (await second).status_code == 200
            finally:
                release.set()

    app = create_app(Blocking())
    with TestClient(app):
        asyncio.run(run())
