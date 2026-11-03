"""Compare the cached encoder with CLIP.forward offline, without model weights.

The dedicated CI job installs the neural service's pinned torch/transformers.
Only CUDA transfers and pretrained loading are replaced; tokenization, image
processing, transformer layers, projections, logits and softmax are real.
"""

import asyncio
import base64
import io
import itertools
from threading import Event
from types import SimpleNamespace

import httpx
import numpy as np
import pytest
from PIL import Image

from goodvision_harness import neural_service, ppe
from goodvision_harness.videojev_core import OPTIONS


def jpeg(image):
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    return buffer.getvalue()


def payload(image, labels):
    return {
        "request_id": "offline-clip-cache",
        "images": [base64.b64encode(jpeg(image)).decode()] * 2,
        "frame_times": [0.0, 1.0],
        "question_template": [
            {"id": f"scene/{label}", "task": "choice", "options": list(OPTIONS)}
            for label in labels
        ],
    }


@pytest.fixture
def clip(monkeypatch, request):
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    # Avoid network/cache access even if an accidental loader call is introduced.
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    dtype = getattr(torch, getattr(request, "param", "float32"))
    alphabet = list("abcdefghijklmnopqrstuvwxyz.,'-")
    tokens = [*alphabet, *(c + "</w>" for c in alphabet)]
    tokens += ["<|startoftext|>", "<|endoftext|>"]
    tokenizer = transformers.CLIPTokenizer(
        vocab={token: i for i, token in enumerate(tokens)},
        merges=[],
        model_max_length=77,
    )
    processor = transformers.CLIPProcessor(
        tokenizer=tokenizer,
        image_processor=transformers.CLIPImageProcessor(
            size={"shortest_edge": 16}, crop_size={"height": 16, "width": 16}
        ),
    )
    config = transformers.CLIPConfig(
        text_config={
            "vocab_size": len(tokenizer),
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "max_position_embeddings": 77,
            "bos_token_id": tokenizer.bos_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            "pad_token_id": tokenizer.pad_token_id,
        },
        vision_config={
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "image_size": 16,
            "patch_size": 4,
        },
        projection_dim=8,
    )
    # Exercise the service's CUDA-only constructor and transfer requests on CPU.
    tensor_to = torch.Tensor.to

    def cpu_cuda(self, *args, **kwargs):
        if args and str(args[0]).startswith("cuda"):
            args = ("cpu", *args[1:])
        if str(kwargs.get("device", "")).startswith("cuda"):
            kwargs["device"] = "cpu"
        return tensor_to(self, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "to", cpu_cuda)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    loads, models, calls = [], [], []

    def load_processor(name, **kwargs):
        loads.append(("processor", name, kwargs))
        return processor

    def load_model(name, **kwargs):
        loads.append(("model", name, kwargs))
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(33 + len(models))
            model = transformers.CLIPModel(config).to(dtype=dtype)
        models.append(model)
        for part in ("text_model", "vision_model"):

            def record(_module, _inputs, part=part):
                calls.append((part, torch.is_inference_mode_enabled()))

            getattr(model, part).register_forward_pre_hook(record)
        return model

    monkeypatch.setattr(transformers.AutoProcessor, "from_pretrained", load_processor)
    monkeypatch.setattr(transformers.AutoModel, "from_pretrained", load_model)
    try:
        yield SimpleNamespace(
            torch=torch,
            processor=processor,
            calls=calls,
            loads=loads,
            encoder=neural_service.Encoder("unused-offline-cache"),
            image=Image.fromarray(
                np.random.default_rng(5).integers(0, 256, (20, 28, 3), dtype=np.uint8)
            ),
        )
    finally:
        torch.set_num_threads(threads)


def original_forward(clip, image, labels):
    inputs = clip.processor(
        images=image,
        text=[text for label in labels for text in neural_service.PROMPTS[label]],
        return_tensors="pt",
        padding=True,
        truncation=True,
    )
    with clip.torch.inference_mode():
        return clip.encoder.model(**inputs)


@pytest.mark.parametrize("clip", ["float32", "float64"], indirect=True)
def test_cached_logits_and_probabilities_match_every_subset_and_order(
    clip, monkeypatch
):
    torch, encoder = clip.torch, clip.encoder
    captured = []
    softmax = torch.Tensor.softmax

    def record_logits(self, *args, **kwargs):
        captured.append(self.detach().clone())
        return softmax(self, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "softmax", record_logits)
    labels = list(neural_service.PROMPTS)
    subsets = [
        list(subset)
        for size in range(1, len(labels) + 1)
        for subset in itertools.combinations(labels, size)
    ]
    # Cold-cache first request is fall-only; later calls request unseen labels.
    encoder.scores(clip.image, ["fall"])
    cache = encoder._text_features
    assert clip.calls == [("text_model", True), ("vision_model", True)]
    for image in (clip.image, Image.new("RGB", (28, 20), "white")):
        for subset in subsets:
            for ordered in (subset, subset[::-1]):
                expected = original_forward(clip, image, ordered)
                clip.calls.clear()
                actual = encoder.scores(image, ordered)
                logits = captured[-1]
                torch.testing.assert_close(
                    logits,
                    expected.logits_per_image[0].reshape(len(ordered), 2),
                    rtol=1e-5,
                    atol=1e-5,
                )
                torch.testing.assert_close(
                    torch.tensor(actual, dtype=expected.logits_per_image.dtype),
                    softmax(expected.logits_per_image[0].reshape(len(ordered), 2), -1),
                    rtol=1e-5,
                    atol=1e-5,
                )
                assert clip.calls == [("vision_model", True)]
                assert encoder._text_features is cache
    assert all(
        value.dtype == encoder.model.text_projection.weight.dtype
        for value in cache.values()
    )
    assert all(
        value.device == next(encoder.model.parameters()).device
        for value in cache.values()
    )
    assert all(not value.requires_grad for value in cache.values())
    assert all(not module.training for module in encoder.model.modules())
    assert clip.loads == [
        (
            kind,
            neural_service.MODEL,
            {
                "revision": neural_service.REVISION,
                "cache_dir": "unused-offline-cache",
            },
        )
        for kind in ("processor", "model")
    ]


@pytest.mark.parametrize("stage", ["processor", "text_model", "text_projection"])
def test_failed_initialization_retries_without_publishing_partial_cache(
    clip,
    monkeypatch,
    stage,
):
    encoder = clip.encoder
    target = encoder if stage == "processor" else encoder.model
    attribute = "processor" if stage == "processor" else stage
    original = getattr(target, attribute)
    calls = []

    def fail_once(*args, **kwargs):
        calls.append(True)
        if len(calls) == 1:
            raise RuntimeError("temporary text initialization failure")
        return original(*args, **kwargs)

    if stage == "processor":
        monkeypatch.setattr(target, attribute, fail_once)
    else:
        original = original.forward
        monkeypatch.setattr(getattr(target, attribute), "forward", fail_once)
    with pytest.raises(RuntimeError, match="temporary text initialization"):
        encoder.scores(clip.image, ["fall"])
    assert encoder._text_features is None
    expected = original_forward(clip, clip.image, ["weapon", "fall"])
    actual = encoder.scores(clip.image, ["weapon", "fall"])
    clip.torch.testing.assert_close(
        clip.torch.tensor(actual),
        expected.logits_per_image[0].reshape(2, 2).softmax(-1),
    )


def test_cache_belongs_to_each_model_instance(clip):
    encoder = clip.encoder
    encoder.scores(clip.image, ["fall"])
    other = neural_service.Encoder("unused-offline-cache")
    assert other._text_features is None
    other.scores(clip.image, ["fall"])
    assert other._text_features is not encoder._text_features
    assert not clip.torch.equal(
        other._text_features["fall"],
        encoder._text_features["fall"],
    )


def test_first_request_serializes_text_initialization_and_recovers_admission(
    clip,
    monkeypatch,
):
    entered, release = Event(), Event()
    original = clip.encoder.model.text_model.forward

    def blocking(*args, **kwargs):
        entered.set()
        assert release.wait(5)
        return original(*args, **kwargs)

    monkeypatch.setattr(clip.encoder.model.text_model, "forward", blocking)

    async def run():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=neural_service.create_app(clip.encoder)),
            base_url="http://offline.test",
        ) as client:
            body = payload(clip.image, ["fall"])
            first = asyncio.create_task(client.post("/predict", json=body))
            try:
                assert await asyncio.to_thread(entered.wait, 3)
                assert (await client.post("/predict", json=body)).status_code == 429
                assert clip.encoder._text_features is None
            finally:
                release.set()
            assert (await first).status_code == 200
            assert (await client.post("/predict", json=body)).status_code == 200

    asyncio.run(run())
    assert [part for part, _ in clip.calls].count("text_model") == 1
    assert [part for part, _ in clip.calls].count("vision_model") == 2


def test_ppe_cache_and_derived_results_survive_mixed_event_requests(clip, monkeypatch):
    # Real CLIP PPE scoring, with synthetic person crops in place of YOLO.
    monkeypatch.setattr(
        ppe.ClipPPE, "_people", lambda self, image: [({"person_id": "person01"}, image)]
    )
    decoded = Image.open(io.BytesIO(jpeg(clip.image))).convert("RGB")
    adapter = ppe.ClipPPE(clip.encoder)
    expected = adapter.assess([jpeg(clip.image)] * 2, [0.0, 1.0])
    event_expected = original_forward(clip, decoded, ["weapon", "ppe_violation"])
    from fastapi.testclient import TestClient
    from test_ppe_batch import stable

    with TestClient(neural_service.create_app(clip.encoder)) as client:
        assert (
            client.post("/predict", json=payload(clip.image, ["fall"])).status_code
            == 200
        )
        cache = clip.encoder._text_features
        clip.calls.clear()
        first = client.post(
            "/predict", json=payload(clip.image, ["weapon", "ppe_violation"])
        )
        second = client.post(
            "/predict", json=payload(clip.image, ["ppe_violation", "weapon"])
        )
    assert first.status_code == second.status_code == 200
    first_rows, second_rows = first.json()["raw"], second.json()["raw"]
    assert first_rows[0]["id"] == second_rows[1]["id"] == "scene/weapon"
    assert first_rows[1]["id"] == second_rows[0]["id"] == "scene/ppe_violation"
    assert (
        stable(first_rows[1]["derived"])
        == stable(second_rows[0]["derived"])
        == stable(expected)
    )
    clip.torch.testing.assert_close(
        clip.torch.tensor([row["probabilities"][:2] for row in first_rows]),
        event_expected.logits_per_image[0].reshape(2, 2).softmax(-1),
    )
    assert clip.encoder._text_features is cache
    assert [part for part, _ in clip.calls].count("text_model") == 1
