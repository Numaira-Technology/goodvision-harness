"""Synthetic software contracts only; no frozen production thresholds or accuracy."""

import copy
import hashlib
import json
import sys
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from PIL import Image
from test_neural_workflow import Encoder as FakeEncoder
from test_neural_workflow import client_and_request

from goodvision_harness import neural_calibration as contract
from goodvision_harness import neural_service as service


def artifact():
    # Deliberately synthetic fixture: these values are not calibration evidence.
    return {
        "schema_version": 1,
        "model": contract.MODEL,
        "revision": contract.REVISION,
        "prompts": {
            key: list(pair) for key, pair in contract.CALIBRATED_PROMPTS.items()
        },
        "preprocessing": contract.PREPROCESSING,
        "aggregation": "max_all_images",
        "comparison": ">",
        "thresholds": dict(
            zip(contract.CORE_EVENTS, [0.1, -0.1, None, 0.0], strict=True)
        ),
        "provenance": {
            "validation_manifest_sha256": "a" * 64,
            "validation_scores_sha256": "b" * 64,
            "selection_protocol_sha256": "c" * 64,
            "source_commit": "d" * 40,
        },
    }


def test_unconfigured_prompts_preserve_original_event_definitions():
    assert service.PROMPTS["fight"] == (
        "People are physically fighting, punching or kicking each other.",
        "People are walking or talking peacefully.",
    )
    assert service.PROMPTS["smoking"] == (
        "A person is smoking a cigarette or vaping.",
        "A person is eating, drinking or touching their face.",
    )
    assert service.PROMPTS["weapon"][1] == (
        "A person is holding a phone or ordinary object."
    )


def write_artifact(tmp_path, document=None):
    path = tmp_path / "synthetic-thresholds.json"
    path.write_text(json.dumps(artifact() if document is None else document))
    return path


def test_artifact_is_hashed_exactly_and_thresholds_are_immutable(tmp_path):
    path = write_artifact(tmp_path)
    loaded = contract.load_thresholds(path)
    assert dict(loaded.values) == artifact()["thresholds"]
    assert loaded.sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(TypeError):
        loaded.values["fight"] = 1


@pytest.mark.parametrize(
    "key,value",
    [
        ("schema_version", True),
        ("schema_version", 2),
        ("model", "other"),
        ("revision", "a" * 40),
        ("prompts", {}),
        ("preprocessing", "center_crop"),
        ("aggregation", "last_image"),
        ("comparison", ">="),
        ("provenance", {}),
        ("thresholds", {}),
        ("thresholds", []),
        ("extra", "ignored?"),
    ],
)
def test_artifact_rejects_incompatible_contract(tmp_path, key, value):
    document = artifact()
    document[key] = value
    with pytest.raises(ValueError):
        contract.load_thresholds(write_artifact(tmp_path, document))


@pytest.mark.parametrize(
    "value",
    [True, "0.1", [], {}, 2.001, -2.001, float("nan"), float("inf"), -(10**400)],
)
def test_artifact_rejects_invalid_thresholds(tmp_path, value):
    document = artifact()
    document["thresholds"]["fight"] = value
    with pytest.raises(ValueError):
        contract.load_thresholds(write_artifact(tmp_path, document))


@pytest.mark.parametrize(
    "raw",
    [
        "[]",
        "null",
        '{"fight": 0.1}',
        '{"schema_version":1,"schema_version":1}',
        '{"thresholds":{"fight":0,"fight":1}}',
        '{"number":NaN}',
        '{"number":Infinity}',
        " " * (64 * 1024 + 1),
    ],
)
def test_artifact_rejects_ambiguous_or_unversioned_json(tmp_path, raw):
    path = tmp_path / "invalid.json"
    path.write_text(raw)
    with pytest.raises(ValueError):
        contract.load_thresholds(path)


@pytest.mark.parametrize("cli", [False, True])
def test_entrypoint_loads_artifact_before_model_and_warms_up(
    tmp_path, monkeypatch, cli
):
    import uvicorn

    path = write_artifact(tmp_path)
    calls = []

    def encoder(cache, thresholds):
        assert cache == "unused"
        assert thresholds.sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
        calls.append("load")
        return SimpleNamespace(warmup=lambda: calls.append("warmup"))

    monkeypatch.setattr(service, "Encoder", encoder)
    monkeypatch.setattr(service, "create_app", lambda *a, **k: "app")
    monkeypatch.setattr(uvicorn, "run", lambda *a, **k: calls.append("listen"))
    monkeypatch.setenv("NEURAL_THRESHOLDS_FILE", str(path) if not cli else "missing")
    monkeypatch.setattr(
        sys,
        "argv",
        ["neural", "--cache-dir", "unused"]
        + (["--thresholds", str(path)] if cli else []),
    )
    service.main()
    assert calls == ["load", "warmup", "listen"]
    path.write_text('{"fight": 0.1}')
    calls.clear()
    with pytest.raises(ValueError):
        service.main()
    assert calls == []


def test_constructor_rejects_raw_thresholds_before_model_import():
    with pytest.raises(ValueError, match="versioned"):
        service.Encoder("unused", {"fight": 0.1})


def score_row(*, threshold=0.1, score=0.1, frame_scores=None, image_index=1):
    return {
        "score_kind": "cosine_margin",
        "score": score,
        "threshold": threshold,
        "frame_scores": [-0.2, score] if frame_scores is None else frame_scores,
        "image_index": image_index,
        "probabilities": [0.9, 0.1, 0.0],
        "thresholds_sha256": "a" * 64,
    }


def response_for(rows):
    class Scores(FakeEncoder):
        def scores(self, images, labels):
            return copy.deepcopy(rows)

    inspector, payload, expected = client_and_request()
    with TestClient(service.create_app(Scores())) as client:
        response = client.post("/predict", json=payload)
    return inspector, expected, response


@pytest.mark.parametrize(
    "threshold,expected_positive",
    [
        (0.1, False),
        (0.10001, False),
        (0.09999, True),
        (None, None),
    ],
)
def test_calibrated_roundtrip_preserves_pair_and_uses_strict_threshold(
    threshold,
    expected_positive,
):
    inspector, expected, response = response_for(
        [
            score_row(threshold=threshold),
            score_row(threshold=None),
        ]
    )
    assert response.status_code == 200
    body = response.json()
    assert body["raw"][0]["probabilities"] == [0.9, 0.1, 0]
    verdicts = inspector._decode(body, expected, ("smoking", "fall"))
    assert [v.detected for v in verdicts] == [expected_positive, None]
    assert all(v.confidence is None for v in verdicts)
    assert inspector.last_response_diagnostics["calibrated_scores"] == body["raw"]


@pytest.mark.parametrize(
    "mutation",
    [
        {"score": True},
        {"threshold": False},
        {"threshold": "0.1"},
        {"score": float("nan")},
        {"score": 3},
        {"frame_scores": [float("nan"), 0.1]},
        {"frame_scores": [0.2, 0.1]},
        {"frame_scores": [0.1]},
        {"frame_scores": [0.1, 0.1]},
        {"image_index": True},
        {"image_index": -1},
        {"image_index": 2},
        {"probabilities": [1, 1, 0]},
        {"probabilities": [True, 0, 0]},
        {"probabilities": [0, 0, 1]},
        {"probabilities": [float("nan"), 0, 0]},
        {"thresholds_sha256": "invalid"},
        {"score_kind": "unknown"},
    ],
)
def test_invalid_calibrated_evidence_fails_service_and_client(mutation):
    bad = {**score_row(), **mutation}
    _, _, failed = response_for([bad, score_row()])
    assert failed.status_code == 503
    inspector, expected, valid = response_for([score_row(), score_row()])
    body = valid.json()
    body["raw"][0].update(mutation)
    with pytest.raises(ValueError):
        inspector._decode(body, expected, ("smoking", "fall"))


@pytest.mark.parametrize(
    "mutation",
    [
        lambda rows: rows[0].pop("threshold"),
        lambda rows: rows[0].pop("score_kind"),
        lambda rows: rows[0].update(answer_index=0, answer="yes"),
        lambda rows: rows[0].update(id="scene/fall"),
        lambda rows: rows.reverse(),
        lambda rows: rows.pop(),
        lambda rows: rows[0].update(thresholds_sha256="b" * 64),
    ],
)
def test_client_rejects_incomplete_or_inconsistent_choices(mutation):
    inspector, expected, response = response_for([score_row(), score_row()])
    body = response.json()
    mutation(body["raw"])
    with pytest.raises(ValueError):
        inspector._decode(body, expected, ("smoking", "fall"))


def test_ppe_and_litter_keep_legacy_decoding_with_calibrated_core():
    inspector, expected, response = response_for([score_row(), score_row()])
    body = response.json()
    options = expected[0]["options"]
    extra = [
        {"id": "scene/" + label, "options": options}
        for label in ("ppe_violation", "ground_litter")
    ]
    body["raw"] += [
        {
            "id": "scene/ppe_violation",
            "pair_scores": ["nan", 0],
            "derived": {"detected": None},
        },
        {
            "id": "scene/ground_litter",
            "task": "choice",
            "answer_index": 0,
            "answer": options[0],
            "probabilities": [0.8, 0.2, 0],
        },
    ]
    verdicts = inspector._decode(
        body, [*expected, *extra], ("smoking", "fall", "ppe_violation", "ground_litter")
    )
    assert [v.detected for v in verdicts] == [False, False, None, True]
    body["raw"][-2].update(score_row())
    with pytest.raises(ValueError):
        inspector._decode(
            body,
            [*expected, *extra],
            ("smoking", "fall", "ppe_violation", "ground_litter"),
        )


@pytest.mark.parametrize(
    "pair", [{}, score_row(), {**score_row(), "score": float("nan")}]
)
def test_warmup_does_not_accept_arbitrary_dictionaries(pair):
    encoder = object.__new__(service.Encoder)
    encoder.scores = lambda *args: [pair] * 6
    with pytest.raises(ValueError, match="invalid model scores"):
        encoder.warmup()


def cpu_transfers(monkeypatch, torch):
    original = torch.Tensor.to
    monkeypatch.setattr(
        torch.Tensor,
        "to",
        lambda self, *a, **k: self if a == ("cuda",) else original(self, *a, **k),
    )


def test_margin_winner_survives_probability_saturation_and_batch_offsets(
    tmp_path,
    monkeypatch,
):
    torch = pytest.importorskip("torch")
    cpu_transfers(monkeypatch, torch)
    encoder = object.__new__(service.Encoder)
    encoder.torch = torch
    encoder.thresholds = contract.load_thresholds(write_artifact(tmp_path))
    encoder.prompts = {**service.PROMPTS, **contract.CALIBRATED_PROMPTS}
    encoder._text_features = torch.eye(12)
    # Both positive probabilities round to 1, but their negative probabilities
    # differ. The margin winner must supply the returned pair, not softmax argmax.
    frames = torch.zeros((3, 12))
    frames[0, 0], frames[0, 2] = 0.6, 0.8
    frames[1, 0], frames[1, 2] = 0.8, 0.6
    frames[2, 0] = 1
    encoder.model = SimpleNamespace(
        vision_model=lambda **k: SimpleNamespace(
            pooler_output=frames[k["pixel_values"]]
        ),
        visual_projection=lambda value: value,
        logit_scale=torch.tensor(100.0).log(),
    )
    output = encoder.score_batch(
        [torch.tensor([0, 1]), torch.tensor([2, 2])], [["fight", "smoking"], ["fight"]]
    )
    first, smoking = output[0]
    assert first["image_index"] == 1 and first["score"] == pytest.approx(0.8)
    assert first["frame_scores"] == pytest.approx([0.6, 0.8])
    expected = (
        (torch.tensor([0.8, 0]) * encoder.model.logit_scale.exp())
        .softmax(dim=-1)
        .tolist()
    )
    assert first["probabilities"] == [*expected, 0]
    assert smoking["image_index"] == 0
    assert output[1][0]["image_index"] == 0  # first tied maximum within this window
    frames[0, 0] = float("nan")
    with pytest.raises(ValueError, match="invalid calibrated frame"):
        encoder.score_batch([torch.tensor([0, 1])], [["fight"]])


@pytest.mark.parametrize("configured", [False, True])
def test_pinned_clip_interfaces_with_cpu_tiny_random_weights(
    tmp_path, monkeypatch, configured
):
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    cpu_transfers(monkeypatch, torch)
    torch.manual_seed(17)
    model = transformers.CLIPModel(
        transformers.CLIPConfig(
            text_config={
                "vocab_size": 32,
                "hidden_size": 16,
                "intermediate_size": 32,
                "num_hidden_layers": 1,
                "num_attention_heads": 2,
                "max_position_embeddings": 8,
                "eos_token_id": 2,
            },
            vision_config={
                "hidden_size": 16,
                "intermediate_size": 32,
                "num_hidden_layers": 1,
                "num_attention_heads": 2,
                "image_size": 224,
                "patch_size": 32,
            },
            projection_dim=12,
        )
    ).eval()
    image_processor = transformers.CLIPImageProcessor()
    text_calls = []

    def processor(*, text=None, images=None, **kwargs):
        if text is not None:
            text_calls.append(text)
            return {
                "input_ids": torch.tensor([[1, index + 3, 2] for index in range(12)])
            }
        return image_processor(images=images, return_tensors="pt")

    # Exercise the actual constructor's prompt selection without downloads or CUDA.
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        transformers.AutoProcessor, "from_pretrained", lambda *a, **k: processor
    )
    monkeypatch.setattr(
        transformers.AutoModel, "from_pretrained", lambda *a, **k: model
    )
    original_to = model.to
    monkeypatch.setattr(
        model, "to", lambda device: model if device == "cuda" else original_to(device)
    )
    thresholds = (
        contract.load_thresholds(write_artifact(tmp_path)) if configured else None
    )
    encoder = service.Encoder("unused", thresholds)
    expected_prompts = (
        {**service.PROMPTS, **contract.CALIBRATED_PROMPTS}
        if configured
        else service.PROMPTS
    )
    assert encoder.prompts == expected_prompts
    encoder.warmup()
    pixels = [
        encoder.prepare(
            [Image.new("RGB", (640, 360), "red"), Image.new("RGB", (360, 640), "blue")]
        )
    ]
    rows = encoder.score_batch(pixels, [["fight", "fall", "ground_litter"]])[0]
    assert text_calls == [[text for pair in expected_prompts.values() for text in pair]]
    with torch.inference_mode():
        features = model.visual_projection(
            model.vision_model(pixel_values=pixels[0]).pooler_output
        )
        features /= features.norm(dim=-1, keepdim=True)
        cosine = features @ encoder._text_features.t()
        margins = (cosine[:, 0] - cosine[:, 1]).tolist()
    if configured:
        assert rows[0]["frame_scores"] == pytest.approx(margins, abs=1e-6)
        assert contract.calibrated_choice(rows[1], "fall", 2) == 2
        assert rows[0]["image_index"] == max(range(2), key=margins.__getitem__)
    else:
        assert all(service._valid_pair(row) for row in rows)
    assert service._valid_pair(rows[-1])
