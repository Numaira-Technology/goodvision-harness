"""Independent HF-native oracle using tiny random CPU weights, never downloads."""

import os
from types import MappingProxyType

import pytest
from PIL import Image, ImageDraw

# The light regression matrix may omit this optional stack. Explicit CPU
# validation sets this flag so missing dependencies fail instead of skipping.
if os.environ.get("NEURAL_REQUIRE_NUMERICAL_ORACLE") == "1":
    import torch
    import transformers
else:
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")

from goodvision_harness import neural_service as service
from goodvision_harness.neural_calibration import FrozenThresholds


@pytest.fixture(params=[False, True], ids=["legacy-pairs", "calibrated-margins"])
def tiny_encoder(monkeypatch, request):
    torch.manual_seed(17)
    model = transformers.CLIPModel(
        transformers.CLIPConfig(
            text_config={
                "vocab_size": 128,
                "hidden_size": 24,
                "intermediate_size": 48,
                "num_hidden_layers": 1,
                "num_attention_heads": 4,
                "max_position_embeddings": 16,
                "bos_token_id": 0,
                "eos_token_id": 127,
            },
            vision_config={
                "hidden_size": 24,
                "intermediate_size": 48,
                "num_hidden_layers": 1,
                "num_attention_heads": 4,
                "image_size": 224,
                "patch_size": 32,
            },
            projection_dim=16,
        )
    ).eval()
    image_processor = transformers.CLIPImageProcessor()
    from goodvision_harness.neural_calibration import CALIBRATED_PROMPTS

    prompts = (
        {**service.PROMPTS, **CALIBRATED_PROMPTS}
        if request.param else service.PROMPTS
    )
    rows = {}
    for index, prompt in enumerate(text for pair in prompts.values() for text in pair):
        ids = torch.randint(
            1, 127, (8,), generator=torch.Generator().manual_seed(1000 + index)
        )
        ids[0], ids[-1] = 0, 127
        rows[prompt] = ids
    text_calls = []

    def text_inputs(text):
        ids = torch.stack([rows[prompt] for prompt in text])
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}

    def processor(*, images=None, text=None, **kwargs):
        if text is not None:
            text_calls.append(text)
            return text_inputs(text)
        return image_processor(images=images, return_tensors="pt")

    original_to = torch.Tensor.to

    def cpu_transfer(value, *args, **kwargs):
        args = tuple("cpu" if arg == "cuda" else arg for arg in args)
        if kwargs.get("device") == "cuda":
            kwargs["device"] = "cpu"
        return original_to(value, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "to", cpu_transfer)
    original_model_to = model.to
    monkeypatch.setattr(
        model, "to",
        lambda device: model if device == "cuda" else original_model_to(device),
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        transformers.AutoModel, "from_pretrained", lambda *args, **kwargs: model
    )
    monkeypatch.setattr(
        transformers.AutoProcessor, "from_pretrained", lambda *args, **kwargs: processor
    )
    # This is synthetic configuration, not a calibrated production artifact.
    thresholds = FrozenThresholds(
        MappingProxyType({"fight": 0.1, "smoking": -0.1, "fall": None, "weapon": 0.0}),
        "f" * 64,
        MappingProxyType({
            "policy": "two-frame-target-spacing-v1",
            "candidate_fps": 4.0,
            "observation_window_frames": 2,
            "observation_frame_delta_seconds": 0.5,
            "observation_fps": 1.25,
        }),
    ) if request.param else None
    encoder = service.Encoder("unused-no-download", thresholds)
    assert encoder.prompts == prompts
    assert next(model.parameters()).device.type == "cpu"
    return encoder, model, text_inputs, text_calls


def frames(count, offset):
    result = []
    for index in range(offset, offset + count):
        image = Image.new(
            "RGB", (280 + index * 9, 180 + index * 17),
            ((index * 71) % 256, (index * 29) % 256, (index * 113) % 256),
        )
        draw = ImageDraw.Draw(image)
        draw.rectangle(
            (index * 11, 10, image.width // 2, image.height - 10),
            fill=(255 - (index * 29) % 256, 80, (index * 67) % 256),
        )
        result.append(image)
    return result


def native_window(model, text_inputs, prompts, pixels, labels):
    """Use the public HF forward, not the production manual decomposition."""
    with torch.inference_mode():
        output = model(
            pixel_values=pixels,
            **text_inputs([text for label in labels for text in prompts[label]]),
        )
        pairs = output.logits_per_image.reshape(len(pixels), len(labels), 2)
        pairs = pairs.softmax(dim=-1)
        # HF's output embeddings are normalized by its own full forward.
        cosine = (output.image_embeds @ output.text_embeds.t()).reshape(
            len(pixels), len(labels), 2
        )
        margins = cosine[:, :, 0] - cosine[:, :, 1]
    return pairs, margins


@pytest.mark.parametrize(
    "counts,labels",
    [
        ([1], [list(service.PROMPTS)]),
        ([2], [["weapon", "smoking", "fall"]]),
        ([1, 2, 4], [["weapon", "fall"], ["smoking", "weapon", "fight"],
                     ["ground_litter", "ppe_violation", "fight"]]),
    ],
    ids=["all-labels", "reordered-subset", "mixed-one-two-four-frame-windows"],
)
def test_manual_forward_matches_native_complete_windows(
    tiny_encoder, counts, labels, monkeypatch,
):
    encoder, model, text_inputs, text_calls = tiny_encoder
    pixels = [encoder.prepare(frames(count, index * 4))
              for index, count in enumerate(counts)]
    assert [value.shape[0] for value in pixels] == counts
    assert all(value.shape[-2:] == (224, 224) for value in pixels)
    actual = encoder.score_batch(pixels, labels)
    assert len(actual) == len(counts)

    earlier_winners = []
    for window, requested, got in zip(pixels, labels, actual, strict=True):
        pairs, margins = native_window(
            model, text_inputs, encoder.prompts, window, requested
        )
        assert len(got) == len(requested)
        for column, (label, row) in enumerate(zip(requested, got, strict=True)):
            if encoder.thresholds is not None and label in encoder.thresholds.values:
                expected_scores = margins[:, column].tolist()
                winner = int(margins[:, column].argmax())
                assert row["frame_scores"] == pytest.approx(expected_scores, abs=1e-6)
                assert row["image_index"] == winner
                assert row["score"] == pytest.approx(expected_scores[winner], abs=1e-6)
                assert row["threshold"] == encoder.thresholds.values[label]
                assert row["thresholds_sha256"] == encoder.thresholds.sha256
                assert row["sampling"] == dict(encoder.thresholds.sampling)
                assert row["probabilities"] == pytest.approx(
                    [*pairs[winner, column].tolist(), 0.0], abs=1e-6
                )
            else:
                winner = int(pairs[:, column, 0].argmax())
                assert row == pytest.approx(pairs[winner, column].tolist(), abs=1e-6)
            earlier_winners.append(winner < len(window) - 1)
    if counts == [1, 2, 4]:
        assert any(earlier_winners), (
            "fixture must distinguish full windows from last frames"
        )

    assert text_calls == [[text for pair in encoder.prompts.values() for text in pair]]
    cached = encoder._text_features

    def no_text_forward(*args, **kwargs):
        pytest.fail("a subsequent request recomputed cached text features")

    monkeypatch.setattr(model.text_model, "forward", no_text_forward)
    repeated = encoder.score_batch(pixels, labels)
    assert encoder._text_features is cached
    assert repeated == actual
    assert len(text_calls) == 1
