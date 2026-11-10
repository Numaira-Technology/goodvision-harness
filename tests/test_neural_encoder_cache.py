"""CLIP cache contracts with deterministic CPU arrays, no weights or provider calls."""

from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from goodvision_harness.neural_calibration import CALIBRATED_PROMPTS, FrozenThresholds
from goodvision_harness.neural_service import Encoder, PROMPTS


class Tensor(np.ndarray):
    device = "cuda"

    def to(self, device):
        assert device == "cuda"
        return self

    def t(self):
        return self.T

    def exp(self):
        return np.exp(self)

    def norm(self, dim, keepdim):
        return np.linalg.norm(self, axis=dim, keepdims=keepdim)

    def softmax(self, dim):
        values = np.exp(self - self.max(axis=dim, keepdims=True))
        return values / values.sum(axis=dim, keepdims=True)

    def cpu(self):
        return self


def tensor(value):
    return np.asarray(value, dtype=np.float64).view(Tensor)


def encoder_fixture(thresholds=None):
    prompts = {**PROMPTS, **CALIBRATED_PROMPTS} if thresholds else PROMPTS
    texts = [text for pair in prompts.values() for text in pair]
    # Distinct, non-unit vectors exercise projection, normalization and ordering.
    vectors = np.arange(1, len(texts) * 3 + 1).reshape(-1, 3) % 7 - 2
    text_projection = np.array([[1, 0, 2], [0, 3, 1], [2, 1, 0]])
    image_projection = np.array([[2, 1, 0], [0, 1, 3], [1, 0, 2]])
    calls = []

    def processor(*, text=None, images=None, **kwargs):
        if text is not None:
            calls.append(("text", tuple(text)))
            return {"input_ids": tensor([vectors[texts.index(t)] for t in text])}
        assert all(image.mode == "RGB" and image.size == (224, 224) for image in images)
        colors = [image.getpixel((112, 112)) for image in images]
        calls.append(("image", colors))
        return {"pixel_values": tensor(colors)}

    encoder = Encoder.__new__(Encoder)
    encoder.torch = SimpleNamespace(
        inference_mode=nullcontext,
        cat=lambda values: tensor(np.concatenate(values)),
    )
    encoder.processor = processor
    encoder.model = SimpleNamespace(
        text_model=lambda input_ids: SimpleNamespace(pooler_output=input_ids),
        vision_model=lambda pixel_values: SimpleNamespace(pooler_output=pixel_values),
        text_projection=lambda value: tensor(value @ text_projection),
        visual_projection=lambda value: tensor(value @ image_projection),
        logit_scale=tensor(np.log(3)),
    )
    encoder._text_features = None
    encoder.prompts = prompts
    encoder.thresholds = thresholds

    def reference(image, labels, *, margins=False):
        # Independent full-forward formula: project each requested prompt, then
        # score and normalize only its own positive/negative pair.
        projected_image = np.asarray(image.getpixel((112, 112)), dtype=float) @ image_projection
        projected_image /= np.linalg.norm(projected_image)
        result = []
        for label in labels:
            pair = []
            cosines = []
            for text in prompts[label]:
                projected = vectors[texts.index(text)] @ text_projection
                cosine = np.dot(projected / np.linalg.norm(projected), projected_image)
                cosines.append(cosine)
                pair.append(np.exp(3 * cosine))
            result.append(cosines[0] - cosines[1] if margins else np.asarray(pair) / sum(pair))
        return result

    return encoder, calls, reference


def test_cached_scores_match_full_forward_for_reordered_subsets_and_new_images():
    encoder, calls, reference = encoder_fixture()
    for color, labels in [
        ([3.0, 1.0, 2.0], ["smoking", "fall"]),
        ([1.0, 4.0, 2.0], ["weapon", "fight", "smoking"]),
        ([2.0, 3.0, 1.0], list(reversed(PROMPTS))),
    ]:
        image = Image.new("RGB", (224, 224), tuple(int(value) for value in color))
        actual = encoder.scores(image, labels)
        np.testing.assert_allclose(actual, reference(image, labels), rtol=1e-12)
    assert [kind for kind, _ in calls] == ["image", "text", "image", "image"]
    assert calls[1][1] == tuple(text for pair in PROMPTS.values() for text in pair)


def test_failed_text_encoding_does_not_publish_a_partial_cache():
    encoder, calls, reference = encoder_fixture()
    projection = encoder.model.text_projection
    encoder.model.text_projection = lambda _: (_ for _ in ()).throw(
        RuntimeError("text projection failed")
    )
    image = Image.new("RGB", (224, 224), (1, 2, 3))
    with pytest.raises(RuntimeError, match="text projection failed"):
        encoder.scores(image, ["fight"])
    assert encoder._text_features is None
    assert [kind for kind, _ in calls] == ["image", "text"]
    encoder.model.text_projection = projection
    actual = encoder.scores(image, ["fight"])
    np.testing.assert_allclose(actual, reference(image, ["fight"]))
    assert [kind for kind, _ in calls] == ["image", "text", "image", "text"]


@pytest.mark.parametrize("configured", [False, True])
def test_cached_batch_matches_independent_frame_scores_and_window_boundaries(configured):
    thresholds = FrozenThresholds(
        {"fight": 0.1, "smoking": 0.0, "fall": None, "weapon": 0.2}, "f" * 64,
        sampling={
            "policy": "two-frame-target-spacing-v1",
            "candidate_fps": 4.0,
            "observation_window_frames": 2,
            "observation_frame_delta_seconds": 0.5,
            "observation_fps": 1.25,
        },
    ) if configured else None
    encoder, calls, reference = encoder_fixture(thresholds)
    # Exercise encoder batch offsets directly. The calibrated HTTP boundary
    # separately enforces two-frame windows before any encoder work.
    windows = [
        [Image.new("RGB", (640, 360), color) for color in [(3, 1, 2), (1, 4, 2)]],
        [Image.new("RGB", (360, 640), (2, 3, 1))],
        [Image.new("RGB", (224, 224), color)
         for color in [(4, 1, 2), (3, 2, 1), (2, 1, 4), (1, 2, 3)]],
    ]
    labels = [["smoking", "fall", "ground_litter"], ["weapon", "fight"], ["fight", "smoking"]]
    # The fourth frame is decisive: truncating this window must change its score.
    smoking = [reference(image, ["smoking"])[0][0] for image in windows[2]]
    assert max(range(4), key=smoking.__getitem__) == 3
    pixels = [encoder.prepare(images) for images in windows]
    batches = encoder.score_batch(pixels, labels)
    for images, requested, actual in zip(windows, labels, batches, strict=True):
        probabilities = [reference(image, requested) for image in images]
        margins = [reference(image, requested, margins=True) for image in images]
        for index, (label, row) in enumerate(zip(requested, actual, strict=True)):
            if thresholds is not None and label in thresholds.values:
                values = [scores[index] for scores in margins]
                winner = max(range(len(values)), key=values.__getitem__)
                assert row["frame_scores"] == pytest.approx(values, abs=1e-12)
                assert row["image_index"] == winner
                assert row["score"] == pytest.approx(values[winner], abs=1e-12)
                assert row["threshold"] == thresholds.values[label]
                assert row["thresholds_sha256"] == thresholds.sha256
                assert row["sampling"] == dict(thresholds.sampling)
                np.testing.assert_allclose(row["probabilities"], [*probabilities[winner][index], 0.0])
            else:
                expected = max((scores[index] for scores in probabilities), key=lambda pair: pair[0])
                np.testing.assert_allclose(row, expected, rtol=1e-12)
    # Reordered subsets in a later request must reuse the same fixed prompt cache.
    encoder.score_batch([pixels[0]], [["weapon", "smoking"]])
    assert [values for kind, values in calls if kind == "text"] == [
        tuple(text for pair in encoder.prompts.values() for text in pair)
    ]
