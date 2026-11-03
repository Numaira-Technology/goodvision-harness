"""CLIP cache contracts with deterministic CPU arrays, no weights or provider calls."""

from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pytest

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

    def softmax(self, dim):
        values = np.exp(self - self.max(axis=dim, keepdims=True))
        return values / values.sum(axis=dim, keepdims=True)

    def cpu(self):
        return self


def tensor(value):
    return np.asarray(value, dtype=np.float64).view(Tensor)


def encoder_fixture():
    texts = [text for pair in PROMPTS.values() for text in pair]
    # Distinct, non-unit vectors exercise projection, normalization and ordering.
    vectors = np.arange(1, len(texts) * 3 + 1).reshape(-1, 3) % 7 - 2
    text_projection = np.array([[1, 0, 2], [0, 3, 1], [2, 1, 0]])
    image_projection = np.array([[2, 1, 0], [0, 1, 3], [1, 0, 2]])
    calls = []

    def processor(*, text=None, images=None, **kwargs):
        if text is not None:
            calls.append(("text", tuple(text)))
            return {"input_ids": tensor([vectors[texts.index(t)] for t in text])}
        calls.append(("image", tuple(images)))
        return {"pixel_values": tensor([images])}

    encoder = Encoder.__new__(Encoder)
    encoder.torch = SimpleNamespace(
        inference_mode=nullcontext,
        pow=np.power,
        sum=lambda value, dim, keepdim: value.sum(axis=dim, keepdims=keepdim),
        cat=lambda values: tensor(np.concatenate(values)),
        matmul=np.matmul,
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

    def reference(image, labels):
        # Independent full-forward formula: project each requested prompt, then
        # score and normalize only its own positive/negative pair.
        projected_image = np.asarray(image) @ image_projection
        projected_image /= np.linalg.norm(projected_image)
        result = []
        for label in labels:
            pair = []
            for text in PROMPTS[label]:
                projected = vectors[texts.index(text)] @ text_projection
                cosine = np.dot(projected / np.linalg.norm(projected), projected_image)
                pair.append(np.exp(3 * cosine))
            result.append(np.asarray(pair) / sum(pair))
        return result

    return encoder, calls, reference


def test_cached_scores_match_full_forward_for_reordered_subsets_and_new_images():
    encoder, calls, reference = encoder_fixture()
    for image, labels in [
        ([3.0, 1.0, 2.0], ["smoking", "fall"]),
        ([1.0, 4.0, 2.0], ["weapon", "fight", "smoking"]),
        ([2.0, 3.0, 1.0], list(reversed(PROMPTS))),
    ]:
        actual = encoder.scores(image, labels)
        np.testing.assert_allclose(actual, reference(image, labels), rtol=1e-12)
    assert [kind for kind, _ in calls] == ["text", "image", "image", "image"]
    assert calls[0][1] == tuple(text for pair in PROMPTS.values() for text in pair)


def test_failed_text_encoding_does_not_publish_a_partial_cache():
    encoder, calls, reference = encoder_fixture()
    projection = encoder.model.text_projection
    encoder.model.text_projection = lambda _: (_ for _ in ()).throw(
        RuntimeError("text projection failed")
    )
    with pytest.raises(RuntimeError, match="text projection failed"):
        encoder.scores([1.0, 2.0, 3.0], ["fight"])
    assert encoder._text_features is None
    assert [kind for kind, _ in calls] == ["text"]
    encoder.model.text_projection = projection
    actual = encoder.scores([1.0, 2.0, 3.0], ["fight"])
    np.testing.assert_allclose(actual, reference([1.0, 2.0, 3.0], ["fight"]))
    assert [kind for kind, _ in calls] == ["text", "text", "image"]
