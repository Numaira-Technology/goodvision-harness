"""Forward numerical-equivalence tests for the CLIP neural service.

These run on CPU with a tiny randomly-initialised CLIP model and a stub
processor, so they need no GPU, no CUDA and no downloaded weights. They prove
that ``Encoder.score_batch``'s hand-written forward
(``vision_model`` pooler -> ``visual_projection`` -> L2 normalize, cached
``text_model`` -> ``text_projection`` -> normalize, then
``logit_scale.exp() * text @ image.T``) reproduces Hugging Face's native
``logits_per_image`` pair-softmax to a tight tolerance, exercising the *real*
``score_batch`` source on CPU.

(The queue-backpressure 429 behaviour lives in test_neural_workflow.py with the
other HTTP-contract tests, since it needs no torch and runs in the main job.)
"""

import pytest

# These tests need a (tiny, random, CPU) CLIP model. The main regression job
# does not install torch/transformers, so skip cleanly there; the dedicated
# neural-forward CI job installs the pinned CPU stack and runs them.
torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from goodvision_harness.neural_service import PROMPTS, Encoder  # noqa: E402


def _tiny_clip():
    """Build a small, deterministic CLIP model and a stub processor on CPU."""
    from transformers import CLIPConfig, CLIPModel

    torch.manual_seed(0)
    config = CLIPConfig(
        text_config=dict(
            vocab_size=120,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=2,
            max_position_embeddings=32,
            bos_token_id=0,
            eos_token_id=119,
        ),
        vision_config=dict(
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=2,
            image_size=30,
            patch_size=10,
            num_channels=3,
        ),
        projection_dim=16,
    )
    model = CLIPModel(config).eval()

    text_config = model.config.text_config
    image_size = model.config.vision_config.image_size
    all_prompts = [text for pair in PROMPTS.values() for text in pair]
    seq_len = 8

    # Deterministic, per-prompt token ids so the same prompt yields the same row
    # whether it is requested alone or inside the full set. bos first, eos last.
    rows = {}
    for index, prompt in enumerate(all_prompts):
        generator = torch.Generator().manual_seed(1000 + index)
        ids = torch.randint(
            1, text_config.vocab_size - 1, (seq_len,), generator=generator
        )
        ids[0] = text_config.bos_token_id
        ids[-1] = text_config.eos_token_id
        rows[prompt] = ids

    pixel_values = torch.randn(
        1, 3, image_size, image_size, generator=torch.Generator().manual_seed(7)
    )

    class StubProcessor:
        def __call__(self, *, images=None, text=None, return_tensors=None, **kwargs):
            if text is not None:
                input_ids = torch.stack([rows[prompt] for prompt in text])
                return {
                    "input_ids": input_ids,
                    "attention_mask": torch.ones_like(input_ids),
                }
            return {"pixel_values": pixel_values.clone()}

    return model, StubProcessor(), pixel_values, rows


def _real_encoder(model, processor):
    encoder = object.__new__(Encoder)
    encoder.torch = torch
    encoder.model = model
    encoder.processor = processor
    encoder._text_features = None
    return encoder


def _on_cpu(callable_):
    """Run ``callable_`` with ``Tensor.to("cuda")`` redirected to CPU.

    ``Encoder.score_batch`` hard-codes ``.to("cuda")``; redirecting it lets the
    real source run unchanged on a CPU-only CI runner.
    """
    original = torch.Tensor.to

    def to_cpu(self, *args, **kwargs):
        args = tuple("cpu" if arg == "cuda" else arg for arg in args)
        if kwargs.get("device") == "cuda":
            kwargs["device"] = "cpu"
        return original(self, *args, **kwargs)

    torch.Tensor.to = to_cpu
    try:
        return callable_()
    finally:
        torch.Tensor.to = original


def _native_pairs(model, processor, labels):
    """Hugging Face's own logits_per_image pair-softmax for the given labels."""
    prompts = [text for label in labels for text in PROMPTS[label]]
    text_inputs = processor(text=prompts)
    pixel_values = processor(images=None)["pixel_values"]
    with torch.inference_mode():
        output = model(
            input_ids=text_inputs["input_ids"],
            attention_mask=text_inputs["attention_mask"],
            pixel_values=pixel_values,
        )
        pairs = output.logits_per_image[0].reshape(len(labels), 2).softmax(dim=-1)
        return pairs.tolist()


def _max_abs_diff(a, b):
    return max(
        abs(x - y)
        for row_a, row_b in zip(a, b, strict=True)
        for x, y in zip(row_a, row_b, strict=True)
    )


def test_manual_forward_matches_native_logits_for_all_events():
    model, processor, _, _ = _tiny_clip()
    labels = list(PROMPTS)

    native = _native_pairs(model, processor, labels)
    encoder = _real_encoder(model, processor)
    got = _on_cpu(lambda: encoder.scores(object(), labels))

    assert len(got) == len(labels)
    for pair in got:
        assert abs(sum(pair) - 1) < 1e-5
    assert _max_abs_diff(native, got) < 1e-4


def test_manual_forward_matches_native_for_requested_subset():
    # The worker caches features for every prompt and indexes the requested
    # labels out; a subset request must still match a native run over just those
    # labels' prompts, in the requested order.
    model, processor, _, _ = _tiny_clip()
    labels = ["weapon", "smoking", "fall"]

    native = _native_pairs(model, processor, labels)
    encoder = _real_encoder(model, processor)
    got = _on_cpu(lambda: encoder.scores(object(), labels))

    assert _max_abs_diff(native, got) < 1e-4


def test_cached_text_features_are_reused_across_calls():
    model, processor, _, _ = _tiny_clip()
    encoder = _real_encoder(model, processor)

    first = _on_cpu(lambda: encoder.scores(object(), list(PROMPTS)))
    cached = encoder._text_features
    assert cached is not None
    # A second call reuses the cached, normalised text features unchanged.
    second = _on_cpu(lambda: encoder.scores(object(), list(PROMPTS)))
    assert encoder._text_features is cached
    assert _max_abs_diff(first, second) < 1e-6
