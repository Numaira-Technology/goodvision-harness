"""Memory specialization for the fixed-person Contour API on Jetson."""

from __future__ import annotations


def cache_person_text(backbone, torch):
    """Cache actual model features; reject prompts outside this API's contract."""
    original = backbone.forward_text
    cached = None

    def forward(captions, input_boxes=None, additional_text=None, device="cuda"):
        nonlocal cached
        if (
            tuple(captions) != ("person", "visual", "geometric")
            or input_boxes is not None
            or additional_text is not None
            or str(device) not in {"cuda", "cuda:0"}
        ):
            raise RuntimeError("Contour Jetson supports the fixed person prompt only")
        if cached is None:
            result = original(captions, device=device)
            if set(result) != {"language_features", "language_mask", "language_embeds"}:
                raise RuntimeError("Unexpected text encoder output")
            cached = {key: value.detach().clone() for key, value in result.items()}
            encoder = backbone.language_backbone
            released = sum(p.numel() * p.element_size() for p in encoder.parameters())
            # The full original checkpoint was strictly validated before inference.
            # Only its already-computed, fixed prompt features are needed afterward.
            encoder.to(device="meta")
            torch.cuda.empty_cache()
            print(
                f"Contour: fixed person features cached; released {released // 1048576}"
                " MiB of text weights",
                flush=True,
            )
        # Isolate each session from any in-place changes in downstream code.
        return {key: value.clone() for key, value in cached.items()}

    backbone.forward_text = forward
