"""Lossless context packing for Cerebras' two-image OpenRouter route."""

import base64
import hashlib
import io
import json
import time
from contextlib import ExitStack

from PIL import Image

CONTEXT_LATEST_LAYOUT = "context-strip-latest-v1"
INDEPENDENT_LAYOUT = "independent-frames-v1"


def image_layout(provider, model, parameters):
    if (
        provider == "openrouter"
        and model == "qwen/qwen3.8-27b"
        and (
            parameters.get("provider", {}).get("only") == ["cerebras"]
            or parameters.get("provider", {}).get("order", [])[:1] == ["cerebras"]
        )
    ):
        return CONTEXT_LATEST_LAYOUT
    return INDEPENDENT_LAYOUT


def pack_context_latest(messages):
    """Pack already encoded context pixels; leave the latest JPEG unchanged.

    Source evidence, frame IDs and timestamps remain intact. No image generation,
    resizing, cropping or second lossy encode is performed here. The first PNG's
    panels are ordered top to bottom; boxes still refer to the standalone latest
    image. Diagnostic source JPEGs remain the components, not transport images.
    """
    started = time.monotonic()
    content = messages[1]["content"]
    count = (len(content) - 1) // 2
    if len(content) != 1 + 2 * count or not 2 <= count <= 4:
        raise ValueError("context layout requires two through four source frames")
    pairs = []
    for i in range(count):
        meta, part = content[1 + 2 * i : 3 + 2 * i]
        if meta["type"] != "text" or part["type"] != "image_url":
            raise ValueError("context layout requires ordered frame metadata/images")
        pairs.append((json.loads(meta["text"]), part))
    prefix = "data:image/jpeg;base64,"
    panels = []
    with ExitStack() as stack:
        images = []
        for meta, part in pairs[:-1]:
            url = part["image_url"]["url"]
            if not url.startswith(prefix):
                raise ValueError("context layout requires sanitized JPEG sources")
            raw = base64.b64decode(url[len(prefix) :], validate=True)
            source = stack.enter_context(Image.open(io.BytesIO(raw)))
            source.load()
            images.append(source)
            panels.append(
                {
                    **meta,
                    "x": 0,
                    "y": sum(im.height for im in images[:-1]),
                    "width": source.width,
                    "height": source.height,
                    "component_jpeg_sha256": hashlib.sha256(raw).hexdigest(),
                }
            )
        if count > 2:
            canvas = stack.enter_context(
                Image.new(
                    "RGB",
                    (max(im.width for im in images), sum(im.height for im in images)),
                )
            )
            for panel, source in zip(panels, images, strict=True):
                canvas.paste(source, (panel["x"], panel["y"]))
            data = io.BytesIO()
            canvas.save(data, format="PNG")
            raw_context = data.getvalue()
            context_part = {
                "type": "image_url",
                "image_url": {
                    "url": "data:image/png;base64,"
                    + base64.b64encode(raw_context).decode("ascii")
                },
            }
        else:
            context_part = pairs[0][1]
            raw_context = base64.b64decode(
                context_part["image_url"]["url"].split(",", 1)[1]
            )
    # Keep hashes in diagnostics, not in the model prompt.
    prompt_panels = [
        {k: v for k, v in p.items() if k != "component_jpeg_sha256"} for p in panels
    ]
    context = json.loads(content[0]["text"])
    context.update(
        image_layout=CONTEXT_LATEST_LAYOUT,
        image_layout_instruction=(
            "The FIRST image contains earlier context frames stacked vertically "
            "in chronological order, top to bottom, with panel bounds given below. "
            "Each panel is a different time, not a different person or camera. "
            "The SECOND and LAST image is the standalone latest frame. Assess the "
            "latest time using the earlier panels as context; all grounding boxes "
            "refer only to the standalone LAST image, never to the context strip."
        ),
    )
    messages[1]["content"] = [
        {"type": "text", "text": json.dumps(context, separators=(",", ":"))},
        {
            "type": "text",
            "text": json.dumps(
                {"context_panels": prompt_panels}, separators=(",", ":")
            ),
        },
        context_part,
        *content[-2:],
    ]
    metadata = {
        "version": CONTEXT_LATEST_LAYOUT,
        "source_frame_count": count,
        "wire_image_count": 2,
        "context_panels": panels,
        "context_image_sha256": hashlib.sha256(raw_context).hexdigest(),
        "latest_frame_sequence": pairs[-1][0]["frame_sequence"],
        "diagnostic_images": "source_components_before_lossless_packing",
    }
    timing = {
        "image_pack_seconds": time.monotonic() - started,
        "source_frame_count": count,
        "wire_image_count": 2,
    }
    return metadata, timing
