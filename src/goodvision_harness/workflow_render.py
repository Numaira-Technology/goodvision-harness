"""Compose only wired, current workflow results onto a source snapshot."""

import io
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from .workflow_modules import display_settings


def recent(timestamp, now, age):
    return (
        type(timestamp) in (int, float)
        and math.isfinite(timestamp)
        and 0 <= now - timestamp <= age
    )


def render_output(jpeg, output, timestamp, *, language="zh"):
    with Image.open(io.BytesIO(jpeg)) as original:
        image = original.convert("RGB")
    image.thumbnail((1280, 1280))
    draw = ImageDraw.Draw(image)
    width, height = image.size
    size = max(12, min(22, width // 40))
    fonts = list((Path(__file__).parent / "fonts").glob("*.otf"))
    font = (
        ImageFont.truetype(str(fonts[0]), size)
        if fonts
        else ImageFont.load_default(size=size)
    )
    settings = display_settings(output.get("display", {}))
    offsets = {
        position: 12
        for position in ("top-left", "top-right", "bottom-left", "bottom-right")
    }

    def panel(text, position, color):
        # Wrap by rendered width, including Chinese text, and bound the panel.
        limit, lines, line = max(40, min(420, width // 2 - 24)), [], ""
        for char in str(text)[:1200]:
            if char == "\n" or draw.textlength(line + char, font=font) > limit:
                lines.append(line)
                line = "" if char == "\n" else char
            else:
                line += char
        if line:
            lines.append(line)
        offset = offsets[position]
        max_lines = min(8, (height - offset - 20) // (size + 5))
        if max_lines < 1:
            return
        clipped = len(lines) > max_lines
        lines = lines[:max_lines]
        if clipped:
            lines[-1] = lines[-1][:-1] + "…"
        box_width = min(width - 8, limit + 16)
        box_height = min(height - 8, len(lines) * (size + 5) + 16)
        x = max(4, width - box_width - 12) if position.endswith("right") else 12
        y = (
            max(4, height - box_height - offset)
            if position.startswith("bottom")
            else offset
        )
        if y + box_height > height - 4:
            return
        draw.rounded_rectangle(
            (x, y, x + box_width, y + box_height), radius=6, fill=color
        )
        draw.multiline_text(
            (x + 8, y + 6), "\n".join(lines), fill="white", font=font, spacing=5
        )
        offsets[position] += box_height + 8

    for item in output["inputs"]:
        layer, value = item["port"], item.get("value")
        if layer not in settings or not settings[layer]["visible"] or not value:
            continue
        if layer == "tracks":
            for anchor in value.get("anchors", [])[-1:]:
                if not recent(anchor.get("media_timestamp"), timestamp, 0.04):
                    continue
                for subject in anchor.get("subjects", []):
                    for polygon in subject.get("polygons", []):
                        if len(polygon) >= 3:
                            draw.line(
                                [(p[0] * width, p[1] * height) for p in polygon]
                                + [(polygon[0][0] * width, polygon[0][1] * height)],
                                fill="#32e7a0",
                                width=3,
                            )
            if recent(value.get("timestamp_seconds"), timestamp, 0.04):
                for person in value.get("objects", []):
                    box = person.get("bbox", {})
                    if (
                        all(
                            type(box.get(k)) in (int, float)
                            and math.isfinite(box[k])
                            and 0 <= box[k] <= 1
                            for k in ("x1", "y1", "x2", "y2")
                        )
                        and box["x2"] > box["x1"]
                        and box["y2"] > box["y1"]
                    ):
                        draw.rectangle(
                            (
                                box["x1"] * width,
                                box["y1"] * height,
                                box["x2"] * width,
                                box["y2"] * height,
                            ),
                            outline="#32e7a0",
                            width=3,
                        )
        elif layer == "events":
            labels = []
            for record in value.get("results", []):
                if not recent(
                    record.get("frame", {}).get("timestamp_seconds"), timestamp, 15
                ):
                    continue
                labels.extend(
                    str(j["event_label"]).replace("_", " ").title()
                    for j in record.get("judgments", [])
                    if j.get("detected") is True
                )
            if labels:
                panel(
                    " · ".join(dict.fromkeys(labels)),
                    settings[layer]["position"],
                    "#bf2637",
                )
        elif not value.get("error") and recent(
            value.get("timestamp_seconds"), timestamp, value.get("display_max_age", 120)
        ):
            text = value.get("summary_translations", {}).get(language) or value.get(
                "summary"
            )
            if text:
                panel(text, settings[layer]["position"], "#23334d")
    output_bytes = io.BytesIO()
    image.save(output_bytes, format="JPEG", quality=90)
    return output_bytes.getvalue()
