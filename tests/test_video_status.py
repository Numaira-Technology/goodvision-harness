from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from goodvision_harness import overlay
from goodvision_harness.contracts import NormalizedBBox


@pytest.mark.parametrize("kind", ["event", "face_recognition", "face_reference_match"])
@pytest.mark.parametrize("detected,incomplete,stale,visible", [
    (False, False, False, True),
    (False, True, False, False),
    (True, True, False, True),
    (True, False, False, True),
    (False, False, True, False),
    (True, True, True, False),
])
def test_video_only_shows_current_clear_or_event_status(
    monkeypatch, kind, detected, incomplete, stale, visible,
):
    captions = []
    monkeypatch.setattr(
        overlay, "_text_with_background",
        lambda image, text, *args, **kwargs: captions.append(text),
    )
    source = np.zeros((100, 300, 3), dtype=np.uint8)
    rendered = overlay.draw_overlay(
        source, event_label="fall", detected=detected, confidence=0.9,
        description="人员跌倒 / Person fell" if detected else "无异常",
        objects=(), analysis_kind=kind, incomplete=incomplete, stale=stale,
    )
    assert bool(captions) is visible
    if visible:
        expected = "CLEAR" if not detected else (
            "FALL" if kind == "event" else "UNKNOWN"
        )
        assert captions[0].startswith(expected)
        assert "人员跌倒" in captions[1] if detected else captions[1] == "无异常"
    assert not source.any()
    assert rendered is not source


def test_unusable_face_does_not_report_clear(monkeypatch):
    captions = []
    monkeypatch.setattr(
        overlay, "_text_with_background",
        lambda image, text, *args, **kwargs: captions.append(text),
    )
    face = SimpleNamespace(
        bbox=NormalizedBBox(0.2, 0.4, 0.5, 0.9),
        attributes={"status": "unusable"},
    )
    overlay.draw_overlay(
        np.zeros((100, 300, 3), dtype=np.uint8),
        event_label="stranger", detected=False, confidence=0.0,
        description="Face is unusable", objects=(face,),
        analysis_kind="face_recognition",
    )
    assert captions == ["unusable"]


@pytest.mark.parametrize("shape", [(100, 640, 3), (20, 45, 3), (1, 1, 3)])
def test_bundled_chinese_font_renders_and_clips_to_video(shape):
    font = overlay._unicode_font(20)
    missing = bytes(font.getmask("\u0378"))
    for character in "中文人员跌倒张三":
        assert bytes(font.getmask(character)) != missing
    image = np.zeros(shape, dtype=np.uint8)
    overlay._text_with_background(
        image, "人员跌倒 / Person fell", (12, 28), (40, 40, 220),
    )
    assert image.shape == shape
    if shape[1] > 8:
        assert image.any()
    fitted = overlay._fit_hud_text("人员跌倒" * 20, 100, 0.55)
    assert "?" not in fitted
    assert overlay._text_size(fitted, 0.55)[0][0] <= 100


def test_chinese_face_name_and_rich_card_are_preserved(monkeypatch):
    drawn = []
    original = overlay.ImageDraw.ImageDraw.text

    def capture(self, xy, text, *args, **kwargs):
        drawn.append(text)
        return original(self, xy, text, *args, **kwargs)

    monkeypatch.setattr(overlay.ImageDraw.ImageDraw, "text", capture)
    image = np.zeros((360, 640, 3), dtype=np.uint8)
    overlay.draw_face_boxes(image, (SimpleNamespace(
        bbox=NormalizedBBox(0.2, 0.4, 0.6, 0.9),
        attributes={"status": "known", "person_id": "007", "name": "张三"},
    ),))
    overlay._draw_rich_card(
        image, (50, 80, 180, 280), "人员 张三", "红色上衣", "背包", (55, 180, 70),
    )
    assert "ID: 007 | 张三" in drawn
    assert "人员 张三" in drawn
    assert "红色上衣" in drawn
