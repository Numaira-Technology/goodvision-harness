"""Bag-demo VideoJev client: question assembly, request contract, verdict decoding.

No GPU, no network: a synthetic `raw` stands in for the scorer, built so it satisfies
the same choice contract that the real server enforces.
"""

import json

import pytest

from goodvision_harness.videojev_core import MODEL, OPTIONS, REVISION
from tools.bag_demo import videojev_bag as vb
from tools.videojev_probe.contract import POLICY_SHA

WINDOW = vb.window_for([0.0, 0.4, 0.8])


def _synth_raw(questions, chosen, confs=None):
    """Valid scorer rows; chosen maps IDs to indices, confs maps IDs to scores."""
    confs = confs or {}
    raw = []
    for q in questions:
        idx = chosen.get(q["id"], 1)  # default: B (absent)
        c = confs.get(q["id"], 0.8)
        rest = (1.0 - c) / 2
        scores = [rest, rest, rest]
        scores[idx] = c
        raw.append(
            {
                "id": q["id"],
                "task": "choice",
                "answer": q["options"][idx],
                "answer_index": idx,
                "probabilities": scores,
            }
        )
    return raw


def test_build_questions_shape_and_options():
    qs = vb.build_questions(WINDOW)
    assert 6 <= len(qs) <= 326
    ids = [q["id"] for q in qs]
    assert vb.BAG_KNIFE_ID in ids and vb.BAG_ALCOHOL_ID in ids
    assert "scene/weapon" in ids  # native weapon cross-check present
    assert len(set(ids)) == len(ids)
    for q in qs:
        assert q["task"] == "choice" and len(q["options"]) == 3
        if q["id"] in (vb.BAG_KNIFE_ID, vb.BAG_ALCOHOL_ID):
            assert q["options"] == list(
                OPTIONS
            )  # bag questions reuse canonical OPTIONS


def test_request_body_pins_contract():
    qs = vb.build_questions(WINDOW)
    body = vb.request_body(["img0", "img1"], qs, request_id="r1", session="s1")
    assert (body["model"], body["revision"], body["policy_sha256"], body["fps"]) == (
        MODEL,
        REVISION,
        POLICY_SHA,
        2.5,
    )
    assert body["lane"] == "events" and 2 <= len(body["images"]) <= 4
    json.dumps(body, allow_nan=False)


@pytest.mark.parametrize("n", [0, 1, 5])
def test_request_body_rejects_bad_frame_count(n):
    qs = vb.build_questions(WINDOW)
    with pytest.raises(ValueError):
        vb.request_body(["x"] * n, qs, request_id="r", session="s")


def test_knife_present_above_gate():
    qs = vb.build_questions(WINDOW)
    raw = _synth_raw(qs, {vb.BAG_KNIFE_ID: 0}, {vb.BAG_KNIFE_ID: 0.99})
    v = vb.parse_bag(raw, qs)
    assert v["knife"] is True and v["knife_present"] is True
    assert v["knife_p"] == pytest.approx(0.99)
    assert vb.demo_label(v) == "刀/武器候选"


def test_knife_below_gate_filtered():
    # Below-gate candidate stays uncertain; the small-sample gate is unchanged.
    qs = vb.build_questions(WINDOW)
    raw = _synth_raw(qs, {vb.BAG_KNIFE_ID: 0}, {vb.BAG_KNIFE_ID: 0.88})
    v = vb.parse_bag(raw, qs)
    assert v["knife"] is True and v["knife_present"] is False
    assert vb.demo_label(v) == "存疑"


def test_native_weapon_above_gate_counts_as_knife():
    qs = vb.build_questions(WINDOW)
    raw = _synth_raw(qs, {"scene/weapon": 0}, {"scene/weapon": 0.98})
    v = vb.parse_bag(raw, qs)
    assert v["weapon_native"] is True and v["knife_present"] is True
    assert vb.demo_label(v) == "刀/武器候选"


def test_alcohol_never_in_headline():
    qs = vb.build_questions(WINDOW)
    raw = _synth_raw(qs, {vb.BAG_ALCOHOL_ID: 0}, {vb.BAG_ALCOHOL_ID: 0.9})
    v = vb.parse_bag(raw, qs)
    assert v["alcohol"] is True and v["alcohol_p"] == pytest.approx(0.9)
    assert vb.demo_label(v) == "未检出刀/武器"  # alcohol is experimental, not a verdict


def test_absent_only_reports_knife_axis():
    qs = vb.build_questions(WINDOW)
    v = vb.parse_bag(_synth_raw(qs, {}), qs)
    assert v["knife_present"] is False and vb.demo_label(v) == "未检出刀/武器"


def test_insufficient_knife_is_unknown():
    qs = vb.build_questions(WINDOW)
    raw = _synth_raw(qs, {vb.BAG_KNIFE_ID: 2, "scene/weapon": 2})
    v = vb.parse_bag(raw, qs)
    assert v["knife"] is None and vb.demo_label(v) == "存疑"
