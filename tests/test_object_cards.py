"""Object recipe cards: schema, migration parity, verify tool, served catalog.

CPU only; reads no media or labels. Design: docs/guides/object-recipe-cards.md.
"""

from __future__ import annotations

import copy
import hashlib
import json
import shutil
from pathlib import Path

import pytest

from goodvision_harness import object_cards as oc
from goodvision_harness import object_presence as op
from tools import object_cards as tool

ROOT = Path(__file__).resolve().parents[1]
BEFORE = json.loads((ROOT / "tests/fixtures/object_presets_before_cards.json").read_text())


def card(cid="bag_knife"):
    return copy.deepcopy(oc.CARDS[cid])


# --- migration parity ------------------------------------------------------


def test_presets_derived_from_cards_equal_the_pre_card_presets():
    # Snapshot taken from object_presence before the presets moved into cards.
    assert op.PRESETS == BEFORE["PRESETS"]
    assert op.PRESET_STATUS == BEFORE["PRESET_STATUS"]
    assert {k: op.preset_status(k) for k in op.PRESETS} == BEFORE["status"]
    recipe = op.RECIPES["bench_bag_v1"]
    assert {
        "frame_step_seconds": recipe["frame_step_seconds"],
        "questions": recipe["questions"],
        "template_len": len(recipe["template"]),
        "window_frames": recipe["window_frames"],
    } == BEFORE["RECIPES"]["bench_bag_v1"]


def test_every_shipped_card_loads_and_stays_experimental():
    files = sorted(p.stem for p in oc.CARD_DIR.glob("*.json"))
    assert files == sorted(oc.CARDS)
    for c in oc.CARDS.values():
        assert c["status"]["state"] == "experimental"
        assert c["status"]["evidence"] == []


def test_offline_only_cards_are_not_live_presets_or_canvas_choices():
    offline = {cid for cid, c in oc.CARDS.items() if not c["live"]["supported"]}
    assert offline == {"clip_bag_knife", "clip_bag_alcohol_bottle"}
    assert not offline & set(op.PRESETS)
    assert not offline & {c["id"] for c in oc.catalog(oc.CARDS)}
    assert op.preset_status("clip_bag_knife") is None


def test_catalog_entries_carry_the_live_preset_and_status():
    entries = {e["id"]: e for e in oc.catalog(oc.CARDS)}
    assert set(entries) == set(op.PRESETS)
    for cid, entry in entries.items():
        assert entry["preset"] == op.PRESETS[cid]
        assert entry["validated"] is False and entry["status"] == "experimental"
        assert entry["note"] == op.preset_status(cid)["note"]
    assert entries["bag_knife"]["recipe"] == "bench_bag_v1"
    assert entries["knife"]["recipe"] is None


# --- schema ---------------------------------------------------------------


@pytest.mark.parametrize("mutate,message", [
    (lambda c: c.update(extra=1), "unknown"),
    (lambda c: c.update(engine="clip"), "does not fit engine"),
    (lambda c: c["recipe"].update(text={"criterion": "x", "counter_examples": None}),
     "exactly one of text"),
    (lambda c: c["decision"].update(rule="margin"), "decision rule"),
    (lambda c: c["status"].update(state="validated"), "live_acceptance"),
    (lambda c: c["status"].update(state="validated", evidence=[
        {"kind": "offline_benchmark", "score": 100}]), "offline scores alone"),
    (lambda c: c["datasets"]["test"].update(pinned_by="not a pin"), "module:CONSTANT"),
    (lambda c: c["recipe"]["window"].update(frames=9), "1-4 frames"),
])
def test_invalid_cards_are_rejected(mutate, message):
    value = card()
    mutate(value)
    with pytest.raises(oc.CardError, match=message):
        oc.validate_card(value)


def test_validated_needs_complete_live_acceptance_evidence():
    value = card()
    value["status"].update(state="validated", evidence=[{
        "kind": "live_acceptance", "protocol": "knife-protocol-v8", "samples": 30,
        "pass_line": "27/30", "reviewers": 2, "record_sha256": "0" * 64,
    }])
    assert oc.validate_card(value)["status"]["state"] == "validated"


def test_margin_threshold_and_its_frozen_artifact_travel_together():
    value = card("clip_bag_knife")
    value["decision"]["threshold"] = 0.01
    with pytest.raises(oc.CardError, match="sha256 of its frozen"):
        oc.validate_card(value)
    value["decision"]["thresholds_sha256"] = "a" * 64
    assert oc.validate_card(value)["decision"]["threshold"] == 0.01


def test_only_live_engines_can_be_offered_live():
    value = card("clip_bag_knife")
    value["live"]["supported"] = True
    with pytest.raises(oc.CardError, match="live.supported"):
        oc.validate_card(value)


def test_cards_sharing_a_recipe_must_share_its_window(tmp_path):
    for cid in ("bag_knife", "bag_alcohol_bottle"):
        shutil.copy(oc.CARD_DIR / f"{cid}.json", tmp_path)
    other = json.loads((tmp_path / "bag_knife.json").read_text())
    other["recipe"]["window"]["step_seconds"] = 0.5
    (tmp_path / "bag_knife.json").write_text(json.dumps(other))
    with pytest.raises(oc.CardError, match="disagree"):
        oc.load_cards(tmp_path)


def test_card_file_name_must_match_its_id(tmp_path):
    shutil.copy(oc.CARD_DIR / "knife.json", tmp_path / "blade.json")
    with pytest.raises(oc.CardError, match="must be named knife.json"):
        oc.load_cards(tmp_path)


# --- verify tool ----------------------------------------------------------


def test_verify_passes_on_the_shipped_cards(capsys):
    assert tool.main(["verify"]) == 0
    assert "FAIL" not in capsys.readouterr().out


def test_verify_detects_a_drifted_benchmark_question(tmp_path):
    profiles = tmp_path / "profiles"
    shutil.copytree(tool.PROFILES, profiles)
    profile = json.loads((profiles / "videojev.json").read_text())
    profile["question_template"][4]["query_parts"][-1] += " "
    (profiles / "videojev.json").write_text(json.dumps(profile))
    assert any("drifted" in p for p in tool.check_card(card(), profiles=profiles))
    assert tool.check_card(card("knife"), profiles=profiles) == []  # own wording


def test_verify_detects_a_drifted_clip_text_pair(tmp_path):
    profiles = tmp_path / "profiles"
    shutil.copytree(tool.PROFILES, profiles)
    profile = json.loads((profiles / "clip.json").read_text())
    profile["prompts"]["bag_knife"][1] = "An empty bag."
    (profiles / "clip.json").write_text(json.dumps(profile))
    assert any("drifted" in p for p in tool.check_card(card("clip_bag_knife"), profiles=profiles))
    # The alcohol pair is hashed on its own, so it is unaffected.
    assert tool.check_card(card("clip_bag_alcohol_bottle"), profiles=profiles) == []


def test_verify_checks_a_threshold_against_its_frozen_artifact(tmp_path):
    model = json.loads((tool.PROFILES / "clip.json").read_text())["model"]
    artifact = tmp_path / "thresholds.json"
    artifact.write_text(json.dumps({"validation_only": True, "model": model,
                                    "thresholds": {"bag_knife": 0.0125}}))
    digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
    value = card("clip_bag_knife")
    # No frozen threshold on the card yet: the tool reports the artifact hash.
    assert "no frozen threshold" in tool.check_card(value, thresholds=artifact)[0]
    value["decision"].update(threshold=0.0125, thresholds_sha256=digest)
    assert tool.check_card(value, thresholds=artifact) == []
    value["decision"]["threshold"] = 0.02
    assert any("card has 0.02" in p for p in tool.check_card(value, thresholds=artifact))
    value["decision"].update(threshold=0.0125, thresholds_sha256="b" * 64)
    assert tool.check_card(value, thresholds=artifact) == [
        "thresholds.json sha256 differs from the card"]


# --- served catalog -------------------------------------------------------


def test_workflow_state_serves_the_card_catalog(monkeypatch, tmp_path):
    from fastapi.testclient import TestClient
    from test_workflow_routes import workflow_app

    with TestClient(workflow_app(monkeypatch, tmp_path)) as client:
        data = client.get("/v1/workflows").json()
    assert data["object_cards"] == oc.catalog(oc.CARDS)
