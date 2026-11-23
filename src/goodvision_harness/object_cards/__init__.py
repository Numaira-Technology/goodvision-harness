"""Object recipe cards: one JSON card per recognizable object.

A card is data, not code. It names the engine, the question or text pair, the
frame window, the decision rule, dataset references (by pin, never content)
and the validation status. The live object node and the Canvas read the same
cards; ``tools/object_cards.py verify`` checks that each card still matches the
frozen benchmark profile it references. Design: docs/guides/object-recipe-cards.md.

A card becomes ``validated`` only with live-acceptance evidence. Offline
benchmark scores may be recorded, but alone they keep a card ``experimental``.
"""

from __future__ import annotations

import copy
import json
import math
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

CARD_DIR = Path(__file__).parent
ENGINES = ("videojev", "decision3", "clip", "siglip2", "hosted_vlm", "luna")
KINDS = {
    "native_choice": {"videojev", "decision3"},
    "text_pair": {"clip", "siglip2"},
    "vlm_boolean": {"hosted_vlm"},
    "probability": {"luna"},
}
RULES = {
    "native_choice": {"argmax", "gate"},
    "text_pair": {"margin"},
    "vlm_boolean": {"boolean"},
    "probability": {"probability"},
}
STATES = ("experimental", "validated", "retired")
# Engines the live object node can run today; others are offline-only cards.
LIVE_ENGINES = {"videojev"}
_SLUG = re.compile(r"[a-z][a-z0-9_]{0,47}\Z")
_HEX64 = re.compile(r"[0-9a-f]{64}\Z")
_PIN = re.compile(r"[a-z_][a-z0-9_.]*:[A-Z_][A-Z0-9_]*\Z")
_LIVE_EVIDENCE = {"kind", "protocol", "samples", "pass_line", "reviewers", "record_sha256"}


class CardError(ValueError):
    """A card that does not satisfy the schema."""


def _fail(card_id: str, message: str):
    raise CardError(f"object card {card_id}: {message}")


def _keys(cid, value, required, optional=(), label="card"):
    if not isinstance(value, Mapping):
        _fail(cid, f"{label} must be an object")
    missing = set(required) - set(value)
    extra = set(value) - set(required) - set(optional)
    if missing or extra:
        _fail(cid, f"{label} fields: missing {sorted(missing)}, unknown {sorted(extra)}")


def _text(cid, value, label, *, allow_none=False):
    if value is None and allow_none:
        return
    if not isinstance(value, str) or not value.strip() or len(value) > 600:
        _fail(cid, f"{label} must be nonempty text up to 600 characters")


def validate_card(card: Any) -> dict:
    """Return a deep copy of a valid card; raise CardError otherwise."""
    cid = card.get("id") if isinstance(card, Mapping) else None
    if not isinstance(cid, str) or not _SLUG.fullmatch(cid):
        raise CardError("object card id must be a lowercase slug")
    _keys(cid, card, ("id", "version", "names", "scope", "engine", "recipe",
                      "decision", "hard_negatives", "datasets", "status", "live"))
    if type(card["version"]) is not int or card["version"] < 1:
        _fail(cid, "version must be a positive integer")
    _keys(cid, card["names"], ("en", "zh"), label="names")
    _text(cid, card["names"]["en"], "names.en")
    _text(cid, card["names"]["zh"], "names.zh", allow_none=True)
    _text(cid, card["scope"], "scope")
    engine = card["engine"]
    if engine not in ENGINES:
        _fail(cid, f"engine must be one of {ENGINES}")

    recipe = card["recipe"]
    _keys(cid, recipe, ("kind",), ("text", "summary", "source", "window"), "recipe")
    kind = recipe["kind"]
    if kind not in KINDS or engine not in KINDS[kind]:
        _fail(cid, f"recipe kind {kind!r} does not fit engine {engine!r}")
    if ("text" in recipe) == ("source" in recipe):
        _fail(cid, "recipe needs exactly one of text (own wording) or source "
                   "(a reference into a frozen benchmark profile)")
    if "text" in recipe:
        _keys(cid, recipe["text"], ("criterion", "counter_examples"), label="recipe.text")
        _text(cid, recipe["text"]["criterion"], "recipe.text.criterion")
        _text(cid, recipe["text"]["counter_examples"], "recipe.text.counter_examples",
              allow_none=True)
        if "summary" in recipe:
            _fail(cid, "recipe.summary is only for source recipes")
    else:
        source = recipe["source"]
        ref = "question_id" if kind == "native_choice" else "prompt_key"
        _keys(cid, source, ("profile", "field", "sha256", ref),
              ("recipe_id", "revision"), "recipe.source")
        if not _SLUG.fullmatch(str(source["profile"])) or not _HEX64.fullmatch(
            str(source["sha256"])
        ):
            _fail(cid, "recipe.source needs a profile alias and a sha256")
        if source.get("recipe_id") is not None and not _SLUG.fullmatch(
            str(source["recipe_id"])
        ):
            _fail(cid, "recipe.source.recipe_id must be a slug")
        _text(cid, recipe.get("summary"), "recipe.summary")
    if "window" in recipe:
        window = recipe["window"]
        _keys(cid, window, ("frames", "step_seconds"), label="recipe.window")
        step = window["step_seconds"]
        if (type(window["frames"]) is not int or not 1 <= window["frames"] <= 4
                or type(step) not in (int, float) or not math.isfinite(step)
                or not 0.1 <= step <= 2):
            _fail(cid, "recipe.window needs 1-4 frames and 0.1-2 s spacing")

    decision = card["decision"]
    rule = decision.get("rule") if isinstance(decision, Mapping) else None
    if rule not in RULES[kind]:
        _fail(cid, f"decision rule must be one of {sorted(RULES[kind])} for {kind}")
    if rule == "gate":
        _keys(cid, decision, ("rule", "gate"), label="decision")
        gate = decision["gate"]
        if type(gate) not in (int, float) or not math.isfinite(gate) or not 0.34 <= gate <= 1:
            _fail(cid, "decision.gate must be between 0.34 and 1")
    elif rule == "margin":
        _keys(cid, decision, ("rule", "comparator", "threshold", "thresholds_sha256"),
              label="decision")
        if decision["comparator"] != ">":
            _fail(cid, "margin comparator must be '>' (the benchmark's rule)")
        threshold, digest = decision["threshold"], decision["thresholds_sha256"]
        if (threshold is None) != (digest is None):
            _fail(cid, "a margin threshold needs the sha256 of its frozen "
                       "validation-only thresholds.json, and vice versa")
        if threshold is not None and (
            type(threshold) not in (int, float) or not math.isfinite(threshold)
            or not _HEX64.fullmatch(str(digest))
        ):
            _fail(cid, "margin threshold must be finite with a sha256")
    else:
        _keys(cid, decision, ("rule",), label="decision")

    if not isinstance(card["hard_negatives"], list):
        _fail(cid, "hard_negatives must be a list")
    for item in card["hard_negatives"]:
        _text(cid, item, "hard_negatives[]")
    _keys(cid, card["datasets"], (), ("validation", "test"), "datasets")
    for split, ref in card["datasets"].items():
        _keys(cid, ref, ("name", "pinned_by"), label=f"datasets.{split}")
        _text(cid, ref["name"], f"datasets.{split}.name")
        if not _PIN.fullmatch(str(ref["pinned_by"])):
            _fail(cid, f"datasets.{split}.pinned_by must be module:CONSTANT")

    status = card["status"]
    _keys(cid, status, ("state", "note", "evidence"), label="status")
    if status["state"] not in STATES:
        _fail(cid, f"status.state must be one of {STATES}")
    _text(cid, status["note"], "status.note")
    if not isinstance(status["evidence"], list) or any(
        not isinstance(e, Mapping) or not isinstance(e.get("kind"), str)
        for e in status["evidence"]
    ):
        _fail(cid, "status.evidence must be a list of objects with a kind")
    if status["state"] == "validated" and not any(
        e.get("kind") == "live_acceptance" and _LIVE_EVIDENCE <= set(e)
        for e in status["evidence"]
    ):
        _fail(cid, "validated needs live_acceptance evidence with "
                   f"{sorted(_LIVE_EVIDENCE)}; offline scores alone stay experimental")

    _keys(cid, card["live"], ("supported",), label="live")
    if card["live"]["supported"] is not False and (
        card["live"]["supported"] is not True or engine not in LIVE_ENGINES
    ):
        _fail(cid, f"live.supported is true only for engines in {sorted(LIVE_ENGINES)}")
    return copy.deepcopy(dict(card))


def load_cards(directory: Path = CARD_DIR) -> dict[str, dict]:
    cards = {}
    for path in sorted(Path(directory).glob("*.json")):
        card = validate_card(json.loads(path.read_text(encoding="utf-8")))
        if path.stem != card["id"]:
            raise CardError(f"object card file {path.name} must be named {card['id']}.json")
        cards[card["id"]] = card
    _check_recipes(cards)
    return cards


def _check_recipes(cards: Mapping[str, dict]) -> None:
    """Cards sharing a benchmark recipe id must share its source and window."""
    seen: dict[str, tuple] = {}
    for card in cards.values():
        source = card["recipe"].get("source") or {}
        rid = source.get("recipe_id")
        if rid is None:
            continue
        key = (card["engine"], source["profile"], source["field"], source["sha256"],
               json.dumps(card["recipe"].get("window"), sort_keys=True))
        if seen.setdefault(rid, key) != key:
            raise CardError(f"object cards sharing recipe {rid} disagree on its source or window")


def preset(card: Mapping[str, Any]) -> dict:
    """The object spec a live node stores for this card (the old preset shape)."""
    recipe, decision = card["recipe"], card["decision"]
    text = recipe.get("text")
    spec = {
        "id": card["id"],
        "name": card["names"]["en"],
        "zh_name": card["names"]["zh"],
        "criterion": text["criterion"] if text else recipe["summary"],
        "counter_examples": text["counter_examples"] if text else None,
        "gate": decision.get("gate") if decision["rule"] == "gate" else None,
    }
    rid = (recipe.get("source") or {}).get("recipe_id")
    if rid is not None:
        spec["recipe"] = rid
    return spec


def status_of(card: Mapping[str, Any]) -> dict:
    status = {
        "status": card["status"]["state"],
        "validated": card["status"]["state"] == "validated",
    }
    rid = (card["recipe"].get("source") or {}).get("recipe_id")
    if rid is not None:
        status["recipe"] = rid
    return status


def catalog(cards: Mapping[str, dict]) -> list[dict]:
    """What the Canvas shows: live cards only, with their status badge data."""
    return [
        {
            "id": card["id"],
            "version": card["version"],
            "engine": card["engine"],
            "names": dict(card["names"]),
            "scope": card["scope"],
            "status": card["status"]["state"],
            "validated": card["status"]["state"] == "validated",
            "note": card["status"]["note"],
            "recipe": (card["recipe"].get("source") or {}).get("recipe_id"),
            "preset": preset(card),
        }
        for card in cards.values()
        if card["live"]["supported"]
    ]


CARDS = load_cards()
