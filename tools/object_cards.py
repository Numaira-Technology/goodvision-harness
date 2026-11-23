"""Check object recipe cards against the frozen benchmark they reference.

    python -m tools.object_cards verify
    python -m tools.object_cards verify --thresholds clip_bag_knife=path/to/thresholds.json

The benchmark profiles stay the single source of truth; a card only points at
one question or text pair by profile alias, field and content hash. This tool
fails when a referenced item drifted, a dataset pin is missing, or a margin
threshold does not match its frozen validation-only thresholds.json. It reads
no media and no labels. Design: docs/guides/object-recipe-cards.md.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import sys
from pathlib import Path

from goodvision_harness import object_cards

PROFILES = Path(__file__).parent / "unified_benchmark" / "profiles"


def canonical_sha256(value) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _profile(alias: str, profiles: Path) -> dict:
    path = profiles / f"{alias}.json"
    if not path.is_file():
        raise ValueError(f"benchmark profile {alias} not found")
    profile = json.loads(path.read_text())
    if profile.get("alias") != alias:
        raise ValueError(f"benchmark profile {alias} has a different alias")
    return profile


def check_card(card: dict, *, profiles: Path = PROFILES, thresholds: Path | None = None) -> list[str]:
    """Problems with one card; an empty list means it matches the benchmark."""
    problems = []
    recipe, decision = card["recipe"], card["decision"]
    source = recipe.get("source")
    if source:
        try:
            profile = _profile(source["profile"], profiles)
        except ValueError as exc:
            return [str(exc)]
        field = profile.get(source["field"])
        # A native question is sent with its whole template, so the template is
        # hashed; a text pair is used on its own, so only that pair is hashed.
        hashed = field
        if "prompt_key" in source and isinstance(field, dict):
            hashed = field.get(source["prompt_key"])
        if field is None:
            problems.append(f"profile {source['profile']} has no field {source['field']}")
        elif canonical_sha256(hashed) != source["sha256"]:
            problems.append(f"profile {source['profile']}.{source['field']} drifted "
                            "from the card's sha256")
        if "question_id" in source and field is not None and source["question_id"] not in {
            q.get("id") for q in field
        }:
            problems.append(f"question {source['question_id']} not in the profile template")
        if "prompt_key" in source and field is not None:
            pair = field.get(source["prompt_key"])
            if not (isinstance(pair, list) and len(pair) == 2):
                problems.append(f"prompt {source['prompt_key']} is not a text pair")
        revision = source.get("revision")
        if revision is not None and profile.get("revision") != revision:
            problems.append(f"profile {source['profile']} revision is not {revision}")
        window = recipe.get("window")
        if window and window["frames"] != profile.get("window_frames", window["frames"]):
            problems.append(f"card window has {window['frames']} frames, profile "
                            f"{profile.get('window_frames')}")
    for split, ref in card["datasets"].items():
        module, constant = ref["pinned_by"].split(":")
        try:
            pins = getattr(importlib.import_module(module), constant)
        except (ImportError, AttributeError):
            problems.append(f"datasets.{split} pin {ref['pinned_by']} not found")
            continue
        if not isinstance(pins, dict) or not pins:
            problems.append(f"datasets.{split} pin {ref['pinned_by']} is empty")
    if decision["rule"] == "margin" and thresholds is not None:
        problems += _check_thresholds(card, thresholds, profiles)
    return problems


def _check_thresholds(card: dict, path: Path, profiles: Path) -> list[str]:
    decision, source = card["decision"], card["recipe"]["source"]
    digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    if decision["thresholds_sha256"] is None:
        return [f"card has no frozen threshold yet; {path} sha256 is {digest}"]
    if digest != decision["thresholds_sha256"]:
        return ["thresholds.json sha256 differs from the card"]
    artifact = json.loads(Path(path).read_text())
    profile = _profile(source["profile"], profiles)
    problems = []
    if artifact.get("validation_only") is not True:
        problems.append("thresholds.json is not a validation-only calibration")
    if artifact.get("model") != profile.get("model"):
        problems.append("thresholds.json model differs from the profile")
    value = (artifact.get("thresholds") or {}).get(source["prompt_key"])
    if value != decision["threshold"]:
        problems.append(f"threshold for {source['prompt_key']} is {value}, card has "
                        f"{decision['threshold']}")
    return problems


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    verify = sub.add_parser("verify")
    verify.add_argument("--profiles", type=Path, default=PROFILES)
    verify.add_argument("--thresholds", action="append", default=[],
                        metavar="CARD=PATH", help="frozen thresholds.json for a margin card")
    args = parser.parse_args(argv)
    thresholds = {}
    for item in args.thresholds:
        card_id, _, path = item.partition("=")
        if card_id not in object_cards.CARDS or not path:
            parser.error(f"--thresholds expects CARD=PATH with a known card: {item}")
        thresholds[card_id] = Path(path)
    failed = False
    for cid, card in object_cards.CARDS.items():
        problems = check_card(card, profiles=args.profiles, thresholds=thresholds.get(cid))
        state = card["status"]["state"] + ("" if card["live"]["supported"] else ", offline only")
        print(f"{'FAIL' if problems else 'ok  '} {cid} ({state})")
        for problem in problems:
            print(f"     - {problem}")
        failed |= bool(problems)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
