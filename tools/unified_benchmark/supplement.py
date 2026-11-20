"""Validate optional test inputs and bind their membership to frozen bundles."""

import hashlib
import json
from pathlib import Path, PureWindowsPath

from .contract import CURRENT_VERSION, EVENTS

SCHEMA = "event-test-supplement-v1"
REVIEWED_SCHEMA = "event-test-supplement-v2"
MANIFEST = "inputs/test-supplement.json"


def parse_test_supplement(raw):
    """Validate before saving; review text is provenance, never inferred approval."""
    doc = json.loads(raw)
    if (
        not isinstance(doc, dict)
        or doc.get("schema") not in (SCHEMA, REVIEWED_SCHEMA)
        or doc.get("split") != "test"
    ):
        raise ValueError("a test-only event supplement is required")
    dataset = doc.get("dataset")
    if not isinstance(dataset, str) or not dataset or dataset != dataset.strip():
        raise ValueError("supplement dataset identity required")
    rows = doc.get("samples")
    if not isinstance(rows, list) or not rows:
        raise ValueError("nonempty supplementary test questions required")
    media, hashes, questions, added = {}, {}, set(), []
    for number, row in enumerate(rows, 203):
        if (
            not isinstance(row, dict)
            or type(row.get("number")) is not int
            or row["number"] != number
        ):
            raise ValueError("supplement question numbers must follow original202")
        name, sha = row.get("filename"), row.get("sha256")
        group = row.get("source_video_group", row.get("source_group"))
        review, event = row.get("human_review"), row.get("event")
        if (
            not isinstance(name, str)
            or not name
            or name in {".", ".."}
            or Path(name).name != name
            or PureWindowsPath(name).name != name
            or ":" in name
            or "\\" in name
            or "\0" in name
        ):
            raise ValueError("invalid supplementary video filename")
        if (
            not isinstance(sha, str)
            or len(sha) != 64
            or any(c not in "0123456789abcdef" for c in sha)
        ):
            raise ValueError("invalid supplementary video SHA256")
        if (
            not isinstance(group, str)
            or not group
            or group != group.strip()
            or any(
                row[key] != group
                for key in ("source_video_group", "source_group")
                if key in row
            )
        ):
            raise ValueError("invalid or conflicting supplementary source group")
        if (
            event not in EVENTS
            or row.get("proposed_answer") not in ("yes", "no")
            or not isinstance(review, str)
            or not review.strip()
        ):
            raise ValueError(
                "supported event, binary reference and review text required"
            )
        if media.setdefault(name, (sha, group)) != (sha, group):
            raise ValueError("conflicting supplementary video identity")
        if hashes.setdefault(sha, name) != name:
            raise ValueError(
                "duplicate supplementary video bytes under different names"
            )
        if (name, event) in questions:
            raise ValueError("duplicate supplementary video/event question")
        questions.add((name, event))
        added.append(row | {"source_video_group": group})
    overlay = {
        "schema": doc["schema"],
        "sha256": hashlib.sha256(raw).hexdigest(),
        "dataset": dataset,
        "question_numbers": [r["number"] for r in added],
        "review_status": sorted({r["human_review"] for r in added}),
    }
    reviewed = doc.get("reviewed_answers", [])
    if not isinstance(reviewed, list) or bool(reviewed) != (doc["schema"] == REVIEWED_SCHEMA):
        raise ValueError("reviewed answers require a v2 test supplement")
    seen = set()
    for row in reviewed:
        number = row["number"]
        if (type(number) is not int or not 1 <= number <= 202 or number in seen
                or row["proposed_answer"] not in ("yes", "no")
                or row["previous_answer"] not in ("yes", "no")):
            raise ValueError("invalid reviewed test answer")
        seen.add(number)
    if reviewed:
        overlay["reviewed_question_numbers"] = sorted(seen)
    return added, overlay, reviewed


def bundle_test_supplement(bundle, hashes, report, protocol):
    """Derive the test denominator from the retained input, not report metadata."""
    context = report.get("dataset_context") or {}
    overlay = context.get("test_supplement")
    rows, reviewed = [], []
    if overlay is not None:
        if (
            protocol != CURRENT_VERSION
            or not isinstance(overlay, dict)
            or not hashes.get(MANIFEST)
            or hashes[MANIFEST] != overlay.get("sha256")
        ):
            raise ValueError("test supplement manifest fingerprint or protocol differs")
        rows, expected, reviewed = parse_test_supplement((bundle / MANIFEST).read_bytes())
        if overlay != expected or context.get("dataset") != expected["dataset"]:
            raise ValueError("test supplement metadata differs from retained input")
    elif MANIFEST in hashes or (bundle / MANIFEST).exists():
        raise ValueError("unrecorded test supplement manifest")
    if protocol == CURRENT_VERSION:
        count = context.get("test_questions", 202)
        if type(count) is not int or count != 202 + len(rows):
            raise ValueError("test question count differs from frozen cohort")
    return rows, reviewed


def verify_supplement_plan(plan, rows, report):
    """Keep appended references and their cohort out of validation plans."""
    overlay = (plan.get("dataset_context") or {}).get("test_supplement")
    expected = (report.get("dataset_context") or {}).get("test_supplement")
    if plan["split"] == "validation":
        if overlay is not None:
            raise ValueError("test supplement must not appear in validation")
        return
    if overlay != expected:
        raise ValueError("plan test supplement differs from preflight")
    if not rows:
        return
    if plan.get("dataset_context") != report.get("dataset_context"):
        raise ValueError("plan test dataset context differs from preflight")
    questions = {q["number"]: q for q in plan["questions"]}
    units = {u["id"]: u for u in plan["units"]}
    for row in rows:
        question = questions.get(row["number"], {})
        if (
            any(
                question.get(k) != row[k]
                for k in ("filename", "event", "source_video_group")
            )
            or question.get("reference") != row["proposed_answer"]
            or not question.get("unit_ids")
            or any(
                units.get(uid, {}).get("source_id") != "test/" + row["sha256"]
                for uid in question["unit_ids"]
            )
        ):
            raise ValueError("plan supplementary question differs from retained input")
