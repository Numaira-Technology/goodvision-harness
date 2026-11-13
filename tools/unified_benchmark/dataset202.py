"""Pinned 202-question test extension. No inference, downloads or label edits."""
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

from .contract import BAG_VALIDATION_SCHEMA, EVENTS, file_hash

PINNED_202_INPUTS = {
    "manifest_202.json": "d5c019130d672afd40638a6ca51c1c84ca8e8c7fad0dbe3faf493187002bb11d",
    "samples.json": "bfb685074787eeb449e2b8c6d89ad6cb055a861363cdb4afae898d5c93631545",
    "videos.json": "71a9e6b4b1e59ba59cc584c874b99ae2907ab10dc4459fb251fbc54d0f31cf2c",
}
EVENT_COUNTS = dict(zip(EVENTS, (32, 22, 62, 66, 10, 10)))
REVIEW_STATUS = {"questions_1_120": "human_reviewed",
                "questions_121_202": "inherited_source_labels_pending_single_view_review"}


def indexed(rows, key):
    result = {r[key]: r for r in rows}
    if len(result) != len(rows):
        raise ValueError("duplicate dataset identity: " + key)
    return result


def extend_test(parts, manifest, samples, registry):
    """Join source groups without rewriting the original frozen split sidecar."""
    for doc in (manifest, samples):
        if (doc.get("status") != "draft_pending_user_review"
                or doc.get("review_status_by_cohort") != REVIEW_STATUS
                or doc.get("dataset") != "goodvision-events-202-20261110"):
            raise ValueError("202 reference review scope differs; never infer human approval")
    if (manifest.get("n_questions") != 202 or manifest.get("n_files_hashed") != 192
            or manifest.get("missing_files") != [] or samples.get("question_count") != 202
            or samples.get("video_count") != 192
            or manifest.get("samples_json_sha256") != PINNED_202_INPUTS["samples.json"]):
        raise ValueError("202 manifest counts or sample fingerprint differ")
    files = indexed(manifest["files"], "number")
    questions = indexed(samples["samples"], "number")
    videos = indexed(registry["videos"], "filename")
    if (set(files) != set(range(1, 203)) or set(questions) != set(files)
            or len(videos) != 192 or set(videos) != {r["filename"] for r in files.values()}
            or Counter(r["event"] for r in files.values()) != EVENT_COUNTS):
        raise ValueError("202 membership, video coverage or event counts differ")
    previous = indexed(parts["test"]["rows"], "number")
    if set(previous) != set(range(1, 121)):
        raise ValueError("original120 membership required")
    joined, pairs = [], defaultdict(list)
    for number in sorted(files):
        row, q = files[number], questions[number]
        for field in ("filename", "event", "question", "proposed_answer", "human_review"):
            if row[field] != q[field]:
                raise ValueError("202 manifest/question/answer mismatch")
        if row["proposed_answer"] not in ("yes", "no"):
            raise ValueError("binary reference required")
        if number <= 120:
            old = previous[number]
            if any(row[k] != old[k] for k in ("filename", "sha256", "event", "question", "proposed_answer", "human_review")):
                raise ValueError("original120 question, answer or video changed")
            joined.append(row | {"source_video_group": old["source_video_group"]})
            continue
        video = videos[row["filename"]]
        group = q.get("source_group")
        if (not isinstance(group, str) or not group.strip() or group != group.strip()
                or row["human_review"] != "pending_view_review"
                or q.get("reference_review") != "inherited_from_human_reviewed_source"
                or q.get("view") not in ("left", "right") or not q.get("source_sample_id")
                or any(video.get(k) != q.get(k) for k in ("source_group", "source_sample_id", "view"))):
            raise ValueError("robot82 source/view/review metadata differs")
        pairs[q["source_sample_id"]].append(q)
        joined.append(row | {"source_video_group": group, "source_sample_id": q["source_sample_id"],
                             "view": q["view"], "reference_review": q["reference_review"]})
    if len(pairs) != 41:
        raise ValueError("expected 41 source clips, each with two single views")
    for pair in pairs.values():
        if (len(pair) != 2 or {q["view"] for q in pair} != {"left", "right"}
                or any(pair[0].get(k) != pair[1].get(k) for k in
                       ("source_group", "source_interval_seconds", "event", "proposed_answer"))):
            raise ValueError("left/right source grouping or inherited labels differ")
    for field in ("filename", "sha256", "source_video_group"):
        if {r[field] for r in joined} & {r[field] for r in parts["validation"]["rows"]}:
            raise ValueError("validation/test overlap by " + field)
    old_hashes = {r["sha256"] for r in previous.values()}
    if old_hashes & {r["sha256"] for r in joined if r["number"] > 120}:
        raise ValueError("robot82 duplicates an original120 video")
    context = {"dataset": samples["dataset"], "test_questions": 202, "test_videos": 192,
               "event_counts": EVENT_COUNTS, "review_status_by_cohort": REVIEW_STATUS,
               "robot82_source_recordings": len({r["source_video_group"] for r in joined if r["number"] > 120}),
               "new_questions_split": "test", "validation_questions": 24,
               "limits": ["82 single-view labels inherited, not individually re-reviewed; provisional reference agreement.",
                          "Related left/right views and recording cuts all stay in test; old stereo41 is not counted again.",
                          "Original120 has previous evaluation/development use; this is not a fresh held-out test.",
                          "Train/development/actor/scene isolation remains unproven; validation24 has no bag examples.",
                          "Original and robot weapon reference questions have different semantics; frozen current-image prompts do not establish relabeling equivalence.",
                          "All 36 additional fall views come from one recording; correlated views are not independent trials."]}
    return parts | {"test": {"rows": joined, "dataset_context": context,
                    "split_context": {"original120": parts["test"]["split_context"],
                                      "extension": "questions121-202 all test; source_group from pinned samples.json"}}}


def load_extension(out, frozen, parts):
    for name, expected in PINNED_202_INPUTS.items():
        path = frozen / name
        if file_hash(path) != expected:
            raise ValueError("frozen input changed: " + name)
        with (out / "inputs" / name).open("xb") as f:
            f.write(path.read_bytes())
    return extend_test(parts, *(json.loads((frozen / name).read_text())
                               for name in ("manifest_202.json", "samples.json", "videos.json")))


def add_bag_validation(out, parts, path, expected_hash):
    """Append two validation questions per original bag video. Test membership is frozen."""
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_hash:
        raise ValueError("bag validation manifest SHA256 differs")
    doc = json.loads(raw)
    if doc["schema"] != BAG_VALIDATION_SCHEMA or len(doc["videos"]) != 10:
        raise ValueError("bag-validation10-v1 requires exactly ten videos")
    original = parts["validation"]["rows"]
    test = parts["test"]["rows"]
    filenames = {r["filename"] for r in original + test}
    hashes = {r["sha256"] for r in original + test}
    test_groups = {r["source_video_group"] for r in test}
    added = []
    for video in doc["videos"]:
        name, sha, group = (video[k] for k in ("filename", "sha256", "source_video_group"))
        if (not isinstance(name, str) or not name or Path(name).name != name or "\\" in name
                or name in {".", ".."} or name in filenames):
            raise ValueError("invalid or overlapping bag validation filename")
        if (not isinstance(sha, str) or len(sha) != 64 or any(c not in "0123456789abcdef" for c in sha)
                or sha in hashes):
            raise ValueError("invalid or overlapping bag validation media SHA256")
        if not isinstance(group, str) or not group or group != group.strip() or group in test_groups:
            raise ValueError("invalid or test-overlapping bag validation source group")
        filenames.add(name)
        hashes.add(sha)
        questions = indexed(video["questions"], "event")
        if set(questions) != set(EVENTS[4:]):
            raise ValueError("each bag validation video requires both bag questions")
        for event in EVENTS[4:]:
            question = questions[event]
            review = question["human_review"]
            if question["proposed_answer"] not in ("yes", "no") or not isinstance(review, str) or not review.strip():
                raise ValueError("binary bag validation reference and review provenance required")
            added.append({"number": len(original) + len(added) + 1, "filename": name, "sha256": sha,
                          "source_video_group": group, "event": event, "proposed_answer": question["proposed_answer"],
                          "human_review": review})
    for event in EVENTS[4:]:
        if {r["proposed_answer"] for r in added if r["event"] == event} != {"yes", "no"}:
            raise ValueError("both validation reference labels required for " + event)
    with (out / "inputs" / "bag-validation.json").open("xb") as f:
        f.write(raw)
    overlay = {"schema": BAG_VALIDATION_SCHEMA, "sha256": expected_hash,
               "review_status": sorted({r["human_review"] for r in added})}
    context = {"bag_validation": overlay, "validation_videos": 34, "validation_questions": 44}
    parts["validation"] = {"rows": original + added, "dataset_context": context,
                           "split_context": {"original24": parts["validation"]["split_context"], "bag_validation": overlay}}
    parts["test"]["dataset_context"].update(context)
