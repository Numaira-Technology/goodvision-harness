"""正式评测离线准备：冻结公开题、私有答案、媒体身份、队列和历史重评分。"""

from __future__ import annotations

import hashlib
import importlib.util
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parents[1]
OLD = WORKSPACE / "harness-experiments/main-event-supplement30-20261006"
REPORT = OLD / "report/182题逐题记录.json"
MEDIA_SOURCES = [
    WORKSPACE / "harness-experiments/ashley-core116-native-resume21-20261004/media-resolved.json",
    WORKSPACE / "harness-experiments/ashley-core116-native-20261002/source/bench_v1_core/media.jsonl",
    WORKSPACE / "harness-experiments/event-supplement12-20261005/media-paths.jsonl",
    WORKSPACE / "harness-experiments/balanced-supplement-20261006/media-paths.jsonl",
    OLD / "media-paths.jsonl",
]
SEED = "goodvision-detector-ablation182-pilot20-20261012-v1"
PUBLIC_FIELDS = (
    "answer_type", "benchmark_id", "benchmark_version", "capability",
    "component_role", "item_role", "membership_order", "public_context",
    "qa_id", "qa_version", "question", "response_options", "review_status",
    "schema_version", "source_dataset", "target_kind", "target_label", "video_id",
)


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def encoded(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def write_jsonl_new(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as out:
        for row in rows:
            out.write(encoded(row) + "\n")


def write_json_new(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as out:
        json.dump(value, out, ensure_ascii=False, sort_keys=True, indent=2)
        out.write("\n")


def digest_file(path: Path) -> tuple[str, int]:
    hasher = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            hasher.update(chunk)
            size += len(chunk)
    return hasher.hexdigest(), size


def rank(qa_id: str) -> str:
    return hashlib.sha256(f"{SEED}:{qa_id}".encode()).hexdigest()


def category(row: dict) -> str:
    cap = row["capability"]
    if cap == "counting":
        return "count"
    if cap == "location_tracking":
        return "location"
    if cap == "event_recognition":
        return "event"
    if cap != "temporal_questions":
        raise ValueError((row["qa_id"], cap))
    if row["question_form"] in {
        "first_exact_visible_count", "last_exact_visible_count", "longest_at_or_above_threshold"
    }:
        return "count_time"
    return "event_time"


def public_row(row: dict) -> dict:
    result = {key: row[key] for key in PUBLIC_FIELDS}
    # These two are question/source metadata, not answer labels or old scores.
    result["question_form"] = row["question_form"]
    result["source_group"] = row["source_group"]
    return result


def pick(rows: list[dict], used: set[str], category_name: str,
         source: str | None = None, form: str | None = None) -> dict:
    choices = [r for r in rows if r["qa_id"] not in used and category(r) == category_name
               and (source is None or r["source_dataset"] == source)
               and (form is None or r["question_form"] == form)]
    if not choices:
        raise ValueError(f"empty pilot stratum: {category_name} {source} {form}")
    winner = min(choices, key=lambda r: rank(r["qa_id"]))
    used.add(winner["qa_id"])
    return winner


def pilot_rows(rows: list[dict]) -> list[dict]:
    used: set[str] = set()
    selected = [
        pick(rows, used, "count", "vscrowd"),
        pick(rows, used, "count", "vscrowd"),
        pick(rows, used, "count", "mot17"),
        pick(rows, used, "count", "mot20"),
        pick(rows, used, "location", "caviar"),
        pick(rows, used, "location", "caviar"),
        pick(rows, used, "location", "meva"),
        pick(rows, used, "location", "mot20"),
        pick(rows, used, "count_time", "mot17"),
        pick(rows, used, "count_time", "mot20"),
        pick(rows, used, "event_time", "ntu_cctv_fights"),
        pick(rows, used, "event_time", "meva"),
        pick(rows, used, "event", "ntu_cctv_fights", "event_presence_binary"),
        pick(rows, used, "event", "rwf2000", "event_presence_binary"),
        pick(rows, used, "event", "surveillance_camera_fight", "event_presence_binary"),
        pick(rows, used, "event", "ubi_fights", "event_presence_binary"),
        pick(rows, used, "event", "caviar", "event_presence_combination"),
        pick(rows, used, "event", "meva", "event_presence_combination"),
        pick(rows, used, "event", "caviar", "event_single_action_selection"),
        pick(rows, used, "event", "meva", "event_single_action_selection"),
    ]
    return selected


def media_index() -> dict[str, dict]:
    resolved = json.loads(MEDIA_SOURCES[0].read_text())
    bank = {r["video_id"]: r for r in read_jsonl(MEDIA_SOURCES[1])}
    records: dict[str, dict] = {}
    for row in resolved:
        video_id = row["video_id"]
        meta = bank.get(video_id, {}).get("media", {})
        records[video_id] = {
            "path": row["source_path"],
            "bank_media_identity": meta.get("sha256"),
            "bank_media_identity_kind": meta.get("identity_kind"),
            "bank_media_bytes": meta.get("bytes"),
            "bank_media_path_or_name": meta.get("path_or_name"),
            "bank_commit": "32da3d422f193667d9e99eb866f362fc4f6b1b36",
            "source_record": str(MEDIA_SOURCES[0]),
            "bank_record": str(MEDIA_SOURCES[1]),
        }
    for source in MEDIA_SOURCES[2:]:
        for row in read_jsonl(source):
            video_id = row["video_id"]
            data = records.setdefault(video_id, {})
            if not data.get("path"):
                data["path"] = row["path"]
                data["source_record"] = str(source)
            for key in ("bank_media_identity", "bank_media_identity_kind", "bank_media_bytes",
                        "bank_media_path_or_name", "bank_commit"):
                if data.get(key) is None and row.get(key) is not None:
                    data[key] = row[key]
                    data["bank_record"] = str(source)
    return records


def media_rows(rows: list[dict]) -> list[dict]:
    index = media_index()
    by_path: dict[str, tuple[str, int]] = {}
    result = []
    for row in rows:
        video_id = row["video_id"]
        item = index.get(video_id)
        if not item or not item.get("path"):
            raise ValueError(f"missing media path: {video_id}")
        path = Path(item["path"])
        if not path.is_file():
            raise FileNotFoundError(path)
        if str(path) not in by_path:
            by_path[str(path)] = digest_file(path)
        actual_sha, actual_bytes = by_path[str(path)]
        bank_sha = item.get("bank_media_identity")
        bank_kind = item.get("bank_media_identity_kind")
        is_file_identity = bank_kind not in {"image_sequence_tree_sha256", "directory_tree_sha256"}
        if bank_sha and is_file_identity and bank_sha != actual_sha:
            raise ValueError(f"bank/media SHA mismatch: {video_id}")
        if is_file_identity and item.get("bank_media_bytes") is not None and item["bank_media_bytes"] != actual_bytes:
            raise ValueError(f"bank/media byte mismatch: {video_id}")
        result.append({
            "qa_id": row["qa_id"], "video_id": video_id, "source_dataset": row["source_dataset"],
            "source_group": row["source_group"], "path": str(path),
            "exists": True, "bytes": actual_bytes, "actual_sha256": actual_sha,
            "media_file_sha256": actual_sha, "bank_media_identity": bank_sha,
            "bank_media_identity_kind": bank_kind, "bank_media_bytes": item.get("bank_media_bytes"),
            "bank_media_path_or_name": item.get("bank_media_path_or_name"),
            "bank_commit": item.get("bank_commit"), "source_record": item.get("source_record"),
            "bank_record": item.get("bank_record"),
            "identity_status": "file_sha256_matches_bank" if bank_sha and is_file_identity
            else "converted_mp4_sha256_separate_from_bank_tree" if not is_file_identity
            else "file_hashed_bank_identity_unavailable",
        })
    return result


def old_rescores(rows: list[dict]) -> list[dict]:
    spec = importlib.util.spec_from_file_location("pr63_scoring", ROOT / "scoring/scoring.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    result = []
    for row in rows:
        for method in ("raw", "sam", "ashley"):
            old = row[method]
            prediction = old.get("prediction")
            status = old.get("status")
            scored = status == "scored"
            doc = {
                "capability": row["capability"],
                "public_context_json": encoded(row["public_context"]),
                "ground_truth_json": encoded(row["ground_truth"]),
            }
            result.append({
                "qa_id": row["qa_id"], "method": method, "status": status,
                "prediction": prediction, "old_metrics": old.get("metrics"),
                "old_source": old.get("source"),
                "pr63_metrics": module.benchmark_v1_process_results(doc, [str(prediction or "")]) if scored else None,
                "rescore_status": "scored" if scored else "historical_pending_retained",
            })
    return result


def main() -> None:
    rows = json.loads(REPORT.read_text())
    if len(rows) != 182 or len({r["qa_id"] for r in rows}) != 182:
        raise ValueError("cohort is not exactly 182 unique questions")
    public = [public_row(r) for r in rows]
    pilot = pilot_rows(public)
    pilot_ids = {r["qa_id"] for r in pilot}
    expected = {"count": 4, "location": 4, "count_time": 2, "event_time": 2, "event": 8}
    if dict(Counter(category(r) for r in pilot)) != expected:
        raise ValueError("pilot quota mismatch")
    ordered = pilot + sorted((r for r in public if r["qa_id"] not in pilot_ids),
                             key=lambda r: (category(r), rank(r["qa_id"])))
    queue = [{"qa_id": r["qa_id"], "video_id": r["video_id"],
              "phase": "pilot" if r["qa_id"] in pilot_ids else "remaining",
              "category": category(r), "geometry_only": category(r) in {"count", "location", "count_time"},
              "vlm_enabled": category(r) in {"event", "event_time"},
              "source_dataset": r["source_dataset"]} for r in ordered]
    media = media_rows(public)
    rescores = old_rescores(rows)
    frozen = [ROOT / "public/questions.jsonl", ROOT / "cohort/media-paths.jsonl",
              ROOT / "private/answers.jsonl", ROOT / "cohort/selection.json",
              ROOT / "cohort/queue.json", ROOT / "scoring/old-pr63-rescore.jsonl"]
    if any(p.exists() for p in frozen):
        raise FileExistsError("frozen cohort output exists; do not overwrite")
    write_jsonl_new(frozen[0], public)
    write_jsonl_new(frozen[1], media)
    write_jsonl_new(frozen[2], [{"qa_id": r["qa_id"], "ground_truth": r["ground_truth"]} for r in rows])
    write_json_new(frozen[3], {
        "seed": SEED, "selection_inputs": ["source_dataset", "question_form", "capability", "qa_id"],
        "selection_does_not_use": ["ground_truth", "event_polarity", "old_prediction", "old_metrics"],
        "pilot_qa_ids": [r["qa_id"] for r in pilot],
        "pilot": [{"qa_id": r["qa_id"], "category": category(r),
                   "source_dataset": r["source_dataset"], "question_form": r["question_form"]} for r in pilot],
        "quota": expected, "all_qa_ids": [r["qa_id"] for r in public],
    })
    write_json_new(frozen[4], {
        "seed": SEED, "total_questions": 182, "pilot_count": 20,
        "geometry_only_count": 90, "event_vlm_count": 92,
        "questions": queue,
    })
    write_jsonl_new(frozen[5], rescores)
    print(encoded({"questions": len(public), "media": len(media), "distinct_media_paths": len({m["path"] for m in media}),
                   "pilot": len(pilot), "queue": len(queue), "old_rescores": len(rescores),
                   "media_identity_status": dict(Counter(m["identity_status"] for m in media))}))


if __name__ == "__main__":
    main()
