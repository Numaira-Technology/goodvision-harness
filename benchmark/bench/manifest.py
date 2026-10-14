"""Fetch manifest + answerability report for a Benchmark V1 question set.

Two operator-facing tables, both derived purely from the public questions/media
(no answers, no GPU):

* **fetch manifest** — every media item the run needs, with its download handle
  (Google Drive id or R2 path), sha256 to verify, and, for image sequences, the
  frame rate the clip must be assembled at (fps varies: MOT17-05 is 14, MOT20 is
  25, MOT17-02 is 30). vscrowd sequences carry no clip; counting is fps-agnostic
  so they default to 25.

* **answerability report** — which questions this harness can actually answer,
  bucketed so the eval side knows expected coverage up front:
    - `cli_fight`   fight event_recognition / fight-end temporal (no GPU; CLI path)
    - `gpu_tracks`  counting / location / MOT count-temporal (needs SAM capture)
    - `no_detector` meva/caviar semantic events — the scene engine has no such
                    detector, so these are structurally unanswerable here.

Usage:
    python -m bench.manifest --questions ../bench_v1_sample80/questions.jsonl \
        --media ../bench_v1_sample80/media.jsonl \
        --fetch-out fetch_manifest.tsv --report-out answerability.tsv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .extract import parse_count_temporal
from .schema import EVENT_TARGET_TO_HARNESS_LABEL, Question, load_jsonl, load_questions

_DEFAULT_SEQ_FPS = 25.0


def fps_by_video(questions: list[Question]) -> dict[str, float]:
    """Derive per-video fps from any question clip (end_frame_number / end_sec)."""
    out: dict[str, float] = {}
    for q in questions:
        clip = q.public_context.get("clip") or {}
        f, s = clip.get("end_frame_number"), clip.get("end_sec")
        if f and s:
            out[q.video_id] = round(float(f) / float(s), 3)
    return out


def classify(question: Question) -> str:
    """Bucket a question by whether/how this harness can answer it."""
    cap = question.capability
    target = str(question.raw.get("target_label", ""))
    if cap == "counting" or cap == "location_tracking":
        return "gpu_tracks"
    if cap == "event_recognition":
        return "cli_fight" if target in EVENT_TARGET_TO_HARNESS_LABEL else "no_detector"
    if cap == "temporal_questions":
        if parse_count_temporal(question.question) is not None:
            return "gpu_tracks"  # MOT count-over-time, computed from tracks
        if target in EVENT_TARGET_TO_HARNESS_LABEL:
            return "cli_fight"  # fight-end
        return "no_detector"  # meva/caviar event-duration
    return "no_detector"


def fetch_rows(media_rows: list[dict], fps: dict[str, float]) -> list[dict]:
    rows = []
    for m in media_rows:
        md = m["media"]
        vid = m["video_id"]
        seq = md["media_type"] == "image_sequence"
        rows.append({
            "video_id": vid,
            "dataset": m["source_dataset"],
            "media_type": md["media_type"],
            "path_or_name": md.get("path_or_name", ""),
            "drive_file_id": md.get("drive_file_id", ""),
            "sha256": md.get("sha256", ""),
            "bytes": md.get("bytes", ""),
            "assemble_fps": (fps.get(vid, _DEFAULT_SEQ_FPS) if seq else ""),
        })
    return rows


def _write_tsv(path: Path | None, header: list[str], rows: list[list]) -> None:
    sink = path.open("w", encoding="utf-8") if path else sys.stdout
    try:
        sink.write("\t".join(header) + "\n")
        for r in rows:
            sink.write("\t".join("" if v is None else str(v) for v in r) + "\n")
    finally:
        if path:
            sink.close()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--questions", type=Path, required=True)
    ap.add_argument("--media", type=Path, required=True)
    ap.add_argument("--fetch-out", type=Path, help="TSV of media to download (default: stdout)")
    ap.add_argument("--report-out", type=Path, help="TSV of per-question answerability")
    args = ap.parse_args(argv)

    questions = load_questions(args.questions)
    media_rows = list(load_jsonl(args.media))
    fps = fps_by_video(questions)

    fr = fetch_rows(media_rows, fps)
    _write_tsv(
        args.fetch_out,
        ["video_id", "dataset", "media_type", "path_or_name", "drive_file_id",
         "sha256", "bytes", "assemble_fps"],
        [[r[k] for k in ("video_id", "dataset", "media_type", "path_or_name",
                         "drive_file_id", "sha256", "bytes", "assemble_fps")] for r in fr],
    )

    buckets: dict[str, int] = {}
    report_rows = []
    for q in questions:
        b = classify(q)
        buckets[b] = buckets.get(b, 0) + 1
        report_rows.append([q.qa_id, q.raw.get("source_dataset", ""), q.capability, b])
    if args.report_out:
        _write_tsv(args.report_out, ["qa_id", "dataset", "capability", "bucket"], report_rows)

    answerable = buckets.get("cli_fight", 0) + buckets.get("gpu_tracks", 0)
    print(f"[manifest] {len(fr)} media items", file=sys.stderr)
    print(
        f"[answerability] {answerable}/{len(questions)} answerable "
        f"(cli_fight={buckets.get('cli_fight', 0)}, gpu_tracks={buckets.get('gpu_tracks', 0)}); "
        f"no_detector={buckets.get('no_detector', 0)}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
