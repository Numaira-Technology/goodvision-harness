"""Join questions to per-video captures and emit benchmark predictions.

Input:
  --questions  bench_v1_sample/questions.jsonl   (public questions, no answers)
  --media      bench_v1_sample/media.jsonl        (video_id -> media metadata)
  --captures   a directory of <video_id>.json capture files (see schema.py),
               produced by the GPU-side capture step (capture_live.py)

Output:
  predictions JSONL, one {"qa_id": ..., "prediction": ...} per question, exactly
  the shape bench_v1_sample/how_to_feed.md asks for. The private answers live on
  the eval side; scoring happens there.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .extract import answer_question
from .schema import Capture, load_media, load_questions


def run(
    questions_path: Path, media_path: Path, captures_dir: Path, output: Path | None
) -> int:
    questions = load_questions(questions_path)
    media = load_media(media_path)
    captures: dict[str, Capture] = {}

    predictions: list[dict[str, str]] = []
    errors = 0
    for question in questions:
        if question.video_id not in media:
            print(f"[warn] {question.qa_id}: video_id {question.video_id} not in media", file=sys.stderr)
        capture = captures.get(question.video_id)
        if capture is None:
            capture_path = captures_dir / f"{question.video_id}.json"
            if not capture_path.exists():
                print(f"[skip] {question.qa_id}: missing capture {capture_path}", file=sys.stderr)
                errors += 1
                continue
            capture = captures[question.video_id] = Capture.load(capture_path)
        try:
            prediction = answer_question(question, capture)
        except Exception as exc:  # noqa: BLE001 - one bad question must not sink the run
            print(f"[error] {question.qa_id}: {exc}", file=sys.stderr)
            errors += 1
            continue
        predictions.append({"qa_id": question.qa_id, "prediction": prediction})

    sink = output.open("w", encoding="utf-8") if output else sys.stdout
    try:
        for row in predictions:
            sink.write(json.dumps(row, ensure_ascii=False) + "\n")
    finally:
        if output:
            sink.close()

    print(
        f"[done] {len(predictions)} predictions, {errors} skipped/errored",
        file=sys.stderr,
    )
    return 1 if errors else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--media", type=Path, required=True)
    parser.add_argument("--captures", type=Path, required=True, help="dir of <video_id>.json")
    parser.add_argument("--output", type=Path, help="predictions JSONL (default: stdout)")
    args = parser.parse_args(argv)
    return run(args.questions, args.media, args.captures, args.output)


if __name__ == "__main__":
    raise SystemExit(main())
