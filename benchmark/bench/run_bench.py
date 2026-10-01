"""Run full offline Harness sessions or extract answers from legacy captures.

Input:
  --questions  bench_v1_sample/questions.jsonl   (public questions, no answers)
  --media      bench_v1_sample/media.jsonl        (video_id -> media metadata)
  --captures   a directory of <video_id>.json capture files (see schema.py),
               produced by the GPU-side capture step (capture_live.py)
  --offline    instead process server-side video files with the full Harness;
               use --media-root and --output-dir (see benchmark/README.md)

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
from urllib.parse import quote

from .capture_live import capture_offline
from .extract import answer_question
from .schema import Capture, load_media, load_questions
from .vocabulary import session_features, supported_labels


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
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--captures", type=Path, help="legacy dir of <video_id>.json")
    mode.add_argument("--offline", action="store_true", help="run full Harness from server-side files")
    parser.add_argument("--output", type=Path, help="predictions JSONL (offline default: output-dir/predictions.jsonl; legacy: stdout)")
    parser.add_argument("--base-url", default="http://localhost:8080")
    parser.add_argument("--media-root", help="video root on the Harness server, not the runner machine")
    parser.add_argument("--output-dir", type=Path)
    reuse = parser.add_mutually_exclusive_group()
    reuse.add_argument("--reuse-results", action="store_true", help="reuse complete local results; run missing videos")
    reuse.add_argument("--reanswer", action="store_true", help="make a new text answer from saved server sessions; no replay")
    parser.add_argument(
        "--no-auto-features", action="store_true",
        help="do not derive per-video events from the questions; "
             "leave every session on the harness default",
    )
    parser.add_argument("--observation-max-tokens", type=int)
    parser.add_argument("--answer-max-tokens", type=int)
    args = parser.parse_args(argv)
    if args.offline:
        if args.output_dir is None or (not args.reanswer and args.media_root is None):
            parser.error("--offline requires --output-dir and, unless --reanswer, --media-root")
        return run_offline(args)
    return run(args.questions, args.media, args.captures, args.output)


def _reusable_result(result_path: Path, public: list, args) -> dict | None:
    """A cached result complete enough to answer from, or None to re-run.

    An unreadable or half-written cache means re-run, not fail: the video is
    still answerable, and the run is cheaper to repeat than to debug.
    """
    if not args.reuse_results or not result_path.exists():
        return None
    try:
        cached = json.loads(result_path.read_text(encoding="utf-8"))
        expected_questions = [
            {key: value for key, value in row.items() if key != "video_id"}
            for row in public
        ]
        if (cached["status"] == "complete" and cached.get("records_downloaded")
                and cached["metadata"]["questions"] == expected_questions):
            return cached
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"[warn] unusable cached result {result_path}: {exc}", file=sys.stderr)
    return None


def run_offline(args) -> int:
    questions = load_questions(args.questions)
    media = load_media(args.media)
    groups = {}
    for question in questions:
        groups.setdefault(question.video_id, []).append(question)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = args.output or args.output_dir / "predictions.jsonl"
    failures = 0
    preflight_done = False
    available_features = None
    with predictions_path.open("w", encoding="utf-8") as predictions, (
        args.output_dir / "details.jsonl"
    ).open("w", encoding="utf-8") as details:
        for video_id, video_questions in groups.items():
            # URL encoding keeps dataset IDs (including ':' and '/') in one directory.
            directory = args.output_dir / "videos" / ("video-" + quote(video_id, safe=""))
            directory.mkdir(parents=True, exist_ok=True)
            answers, error = {}, None
            public = [question.public_payload() for question in video_questions]
            result = _reusable_result(directory / "result.json", public, args)
            if result is None and not preflight_done:
                # Read the installed vocabulary once. Questions outside it
                # still reach the VLM through question-aware observations.
                if not args.reanswer and not args.no_auto_features:
                    available_features = supported_labels(args.base_url)
                preflight_done = True
            try:
                if result is None:
                    video_path = ""
                    if not args.reanswer:
                        relative_path = Path(media[video_id]["path"])
                        if relative_path.is_absolute() or ".." in relative_path.parts:
                            raise ValueError("media path must be relative to --media-root")
                        video_path = str(Path(args.media_root) / relative_path)
                    result = capture_offline(
                        args.base_url, video_id, video_path,
                        public, directory,
                        observation_max_tokens=args.observation_max_tokens,
                        answer_max_tokens=args.answer_max_tokens,
                        reanswer=args.reanswer,
                        features=(
                            None
                            if args.no_auto_features
                            else session_features(video_questions, supported=available_features)
                        ),
                    )
                answers = {row["qa_id"]: row["prediction"] for row in result["predictions"]}
                error = result.get("error")
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                print(f"[error] {video_id}: {error}", file=sys.stderr, flush=True)
            for question in video_questions:
                prediction = answers.get(question.qa_id, "")
                failed = error is not None or not prediction.strip()
                failures += int(failed)
                row = {"qa_id": question.qa_id, "prediction": prediction}
                predictions.write(json.dumps(row, ensure_ascii=False) + "\n")
                details.write(json.dumps({
                    **row, "video_id": video_id, "capability": question.capability,
                    "status": "failed" if failed else "answered",
                    "error": error or ("no_answer" if not prediction else None),
                    "artifact_error": result.get("artifact_error") if result else None,
                    "artifacts": result.get("artifacts", str(directory)) if result else str(directory),
                }, ensure_ascii=False) + "\n")
            predictions.flush()
            details.flush()
    print(f"[done] {len(questions)} predictions, {failures} failed/empty", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
