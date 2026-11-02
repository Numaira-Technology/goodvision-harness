#!/usr/bin/env python3
"""Common streaming event evaluation, with an explicit historical v2 mode."""
import argparse
import base64
import hashlib
import json
import math
import re
import subprocess
import sys
import time
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------- frozen rules
SCORED_EVENTS = ["fight", "smoking", "fall", "weapon"]  # §2
FRAME_MAX = 14            # §4  n = max(3, min(14, int(dur)))
FRAME_MIN = 3
FRAME_WIDTH = 720         # §4  scale=720:-2
GATE = 0.5               # §5  uniform, never tuned on the set
TEMPERATURE = 0           # §6
CHAT_MAX_TOKENS = 30      # §6
TIMEOUT = 90              # §6
RETRIES = 3               # §6
CHAT_URL = "https://openrouter.ai/api/v1/chat/completions"
LUNA_URL = "https://openrouter.ai/api/alpha/decisions"
# model ID + backend class for each named candidate (§3, §6). Three classes:
#   language  - chat VLMs (qwen, gemini): prompt -> JSON yes/no
#   decision  - "system-one" decision API (Luna; videojev would join here): noul
#   threshold - CLIP / YOLO etc: score vs fixed threshold (not yet wired)
# Output format is fixed WITHIN a class; classes are not aligned to each other.
CANDIDATES = {
    "qwen":   ("qwen/qwen3-vl-8b-instruct", "language"),
    "gemini": ("google/gemini-2.5-flash",   "language"),
    "luna":   ("openai/gpt-6-luna-decisions", "decision"),
}

REPO = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = Path(__file__).parent / "datasets/manifest_50.json"
SPEC_VERSION = "v2"


def load_key(env_file):
    import os
    if os.environ.get("OPENROUTER_API_KEY"):
        return os.environ["OPENROUTER_API_KEY"]
    for line in open(env_file):
        if line.startswith("OPENROUTER_API_KEY="):
            return line.split("=", 1)[1].strip()
    sys.exit("OPENROUTER_API_KEY not found (env or %s)" % env_file)


def question_core(question):
    """§1: the exam question minus the trailing 'Answer yes or no.'"""
    return re.sub(r"\s*answer yes or no\.?\s*$", "", question, flags=re.I).strip()


# ---------------------------------------------------------------- frame sampling (§4)
def probe_duration(path):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "csv=p=0", str(path)], capture_output=True, text=True,
        check=True, timeout=TIMEOUT).stdout.strip()
    duration = float(out)
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("invalid video duration")
    return duration


def frame_times(dur, interval=None):
    """Dense mode (default): n = max(3, min(14, int(dur))) uniform frames.
    Interval mode (--interval S): one frame per S seconds, like the product's
    PREVIEW_SELECT_INTERVAL — short clips get as few as 1 frame (§4)."""
    if not math.isfinite(dur) or dur <= 0:
        raise ValueError("duration must be finite and positive")
    if interval is not None:
        if not math.isfinite(interval) or interval <= 0:
            raise ValueError("interval must be finite and positive")
        n = max(1, round(dur / interval))
    else:
        n = max(FRAME_MIN, min(FRAME_MAX, int(dur)))
    return [dur * (i + 0.5) / n for i in range(n)]


def frame_uri(path, t):
    raw = subprocess.run(
        ["ffmpeg", "-nostdin", "-loglevel", "error", "-ss", str(t), "-i", str(path),
         "-vframes", "1", "-vf", f"scale={FRAME_WIDTH}:-2", "-f", "image2pipe",
         "-vcodec", "mjpeg", "pipe:1"], capture_output=True,
        check=True, timeout=TIMEOUT).stdout
    return "data:image/jpeg;base64," + base64.b64encode(raw).decode() if raw else None


# ---------------------------------------------------------------- backends (§3)
def _unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate response field")
        result[key] = value
    return result


def _post(url, body, key):
    req = urllib.request.Request(
        url, data=json.dumps(body).encode(),
        headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=TIMEOUT) as response:
        encoded = response.read(1024 * 1024 + 1)
    if len(encoded) > 1024 * 1024:
        raise ValueError("oversized response")
    return json.loads(encoded, object_pairs_hook=_unique)


def frame_score(backend, model, qcore, uri, key):
    """Return a per-frame score in [0,1], or None on hard failure after retries."""
    if backend == "language":
        prompt = (f'{qcore} Answer only with compact JSON: '
                  '{"answer": true} or {"answer": false}.')
        body = {"model": model, "messages": [{"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": uri}}]}],
            "max_tokens": CHAT_MAX_TOKENS, "temperature": TEMPERATURE}
        for _ in range(RETRIES):
            try:
                choice = _post(CHAT_URL, body, key)["choices"][0]
                if choice.get("finish_reason") != "stop":
                    raise ValueError("incomplete language answer")
                c = choice["message"]["content"]
                m = re.search(r"\{.*\}", c, re.DOTALL)
                answer = json.loads(m.group(0), object_pairs_hook=_unique) if m else None
                if (not isinstance(answer, dict) or set(answer) != {"answer"}
                        or type(answer["answer"]) is not bool):
                    raise ValueError("language answer must be a JSON boolean")
                return float(answer["answer"])
            except Exception:
                time.sleep(1)
        return None
    if backend == "decision":
        body = {"model": model,
                "state": ["Surveillance video frame.",
                          {"type": "image_url", "image_url": {"url": uri}}],
                "questions": {"answer": {"type": "noul", "instructions": qcore,
                    "criteria": {"true": "Yes — " + qcore, "false": "No — " + qcore}}}}
        for _ in range(RETRIES):
            try:
                answer = _post(LUNA_URL, body, key)["answers"]["answer"]
                score = answer.get("noul")
                if (answer.get("type") != "noul" or answer.get("error") is not None
                        or type(score) not in (int, float) or not math.isfinite(score)
                        or not 0 <= score <= 1):
                    raise ValueError("decision answer must be a probability in [0, 1]")
                return score
            except Exception:
                time.sleep(1)
        return None
    raise ValueError("unknown backend " + backend)


# ---------------------------------------------------------------- per-video (§5)
def eval_video(backend, model, sample, media_index, key, interval=None):
    path = media_index.get(sample["filename"])
    if path is None or not path.exists():
        return sample, {"error": "missing file"}
    qcore = question_core(sample["question"])
    best, seen, failed = 0.0, 0, 0
    try:
        times = frame_times(probe_duration(path), interval)
    except (OSError, ValueError, subprocess.SubprocessError):
        return sample, {"error": "video duration unavailable"}
    for t in times:
        try:
            uri = frame_uri(path, t)
        except (OSError, subprocess.SubprocessError):
            uri = None
        if not uri:
            failed += 1
            continue
        s = frame_score(backend, model, qcore, uri, key)
        if s is None:
            failed += 1
            continue
        seen += 1
        best = max(best, s)
    counts = {"frames_planned": len(times), "frames_ok": seen,
              "frames_failed": failed, "frame_times": times}
    if failed:
        return sample, {"error": "incomplete frame evidence", **counts}
    pred = "yes" if best >= GATE else "no"
    return sample, {"score": round(best, 4), "pred": pred,
                    "correct": pred == sample["proposed_answer"],
                    **counts}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def frozen_inputs(samples_path, manifest_path, media_dirs):
    """Verify the reviewed answers and exact media before any paid request."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "human_reviewed":
        raise ValueError("the dataset manifest must be human_reviewed")
    if sha256(samples_path) != manifest.get("samples_json_sha256"):
        raise ValueError("samples.json does not match the frozen manifest")
    samples = [s for s in json.loads(samples_path.read_text(encoding="utf-8"))["samples"]
               if s["event"] in SCORED_EVENTS]
    if not samples or len({s["number"] for s in samples}) != len(samples):
        raise ValueError("samples must contain unique scored question numbers")
    frozen = {row["number"]: row for row in manifest["files"] if row["event"] in SCORED_EVENTS}
    if set(frozen) != {s["number"] for s in samples}:
        raise ValueError("scored questions do not match the manifest")
    wanted = {s["filename"] for s in samples}
    media_index = {}
    for directory in media_dirs:
        for path in directory.iterdir():
            if not path.is_file() or path.name not in wanted:
                continue
            if path.name in media_index and path.resolve() != media_index[path.name].resolve():
                raise ValueError("ambiguous media filename: " + path.name)
            media_index[path.name] = path
    hashes = {}
    for sample in samples:
        row = frozen[sample["number"]]
        if (sample["proposed_answer"] not in {"yes", "no"}
                or not question_core(sample["question"])
                or any(sample[key] != row[key] for key in
                       ("filename", "event", "question", "proposed_answer"))):
            raise ValueError("invalid or changed frozen answer")
        filename = sample["filename"]
        if filename not in media_index:
            raise ValueError("missing media file: " + filename)
        if filename not in hashes:
            hashes[filename] = sha256(media_index[filename])
        if hashes[filename] != row["sha256"]:
            raise ValueError("media checksum mismatch: " + filename)
    return samples, media_index, hashes


def streaming_inputs(samples_path, manifest_path, media_dirs, events):
    """Read common inputs without extending legacy dataset eligibility gates."""
    document = json.loads(samples_path.read_text(encoding="utf-8"))
    if not isinstance(document, dict) or not isinstance(document.get("samples"), list):
        raise ValueError("samples JSON must contain a samples list")
    samples = [row for row in document["samples"]
               if isinstance(row, dict) and row.get("event") in events]
    numbers = [row.get("number") for row in samples]
    if (not samples or any(type(number) not in (int, str) for number in numbers)
            or len(set(numbers)) != len(numbers)):
        raise ValueError("selected samples require unique question numbers")
    for row in samples:
        if (row.get("proposed_answer") not in ("yes", "no", "unknown")
                or not isinstance(row.get("filename"), str) or not row["filename"]
                or not isinstance(row.get("question"), str) or not row["question"].strip()):
            raise ValueError("each sample requires a filename, question and yes/no/unknown reference")
    source = {"path": str(manifest_path), "status": None}
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if isinstance(manifest, dict):
            source.update({key: manifest.get(key) for key in ("status", "dataset", "n_questions")})
        else:
            source["read_error"] = "manifest metadata is not an object"
    except (OSError, ValueError) as error:
        source["read_error"] = str(error)
    wanted = {row["filename"] for row in samples}
    media_index, ambiguous, media_warnings = {}, set(), []
    for directory in media_dirs:
        try:
            for path in directory.iterdir():
                if not path.is_file() or path.name not in wanted:
                    continue
                if path.name in media_index and path.resolve() != media_index[path.name].resolve():
                    ambiguous.add(path.name)
                else:
                    media_index[path.name] = path
        except OSError as error:
            media_warnings.append({"directory": str(directory), "error": str(error)})
    for filename in ambiguous:
        media_index.pop(filename, None)
    if ambiguous:
        media_warnings.append({"ambiguous_filenames": sorted(ambiguous)})
    return samples, media_index, source, media_warnings


def streaming_run(args, parser, model, backend):
    """Use one input, window, clock and scoring implementation for every model."""
    from backends import ContrastiveBackend, RemoteBackend
    from realtime import BASE_EVENTS, SIX_EVENTS, run_streams

    events = []
    for event in args.events:
        events.extend(BASE_EVENTS if event == "base4" else SIX_EVENTS if event == "six" else [event])
    events = list(dict.fromkeys(events))
    if args.dense:
        parser.error("--dense belongs to --protocol legacy-v2; streaming uses --interval")
    if not 1 <= args.slots <= 12 or not 1 <= args.window_frames <= 4:
        parser.error("slots must be 1..12 and window-frames must be 1..4")
    if not math.isfinite(args.queue_age) or args.queue_age <= 0:
        parser.error("queue-age must be positive and finite")
    if backend in ("videojev", "decision3") and args.window_frames < 2:
        parser.error("native services require at least two distinct original frames")
    if backend == "decision3" and args.window_frames != 2:
        parser.error("the current Decision 3 runtime accepts exactly two original frames")
    if not args.out:
        parser.error("streaming evaluation requires --out to preserve responses as they arrive")
    if args.out.exists() or args.out.with_name(args.out.stem + "-windows.jsonl").exists():
        parser.error("output exists; choose a new path to preserve completed requests")
    try:
        samples, media_index, manifest_source, media_warnings = streaming_inputs(
            args.samples, args.manifest, args.media, events)
        questions, contrast_texts, question_specs = {}, {}, {}
        for sample in samples:
            label = sample["event"]
            question = re.sub(r"\s*answer yes(?:\s*,\s*no\s*,?\s*or\s*unknown| or no)\.?\s*$", "",
                              sample["question"], flags=re.I).strip()
            if label in questions and questions[label] != question:
                raise ValueError("multiple questions for one event require separate explicit runs: " + label)
            questions[label] = question
        if args.questions:
            extra = json.loads(args.questions.read_text(encoding="utf-8"))
            question_specs = extra
            if not isinstance(extra, dict):
                raise ValueError("--questions must be an event-keyed JSON object")
            for label, spec in extra.items():
                question = spec.get("question") if isinstance(spec, dict) else spec
                if not isinstance(label, str) or not isinstance(question, str) or not question.strip():
                    raise ValueError("every question requires a nonempty label and question string")
                if label in questions and questions[label] != question:
                    raise ValueError("extra questions must not override dataset question semantics: " + label)
                if label in events:
                    questions[label] = question
                    if isinstance(spec, dict) and ("positive" in spec or "contrast" in spec):
                        pair = [spec.get("positive"), spec.get("contrast")]
                        if any(not isinstance(text, str) or not text.strip() for text in pair):
                            raise ValueError("contrastive overrides require positive and contrast strings: " + label)
                        contrast_texts[label] = pair
        missing = set(events) - set(questions)
        defaults = {
            "fight": "Does any fighting occur in this video?",
            "smoking": "Does this video show a person smoking?",
            "fall": "Does a person fall in this video?",
            "weapon": "Does this video show a visible weapon, or a visible object being used to strike, stab, or explicitly threaten a person?",
            "bag_knife": "Is a knife visibly exposed inside the bag or taken out of it in this video?",
            "bag_alcohol_bottle": "Is a bottle identifiable as an alcohol bottle visibly exposed inside the bag or taken out of it in this video? Judge its visible appearance only.",
        }
        for label in missing:
            if label not in defaults:
                raise ValueError("no dataset question or --questions definition for " + label)
            questions[label] = defaults[label]
        questions = {label: questions[label] for label in events}
        thresholds = None
        if args.thresholds:
            thresholds = json.loads(args.thresholds.read_text(encoding="utf-8"))
            if isinstance(thresholds, dict) and model in thresholds:
                thresholds = thresholds[model]
            if not isinstance(thresholds, dict):
                raise ValueError("thresholds must be an event-keyed or model-keyed JSON object")
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.error(str(error))
    if backend == "contrastive":
        runtime = ContrastiveBackend(model, questions, cache_dir=args.cache_dir,
                                     thresholds=thresholds, contrast_texts=contrast_texts)
    else:
        if args.thresholds:
            parser.error("--thresholds applies only to cosine-margin backends, never native choices")
        key = load_key(args.env_file) if backend in ("language", "decision") else ""
        runtime = RemoteBackend(model, backend, questions, key=key,
                                endpoint=args.endpoint, timeout=TIMEOUT)
    out = None
    try:
        out = run_streams(samples, media_index, questions, runtime, args.out,
                          slots=args.slots, workers=args.workers, interval=args.interval,
                          window_frames=args.window_frames, frame_width=FRAME_WIDTH,
                          queue_age=args.queue_age)
    finally:
        try:
            runtime.close()
        except Exception as error:
            if out is None:
                raise
            out["shared_errors"].append("backend_close_failed: " + str(error))
            out["execution_complete"] = False
    out.update(candidate=args.candidate, samples_source=str(args.samples),
               manifest_source=manifest_source, media_directories=[str(path) for path in args.media],
               media_warnings=media_warnings, run_at=datetime.now(timezone.utc).isoformat(),
               runner_commit=subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO,
                                             capture_output=True, text=True).stdout.strip() or None)
    if args.questions:
        out["question_specs"] = question_specs
    args.out.write_text(json.dumps(out, ensure_ascii=False, indent=1, allow_nan=False), encoding="utf-8")
    counts = out["overall"]
    print(f"{args.candidate or model}: {counts['correct']}/{counts['total']}, "
          f"unknown={counts['unknown']}, execution_complete={out['execution_complete']}")
    print(f"-> {args.out}")
    return out


def main():
    from backends import CANDIDATES as STREAM_CANDIDATES

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--protocol", choices=["streaming-v3", "legacy-v2"], default="legacy-v2",
                    help="preserve historical commands; select streaming-v3 for common real-time windows")
    ap.add_argument("--candidate", choices=STREAM_CANDIDATES, help="named model+backend preset")
    ap.add_argument("--model", help="explicit model id (with --backend)")
    ap.add_argument("--backend", choices=["language", "decision", "videojev", "decision3", "contrastive"],
                    help="explicit backend class")
    ap.add_argument("--samples", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    ap.add_argument("--media", type=Path, nargs="+", required=True,
                    help="one or more dirs holding the clips; searched by filename "
                         "(the dataset may be split across folders)")
    ap.add_argument("--env-file", type=Path, default=REPO / ".env")
    ap.add_argument("--workers", type=int, default=12,
                    help="bounded inference parallelism; input slots are independent")
    ap.add_argument("--slots", type=int, default=12, help="simultaneous original-speed video inputs")
    ap.add_argument("--window-frames", type=int, default=2,
                    help="distinct sampled original frames in each sliding window; default 2")
    ap.add_argument("--queue-age", type=float, default=2.0,
                    help="drop an unsent window after this many queued seconds")
    ap.add_argument("--events", nargs="+", default=["six"],
                    help="base4, six, or explicit event labels; common across every model")
    ap.add_argument("--questions", type=Path,
                    help="event-keyed JSON: question strings or {question, positive, contrast} specs")
    ap.add_argument("--endpoint", help="API endpoint; native services use http://host:port/predict")
    ap.add_argument("--cache-dir", type=Path, default=Path.home() / ".cache/huggingface/hub")
    ap.add_argument("--thresholds", type=Path,
                    help="pre-frozen cosine margins; omitted means explicit untuned zero baseline")
    ap.add_argument("--interval", type=float, default=1.0,
                    help="seconds between sampled frames; default 1.0 (= 1 fps, the "
                         "frozen spec cadence). e.g. 4 = 0.25 fps product cadence.")
    ap.add_argument("--dense", action="store_true",
                    help="ignore --interval; use the capped dense rule "
                         "(max 14 uniform frames) instead of 1 fps")
    ap.add_argument("--checksums", action="store_true", help="legacy-v2 compatibility flag; unused in streaming-v3")
    ap.add_argument("--out", type=Path, help="write full result JSON here")
    a = ap.parse_args()
    if a.workers < 1 or not math.isfinite(a.interval) or a.interval <= 0:
        ap.error("workers and interval must be positive; interval must be finite")
    if a.candidate and (a.model or a.backend):
        ap.error("use --candidate or --model with --backend, not both")

    if a.candidate:
        model, backend = STREAM_CANDIDATES[a.candidate]
    elif a.model and a.backend:
        model, backend = a.model, a.backend
    else:
        ap.error("give --candidate, or both --model and --backend")

    if a.protocol == "streaming-v3":
        return streaming_run(a, ap, model, backend)
    if backend not in ("language", "decision"):
        ap.error("legacy-v2 supports only language and noul decision backends")
    interval = None if a.dense else a.interval  # None -> dense rule
    try:
        samples, media_index, hashes = frozen_inputs(a.samples, a.manifest, a.media)
    except (OSError, ValueError, KeyError, TypeError) as error:
        ap.error(str(error))
    key = load_key(a.env_file)

    results = []
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        futs = [ex.submit(eval_video, backend, model, s, media_index, key, interval) for s in samples]
        for f in as_completed(futs):
            results.append(f.result())

    # ---- score (§7)
    cnt = defaultdict(lambda: {"TP": 0, "FN": 0, "TN": 0, "FP": 0, "err": 0})
    oc = ot = 0
    for s, r in results:
        ev = s["event"]
        if "error" in r:
            cnt[ev]["err"] += 1
            continue
        gt, pred = s["proposed_answer"], r["pred"]
        key_ = ("TP" if pred == "yes" else "FN") if gt == "yes" else ("FP" if pred == "yes" else "TN")
        cnt[ev][key_] += 1
        oc += r["correct"]; ot += 1

    samp = "dense(≤14)" if interval is None else f"{interval}s (~{1/interval:.3g}fps)"
    print(f"\n=== {a.candidate or model}  (backend={backend}, gate={GATE}, sampling={samp}) ===")
    print(f"{'event':8s} {'acc':>7s} {'TP':>4s} {'FN':>4s} {'TN':>4s} {'FP':>4s} {'err':>4s}  specificity")
    for ev in SCORED_EVENTS:
        c = cnt[ev]; n = c["TP"] + c["FN"] + c["TN"] + c["FP"]
        acc = f"{c['TP']+c['TN']}/{n}" if n else "0/0"
        spec = f"{c['TN']}/{c['TN']+c['FP']}" if (c["TN"] + c["FP"]) else "—(无负例)"
        print(f"{ev:8s} {acc:>7s} {c['TP']:>4d} {c['FN']:>4d} {c['TN']:>4d} {c['FP']:>4d} {c['err']:>4d}  {spec}")
    print(f"OVERALL {oc}/{ot} = {100*oc/ot:.1f}%" if ot else "OVERALL n/a")

    out = {
        "spec_version": SPEC_VERSION,
        "samples_json_sha256": sha256(a.samples),
        "manifest_sha256": sha256(a.manifest),
        "runner_sha256": sha256(Path(__file__)),
        "spec_sha256": sha256(Path(__file__).with_name("SPEC.md")),
        "run_at": datetime.now(timezone.utc).isoformat(),
        "candidate": a.candidate, "model": model, "backend": backend,
        "config": {"gate": GATE, "sampling": samp, "interval": interval,
                   "frame_rule": (f"n=max(3,min({FRAME_MAX},int(dur))); t=dur*(i+0.5)/n"
                                  if interval is None else "n=max(1,round(dur/interval)); t=dur*(i+0.5)/n"),
                   "workers": a.workers,
                   "frame_width": FRAME_WIDTH, "temperature": TEMPERATURE,
                   "retries": RETRIES, "timeout": TIMEOUT, "decision": "max-frame-score >= gate"},
        "overall": {"correct": oc, "total": ot, "attempted": len(samples), "errors": len(samples) - ot},
        "per_event": {ev: dict(cnt[ev]) for ev in SCORED_EVENTS},
        "per_video": [{"number": s["number"], "filename": s["filename"], "event": s["event"],
                       "gt": s["proposed_answer"],
                       "sha256": hashes[s["filename"]],
                       **r}
                      for s, r in sorted(results, key=lambda x: x[0]["number"])],
    }
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(out, ensure_ascii=False, indent=1, allow_nan=False), encoding="utf-8")
        print(f"-> {a.out}")
    return out


if __name__ == "__main__":
    result = main()
    if result.get("spec_version") == "streaming-v3" and not result.get("execution_complete"):
        sys.exit(1)
