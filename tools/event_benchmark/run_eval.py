#!/usr/bin/env python3
"""Canonical event-recognition benchmark runner (see SPEC.md).

A fixed exam: every knob below is frozen and identical for all candidates. The
ONLY thing that varies between runs is the model (--candidate, or --model +
--backend). Do not add per-model prompt/threshold/sampling branches here.
"""
import argparse
import base64
import hashlib
import importlib.util
import json
import math
import os
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
# Load the sibling by path so standalone scripts (including python -I) use the
# same evidence API without depending on the caller's working directory.
_evidence_spec = importlib.util.spec_from_file_location(
    "event_benchmark_evidence", Path(__file__).with_name("evidence.py"))
evidence = importlib.util.module_from_spec(_evidence_spec)
_evidence_spec.loader.exec_module(evidence)
DEFAULT_MANIFEST = Path(__file__).parent / "datasets/manifest_50.json"
SPEC_VERSION = "v2.1"


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


def build_request(backend, model, qcore, uri, provider_context=None):
    """Preserve the frozen envelopes; routing is a separately recorded variant."""
    if backend == "language":
        prompt = (f'{qcore} Answer only with compact JSON: '
                  '{"answer": true} or {"answer": false}.')
        body = {"model": model, "messages": [{"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": uri}}]}],
            "max_tokens": CHAT_MAX_TOKENS, "temperature": TEMPERATURE}
        # Never redirect the selected cloud credential through an ambient URL.
        url = CHAT_URL
    elif backend == "decision":
        body = {"model": model,
                "state": ["Surveillance video frame.",
                          {"type": "image_url", "image_url": {"url": uri}}],
                "questions": {"answer": {"type": "noul", "instructions": qcore,
                    "criteria": {"true": "Yes — " + qcore, "false": "No — " + qcore}}}}
        url = LUNA_URL
    else:
        raise ValueError("unknown backend " + backend)
    if provider_context:
        provider = evidence.provider_body(provider_context)
        if provider is not None:
            body["provider"] = provider
    return url, body


def parse_answer(payload, backend):
    if not isinstance(payload, dict) or payload.get("error") is not None:
        raise ValueError("invalid provider response")
    if backend == "language":
        choice = payload["choices"][0]
        if choice.get("finish_reason") != "stop" or choice.get("error") is not None:
            raise ValueError("incomplete language answer")
        m = re.search(r"\{.*\}", choice["message"]["content"], re.DOTALL)
        answer = json.loads(m.group(0), object_pairs_hook=_unique) if m else None
        if (not isinstance(answer, dict) or set(answer) != {"answer"}
                or type(answer["answer"]) is not bool):
            raise ValueError("language answer must be a JSON boolean")
        return float(answer["answer"])
    answers = payload.get("answers")
    if not isinstance(answers, dict) or not isinstance(answers.get("answer"), dict):
        raise ValueError("invalid decision answer envelope")
    answer = answers["answer"]
    score = answer.get("noul")
    if (answer.get("type") != "noul" or answer.get("error") is not None
            or not evidence.finite_number(score)
            or not 0 <= score <= 1):
        raise ValueError("decision answer must be a probability in [0, 1]")
    return score


def frame_score(backend, model, qcore, uri, key, provider_context=None, attempts=None):
    """Same retry/scoring policy; retain received scores and all attempt metadata."""
    url, body = build_request(backend, model, qcore, uri, provider_context)
    request_hash = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()
    for index in range(RETRIES):
        record = {"attempt": index + 1, "request_sha256": request_hash, "ok": False}
        started = time.perf_counter()
        try:
            payload = _post(url, body, key)
            record.update(evidence.response_metadata(payload))
            score = parse_answer(payload, backend)
            record.update(ok=True, score=score)
            return score
        except Exception as error:
            record["error"] = ("http_" + str(error.code) if hasattr(error, "code")
                               else type(error).__name__)
        finally:
            record["elapsed_s"] = time.perf_counter() - started
            if attempts is not None:
                attempts.append(record)
        time.sleep(1)
    return None


# ---------------------------------------------------------------- per-video (§5)
def eval_video(backend, model, sample, media_index, key, interval=None, provider_context=None):
    path = media_index.get(sample["filename"])
    if path is None or not path.exists():
        return sample, {"error": "missing file"}
    qcore = question_core(sample["question"])
    best, seen, failed = 0.0, 0, 0
    try:
        times = frame_times(probe_duration(path), interval)
    except (OSError, ValueError, subprocess.SubprocessError):
        return sample, {"error": "video duration unavailable"}
    frames = []
    for t in times:
        frame = {"time_s": t, "attempts": []}
        frames.append(frame)
        try:
            uri = frame_uri(path, t)
        except (OSError, subprocess.SubprocessError):
            uri = None
        if not uri:
            frame["error"] = "decode_failed"
            failed += 1
            continue
        s = frame_score(backend, model, qcore, uri, key, provider_context, frame["attempts"])
        if s is None:
            frame["error"] = "provider_answer_failed"
            failed += 1
            continue
        seen += 1
        frame["score"] = s
        best = max(best, s)
    counts = {"frames_planned": len(times), "frames_ok": seen,
              "frames_failed": failed, "frame_times": times, "frames": frames}
    if failed:
        return sample, {"error": "incomplete frame evidence", **counts}
    pred = "yes" if best >= GATE else "no"
    return sample, {"score": best, "score_display": round(best, 4), "pred": pred,
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


def main():
    ap = argparse.ArgumentParser(description="Frozen clip-level event benchmark (see SPEC.md)")
    ap.add_argument("--candidate", choices=CANDIDATES, help="named model+backend preset")
    ap.add_argument("--model", help="explicit model id (with --backend)")
    ap.add_argument("--backend", choices=["language", "decision"], help="explicit backend class")
    ap.add_argument("--samples", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    ap.add_argument("--media", type=Path, nargs="+", required=True,
                    help="one or more dirs holding the clips; searched by filename "
                         "(the dataset may be split across folders)")
    ap.add_argument("--env-file", type=Path, default=REPO / ".env")
    ap.add_argument("--workers", type=int, default=12,
                    help="parallelism, default 12 to align with the 12-stream product "
                         "(does NOT affect scoring)")
    ap.add_argument("--interval", type=float, default=1.0,
                    help="seconds between sampled frames; default 1.0 (= 1 fps, the "
                         "frozen spec cadence). e.g. 4 = 0.25 fps product cadence.")
    ap.add_argument("--dense", action="store_true",
                    help="ignore --interval; use the capped dense rule "
                         "(max 14 uniform frames) instead of 1 fps")
    ap.add_argument("--checksums", action="store_true", help="compatibility flag; checksums are always verified")
    ap.add_argument("--provider-config", type=Path, help="frozen single-provider configuration")
    ap.add_argument("--split-manifest", type=Path, help="frozen source-group sidecar")
    ap.add_argument("--split", choices=["validation", "test"])
    ap.add_argument("--legacy-baseline", action="store_true",
                    help="explicit v2.1 baseline: default provider routing, no validation/test sidecar; not eligible for controlled tuning")
    ap.add_argument("--preflight", action="store_true", help="offline input checks only; no credentials or HTTP")
    ap.add_argument("--out", type=Path, help="write full result JSON here")
    a = ap.parse_args()
    if a.legacy_baseline and (a.provider_config or a.split_manifest or a.split):
        ap.error("--legacy-baseline cannot be combined with controlled provider/split options")
    variant = evidence.LEGACY_VARIANT if a.legacy_baseline else evidence.VARIANT
    if a.workers < 1 or not math.isfinite(a.interval) or a.interval <= 0:
        ap.error("workers and interval must be positive; interval must be finite")
    if a.candidate and (a.model or a.backend):
        ap.error("use --candidate or --model with --backend, not both")

    if a.candidate:
        model, backend = CANDIDATES[a.candidate]
    elif a.model and a.backend:
        model, backend = a.model, a.backend
    else:
        ap.error("give --candidate, or both --model and --backend")

    interval = None if a.dense else a.interval  # None -> dense rule
    try:
        if a.out and a.out.exists():
            raise ValueError("output already exists; choose a new path")
        samples, media_index, hashes = frozen_inputs(a.samples, a.manifest, a.media)
        if bool(a.split_manifest) != bool(a.split):
            raise ValueError("--split-manifest and --split must be supplied together")
        groups, split_context = ({}, None)
        if a.split_manifest:
            groups, split_context = evidence.split_groups(a.split_manifest, a.manifest, a.split)
        provider_context = (evidence.load_provider(a.provider_config, model, backend)
                            if a.provider_config else None)
        provenance = {"samples_json_sha256": sha256(a.samples), "manifest_sha256": sha256(a.manifest),
                      "runner_sha256": sha256(Path(__file__)), "spec_sha256": sha256(Path(__file__).with_name("SPEC.md")),
                      "evidence_sha256": sha256(Path(evidence.__file__)), **evidence.code_provenance()}
    except (OSError, ValueError, KeyError, TypeError) as error:
        ap.error(str(error))
    if a.preflight:
        report = {"artifact_type": "offline-preflight", "model": model, "backend": backend,
                  **provenance,
                  "protocol_variant": variant,
                  "preflight_at_utc": datetime.now(timezone.utc).isoformat(),
                  "questions": len(samples), "videos": len(hashes),
                  "samples_json_sha256": sha256(a.samples), "manifest_sha256": sha256(a.manifest),
                  "runner_sha256": sha256(Path(__file__)), "evidence_sha256": sha256(Path(evidence.__file__)),
                  "split_context": split_context, "provider_context": provider_context,
                  "interval": interval, "workers": a.workers, "inference_calls": 0,
                  "planned_config": {"frame_width": FRAME_WIDTH, "gate": GATE,
                                     "temperature": TEMPERATURE, "chat_max_tokens": CHAT_MAX_TOKENS,
                                     "retries": RETRIES, "timeout": TIMEOUT,
                                     "decision": "max-frame-score >= gate"},
                  "data_checks_passed": True,
                  "pending": ["paid inference authorization", "live provider receipt verification"],
                  "runtime_note": "workers are not proof of twelve-camera end-to-end throughput"}
        if not provider_context and not a.legacy_baseline:
            report["pending"].append("frozen provider configuration")
        if not split_context and not a.legacy_baseline:
            report["pending"].append("frozen split/source-group mapping")
        if a.out:
            evidence.write_new(a.out, report)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return report
    if not a.legacy_baseline and (not provider_context or not split_context):
        ap.error("controlled inference requires --provider-config, --split-manifest and --split")
    key = load_key(a.env_file)
    run_at = datetime.now(timezone.utc).isoformat()

    results = []
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        futs = [ex.submit(eval_video, backend, model, s, media_index, key, interval, provider_context) for s in samples]
        for f in as_completed(futs):
            results.append(f.result())
    attempts = [a for _, r in results for frame in r.get("frames", []) for a in frame["attempts"]]
    evidence.enrich_attempts(attempts, key)

    # ---- score (§7)
    cnt = defaultdict(lambda: {"TP": 0, "FN": 0, "TN": 0, "FP": 0, "err": 0, "total": 0})
    for s in samples:
        cnt[s["event"]]["total"] += 1
    oc, ot = 0, len(samples)
    for s, r in results:
        ev = s["event"]
        if "error" in r:
            cnt[ev]["err"] += 1
            continue
        gt, pred = s["proposed_answer"], r["pred"]
        key_ = ("TP" if pred == "yes" else "FN") if gt == "yes" else ("FP" if pred == "yes" else "TN")
        cnt[ev][key_] += 1
        oc += r["correct"]

    samp = "dense(≤14)" if interval is None else f"{interval}s (~{1/interval:.3g}fps)"
    print(f"\n=== {a.candidate or model}  (backend={backend}, gate={GATE}, sampling={samp}) ===")
    print(f"{'event':8s} {'acc':>7s} {'TP':>4s} {'FN':>4s} {'TN':>4s} {'FP':>4s} {'err':>4s}  specificity")
    for ev in SCORED_EVENTS:
        c = cnt[ev]; n = c["total"]
        acc = f"{c['TP']+c['TN']}/{n}" if n else "0/0"
        spec = f"{c['TN']}/{c['TN']+c['FP']}" if (c["TN"] + c["FP"]) else "—(无负例)"
        print(f"{ev:8s} {acc:>7s} {c['TP']:>4d} {c['FN']:>4d} {c['TN']:>4d} {c['FP']:>4d} {c['err']:>4d}  {spec}")
    print(f"OVERALL {oc}/{ot} = {100*oc/ot:.1f}%" if ot else "OVERALL n/a")

    out = {
        **provenance,
        "spec_version": SPEC_VERSION,
        "protocol_variant": variant,
        "score_precision": "lossless",
        "score_precision_scope": "Received JSON numeric value stored as Python float, without extra rounding; server rounding is unknown.",
        "provider_context": provider_context,
        "provider_observed": evidence.provider_summary(attempts, provider_context),
        "split_context": split_context,
        "run_at": run_at,
        "candidate": a.candidate, "model": model, "backend": backend,
        "config": {"gate": GATE, "sampling": samp, "interval": interval,
                   "frame_rule": (f"n=max(3,min({FRAME_MAX},int(dur))); t=dur*(i+0.5)/n"
                                  if interval is None else "n=max(1,round(dur/interval)); t=dur*(i+0.5)/n"),
                   "workers": a.workers,
                   "frame_width": FRAME_WIDTH, "temperature": TEMPERATURE,
                   "chat_max_tokens": CHAT_MAX_TOKENS,
                   "retries": RETRIES, "timeout": TIMEOUT, "decision": "max-frame-score >= gate"},
        "overall": {"correct": oc, "total": ot, "attempted": ot,
                    "errors": sum(c["err"] for c in cnt.values())},
        "per_event": {ev: dict(cnt[ev]) for ev in SCORED_EVENTS},
        "per_video": [{"number": s["number"], "filename": s["filename"], "event": s["event"],
                       "gt": s["proposed_answer"],
                       "sha256": hashes[s["filename"]],
                       "source_video_group": groups.get(s["number"]),
                       **r}
                      for s, r in sorted(results, key=lambda x: x[0]["number"])],
    }
    if a.out:
        evidence.write_new(a.out, out)
        print(f"-> {a.out}")
    return out


if __name__ == "__main__":
    main()
