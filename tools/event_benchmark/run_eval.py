#!/usr/bin/env python3
"""Canonical event-recognition benchmark runner (see SPEC.md).

A fixed exam: every knob below is frozen and identical for all candidates. The
ONLY thing that varies between runs is the model (--candidate, or --model +
--backend). Do not add per-model prompt/threshold/sampling branches here.
"""
import argparse, base64, json, re, subprocess, sys, time, hashlib, urllib.request
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
DEFAULT_SAMPLES = Path("/Users/ashley/Downloads/media50/samples.json")
DEFAULT_MEDIA = Path("/Users/ashley/Downloads/media50/media/media")


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
         "-of", "csv=p=0", str(path)], capture_output=True, text=True).stdout.strip()
    try:
        return float(out)
    except ValueError:
        return 0.0


def frame_times(dur, interval=None):
    """Dense mode (default): n = max(3, min(14, int(dur))) uniform frames.
    Interval mode (--interval S): one frame per S seconds, like the product's
    PREVIEW_SELECT_INTERVAL — short clips get as few as 1 frame (§4)."""
    if not dur:
        return [0.5] if interval else [0.5, 1.0, 1.5]
    if interval:
        n = max(1, round(dur / interval))
    else:
        n = max(FRAME_MIN, min(FRAME_MAX, int(dur)))
    return [dur * (i + 0.5) / n for i in range(n)]


def frame_uri(path, t):
    raw = subprocess.run(
        ["ffmpeg", "-nostdin", "-loglevel", "error", "-ss", str(t), "-i", str(path),
         "-vframes", "1", "-vf", f"scale={FRAME_WIDTH}:-2", "-f", "image2pipe",
         "-vcodec", "mjpeg", "pipe:1"], capture_output=True).stdout
    return "data:image/jpeg;base64," + base64.b64encode(raw).decode() if raw else None


# ---------------------------------------------------------------- backends (§3)
def _post(url, body, key):
    req = urllib.request.Request(
        url, data=json.dumps(body).encode(),
        headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=TIMEOUT).read())


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
                c = _post(CHAT_URL, body, key)["choices"][0]["message"]["content"]
                m = re.search(r"\{.*\}", c, re.DOTALL)
                return 1.0 if (m and bool(json.loads(m.group(0)).get("answer"))) else 0.0
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
                return float(_post(LUNA_URL, body, key)["answers"]["answer"]["noul"])
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
    for t in frame_times(probe_duration(path), interval):
        uri = frame_uri(path, t)
        if not uri:
            continue
        s = frame_score(backend, model, qcore, uri, key)
        if s is None:
            failed += 1
            continue
        seen += 1
        best = max(best, s)
    if seen == 0:
        return sample, {"error": "all frames failed (%d)" % failed}
    pred = "yes" if best >= GATE else "no"
    return sample, {"score": round(best, 4), "pred": pred,
                    "correct": pred == sample["proposed_answer"],
                    "frames_ok": seen, "frames_failed": failed}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description="Frozen clip-level event benchmark (see SPEC.md)")
    ap.add_argument("--candidate", choices=CANDIDATES, help="named model+backend preset")
    ap.add_argument("--model", help="explicit model id (with --backend)")
    ap.add_argument("--backend", choices=["language", "decision"], help="explicit backend class")
    ap.add_argument("--samples", type=Path, default=DEFAULT_SAMPLES)
    ap.add_argument("--media", type=Path, nargs="+", default=[DEFAULT_MEDIA],
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
                    help="ignore --interval; use the uncapped-ceiling dense rule "
                         "(max 14 uniform frames) instead of 1 fps")
    ap.add_argument("--checksums", action="store_true", help="also record per-file SHA-256 (§8)")
    ap.add_argument("--out", type=Path, help="write full result JSON here")
    a = ap.parse_args()

    if a.candidate:
        model, backend = CANDIDATES[a.candidate]
    elif a.model and a.backend:
        model, backend = a.model, a.backend
    else:
        ap.error("give --candidate, or both --model and --backend")

    key = load_key(a.env_file)
    interval = None if a.dense else a.interval  # None -> dense rule
    samples = [s for s in json.load(open(a.samples))["samples"]
               if s["event"] in SCORED_EVENTS]

    # index clips by filename across all --media dirs (dataset may be split)
    media_index = {}
    for d in a.media:
        for f in d.iterdir() if d.is_dir() else []:
            if f.is_file():
                media_index.setdefault(f.name, f)

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
        "spec_version": "v1",
        "run_at": datetime.now(timezone.utc).isoformat(),
        "candidate": a.candidate, "model": model, "backend": backend,
        "config": {"gate": GATE, "sampling": samp, "interval": interval,
                   "frame_rule": f"max(3,min({FRAME_MAX},int(dur))) @ (i+0.5)/n",
                   "frame_width": FRAME_WIDTH, "temperature": TEMPERATURE,
                   "retries": RETRIES, "timeout": TIMEOUT, "decision": "max-frame-score >= gate"},
        "overall": {"correct": oc, "total": ot},
        "per_event": {ev: dict(cnt[ev]) for ev in SCORED_EVENTS},
        "per_video": [{"number": s["number"], "filename": s["filename"], "event": s["event"],
                       "gt": s["proposed_answer"],
                       **({"sha256": sha256(media_index[s["filename"]])}
                          if a.checksums and s["filename"] in media_index else {}),
                       **r}
                      for s, r in sorted(results, key=lambda x: x[0]["number"])],
    }
    if a.out:
        json.dump(out, open(a.out, "w"), ensure_ascii=False, indent=1)
        print(f"-> {a.out}")
    return out


if __name__ == "__main__":
    main()
