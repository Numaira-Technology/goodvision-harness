#!/usr/bin/env python3
"""Latency harness for the event-recognition benchmark.

Fills the two speed columns of the report format: single inference speed, and
P50 / P95 under 12-way concurrency. This is deliberately SEPARATE from accuracy
(run_eval.py) — it measures the wall-clock of one predicate on one 720px frame,
client -> OpenRouter -> model -> back (network + queue included). It is NOT the
full product pipeline (no YOLO, no track binding, no chips).

Keep the measured unit identical across models so the numbers are comparable:
one frame, one yes/no predicate, same image, same question.

Usage:
  python latency_bench.py --candidate luna   --frame some.jpg
  python latency_bench.py --candidate qwen   --video clip.mp4   # auto-grabs a frame
  python latency_bench.py --candidate gemini --video clip.mp4 --concurrency 12 --rounds 6
"""
import argparse, base64, json, os, statistics, subprocess, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor

CHAT_URL = "https://openrouter.ai/api/v1/chat/completions"
LUNA_URL = "https://openrouter.ai/api/alpha/decisions"
FRAME_WIDTH = 720
TIMEOUT = 90
# same registry as run_eval.py: name -> (model id, backend class)
CANDIDATES = {
    "qwen":   ("qwen/qwen3-vl-8b-instruct", "language"),
    "gemini": ("google/gemini-2.5-flash",   "language"),
    "luna":   ("openai/gpt-6-luna-decisions", "decision"),
    # flagship additions (speed-harness config only; accuracy exam unchanged)
    "qwen235b": ("qwen/qwen3-vl-235b-a22b-instruct", "language"),
    "llama4":   ("meta-llama/llama-4-maverick", "language"),
    "gemini_pro": ("google/gemini-3.1-pro-preview", "language"),
    "gpt":      ("openai/gpt-5.4", "language"),
    "claude":   ("anthropic/claude-opus-5.5", "language"),
}
# one fixed predicate, shared by every model (semantics identical, §1 of SPEC)
QCORE = "Does any fighting occur in this image?"


def load_key(env_file):
    if os.environ.get("OPENROUTER_API_KEY"):
        return os.environ["OPENROUTER_API_KEY"]
    try:
        for line in open(env_file):
            if line.startswith("OPENROUTER_API_KEY="):
                return line.split("=", 1)[1].strip()
    except OSError:
        pass
    sys.exit("OPENROUTER_API_KEY not found (env or %s)" % env_file)


def frame_from_video(path):
    raw = subprocess.run(
        ["ffmpeg", "-nostdin", "-loglevel", "error", "-ss", "0.5", "-i", str(path),
         "-vframes", "1", "-vf", f"scale={FRAME_WIDTH}:-2", "-f", "image2pipe",
         "-vcodec", "mjpeg", "pipe:1"], capture_output=True).stdout
    if not raw:
        sys.exit("could not extract a frame from %s" % path)
    return raw


def frame_uri(jpeg_bytes):
    return "data:image/jpeg;base64," + base64.b64encode(jpeg_bytes).decode()


def build_body(backend, model, uri):
    if backend == "language":
        prompt = (f'{QCORE} Answer only with compact JSON: '
                  '{"answer": true} or {"answer": false}.')
        return CHAT_URL, {"model": model, "messages": [{"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": uri}}]}],
            "max_tokens": 30, "temperature": 0}
    if backend == "decision":
        return LUNA_URL, {"model": model,
            "state": ["Surveillance video frame.",
                      {"type": "image_url", "image_url": {"url": uri}}],
            "questions": {"answer": {"type": "noul", "instructions": QCORE,
                "criteria": {"true": "Yes — " + QCORE, "false": "No — " + QCORE}}}}
    sys.exit("unknown backend " + backend)


def one_call(url, body, key):
    """Return wall-clock seconds for a single request, or None on failure."""
    req = urllib.request.Request(
        url, data=json.dumps(body).encode(),
        headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"})
    t0 = time.perf_counter()
    try:
        urllib.request.urlopen(req, timeout=TIMEOUT).read()
        return time.perf_counter() - t0
    except Exception:
        return None


def pct(xs, p):
    xs = sorted(xs)
    if not xs:
        return float("nan")
    k = (len(xs) - 1) * p / 100
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def main():
    ap = argparse.ArgumentParser(description="Latency harness (speed columns of the benchmark)")
    ap.add_argument("--candidate", choices=CANDIDATES, required=True)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--frame", help="a 720px JPEG to send")
    g.add_argument("--video", help="a clip to auto-grab one frame from")
    ap.add_argument("--single-n", type=int, default=20, help="sequential calls for single-inference speed")
    ap.add_argument("--concurrency", type=int, default=12, help="parallel calls (match the 12-stream product)")
    ap.add_argument("--rounds", type=int, default=6, help="concurrency rounds; total = concurrency*rounds")
    ap.add_argument("--env-file", default=os.path.expanduser("~/Desktop/goodvision-harness/.env"))
    ap.add_argument("--out", help="write raw latencies + summary JSON here")
    a = ap.parse_args()

    key = load_key(a.env_file)
    model, backend = CANDIDATES[a.candidate]
    jpeg = open(a.frame, "rb").read() if a.frame else frame_from_video(a.video)
    url, body = build_body(backend, model, frame_uri(jpeg))

    # 1) single-inference speed: sequential, no overlap
    single = [t for _ in range(a.single_n) if (t := one_call(url, body, key)) is not None]

    # 2) concurrency: <concurrency> at a time, <rounds> rounds
    conc = []
    for _ in range(a.rounds):
        with ThreadPoolExecutor(max_workers=a.concurrency) as ex:
            conc += [t for t in ex.map(lambda _: one_call(url, body, key),
                                       range(a.concurrency)) if t is not None]

    summ = {
        "candidate": a.candidate, "model": model, "backend": backend,
        "single_inference_s": round(statistics.median(single), 3) if single else None,
        "single_mean_s": round(statistics.mean(single), 3) if single else None,
        "single_n_ok": len(single), "single_n": a.single_n,
        "concurrency": a.concurrency, "rounds": a.rounds,
        "conc_n_ok": len(conc), "conc_n": a.concurrency * a.rounds,
        "p50_s": round(pct(conc, 50), 3) if conc else None,
        "p95_s": round(pct(conc, 95), 3) if conc else None,
    }
    print(f"\n=== {a.candidate} ({backend}) latency ===")
    print(f"single inference (median of {summ['single_n_ok']}/{a.single_n}): {summ['single_inference_s']}s "
          f"(mean {summ['single_mean_s']}s)")
    print(f"{a.concurrency}-way concurrency (n={summ['conc_n_ok']}): "
          f"P50 {summ['p50_s']}s  P95 {summ['p95_s']}s")
    if a.out:
        json.dump({**summ, "single_raw": single, "conc_raw": conc},
                  open(a.out, "w"), indent=1)
        print(f"-> {a.out}")


if __name__ == "__main__":
    main()
