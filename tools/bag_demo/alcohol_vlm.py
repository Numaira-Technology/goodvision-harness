"""Alcohol-bottle detection for the 翻包 demo via a hosted vision VLM (Qwen3.5-VL).

VideoJev has no alcohol concept, so the alcohol axis is delegated to a label-reading VLM
over an OpenAI-compatible chat-completions endpoint (OpenRouter / DashScope — keys from
.env, no GPU). Whole-frame rather than a YOLO crop: the COCO detector misses the bottle in
the one true-positive clip, so gating on it would just reproduce the miss. We sample frames
across the clip, ask per frame, and take the max (any frame positive -> clip positive).

Demo-only; not the frozen benchmark.
"""

import base64
import json
import os
import re

import httpx

PROMPT = (
    "You are inspecting a frame from a bag-search video. Question: is a bottle of an "
    "ALCOHOLIC drink (liquor, spirits, wine or beer -- typically a glass bottle with a "
    "brand label) visibly present, whether inside the bag or taken out of it? A plain "
    "water bottle, a soft-drink or plastic bottle, a flask, thermos or cup does NOT count. "
    "Judge only by what is visible. Reply with ONLY compact JSON: "
    '{"answer": true, "conf": 0.0-1.0} or {"answer": false, "conf": 0.0-1.0}.'
)
FRAME_WIDTH = 768  # higher than VideoJev's 720 so labels are a bit more legible


def _provider():
    """(endpoint, api_key, model) from env; prefer OpenRouter, else DashScope."""
    if os.environ.get("OPENROUTER_API_KEY"):
        return (
            os.environ.get("OPENROUTER_ENDPOINT", "https://openrouter.ai/api/v1/chat/completions"),
            os.environ["OPENROUTER_API_KEY"],
            os.environ.get("ALCOHOL_VLM_MODEL", "qwen/qwen3.5-35b-a3b"),
        )
    if os.environ.get("DASHSCOPE_API_KEY"):
        return (
            "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
            os.environ["DASHSCOPE_API_KEY"],
            os.environ.get("ALCOHOL_VLM_MODEL", "qwen-vl-max"),
        )
    raise RuntimeError("Set OPENROUTER_API_KEY or DASHSCOPE_API_KEY in .env")


def _frame_data_uri(video_path, t):
    import subprocess

    proc = subprocess.run(
        ["ffmpeg", "-nostdin", "-ss", f"{t:.3f}", "-i", str(video_path),
         "-vframes", "1", "-vf", f"scale={FRAME_WIDTH}:-2", "-f", "mjpeg", "pipe:1"],
        capture_output=True, check=True,
    )
    return "data:image/jpeg;base64," + base64.b64encode(proc.stdout).decode("ascii")


def ask_frame(data_uri, *, client=None, endpoint=None, api_key=None, model=None):
    """Ask the VLM about one frame; return (answer_bool, confidence_float)."""
    if endpoint is None:
        endpoint, api_key, model = _provider()
    body = {
        "model": model,
        "temperature": 0,
        "max_tokens": 60,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": PROMPT},
                {"type": "image_url", "image_url": {"url": data_uri}},
            ],
        }],
    }
    owns = client is None
    client = client or httpx.Client(timeout=40.0)
    try:
        r = client.post(endpoint, json=body,
                        headers={"Authorization": f"Bearer {api_key}"})
        r.raise_for_status()
        text = r.json()["choices"][0]["message"].get("content") or ""
    finally:
        if owns:
            client.close()
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return False, 0.0
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError:
        return False, 0.0
    return bool(data.get("answer")), float(data.get("conf") or 0.0)


def detect_alcohol(video_path, times, *, conf_gate=0.5, client=None):
    """Return {present, conf, per_frame} by max-pooling the VLM over sampled frames."""
    endpoint, api_key, model = _provider()
    owns = client is None
    client = client or httpx.Client(timeout=40.0)
    per_frame = []
    present, best = False, 0.0
    try:
        for t in times:
            uri = _frame_data_uri(video_path, t)
            ans, conf = ask_frame(uri, client=client, endpoint=endpoint, api_key=api_key, model=model)
            per_frame.append({"t": round(t, 2), "answer": ans, "conf": conf})
            if ans and conf >= conf_gate:
                present = True
            best = max(best, conf if ans else 0.0)
    finally:
        if owns:
            client.close()
    return {"present": present, "conf": round(best, 3), "model": model, "per_frame": per_frame}
