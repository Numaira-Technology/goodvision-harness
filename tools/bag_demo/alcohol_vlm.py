"""Alcohol-bottle detection for the 翻包 demo via a hosted vision VLM (Qwen3.5-VL).

VideoJev has no alcohol concept, so the alcohol axis is delegated to a label-reading VLM
over an OpenAI-compatible endpoint (OpenRouter / DashScope; keys from exported
environment variables). Whole frames are used because the author reported a COCO
miss in the one positive smoke clip. Any valid above-gate positive makes the sampled
clip positive; failures and low scores remain explicitly unknown. This helper is
not wired into the demo server.

Demo-only; not the frozen benchmark.
"""

import base64
import json
import math
import os
from urllib.parse import urlsplit

import httpx

from tools.bag_demo.media import frame_jpeg, validate_times

PROMPT = (
    "You are inspecting a frame from a bag-search video. Question: is a bottle of an "
    "ALCOHOLIC drink (liquor, spirits, wine or beer -- typically a glass bottle with a "
    "brand label) visibly present, whether inside the bag or taken out of it? A plain "
    "water bottle, a soft-drink or plastic bottle, a flask, thermos or cup "
    "does NOT count. "
    "Judge only by what is visible. Reply with ONLY compact JSON: "
    '{"answer": true, "conf": 0.0-1.0} or {"answer": false, "conf": 0.0-1.0}.'
)
FRAME_WIDTH = 768  # higher than VideoJev's 720 so labels are a bit more legible


def _provider():
    """(endpoint, api_key, model) from env; prefer OpenRouter, else DashScope."""
    if os.environ.get("OPENROUTER_API_KEY"):
        return (
            os.environ.get(
                "OPENROUTER_ENDPOINT", "https://openrouter.ai/api/v1/chat/completions"
            ),
            os.environ["OPENROUTER_API_KEY"],
            os.environ.get("ALCOHOL_VLM_MODEL", "qwen/qwen3.5-35b-a3b"),
        )
    if os.environ.get("DASHSCOPE_API_KEY"):
        return (
            "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
            os.environ["DASHSCOPE_API_KEY"],
            os.environ.get("ALCOHOL_VLM_MODEL", "qwen-vl-max"),
        )
    raise RuntimeError("Export OPENROUTER_API_KEY or DASHSCOPE_API_KEY")


def _frame_data_uri(video_path, t):
    frame = frame_jpeg(video_path, t, width=FRAME_WIDTH)
    return "data:image/jpeg;base64," + base64.b64encode(frame).decode("ascii")


def _parse_answer(text):
    """Malformed/refused replies are unknown, never a confident negative."""
    if not isinstance(text, str):
        raise ValueError("invalid alcohol response")
    text = text.strip()
    if text.startswith("```json\n") and text.endswith("\n```"):
        text = text[8:-4]
    try:
        data = json.loads(text)
    except (ValueError, TypeError):
        raise ValueError("invalid alcohol response") from None
    if (
        not isinstance(data, dict)
        or type(data.get("answer")) is not bool
        or type(data.get("conf")) not in (int, float)
        or not math.isfinite(data["conf"])
        or not 0 <= data["conf"] <= 1
    ):
        raise ValueError("invalid alcohol response")
    return data["answer"], float(data["conf"])


def ask_frame(data_uri, *, client=None, endpoint=None, api_key=None, model=None):
    """Ask the VLM about one frame; return (answer_bool, confidence_float)."""
    if endpoint is None:
        endpoint, api_key, model = _provider()
    url = urlsplit(endpoint)
    if url.scheme != "https" or not url.hostname or url.username or url.password:
        raise ValueError(
            "hosted alcohol endpoint must use HTTPS without URL credentials"
        )
    if not api_key or not model:
        raise ValueError("hosted alcohol credentials and model are required")
    body = {
        "model": model,
        "temperature": 0,
        "max_tokens": 60,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": PROMPT},
                    {"type": "image_url", "image_url": {"url": data_uri}},
                ],
            }
        ],
    }
    owns = client is None
    client = client or httpx.Client(timeout=40.0, trust_env=False)
    try:
        r = client.post(
            endpoint, json=body, headers={"Authorization": f"Bearer {api_key}"}
        )
        r.raise_for_status()
        try:
            text = r.json()["choices"][0]["message"]["content"]
        except (ValueError, KeyError, IndexError, TypeError):
            raise ValueError("invalid alcohol response") from None
    finally:
        if owns:
            client.close()
    return _parse_answer(text)


def detect_alcohol(video_path, times, *, conf_gate=0.5, client=None):
    """Pool valid positives; retain unknowns and frame failures explicitly.

    Confidence is the model's self-reported score, not calibrated correctness.
    This standalone helper is intentionally not called by the demo server.
    """
    times = validate_times(times)
    if type(conf_gate) not in (int, float) or not 0 <= conf_gate <= 1:
        raise ValueError("confidence gate must be finite and within 0..1")
    endpoint, api_key, model = _provider()
    owns = client is None
    client = client or httpx.Client(timeout=40.0, trust_env=False)
    per_frame = []
    positive, best, unknown = False, 0.0, False
    try:
        for t in times:
            try:
                uri = _frame_data_uri(video_path, t)
                ans, conf = ask_frame(
                    uri, client=client, endpoint=endpoint, api_key=api_key, model=model
                )
            except Exception as error:
                # Do not expose provider URLs, credentials, media or raw replies.
                per_frame.append(
                    {
                        "t": t,
                        "answer": None,
                        "conf": None,
                        "error": type(error).__name__,
                    }
                )
                unknown = True
                continue
            per_frame.append({"t": round(t, 2), "answer": ans, "conf": conf})
            if ans and conf >= conf_gate:
                positive = True
            if conf < conf_gate:
                unknown = True
            best = max(best, conf if ans else 0.0)
    finally:
        if owns:
            client.close()
    present = True if positive else None if unknown else False
    return {
        "present": present,
        "conf": round(best, 3),
        "model": model,
        "status": "partial" if unknown else "ok",
        "per_frame": per_frame,
    }
