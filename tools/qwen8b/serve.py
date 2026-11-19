#!/usr/bin/env python3
"""OpenAI-compatible loopback server for self-hosted Qwen3-VL-8B.

Loads the R2-pinned snapshot with flash_attention_2. Extra request fields
(provider, OpenRouter routing) are ignored. One in-flight generate() at a time
so twelve eval workers queue instead of OOM on a 24GB 4090.
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

MODEL_ID = "qwen/qwen3-vl-8b-instruct"
REPO = "Qwen/Qwen3-VL-8B-Instruct"
REVISION = "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"


def _snapshot(cache: Path) -> Path:
    from tools.model_source import snapshot_dir
    path = snapshot_dir(cache, REPO, REVISION)
    if not path.is_dir():
        raise SystemExit(f"missing local snapshot {path}; prefetch from R2 first")
    return path


def load_model(cache: Path):
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    snap = _snapshot(cache)
    processor = AutoProcessor.from_pretrained(snap, local_files_only=True, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        snap, local_files_only=True, trust_remote_code=True,
        torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2",
        device_map="cuda:0",
    )
    model.eval()
    return processor, model


def _image(url: str):
    from PIL import Image
    if not isinstance(url, str) or not url.startswith("data:image/"):
        raise ValueError("image_url must be a data URI")
    _, b64 = url.split(",", 1)
    with Image.open(io.BytesIO(base64.b64decode(b64))) as image:
        return image.convert("RGB")


def _messages(body):
    messages = body.get("messages") or []
    if len(messages) != 1 or messages[0].get("role") != "user":
        raise ValueError("one user message required")
    content = messages[0].get("content")
    if not isinstance(content, list):
        raise ValueError("multipart content required")
    parts = []
    for item in content:
        if item.get("type") == "text":
            parts.append({"type": "text", "text": item.get("text") or ""})
        elif item.get("type") == "image_url":
            parts.append({"type": "image", "image": _image((item.get("image_url") or {}).get("url") or "")})
        else:
            raise ValueError("unsupported content part")
    if not any(p["type"] == "image" for p in parts) or not any(p["type"] == "text" for p in parts):
        raise ValueError("text and image parts required")
    return [{"role": "user", "content": parts}]


def generate(processor, model, body):
    import torch
    messages = _messages(body)
    max_tokens = int(body.get("max_tokens") or 30)
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    images = [part["image"] for part in messages[0]["content"] if part["type"] == "image"]
    inputs = processor(text=[text], images=images, return_tensors="pt", padding=True)
    inputs = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in inputs.items()}
    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False)
    trimmed = out[:, inputs["input_ids"].shape[1]:]
    eos_ids = model.generation_config.eos_token_id
    if isinstance(eos_ids, int):
        eos_ids = [eos_ids]
    # EOS can itself be the final allowed token. Only budget exhaustion without
    # EOS is a truncated answer; the benchmark parser rejects that receipt.
    ended_on_eos = bool(trimmed.shape[1] and trimmed[0, -1].item() in (eos_ids or []))
    finish_reason = "length" if trimmed.shape[1] >= max_tokens and not ended_on_eos else "stop"
    answer = processor.batch_decode(trimmed, skip_special_tokens=True)[0].strip()
    return {
        "id": "qwen8b-r2-host",
        "model": MODEL_ID,
        "object": "chat.completion",
        "choices": [{"index": 0, "finish_reason": finish_reason,
                     "message": {"role": "assistant", "content": answer, "refusal": None}}],
        "provider": "self-hosted-r2-4090",
    }


def make_handler(processor, model, ready):
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            print("[qwen8b-serve]", fmt % args, flush=True)

        def _send(self, code, payload):
            raw = json.dumps(payload).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self):
            if self.path.rstrip("/") in ("", "/health"):
                gpu = ""
                try:
                    import torch
                    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else ""
                except Exception:
                    gpu = ""
                return self._send(200, {"ready": True, "model": MODEL_ID, "revision": REVISION, "gpu": gpu})
            self._send(404, {"error": "not found"})

        def do_POST(self):
            if self.path.rstrip("/") != "/v1/chat/completions":
                return self._send(404, {"error": "not found"})
            if not ready.is_set():
                return self._send(503, {"error": "model not ready"})
            length = int(self.headers.get("Content-Length") or 0)
            if length > 8 * 1024 * 1024:
                return self._send(413, {"error": "oversized request"})
            try:
                body = json.loads(self.rfile.read(length))
            except (ValueError, TypeError):
                return self._send(400, {"error": "invalid json"})
            try:
                with lock:
                    payload = generate(processor, model, body)
            except Exception as error:
                return self._send(400, {"error": type(error).__name__ + ": " + str(error)[:500]})
            self._send(200, payload)

    return Handler


def require_packed_prefix(name="qwen8b"):
    import site
    import sys

    from tools import registry

    env = registry.env_of(name)
    if env is None:
        raise SystemExit(f"model {name!r} has no packed env")
    want = os.path.realpath(env["unpack_to"])
    got = os.path.realpath(sys.prefix)
    if got != want:
        raise SystemExit(f"sys.prefix {got!r} != restored env {want!r}")
    if site.ENABLE_USER_SITE:
        raise SystemExit("user site must be disabled (PYTHONNOUSERSITE)")


def main(argv=None):
    require_packed_prefix("qwen8b")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache-dir", type=Path, required=True)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args(argv)
    ready = threading.Event()
    print(f"loading {REPO}@{REVISION[:12]} from {args.cache_dir}", flush=True)
    processor, model = load_model(args.cache_dir)
    ready.set()
    server = ThreadingHTTPServer((args.host, args.port), make_handler(processor, model, ready))
    print(f"qwen8b ready on http://{args.host}:{args.port}/v1/chat/completions", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    main()
