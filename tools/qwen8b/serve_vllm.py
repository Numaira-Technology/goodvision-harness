#!/usr/bin/env python3
"""Start vLLM's OpenAI server for the pinned Qwen3-VL-8B snapshot.

This is the lock alternative: vLLM continuous batching + paged KV, not
transformers.generate() under a process-wide threading.Lock(). FlashAttention
is an optional kernel inside vLLM, not a competing product.

Does not replace tools.qwen8b.serve (HF+FA2).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

MODEL_ID = "qwen/qwen3-vl-8b-instruct"
REPO = "Qwen/Qwen3-VL-8B-Instruct"
REVISION = "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"
DEFAULTS = {
    "dtype": "bfloat16",
    # Headroom for one frozen 720px JPEG + current-image prompt + max_tokens=30.
    # Must not truncate exam requests; do not add mm resize / max_pixels caps.
    "max_model_len": 8192,
    "max_num_seqs": 12,
    "max_num_batched_tokens": 8192,
    "gpu_memory_utilization": 0.90,
    "enforce_eager": True,
    "limit_mm_per_prompt": {"image": 1, "video": 0},
    "served_model_name": MODEL_ID,
}


def snapshot(cache: Path) -> Path:
    from tools.model_source import snapshot_dir
    path = snapshot_dir(cache, REPO, REVISION)
    if not path.is_dir():
        raise SystemExit(f"missing local snapshot {path}; prefetch from R2 first")
    return path


def serve_argv(cache: Path, *, host="127.0.0.1", port=8000, overrides=None):
    cfg = dict(DEFAULTS)
    if overrides:
        cfg.update({k: v for k, v in overrides.items() if v is not None})
    limit = cfg["limit_mm_per_prompt"]
    if isinstance(limit, dict):
        limit = json.dumps(limit, separators=(",", ":"))
    argv = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", str(snapshot(cache)),
        "--served-model-name", str(cfg["served_model_name"]),
        "--host", host, "--port", str(port),
        "--dtype", str(cfg["dtype"]),
        "--max-model-len", str(int(cfg["max_model_len"])),
        "--max-num-seqs", str(int(cfg["max_num_seqs"])),
        "--max-num-batched-tokens", str(int(cfg["max_num_batched_tokens"])),
        "--gpu-memory-utilization", str(cfg["gpu_memory_utilization"]),
        "--limit-mm-per-prompt", limit,
        "--tensor-parallel-size", "1",
        "--generation-config", "vllm",
    ]
    if cfg.get("enforce_eager"):
        argv.append("--enforce-eager")
    return argv, cfg


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache-dir", type=Path, required=True)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--max-num-seqs", type=int)
    ap.add_argument("--gpu-memory-utilization", type=float)
    ap.add_argument("--max-model-len", type=int)
    ap.add_argument("--print-argv", action="store_true")
    args = ap.parse_args(argv)
    overrides = {
        "max_num_seqs": args.max_num_seqs,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "max_model_len": args.max_model_len,
    }
    cmd, cfg = serve_argv(args.cache_dir, host=args.host, port=args.port, overrides=overrides)
    print(json.dumps({"engine": "vllm", "config": cfg, "model": MODEL_ID, "revision": REVISION}), flush=True)
    if args.print_argv:
        print(" ".join(cmd), flush=True)
        return
    os.execvp(cmd[0], cmd)


if __name__ == "__main__":
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("VLLM_NO_USAGE_STATS", "1")
    main()
