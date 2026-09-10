"""Measure the configured production rich client against an authorized image."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from goodvision_harness.config import load_config
from goodvision_harness.model_api import environment_api_key
from goodvision_harness.video_to_stream import encode_jpeg
from goodvision_harness.vlm_rich_labeling import VLMRichLabelClient


def run(args: argparse.Namespace) -> None:
    load_dotenv(args.env_file)
    os.environ["INFERENCE_PROVIDER"] = "openrouter"
    config = load_config(args.config)
    rich = config.rich_labeling
    settings = replace(
        config.model_api, timeout_seconds=rich.timeout_seconds, max_retries=0,
        extra_headers={**config.model_api.extra_headers, "X-OpenRouter-Cache": "false"},
    )
    parameters = dict(rich.vlm_request_parameters)
    model = args.model or rich.model or config.inference.model
    if "-instruct" in model:
        parameters.pop("reasoning", None)
    responses = []
    client = VLMRichLabelClient(
        settings,
        model=model,
        api_key=environment_api_key(settings),
        max_tokens=rich.vlm_max_tokens,
        max_image_side=rich.vlm_max_image_side,
        wire_format=rich.vlm_wire_format,
        execution_mode=rich.vlm_execution_mode,
        parallel_strategy=rich.vlm_parallel_strategy,
        structured_output=rich.vlm_structured_output,
        require_complete=rich.vlm_require_complete,
        prompt_cache=rich.vlm_prompt_cache and not args.no_cache,
        http2=rich.vlm_http2,
        trust_env=rich.vlm_trust_env,
        keepalive_expiry_seconds=rich.vlm_keepalive_expiry_seconds,
        request_parameters=parameters,
        response_observer=responses.append,
    )
    image = cv2.imdecode(
        np.frombuffer(args.image.read_bytes(), dtype=np.uint8), cv2.IMREAD_COLOR
    )
    if image is None:
        raise SystemExit("Cannot decode image")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        for trial in range(args.repeats):
            responses.clear()
            frame = cv2.flip(image, 1) if trial % 2 else image
            encoded, _, _ = encode_jpeg(frame, max_image_side=720, quality=85)
            row = {
                "trial": trial,
                "model": model,
                "mirrored": bool(trial % 2),
                "prompt_cache": client.prompt_cache,
                "execution_mode": client.execution_mode,
            }
            started = time.perf_counter()
            try:
                snapshot = client.infer(
                    encoded,
                    stream_session_id="latency-probe",
                    frame_sequence=trial,
                    timestamp_seconds=trial * 15.0,
                )
                people = [o for o in snapshot.objects if o.label == "person"]
                row.update(
                    {
                        "valid": True,
                        "people": len(people),
                        "items": len(snapshot.objects) - len(people),
                        "objects": [o.as_dict() for o in snapshot.objects],
                    }
                )
            except Exception as exc:
                row.update({"valid": False, "error": str(exc)})
            row["wall_seconds"] = round(time.perf_counter() - started, 6)
            if responses:
                row["response"] = responses[-1]
            with args.output.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            brief = {k: v for k, v in row.items() if k not in ("response", "objects")}
            if responses:
                brief.update(
                    {
                        "usage": responses[-1].get("usage"),
                        "timing": responses[-1].get("_timing"),
                    }
                )
            print(json.dumps(brief, ensure_ascii=False), flush=True)
    finally:
        client.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--env-file", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("config.vlm-rich.yaml"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=4)
    parser.add_argument("--model")
    parser.add_argument("--no-cache", action="store_true")
    run(parser.parse_args())
