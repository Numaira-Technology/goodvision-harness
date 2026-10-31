"""Run two fixed CUDA encoders, pausing between validation and fixed-threshold test."""

import argparse
import json
from pathlib import Path
import time

import cv2
import torch

from runtime import ClipEncoder, REVISIONS, run_streams


class SharedRunFailure(RuntimeError):
    """Stop after a shared failure while retaining completed group files."""


def save(path, value):
    pending = path.with_suffix(path.suffix + ".tmp")
    pending.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    pending.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--thresholds", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    cv2.setNumThreads(1)
    progress = output / "completion.json"
    encoders, errors, outcomes = {}, {}, {}
    try:
        for model_id in REVISIONS:
            short = model_id.rsplit("/", 1)[1]
            save(progress, {"phase": "loading_model", "model": model_id})
            try:
                encoder = ClipEncoder(model_id, args.cache_dir)
                encoders[model_id] = encoder
                save(output / (short + "-loaded.json"), encoder.metadata)
                save(progress, {"phase": "validation", "model": model_id})
                outcomes["validation-" + short] = run_streams(
                    config["validation"], encoder, {}, output / ("validation-" + short),
                    slots=12, interval=.8,
                )
                if outcomes["validation-" + short].get("shared_failure"):
                    raise SharedRunFailure(str(outcomes["validation-" + short]["shared_failure"]))
            except SharedRunFailure:
                raise
            except Exception as error:
                errors[model_id] = {"stage": "validation", "error": str(error)}
                save(output / (short + "-error.json"), errors[model_id])
        save(progress, {"phase": "awaiting_thresholds", "model_errors": errors,
                        "available_models": list(encoders)})
        deadline = time.monotonic() + 600
        while not args.thresholds.is_file():
            if time.monotonic() >= deadline:
                raise TimeoutError("Fixed validation thresholds were not supplied within ten minutes")
            time.sleep(.5)
        frozen = json.loads(args.thresholds.read_text())
        save(output / "frozen-thresholds.json", frozen)
        for model_id, encoder in encoders.items():
            short = model_id.rsplit("/", 1)[1]
            save(progress, {"phase": "test", "model": model_id})
            try:
                outcomes["test-" + short] = run_streams(
                    config["test"], encoder, frozen[model_id], output / ("test-" + short),
                    slots=12, interval=.8,
                )
                if outcomes["test-" + short].get("shared_failure"):
                    raise SharedRunFailure(str(outcomes["test-" + short]["shared_failure"]))
            except SharedRunFailure:
                raise
            except Exception as error:
                errors[model_id] = {"stage": "test", "error": str(error)}
                save(output / (short + "-error.json"), errors[model_id])
        complete = len(outcomes) == 4 and all(
            result.get("execution_complete") for result in outcomes.values()
        ) and not errors
        save(progress, {"phase": "finished", "execution_complete": complete,
                        "model_errors": errors,
                        "groups": {name: {key: result.get(key) for key in (
                            "execution_complete", "primary", "all", "wall_seconds", "shared_failure",
                        )} for name, result in outcomes.items()}})
    finally:
        for encoder in encoders.values():
            encoder.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

