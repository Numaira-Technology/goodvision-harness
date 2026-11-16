"""Stage native CLIP/SigLIP2: offline preflight, validation, selection, frozen test."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.clip_benchmark import controls as c, scoring
from tools.clip_benchmark.protocol import PROMPTS, REVISIONS
from tools.event_benchmark.evidence import read_json, write_new


def select(context, model, bundle_path, output):
    bundle, bundle_hash = c.verify_bundle(bundle_path, context["contract"], model, "validation")
    root = bundle_path.parent
    completion, _ = read_json(root / "completion.json")
    samples = context["parts"]["validation"]["samples"]
    records = root / "clip-records.jsonl"
    c.validation_complete(records, completion, samples, bundle["planned_media"])
    output.mkdir(parents=True, exist_ok=False)
    selected = scoring.select_thresholds(records, samples, output / "analysis", model)
    analysis, _ = read_json(output / "analysis" / "thresholds.json")
    artifact = {"model": model, "selected_at": c.now(), "validation_only": True,
                "contract": context["contract"], "validation_bundle_sha256": bundle_hash,
                "validation_files": bundle["files"], "hardware": bundle["hardware"],
                "encoder": {k: v for k, v in bundle["encoder"].items() if k != "load_and_text_seconds"},
                "thresholds": selected, "selection_rule": scoring.SELECTION_RULE,
                "events": analysis["events"],
                "historical_thresholds": context["historical_thresholds"][model],
                "historical_reference_only": "old 1.25 FPS run; not a current calibration input",
                "bag_status": "zero, uncalibrated: no bag validation"}
    write_new(output / "frozen-thresholds.json", artifact)
    write_new(output / "artifact-sha256.json", {"sha256": c.digest(output / "frozen-thresholds.json")})
    return 0


def collect(context, model, split, output, cache_dir, artifact=None):
    hardware = c.gpu_environment()
    if artifact and hardware != artifact["hardware"]:
        raise ValueError("validation/test GPU or software environment differs")
    # CUDA imports and downloads occur only after all input/frozen-threshold checks.
    import cv2
    import torch
    from tools.clip_benchmark.runtime import ClipEncoder, run_streams
    torch.set_num_threads(1)
    cv2.setNumThreads(1)
    part = context["parts"][split]
    planned = c.inspect_media(part["media"])
    output.mkdir(parents=True, exist_ok=False)
    encoder = None
    try:
        encoder = ClipEncoder(model, cache_dir)
        identity = {k: v for k, v in encoder.metadata.items() if k != "load_and_text_seconds"}
        if artifact and identity != artifact["encoder"]:
            raise ValueError("validation/test encoder identity differs")
        thresholds = artifact["thresholds"] if artifact else dict.fromkeys(PROMPTS, 0.0)
        completion = run_streams(part["media"], encoder, thresholds, output,
                                 slots=12, interval=1.0)
        c.write_bundle(output, context, model, split, hardware, planned, encoder.metadata)
        if split == "validation":
            c.validation_complete(output / "clip-records.jsonl", completion, part["samples"], planned)
        else:
            write_new(output / "frozen-thresholds.json", artifact)
            for variant, values in (("validation-tuned", artifact["thresholds"]),
                                    ("historical-thresholds", artifact["historical_thresholds"]),
                                    ("fixed-zero", dict.fromkeys(PROMPTS, 0.0))):
                for cohort, events in (("base4", scoring.BASE_EVENTS), ("bag", scoring.BAG_EVENTS)):
                    samples = [s for s in part["samples"] if s["event"] in events]
                    if samples:
                        scoring.evaluate(output / "clip-records.jsonl", samples, values,
                                         output / "reports" / variant / cohort, model)
        return 0 if completion.get("execution_complete") else 1
    except Exception as error:
        write_new(output / "failure.json", {"phase": split, "model": model,
                  "error_type": type(error).__name__, "error": str(error), "recorded_at": c.now()})
        raise
    finally:
        if encoder is not None:
            encoder.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("preflight", "validation", "select", "test"))
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", choices=REVISIONS)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--validation-bundle", type=Path)
    parser.add_argument("--thresholds", type=Path)
    parser.add_argument("--thresholds-sha256")
    parser.add_argument("--decode", action="store_true", help="preflight: decode all original frames without pacing")
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError("refusing to overwrite output: " + str(args.output))
    context = c.load_inputs(args.config)
    if args.phase == "preflight":
        media = {part: c.inspect_media(data["media"], args.decode)
                 for part, data in context["parts"].items()}
        result = {"status": "offline_inputs_passed_gpu_run_pending", "recorded_at": c.now(),
                  "contract": context["contract"], "config_sha256": context["config_sha256"],
                  "environment": c.environment(), "models": list(REVISIONS), "media": media,
                  "counts": {part: {"questions": len(data["samples"]), "videos": len(data["media"]),
                        "planned_1fps_observations": sum(v["planned_1fps_observations"] for v in media[part].values())}
                        for part, data in context["parts"].items()},
                  "pending": ["rental and model download/inference authorization", "actual 4090 24GB and CUDA dependencies",
                              "model loading, VRAM peak, paced 12-slot completion and latency",
                              "fresh complete validation scores and frozen thresholds before test"],
                  "inference_calls": 0, "model_downloads": 0,
                  "limits": ["1 FPS is a target; live late/busy misses remain measured and cannot be silently filled.",
                             "12 paced local file readers do not establish 12-camera production throughput.",
                             "Native start-of-second original pixels differ from API centered 720px JPEGs.",
                             "Cross-class results compare systems; accuracy/error denominators also differ."]}
        write_new(args.output, result)
        print(json.dumps({"status": result["status"], "counts": result["counts"]}))
        return 0
    if not args.model:
        parser.error("--model required outside preflight")
    if args.phase == "select":
        if not args.validation_bundle:
            parser.error("select requires --validation-bundle")
        return select(context, args.model, args.validation_bundle, args.output)
    if not args.cache_dir:
        parser.error("validation/test require --cache-dir")
    artifact = None
    if args.phase == "test":
        if not args.thresholds or not args.thresholds_sha256:
            parser.error("test requires --thresholds and --thresholds-sha256")
        artifact = c.verify_threshold_artifact(args.thresholds, args.thresholds_sha256,
                                              context["contract"], args.model)
    return collect(context, args.model, args.phase, args.output, args.cache_dir, artifact)


if __name__ == "__main__":
    raise SystemExit(main())
