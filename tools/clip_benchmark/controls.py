"""Frozen inputs, provenance and complete validation evidence for the native runner."""

from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import math
from pathlib import Path
import platform
import subprocess
import sys

from tools.clip_benchmark import scoring
from tools.clip_benchmark.protocol import PROMPTS, REVISIONS
from tools.event_benchmark.evidence import read_json, split_groups, write_new

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = {
    "variant": "native-clip-1fps-midpoint-v1", "sampling_fps": 1.0,
    "interval_seconds": 1.0, "slots": 12, "window_frames": 1,
    "sampling_origin_seconds": 0.0,
    "sampling_policy": "first decoded frame at/after each source-time second; paced feeder",
    "image_policy": "original decoded RGB through pinned official checkpoint processor",
    "score": "FP32 normalized positive cosine minus contrast cosine; lossless JSON float",
    "aggregation": "maximum valid target margin per primary video",
    "comparator": ">", "precision": "float32", "tf32": False, "cpu_threads": 1,
    "tail_load": "marked repeated completed videos; excluded from quality",
    "quality_denominator": "all questions, including unknown; base4 and bag reported separately",
    "expected_gpu": "NVIDIA RTX 4090", "expected_vram_gib": 24,
    "prompts": PROMPTS, "revisions": REVISIONS,
}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def code_identity():
    paths = sorted(Path(__file__).parent.glob("*.py")) + [
        ROOT / "tools/event_benchmark/evidence.py"]
    return {str(path.relative_to(ROOT)): digest(path) for path in paths}


def git_head():
    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                            capture_output=True, text=True, check=True)
    return result.stdout.strip()


def load_inputs(config_path):
    """Verify every frozen question, including the separate bag cohort; never edit inputs."""
    config_path = Path(config_path).resolve()
    config, config_hash = read_json(config_path)
    if (type(config.get("interval_seconds")) not in (int, float) or config["interval_seconds"] != 1.0
            or type(config.get("slots")) is not int or config["slots"] != 12):
        raise ValueError("controlled run requires interval_seconds=1.0 and slots=12")

    def resolve(value):
        path = Path(value)
        return path if path.is_absolute() else config_path.parent / path

    split_path = resolve(config["split_manifest"])
    parts, provenance = {}, {}
    for part in ("validation", "test"):
        spec = config[part]
        samples_path, manifest_path = resolve(spec["samples"]), resolve(spec["manifest"])
        document, sample_hash = read_json(samples_path)
        manifest, manifest_hash = read_json(manifest_path)
        if manifest.get("status") != "human_reviewed" or manifest.get("samples_json_sha256") != sample_hash:
            raise ValueError(f"{part}: unreviewed or changed samples/manifest")
        samples, frozen = document["samples"], manifest["files"]
        numbers = [row["number"] for row in samples]
        frozen_numbers = [row["number"] for row in frozen]
        if (not samples or len(numbers) != len(set(numbers)) or
                len(frozen_numbers) != len(set(frozen_numbers)) or set(numbers) != set(frozen_numbers)):
            raise ValueError(f"{part}: duplicate or mismatched questions")
        by_number = {row["number"]: row for row in frozen}
        groups, context = split_groups(split_path, manifest_path, part)
        media, hashes = {}, {}
        for sample in samples:
            row = by_number[sample["number"]]
            if (sample["event"] not in PROMPTS or sample["proposed_answer"] not in ("yes", "no", "unknown")
                    or not isinstance(sample["question"], str) or not sample["question"].strip()
                    or any(sample[k] != row[k] for k in ("filename", "event", "question", "proposed_answer"))):
                raise ValueError(f"{part}: changed question identity or invalid reference")
            if sample["event"] in scoring.BASE_EVENTS and sample["proposed_answer"] == "unknown":
                raise ValueError(f"{part}: base-event reference must be binary")
            filename = sample["filename"]
            if Path(filename).name != filename:
                raise ValueError("media filename must be a basename")
            if filename not in media:
                matches = {str((resolve(d) / filename).resolve()) for d in spec["media"]
                           if (resolve(d) / filename).is_file()}
                if len(matches) != 1:
                    raise ValueError(f"{part}: missing or ambiguous media: {filename}")
                media[filename] = matches.pop()
                hashes[filename] = digest(media[filename])
            if hashes[filename] != row["sha256"]:
                raise ValueError(f"{part}: media checksum mismatch: {filename}")
        # Copy the joined source field for reports; never rewrite samples or feed labels to the encoder.
        joined = [row | {"source_video_group": groups[row["number"]]} for row in samples]
        parts[part] = {"samples": joined,
                       "media": [{"filename": name, "video_path": path} for name, path in media.items()]}
        provenance[part] = {"samples_sha256": sample_hash, "manifest_sha256": manifest_hash,
                            "media_sha256": hashes, "source_groups": context}
    references, reference_hash = read_json(resolve(config["reference_thresholds"]))
    for model in REVISIONS:
        validate_thresholds(references.get(model))
    contract = {"protocol": PROTOCOL, "inputs": provenance, "code": code_identity(),
                "source_commit": git_head(), "historical_thresholds_sha256": reference_hash}
    return {"parts": parts, "contract": contract, "config_sha256": config_hash,
            "historical_thresholds": references}


def validate_thresholds(values):
    if not isinstance(values, dict) or set(values) != set(PROMPTS):
        raise ValueError("all six finite event thresholds required")
    if any(type(v) not in (int, float) or not math.isfinite(v) for v in values.values()):
        raise ValueError("all six finite event thresholds required")


def inspect_media(media, decode=False):
    from tools.clip_benchmark.base_timing import probe_video
    result = {}
    for entry in media:
        info = probe_video(entry["video_path"])
        if not math.isfinite(info.source_fps):
            raise ValueError("nonfinite media FPS")
        expected = math.floor((info.frame_count - 1) / info.source_fps + 1e-8) + 1
        detail = asdict(info) | {"planned_1fps_observations": expected}
        if decode:
            import cv2
            capture = cv2.VideoCapture(entry["video_path"])
            decoded = 0
            try:
                while True:
                    ok, frame = capture.read()
                    if not ok:
                        break
                    if frame is None or not frame.size:
                        raise ValueError("empty decoded frame")
                    decoded += 1
            finally:
                capture.release()
            if decoded != info.frame_count:
                raise ValueError(f"incomplete decode: {entry['filename']}")
            detail["decoded_frames"] = decoded
        result[entry["filename"]] = detail
    return result


def environment():
    packages = {}
    for name in ("torch", "torchvision", "transformers", "Pillow", "numpy", "opencv-python-headless"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    resolved = {dist.metadata["Name"]: dist.version for dist in importlib.metadata.distributions()
                if dist.metadata.get("Name")}
    return {"python": sys.version, "platform": platform.platform(), "packages": packages,
            "resolved_packages": dict(sorted(resolved.items()))}


def gpu_environment():
    """Fail before downloading weights if the authorized hardware is unavailable."""
    import torch
    if not torch.cuda.is_available():
        raise ValueError("CUDA RTX 4090 24GB required")
    gpu = torch.cuda.get_device_properties(0)
    if "4090" not in gpu.name or not 23 <= gpu.total_memory / 2**30 <= 25:
        raise ValueError("expected RTX 4090 24GB; changed hardware requires a separate run")
    info = environment() | {"gpu": gpu.name, "vram_bytes": gpu.total_memory,
                             "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version()}
    try:
        query = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                               capture_output=True, text=True, timeout=10, check=True)
        info["driver"] = query.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        info["driver"] = None
    return info


def validation_complete(records_path, completion, samples, planned):
    """Reject missing planned evidence before any fitting; test retains native partial policy."""
    if not completion.get("execution_complete") or completion.get("shared_failure"):
        raise ValueError("validation execution incomplete")
    states = [s for s in completion.get("streams", []) if s.get("quality_reference") is True]
    if len(states) != len(planned) or {s["filename"] for s in states} != set(planned):
        raise ValueError("validation source membership differs")
    records, issues = scoring._read_records(records_path)
    indexed = scoring._index(records)
    if issues or set(indexed) != set(planned):
        raise ValueError("validation records missing, extra or unreadable")
    for state in states:
        name = state["filename"]
        expected = planned[name]["planned_1fps_observations"]
        if (not state.get("execution_complete") or state.get("error") or
                state.get("completed") != expected or state.get("opportunities") != expected or
                any(state.get(k) != 0 for k in ("busy_missed", "late_missed", "local_failures"))):
            raise ValueError(f"incomplete validation opportunities: {name}")
        rows = indexed[name]
        fps = planned[name]["source_fps"]
        bins = []
        for row in rows:
            timestamp, frame_index = row.get("timestamp_seconds"), row.get("frame_index")
            if (row.get("errors") is not None or type(timestamp) not in (int, float) or
                    not math.isfinite(timestamp) or type(frame_index) is not int or
                    not 0 <= frame_index < planned[name]["frame_count"] or
                    not math.isclose(timestamp, frame_index / fps, abs_tol=1e-8)):
                raise ValueError("invalid validation frame evidence")
            bins.append(math.floor(timestamp + 1e-8))
            if any(scoring._margin(row, event) is None for event in PROMPTS):
                raise ValueError("missing/nonfinite validation margin")
        if len(rows) != expected or sorted(bins) != list(range(expected)):
            raise ValueError("duplicate/missing validation frame opportunities")
    for event in scoring.BASE_EVENTS:
        support = Counter(s["proposed_answer"] for s in samples if s["event"] == event)
        if sum(support.values()) < 6 or not support["yes"] or not support["no"]:
            raise ValueError("validation requires six reviewed questions and both classes per base event")


def verify_bundle(bundle_path, contract, model, split):
    bundle, bundle_hash = read_json(bundle_path)
    if bundle.get("contract") != contract or bundle.get("model") != model or bundle.get("split") != split:
        raise ValueError("run model/config/code/split contract differs")
    root = Path(bundle_path).parent
    for name in ("run.json", "completion.json", "clip-records.jsonl"):
        if digest(root / name) != bundle["files"].get(name):
            raise ValueError("run evidence checksum mismatch: " + name)
    return bundle, bundle_hash


def verify_threshold_artifact(path, expected_hash, contract, model):
    artifact, actual_hash = read_json(path)
    if actual_hash != expected_hash:
        raise ValueError("frozen threshold artifact checksum mismatch")
    if artifact.get("contract") != contract or artifact.get("model") != model:
        raise ValueError("threshold model/config/code/split contract differs")
    if artifact.get("validation_only") is not True:
        raise ValueError("thresholds must be selected on validation")
    validate_thresholds(artifact.get("thresholds"))
    validate_thresholds(artifact.get("historical_thresholds"))
    return artifact


def write_bundle(output, context, model, split, hardware, media, encoder_metadata):
    bundle = {"model": model, "split": split, "recorded_at": now(),
              "contract": context["contract"], "config_sha256": context["config_sha256"],
              "hardware": hardware, "encoder": encoder_metadata, "planned_media": media,
              "files": {name: digest(output / name) for name in
                        ("run.json", "completion.json", "clip-records.jsonl")}}
    write_new(output / "bundle.json", bundle)
    return bundle
