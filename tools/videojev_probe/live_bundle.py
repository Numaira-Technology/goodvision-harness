"""Derive a live-spacing (0.5 s) VideoJev bundle from a frozen native bundle.

The frozen native protocol samples one frame per second and sends two adjacent
frames (1 s apart, fps 1.0). The live System 1 workbench sends two frames
0.5 s apart (fps 2.0). This tool re-extracts frames every ``step`` seconds from
the same verified source videos and builds windows of two adjacent frames at
that spacing, so gpu_sweep can measure latency and answers at the live spacing.

Kept from the frozen bundle: the six frozen questions (config), the 202
question references and their per-video grouping (any yes -> yes ...). Changed:
the frames, the windows (about twice as many) and the request ``fps``. This is
a diagnostic replay, not the frozen protocol; its accuracy is not comparable to
the frozen table.

The output directory holds frames, ``configs/videojev.json``, ``plans/videojev-test.json``
and ``live-manifest.json`` (parent bundle digest, step, SHA-256 of every file).
Like the frozen bundle it contains reference answers: keep it private.
"""

import argparse
import base64
import io
import json
import math
from pathlib import Path

from tools.event_benchmark import run_eval
from tools.unified_benchmark.contract import digest, file_hash, write_new
from tools.unified_benchmark.preflight import probe_last_frame_time
from tools.unified_benchmark.workflow import verify_bundle

KIND = "videojev-live-spacing-bundle"


def live_times(duration, last_frame_time, step):
    """k * step for every k with the timestamp inside the decodable video."""
    if not (math.isfinite(step) and 0 < step <= 1):
        raise ValueError("step must be in (0, 1]")
    count = math.floor(min(duration, last_frame_time) / step + 1e-9) + 1
    return [round(k * step, 6) for k in range(count)
            if k * step < duration and k * step <= last_frame_time + 1e-9]


def request_body(config, unit, bundle, step):
    """The frozen native body with the live spacing; frames verified by hash."""
    frames = unit["frames"]
    if len(frames) != 2 or not math.isclose(frames[1]["time"] - frames[0]["time"], step,
                                            abs_tol=1e-6):
        raise ValueError("live window requires two adjacent frames at the bundle step")
    images = []
    for frame in frames:
        path = (Path(bundle) / frame["path"]).resolve()
        if not path.is_relative_to(Path(bundle).resolve()) or file_hash(path) != frame["sha256"]:
            raise ValueError("frame path or fingerprint differs")
        images.append(base64.b64encode(path.read_bytes()).decode())
    return {"session": unit["source_id"], "request_id": unit["id"], "fps": 1.0 / step,
            "frame_times": [f["time"] for f in frames], "images": images,
            "question_template": config["question_template"]}


def plan_from_frames(frozen_plan, config, bundle, frames_by_source, step):
    """Windows of adjacent frames per source; questions keep their references."""
    units, ids_by_source = [], {}
    targets = {}
    for unit in frozen_plan["units"]:
        targets[unit["source_id"]] = unit["targets"]
    for source_id, frames in frames_by_source.items():
        if len(frames) < 2:
            raise ValueError("video too short for a two-frame window: " + source_id)
        ids = []
        for index in range(len(frames) - 1):
            unit = {"id": f"videojev-live/{source_id}/all/{index}", "source_id": source_id,
                    "targets": targets[source_id], "frames": frames[index:index + 2],
                    "question": None}
            unit["request_sha256"] = digest(request_body(config, unit, bundle, step))
            units.append(unit)
            ids.append(unit["id"])
        ids_by_source[source_id] = ids
    source_of = {u["id"]: u["source_id"] for u in frozen_plan["units"]}
    questions = []
    for q in frozen_plan["questions"]:
        sources = {source_of[uid] for uid in q["unit_ids"]}
        if len(sources) != 1:
            raise ValueError("a frozen question spans several videos")
        questions.append({**q, "unit_ids": ids_by_source[sources.pop()]})
    return {"protocol": KIND, "split": frozen_plan["split"], "step_seconds": step,
            "parent_plan_sha256": digest(frozen_plan), "config_sha256": digest(config),
            "questions": questions, "units": units}


def build(frozen, media, output, step=0.5, split="test"):
    frozen, media, output = Path(frozen), Path(media), Path(output)
    parent_digest = verify_bundle(frozen)
    plan = json.loads((frozen / "plans" / f"videojev-{split}.json").read_text())
    config = json.loads((frozen / "configs" / "videojev.json").read_text())
    output.mkdir(parents=True, exist_ok=False)
    files = {}
    filename_of = {}
    source_of = {u["id"]: u["source_id"] for u in plan["units"]}
    for q in plan["questions"]:
        filename_of[source_of[q["unit_ids"][0]]] = q["filename"]
    from PIL import Image

    frames_by_source = {}
    for source_id, filename in sorted(filename_of.items()):
        sha = source_id.split("/", 1)[1]
        path = media / split / filename
        if file_hash(path) != sha:
            raise ValueError("source video differs from the frozen bundle: " + filename)
        times = live_times(run_eval.probe_duration(path), probe_last_frame_time(path), step)
        frames = []
        for index, t in enumerate(times):
            uri = run_eval.frame_uri(path, t)
            if not uri:
                raise ValueError(f"planned frame unavailable: {filename}@{t}")
            raw = base64.b64decode(uri.split(",", 1)[1], validate=True)
            with Image.open(io.BytesIO(raw)) as image:
                image.load()
                size = list(image.size)
            rel = f"frames/{split}/{sha}/{index:05}.jpg"
            dest = output / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(raw)
            files[rel] = file_hash(dest)
            frames.append({"path": rel, "time": t, "sha256": files[rel], "size": size})
        frames_by_source[source_id] = frames
        print(json.dumps({"video": filename, "frames": len(frames)}), flush=True)
    live_plan = plan_from_frames(plan, config, output, frames_by_source, step)
    for rel, value in ((f"plans/videojev-{split}.json", live_plan),
                       ("configs/videojev.json", config)):
        (output / rel).parent.mkdir(parents=True, exist_ok=True)
        write_new(output / rel, value)
        files[rel] = file_hash(output / rel)
    manifest = {"kind": KIND, "parent_bundle_sha256": parent_digest, "step_seconds": step,
                "split": split, "units": len(live_plan["units"]),
                "questions": len(live_plan["questions"]), "files": files}
    write_new(output / "live-manifest.json", manifest)
    return manifest


def verify(bundle):
    """Check every file against the manifest; return (digest, step)."""
    bundle = Path(bundle)
    manifest = json.loads((bundle / "live-manifest.json").read_text())
    if manifest.get("kind") != KIND:
        raise ValueError("not a live-spacing bundle")
    for rel, expected in manifest["files"].items():
        path = (bundle / rel).resolve()
        if not path.is_relative_to(bundle.resolve()) or file_hash(path) != expected:
            raise ValueError("live bundle file differs: " + rel)
    return file_hash(bundle / "live-manifest.json"), manifest["step_seconds"]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--frozen-bundle", required=True)
    parser.add_argument("--media", required=True, help="contains test/ videos")
    parser.add_argument("--output", required=True, help="new directory")
    parser.add_argument("--step", type=float, default=0.5)
    args = parser.parse_args(argv)
    manifest = build(args.frozen_bundle, args.media, args.output, args.step)
    print(json.dumps({k: manifest[k] for k in ("units", "questions", "step_seconds")}))


if __name__ == "__main__":
    main()
