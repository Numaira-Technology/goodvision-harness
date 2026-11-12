"""Build and verify an immutable offline bundle. Never loads keys or calls models."""
import argparse
import base64
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import math
import platform
from pathlib import Path
import shutil
import subprocess
import sys

from tools.event_benchmark import evidence, run_eval
from tools.clip_benchmark.protocol import PROMPTS, REVISIONS
from .adapters import CURRENT_IMAGE_PROMPTS, question_core, request_body
from .contract import CURRENT_VERSION, EVENTS, VERSION, canonical, digest, file_hash, rules_for, sample_times, source_files, write_new
from .dataset202 import load_extension

ROOT = Path(__file__).resolve().parents[2]
PROFILES = Path(__file__).parent / "profiles"
PINNED_INPUTS = {
    "manifest_validation_24.json": "41a3ef7cadc16e5a4e511ce779688e07811e450aba168bb62b33d0f42bfa7365",
    "validation-samples.json": "efe43e6760cbbeacc1615fa6f9fce5c9bb6f81475b88c04b1cf02fab5a392e45",
    "manifest_120.json": "51b716bc07ad90a47358accd91ef1289ea6152cdedbacf69f45f0e656b3b56d7",
    "test-samples.json": "fdf04bcd28db87f29d8ab93cf0fce1a83ab644ac72a052b0a466ce639ff8fe58",
    "validation_split.json": "a0375a6a846fe964846a58d94ed68a5a2e18c61a238b0f36940d4c2a83f51b2c",
}


def load_inputs(out, frozen, protocol=CURRENT_VERSION):
    rules_for(protocol)
    for name, expected in PINNED_INPUTS.items():
        if file_hash(frozen / name) != expected:
            raise ValueError("frozen input changed: " + name)
        target = out / "inputs" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as f:
            f.write((frozen / name).read_bytes())
    parts = {}
    for split, manifest_name in [("validation", "manifest_validation_24.json"), ("test", "manifest_120.json")]:
        manifest = json.loads((frozen / manifest_name).read_text())
        samples_name = "validation-samples.json" if split == "validation" else "test-samples.json"
        raw = json.loads((frozen / samples_name).read_text())
        if manifest["status"] != "human_reviewed" or raw["status"] != "human_reviewed" or manifest["samples_json_sha256"] != PINNED_INPUTS[samples_name]:
            raise ValueError("unreviewed or mismatched data")
        groups, context = evidence.split_groups(frozen / "validation_split.json", frozen / manifest_name, split)
        expected = {q["number"]: q for q in manifest["files"]}
        samples = {q["number"]: q for q in raw["samples"]}
        if len(samples) != len(raw["samples"]) or len(expected) != len(manifest["files"]) or set(samples) != set(expected):
            raise ValueError("duplicate or mismatched question IDs")
        for number, m in expected.items():
            if any(samples[number][k] != m[k] for k in ("filename", "event", "question", "proposed_answer", "human_review")):
                raise ValueError("manifest/question/answer mismatch")
            if m["proposed_answer"] not in ("yes", "no") or m["event"] not in EVENTS:
                raise ValueError("unsupported frozen question")
        if len(samples) != (24 if split == "validation" else 120):
            raise ValueError("wrong frozen cohort")
        parts[split] = {"rows": [expected[n] | {"source_video_group": groups[n]} for n in sorted(expected)],
                        "split_context": context}
    return load_extension(out, frozen, parts) if protocol == CURRENT_VERSION else parts


def model_configs(aliases, profiles=PROFILES):
    configs = {}
    for alias in aliases:
        if not alias or Path(alias).name != alias:
            raise ValueError("model must be a profile filename without directories")
        config = json.loads((profiles / (alias + ".json")).read_text(encoding="utf-8"))
        if config["alias"] != alias or config["window_frames"] not in (1, 2):
            raise ValueError("profile identity/window mismatch")
        if config.get("execution") == "self_hosted_r2":
            if config.get("prompt_policy") != "current-image-v1":
                raise ValueError("current-image-v1 prompt policy required")
            if not config.get("weight_revision"):
                raise ValueError("self-hosted profile requires a pinned weight_revision")
        elif config["backend"] in ("language", "decision"):
            c = config["provider_context"]["config"]
            route = c["provider"]
            if (c["model"] != config["model"] or c["backend"] != config["backend"]
                    or len(route.get("only", [])) != 1 or route.get("allow_fallbacks") is not False
                    or not c.get("expected_provider_name")):
                raise ValueError("one pinned provider matching the model is required")
            if config.get("prompt_policy") != "current-image-v1":
                raise ValueError("current-image-v1 prompt policy required")
        elif not config.get("revision"):
            raise ValueError("native model revision required")
        configs[alias] = config
    return configs


def probe_last_frame_time(path):
    """Use decoded video PTS, not audio/container tail or rounded frame counts.

    Input-side ffmpeg -ss is relative to the container start, so normalize PTS
    to the same origin. Missing timestamps are a failed check, not a fallback.
    """
    result = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "format=start_time:frame=best_effort_timestamp_time",
        "-of", "json", str(path)], capture_output=True, text=True, check=True, timeout=300)
    if result.stderr.strip():
        raise ValueError("video timestamp probe failed: " + result.stderr[-1500:])
    data = json.loads(result.stdout)
    start = float(data["format"]["start_time"])
    times = [float(frame["best_effort_timestamp_time"]) for frame in data["frames"]]
    if not times or not math.isfinite(start) or any(not math.isfinite(t) for t in times):
        raise ValueError("missing or nonfinite decoded video timestamps")
    last = max(times) - start
    if not math.isfinite(last) or last < 0:
        raise ValueError("last decoded video timestamp precedes container start")
    return last


def inspect_video(split, row, out, media, protocol=VERSION):
    path = (media / split / row["filename"]).resolve()
    if not path.is_relative_to((media / split).resolve()) or file_hash(path) != row["sha256"]:
        raise ValueError("missing/changed media: " + row["filename"])
    duration = run_eval.probe_duration(path)
    decoded = subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-xerror", "-threads", "1", "-i", str(path),
                              "-map", "0:v:0", "-f", "null", "-"], capture_output=True, timeout=300)
    if decoded.returncode or decoded.stderr.strip():
        raise ValueError("full decode failure: " + row["filename"] + ": " + decoded.stderr.decode(errors="replace")[-1500:])
    planned_times = sample_times(duration)
    sampling_audit = None
    if protocol == CURRENT_VERSION:
        last = probe_last_frame_time(path)
        bounded_times = sample_times(duration, last_frame_time=last)
        sampling_audit = {"last_decoded_frame_time": last,
            "container_grid_times": planned_times, "retained_times": bounded_times,
            "excluded_tail_times": [t for t in planned_times if t > last]}
        planned_times = bounded_times
    frames = []
    for index, timestamp in enumerate(planned_times):
        uri = run_eval.frame_uri(path, timestamp)
        if not uri:
            raise ValueError("planned frame unavailable: " + row["filename"] + "@" + str(timestamp))
        raw = base64.b64decode(uri.split(",", 1)[1], validate=True)
        from PIL import Image
        import io
        with Image.open(io.BytesIO(raw)) as im:
            im.load()
            if im.width != 720 or im.height <= 0:
                raise ValueError("unexpected decoded frame dimensions")
            size = list(im.size)
        rel = f"frames/{split}/{row['sha256']}/{index:05}.jpg"
        dest = out / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        with dest.open("xb") as f:
            f.write(raw)
        frames.append({"path": rel, "time": timestamp, "sha256": file_hash(dest), "size": size})
    if len(frames) < 2:
        raise ValueError("video too short for a two-frame window: " + row["filename"])
    result = {"filename": row["filename"], "sha256": row["sha256"], "duration": duration,
              "full_decode_passed": True, "frames": frames}
    if sampling_audit is not None:
        result["sampling_audit"] = sampling_audit
    return result


def build_plan(split, rows, videos, config, bundle, protocol=VERSION, dataset_context=None):
    units, questions = [], []
    native = config["backend"] not in ("language", "decision")
    byfile = {}
    for row in rows:
        byfile.setdefault(row["filename"], []).append(row)
    groupings = list(byfile.values()) if native else [[r] for r in rows]
    for group in groupings:
        row = group[0]
        frames = videos[row["filename"]]["frames"]
        windows = [frames[i:i + config["window_frames"]] for i in range(len(frames) - config["window_frames"] + 1)]
        source_id = f"{split}/{row['sha256']}"
        targets = sorted({r["event"] for r in group})
        ids = []
        for index, window in enumerate(windows):
            uid = f"{config['alias']}/{source_id}/{'all' if native else row['number']}/{index}"
            unit = {"id": uid, "source_id": source_id, "targets": targets, "frames": window,
                    "question": CURRENT_IMAGE_PROMPTS[row["event"]] if not native else None}
            _, body = request_body(config, unit, bundle)
            unit["request_sha256"] = digest(body)
            units.append(unit)
            ids.append(uid)
        for q in group:
            questions.append({"number": q["number"], "filename": q["filename"], "event": q["event"],
                              "reference": q["proposed_answer"], "source_video_group": q["source_video_group"],
                              "unit_ids": ids})
    return {"protocol": protocol, "split": split, "config_sha256": digest(config), "rules_sha256": digest(rules_for(protocol)),
            "dataset_context": dataset_context,
            "questions": sorted(questions, key=lambda q: q["number"]), "units": units}


def freeze_sources(out):
    result = {}
    for rel, path in source_files().items():
        result[rel] = file_hash(path)
        dest = out / "source-snapshot" / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        with dest.open("xb") as f:
            f.write(path.read_bytes())
    write_new(out / "source-hashes.json", result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True, help="Private frozen inputs: eight JSON files for 202, five for legacy120")
    parser.add_argument("--protocol", choices=(CURRENT_VERSION, VERSION), default=CURRENT_VERSION)
    parser.add_argument("--media", type=Path, required=True, help="contains validation/ and test/ videos")
    parser.add_argument("--profiles", type=Path, default=PROFILES)
    parser.add_argument("--models", nargs="+", required=True, help="profile aliases to prepare")
    args = parser.parse_args(argv)
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    report = {"protocol": args.protocol, "started_at": datetime.now(timezone.utc).isoformat(),
              "inference_calls": 0, "model_downloads": 0, "network_calls": 0, "formal_run_authorized": False}
    try:
        parts = load_inputs(out, args.inputs, args.protocol)
        report["dataset_context"] = parts["test"].get("dataset_context")
        write_new(out / "split-audit.json", {split: {"context": part["split_context"],
            "members": [{k: row[k] for k in ("number", "filename", "sha256", "source_video_group", "human_review")}
                        for row in part["rows"]]} for split, part in parts.items()})
        configs = model_configs(args.models, args.profiles)
        write_new(out / "rules.json", rules_for(args.protocol))
        tasks, inspected = [], {"validation": {}, "test": {}}
        for split, part in parts.items():
            unique = {r["filename"]: r for r in part["rows"]}
            if len(unique) != (24 if split == "validation" else 192 if args.protocol == CURRENT_VERSION else 110):
                raise ValueError("unique video count differs")
            tasks.extend((split, r) for r in unique.values())
        errors = []
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = {pool.submit(inspect_video, split, row, out, args.media, args.protocol): (split, row) for split, row in tasks}
            for done, future in enumerate(as_completed(futures), 1):
                split, row = futures[future]
                try:
                    inspected[split][row["filename"]] = future.result()
                except Exception as error:
                    errors.append({"split": split, "filename": row["filename"], "error": str(error)})
                if done % 20 == 0 or done == len(tasks):
                    print(json.dumps({"videos_checked": done, "planned": len(tasks), "errors": len(errors)}), flush=True)
        write_new(out / "media-audit.json", {"videos": inspected, "errors": errors})
        if errors:
            raise ValueError(f"{len(errors)} video checks failed; see media-audit.json")
        report["models"] = {}
        for alias, config in configs.items():
            write_new(out / "configs" / (alias + ".json"), config)
            counts = {}
            for split, part in parts.items():
                plan = build_plan(split, part["rows"], inspected[split], config, out, args.protocol, part.get("dataset_context"))
                write_new(out / "plans" / f"{alias}-{split}.json", plan)
                counts[split] = {"questions": len(plan["questions"]), "unique_units": len(plan["units"]),
                                 "target_evaluations": sum(len(q["unit_ids"]) for q in plan["questions"]),
                                 "plan_sha256": digest(plan)}
            report["models"][alias] = counts
        write_new(out / "weights-plan.json", {"source": "official_Hugging_Face_pinned_revision",
            "r2": "optional verified cache; see R2 PR/skill availability, never assume credentials", "downloads_started": False,
            "models": {a: {k: c[k] for k in ("model", "revision", "precision")} for a, c in configs.items() if "revision" in c},
            "integrity": "verify official LFS SHA256 or Git blob SHA1 for every file; preserve manifest; no revision fallback"})
        sources = freeze_sources(out)
        report.update(offline_passed=True, sources_sha256=digest(sources),
            unique_videos=sum(len(v) for v in inspected.values()),
            sampled_unique_frames=sum(len(v["frames"]) for part in inspected.values() for v in part.values()),
            environment={"platform": platform.platform(), "python": platform.python_version(),
                         "ffmpeg": subprocess.check_output(["ffmpeg", "-version"], text=True).splitlines()[0]},
            formal_ready=False,
            pending=["user confirmation for formal execution", "available 4090 24GB SSH/environment",
                     "official model downloads and file checksums on deployment host", "model load and original warmup",
                     "actual GPU/VRAM/dtype and native service identity verification",
                     "current API access/credits/cost/provider verification before paid execution",
                     "complete validation and frozen thresholds before calibrated test"],
            limits=["No proof of training/development/actor/scene isolation; test has prior uses.",
                    "No bag validation: zero margin / 0.5 Luna bag thresholds remain uncalibrated.",
                    "Exact t=0,1,... and shared 720px JPEGs differ from old API centered sampling and native pixels.",
                    "Single-frame and native two-frame interfaces, prompts and native precisions remain different.",
                    "Three total attempts cannot guarantee full completion; missing evidence is reported separately.",
                    "12 waiting task slots and no-drop offline completion do not prove camera throughput."])
    except Exception as error:
        report.update(offline_passed=False, formal_ready=False, error=type(error).__name__ + ": " + str(error))
    report["finished_at"] = datetime.now(timezone.utc).isoformat()
    write_new(out / "preflight.json", report)
    write_new(out / "bundle-files-sha256.json", {p.relative_to(out).as_posix(): file_hash(p) for p in sorted(out.rglob("*")) if p.is_file()})
    print(json.dumps({k: report[k] for k in ("offline_passed", "formal_ready", "inference_calls")} | {"error": report.get("error")}), flush=True)
    return 0 if report["offline_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
