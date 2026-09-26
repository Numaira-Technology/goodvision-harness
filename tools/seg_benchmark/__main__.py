"""Run with: .venv-seg-benchmark/bin/python -m tools.seg_benchmark --help."""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import importlib.metadata
import json
import os
import platform
import random
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from .metrics import polygons_to_masks, replay_delay, score_frame, summarize
from .models import MODELS, Adapter, resolve_device, synchronize

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA = ROOT / "outputs/seg-benchmark/dataset"


def setup_cache():
    # No hidden CPU fallback: an unsupported MPS op must be visible as a failure.
    if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "1":
        raise ValueError(
            "Unset PYTORCH_ENABLE_MPS_FALLBACK for comparable GPU measurements"
        )
    locations = {
        "RF_HOME": ROOT / ".models/seg-benchmark/rfdetr",
        "TORCH_HOME": ROOT / ".models/seg-benchmark/torch",
        "HF_HOME": ROOT / ".models/seg-benchmark/huggingface",
        "YOLO_CONFIG_DIR": ROOT / ".cache/seg-benchmark/ultralytics",
        "MPLCONFIGDIR": ROOT / ".cache/seg-benchmark/matplotlib",
    }
    for key, path in locations.items():
        path.mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(path)
    os.environ["YOLO_AUTOINSTALL"] = "false"


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)
    )
    temporary.replace(path)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def environment():
    import torch

    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "mps_available": torch.backends.mps.is_available(),
        "cuda_available": torch.cuda.is_available(),
        "versions": {
            name: importlib.metadata.version(name)
            for name in (
                "torch",
                "torchvision",
                "rfdetr",
                "ultralytics",
                "numpy",
                "scipy",
            )
        },
    }


def doctor(_args):
    setup_cache()
    import rfdetr
    from ultralytics import YOLO  # noqa: F401

    info = environment()
    info["device"] = resolve_device("auto")
    info["rf_classes_available"] = all(
        hasattr(rfdetr, value[1]) for value in MODELS.values() if value[0] == "rf"
    )
    info["model_forward_tested"] = False
    print(json.dumps(info, ensure_ascii=False, indent=2))


def prepare(args):
    destination = Path(args.output).resolve()
    if destination.exists():
        raise ValueError(f"Output already exists; choose a new --output: {destination}")
    videos = [Path(p).expanduser().resolve() for p in (args.video or [])]
    provenance = {}
    if not videos and not args.video_dir:
        source_manifest = Path(
            getattr(args, "source_manifest", None)
            or ROOT / "tmp/eval/benchmark-12clips-360x202/manifest.json"
        ).resolve()
        if not source_manifest.is_file():
            raise ValueError("Supply --video, --video-dir or --source-manifest")
        source_data = json.loads(source_manifest.read_text())
        for clip in source_data["clips"]:
            path = Path(clip["path"])
            path = (source_manifest.parent / path).resolve()
            videos.append(path)
            provenance[str(path)] = {
                k: clip[k]
                for k in ("id", "description", "public_source_url", "license", "sha256")
                if k in clip
            }
    if not videos:
        videos = sorted(
            p
            for p in Path(args.video_dir).expanduser().resolve().rglob("*")
            if p.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv", ".webm"}
        )
    if not videos:
        raise ValueError(
            "No local videos found. Supply --video /absolute/path/to/video.mp4"
        )
    (destination / "frames").mkdir(parents=True)
    manifest = {
        "schema": 1,
        "sample_fps": args.sample_fps,
        "max_side": args.max_side,
        "sources": [],
        "frames": [],
    }
    seen = set()
    for video in videos:
        if len(manifest["sources"]) >= args.max_videos:
            break
        digest = sha256(video)
        original = provenance.get(str(video), {})
        if original.get("sha256") and original["sha256"] != digest:
            raise ValueError(f"Source checksum differs from manifest: {video}")
        if digest in seen:
            continue
        seen.add(digest)
        cap = cv2.VideoCapture(str(video))
        try:
            fps = cap.get(cv2.CAP_PROP_FPS)
            if not cap.isOpened() or not np.isfinite(fps) or fps <= 0:
                raise ValueError(f"Cannot read video/FPS: {video}")
            clip = f"clip-{len(manifest['sources']) + 1:02d}"
            saved, index, next_time = 0, 0, 0.0
            while index / fps < args.seconds:
                ok, frame = cap.read()
                if not ok:
                    break
                timestamp = index / fps
                source_index = index
                index += 1
                if timestamp + 1e-8 < next_time:
                    continue
                next_time = (saved + 1) / min(args.sample_fps, fps)
                h, w = frame.shape[:2]
                scale = min(1, args.max_side / max(w, h))
                if scale < 1:
                    frame = cv2.resize(
                        frame,
                        (round(w * scale), round(h * scale)),
                        interpolation=cv2.INTER_AREA,
                    )
                key = f"{clip}-{saved:05d}"
                relative = f"frames/{key}.jpg"
                if not cv2.imwrite(
                    str(destination / relative), frame, [cv2.IMWRITE_JPEG_QUALITY, 95]
                ):
                    raise ValueError(f"Failed to write {relative}")
                manifest["frames"].append(
                    {
                        "id": key,
                        "clip": clip,
                        "image": relative,
                        "timestamp": timestamp,
                        "source_frame_index": source_index,
                        "width": frame.shape[1],
                        "height": frame.shape[0],
                        "sha256": sha256(destination / relative),
                    }
                )
                saved += 1
            if saved == 0:
                raise ValueError(f"No decodable frames: {video}")
            manifest["sources"].append(
                {
                    "id": clip,
                    "path": str(video),
                    "sha256": digest,
                    "source_fps": fps,
                    "frames": saved,
                    "timestamp_basis": "frame_index / container_fps",
                    "provenance": original,
                }
            )
        finally:
            cap.release()
    write_json(destination / "manifest.json", manifest)
    identity = sha256(destination / "manifest.json")
    write_json(
        destination / "annotations.json",
        {
            "schema": 1,
            "dataset_sha256": identity,
            "frames": {},
        },
    )
    template = Path(__file__).with_name("annotate.html").read_text()
    payload = json.dumps(
        {"manifest": manifest, "dataset_sha256": identity}, ensure_ascii=False
    ).replace("<", "\\u003c")
    (destination / "annotate.html").write_text(
        template.replace("__DATASET_JSON__", payload)
    )
    print(
        f"Prepared {len(manifest['frames'])} frames "
        f"from {len(manifest['sources'])} clips"
    )
    print(f"Annotate: {destination / 'annotate.html'}")


def load_dataset(directory, annotations=None):
    directory = Path(directory).resolve()
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest.get("schema") != 1 or not manifest.get("frames"):
        raise ValueError("Invalid/empty dataset manifest")
    path = (
        Path(annotations).resolve() if annotations else directory / "annotations.json"
    )
    labels = json.loads(path.read_text())
    if labels.get("schema") != 1 or labels.get("dataset_sha256") != sha256(
        directory / "manifest.json"
    ):
        raise ValueError("Annotations belong to a different dataset")
    known = {frame["id"] for frame in manifest["frames"]}
    if len(known) != len(manifest["frames"]) or set(labels["frames"]) - known:
        raise ValueError("Duplicate frame IDs or unknown annotation frames")
    for frame in manifest["frames"]:
        image_path = (directory / frame["image"]).resolve()
        if (
            not image_path.is_relative_to(directory)
            or sha256(image_path) != frame["sha256"]
        ):
            raise ValueError(f"Dataset image changed: {frame['id']}")
        item = labels["frames"].get(frame["id"])
        if item and item.get("reviewed") is True:
            polygons_to_masks(item["people"], frame["width"], frame["height"])
    return manifest, labels


def mask_contours(masks):
    return [
        cv2.findContours(mask.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[
            0
        ]
        for mask in masks
    ]


def worker(args):
    setup_cache()
    import torch

    job = json.loads(Path(args.job).read_text())
    output = Path(job["output"])
    output.mkdir(parents=True, exist_ok=True)
    rows, report = [], {"model": job["model"], "status": "failed"}
    try:
        manifest, labels = load_dataset(job["dataset"], job["annotations"])
        report["environment"] = environment()
        device = resolve_device(job["device"])
        report["device"] = device
        torch.manual_seed(job["seed"])
        np.random.seed(job["seed"])
        load_started = time.perf_counter()
        model = Adapter(
            job["model"],
            device,
            ROOT / ".models/seg-benchmark/yolo",
            job["confidence"],
            job["size"],
        )
        report["load_seconds"] = time.perf_counter() - load_started
        report["input_size"] = model.size
        report["weight_sha256"] = sha256(model.weight)
        report["precision"] = "fp32_requested_no_external_autocast"
        first = cv2.imread(str(Path(job["dataset"]) / manifest["frames"][0]["image"]))
        warmup_started = time.perf_counter()
        with torch.inference_mode():
            for _ in range(job["warmup"]):
                model.predict(first)
        synchronize(device)
        report["warmup_seconds"] = time.perf_counter() - warmup_started
        if device.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats(device)
        peak_allocated, peak_rss = 0, 0
        import psutil

        with (output / "frames.jsonl").open("w") as stream:
            for repeat in range(job["repeats"]):
                finishes = {}
                for frame in manifest["frames"]:
                    # Disk decode, scoring and preview output are outside the timer.
                    image = cv2.imread(str(Path(job["dataset"]) / frame["image"]))
                    if image is None or image.shape[:2] != (
                        frame["height"],
                        frame["width"],
                    ):
                        raise ValueError(f"Bad image {frame['id']}")
                    synchronize(device)
                    started = time.perf_counter()
                    with torch.inference_mode():
                        masks = model.predict(image)
                    contours = mask_contours(masks)
                    synchronize(device)
                    elapsed = (time.perf_counter() - started) * 1000
                    finished, delay = replay_delay(
                        frame["timestamp"], elapsed, finishes.get(frame["clip"])
                    )
                    finishes[frame["clip"]] = finished
                    row = {
                        "frame": frame["id"],
                        "clip": frame["clip"],
                        "repeat": repeat,
                        "pipeline_ms": elapsed,
                        "simulated_ready_ms": delay,
                        "predicted_people": len(masks),
                        "quality": None,
                    }
                    annotation = labels["frames"].get(frame["id"])
                    if annotation and annotation.get("reviewed") is True:
                        truth = polygons_to_masks(
                            annotation["people"], frame["width"], frame["height"]
                        )
                        row["quality"] = score_frame(truth, masks, job["iou"])
                    rows.append(row)
                    stream.write(json.dumps(row) + "\n")
                    stream.flush()
                    if repeat == 0:
                        np.savez_compressed(output / f"{frame['id']}.npz", masks=masks)
                        for person in contours:
                            cv2.drawContours(image, person, -1, (80, 240, 80), 2)
                        cv2.putText(
                            image,
                            f"{job['model']} {elapsed:.1f} ms",
                            (10, 25),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (80, 240, 80),
                            2,
                        )
                        if not cv2.imwrite(str(output / f"{frame['id']}.jpg"), image):
                            raise ValueError("Cannot write preview")
                    allocated = (
                        torch.mps.current_allocated_memory()
                        if device == "mps"
                        else torch.cuda.max_memory_allocated(device)
                        if device.startswith("cuda")
                        else 0
                    )
                    peak_allocated = max(peak_allocated, allocated)
                    peak_rss = max(peak_rss, psutil.Process().memory_info().rss)
                    if len(rows) % 20 == 0:
                        print(f"{job['model']}: {len(rows)} frames", flush=True)
        report.update(
            status="ok",
            sampled_device_allocated_bytes=peak_allocated,
            sampled_process_rss_bytes=peak_rss,
            summary=summarize(
                rows,
                job["budget_ms"],
                job["target"],
                len(manifest["frames"]) * job["repeats"],
            ),
        )
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
        traceback.print_exc()
    write_json(output / "result.json", report)
    return 0 if report["status"] == "ok" else 1


def build_report(output, results, manifest):
    columns = [
        "model",
        "status",
        "device",
        "input_size",
        "p50_pipeline_ms",
        "p95_pipeline_ms",
        "p95_simulated_ready_ms",
        "mask_recall",
        "effective_coverage",
        "mean_boundary_f1_all_gt",
        "passes_target",
        "error",
    ]
    flattened = [{**r, **r.get("summary", {})} for r in results]
    with (output / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(flattened)
    eligible = [r for r in flattened if r["status"] == "ok" and r.get("passes_target")]
    eligible.sort(key=lambda r: r["p95_simulated_ready_ms"])
    write_json(
        output / "summary.json",
        {
            "results": results,
            "fastest_qualified_in_this_run": eligible[0]["model"] if eligible else None,
            "scope": "Offline sampled frames; FIFO simulation, not live display",
        },
    )
    table = "".join(
        "<tr>"
        + "".join(f"<td>{html.escape(str(row.get(c, '—')))}</td>" for c in columns)
        + "</tr>"
        for row in flattened
    )
    successful = [r for r in results if r["status"] == "ok"]
    previews = []
    for frame in manifest["frames"]:
        cards = "".join(
            f'<figure><img loading="lazy" src="{r["model"]}/{frame["id"]}.jpg">'
            f"<figcaption>{r['model']}</figcaption></figure>"
            for r in successful
        )
        previews.append(
            f"<details><summary>{frame['id']}</summary><div>{cards}</div></details>"
        )
    body = """<!doctype html><meta charset="utf-8"><title>分割模型对比</title>
    <style>body{font:15px system-ui;margin:32px;background:#101820;color:#edf3fa}
    table{border-collapse:collapse;font-size:13px}
    td,th{padding:9px;border:1px solid #456}
    .scroll{overflow:auto}details{margin:14px 0}details div{display:flex;flex-wrap:wrap}
    figure{margin:8px;width:420px}img{max-width:100%}summary{cursor:pointer}</style>
    <h1>本机分割模型对比</h1>
    <p>pipeline = 内存图像→预处理→模型→CPU mask→轮廓。不含读盘、网络、跟踪或显示。
    simulated_ready = 按样本时间模拟的单路 FIFO 排队结果。非实测端到端延迟。</p>
    <p>没有完整人工标注时不判定 70% 达标。重复次数不增加独立样本量。
    下方是首轮结果。性能 CSV/JSON 包含全部重复。每个模型的日志在同名目录。</p>"""
    (output / "report.html").write_text(
        body
        + '<div class="scroll"><table><tr>'
        + "".join(f"<th>{c}</th>" for c in columns)
        + "</tr>"
        + table
        + "</table></div><h2>逐帧描边对比</h2>"
        + "".join(previews)
    )


def run(args):
    setup_cache()
    device = resolve_device(args.device)
    dataset = Path(args.dataset).resolve()
    manifest, _ = load_dataset(dataset, args.annotations)
    names = list(MODELS) if args.models == "all" else args.models.split(",")
    if len(set(names)) != len(names) or set(names) - MODELS.keys():
        raise ValueError(f"Choose unique models from: {', '.join(MODELS)}")
    output = Path(
        args.output
        or ROOT
        / "outputs/seg-benchmark"
        / datetime.now(timezone.utc).strftime("run-%Y%m%dT%H%M%S%fZ")
    ).resolve()
    output.mkdir(parents=True, exist_ok=False)
    annotations = (
        Path(args.annotations).resolve()
        if args.annotations
        else dataset / "annotations.json"
    )
    settings = {key: value for key, value in vars(args).items() if key != "func"}
    settings.update(
        dataset=str(dataset),
        device=device,
        annotations=str(annotations),
        dataset_sha256=sha256(dataset / "manifest.json"),
        annotations_sha256=sha256(annotations),
    )
    random.Random(args.seed).shuffle(names)
    settings["execution_order"] = names
    settings["tool_sha256"] = {
        path.name: sha256(path) for path in Path(__file__).parent.glob("*.py")
    }
    write_json(output / "settings.json", settings)
    (output / "dependencies.txt").write_text(
        "\n".join(
            sorted(
                f"{dist.metadata['Name']}=={dist.version}"
                for dist in importlib.metadata.distributions()
                if dist.metadata.get("Name")
            )
        )
        + "\n"
    )
    # Freeze labels across subprocesses, even if the user edits the original later.
    write_json(output / "annotations.json", json.loads(annotations.read_text()))
    settings["annotations"] = str(output / "annotations.json")
    results = []
    for name in names:
        model_dir = output / name
        model_dir.mkdir()
        job = {**settings, "model": name, "output": str(model_dir)}
        write_json(model_dir / "job.json", job)
        print(f"Starting {name}; log: {model_dir / 'run.log'}", flush=True)
        with (model_dir / "run.log").open("w") as log:
            child = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "tools.seg_benchmark",
                    "_worker",
                    "--job",
                    str(model_dir / "job.json"),
                ],
                cwd=ROOT,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
        result_path = model_dir / "result.json"
        result = (
            json.loads(result_path.read_text())
            if result_path.exists()
            else {
                "model": name,
                "status": "failed",
                "error": f"Process exited {child.returncode}; see log",
            }
        )
        if child.returncode and result.get("status") == "ok":
            result.update(status="failed", error=f"Process exited {child.returncode}")
        results.append(result)
        build_report(output, results, manifest)
        print(f"{name}: {result['status']}", flush=True)
    print(f"Report: {output / 'report.html'}")
    return int(any(r["status"] != "ok" for r in results))


def positive(value):
    value = float(value)
    if not np.isfinite(value) or value <= 0:
        raise argparse.ArgumentTypeError("Must be finite and positive")
    return value


def positive_int(value):
    if not value.isdecimal() or int(value) < 1:
        raise argparse.ArgumentTypeError("Must be a positive integer")
    return int(value)


def fraction(value):
    value = positive(value)
    if value > 1:
        raise argparse.ArgumentTypeError("Must be in (0, 1]")
    return value


def main():
    parser = argparse.ArgumentParser(
        description="Local RF-DETR / YOLO person segmentation benchmark"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser(
        "doctor", help="Check packages and device without loading weights"
    ).set_defaults(func=doctor)
    p = commands.add_parser(
        "prepare", help="Extract a fixed local corpus and annotation page"
    )
    p.add_argument("--video", action="append", help="Local video path; can repeat")
    p.add_argument("--video-dir", help="Explicit local directory to search for videos")
    p.add_argument("--source-manifest", help="Existing clips manifest with local paths")
    p.add_argument("--output", default=str(DEFAULT_DATA))
    p.add_argument("--max-videos", type=positive_int, default=6)
    p.add_argument("--seconds", type=positive, default=20)
    p.add_argument("--sample-fps", type=positive, default=2)
    p.add_argument("--max-side", type=positive_int, default=960)
    p.set_defaults(func=prepare)
    p = commands.add_parser(
        "run", help="Run actual models (first run downloads official weights)"
    )
    p.add_argument("--dataset", default=str(DEFAULT_DATA))
    p.add_argument("--annotations", help="JSON exported from the local annotation page")
    p.add_argument(
        "--models",
        default="all",
        help="all or comma-separated IDs: " + ",".join(MODELS),
    )
    p.add_argument("--device", choices=["auto", "mps", "cpu", "cuda:0"], default="auto")
    p.add_argument(
        "--size",
        type=positive_int,
        help="Override native input size (divisibility validated)",
    )
    p.add_argument("--confidence", type=fraction, default=0.25)
    p.add_argument("--iou", type=fraction, default=0.5)
    p.add_argument("--target", type=fraction, default=0.70)
    p.add_argument("--budget-ms", type=positive, default=500)
    p.add_argument("--repeats", type=positive_int, default=3)
    p.add_argument("--warmup", type=positive_int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output")
    p.set_defaults(func=run)
    if len(sys.argv) > 1 and sys.argv[1] == "_worker":
        p = commands.add_parser("_worker")
        p.add_argument("--job", required=True)
        p.set_defaults(func=worker)
    args = parser.parse_args()
    try:
        return args.func(args) or 0
    except (ValueError, OSError) as error:
        parser.exit(2, f"Error: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
