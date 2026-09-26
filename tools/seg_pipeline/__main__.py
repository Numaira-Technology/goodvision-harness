"""Run the production browser pipeline against one already-started GPU worker."""

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit

import httpx

from .report import summarize_trial, write_report
from .service import benchmark_config, display_delay, fixed_protocol

ROOT = Path(__file__).resolve().parents[2]
BASELINE = ROOT / "docs/validation/evidence/sam-display-latency-20260922.json"
SOURCE_HASH = "2ac7a0f3271c4c9937b1796bdc43526617d9b1715de9bdf15713d8d3d80dacb9"
MODELS = tuple(
    f"{family}-{size}" for family in ("rfdetr", "yolov8", "yolo26") for size in "nsml"
)


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def child_environment(output):
    # Do not inherit an unrelated production deployment, webhook or spool path.
    prefixes = (
        "HARNESS_",
        "SAM",
        "RICH_",
        "SCENE_",
        "LOCAL_",
        "TRACKING_",
        "INFERENCE_",
        "FACE_",
        "DISPLAY_",
        "GOODVISION_",
        "VLM_",
        "VIDEO_",
        "PERCEPTION_",
        "PERSON_",
        "EVENT_",
        "CLOUD_",
    )
    env = {
        key: value for key, value in os.environ.items() if not key.startswith(prefixes)
    }
    env.update(
        PYTHONPATH=str(ROOT / "src") + os.pathsep + str(ROOT),
        PYTHON_DOTENV_DISABLED="1",
        PYTHONUNBUFFERED="1",
        GOODVISION_SAM_SPOOL_DIRECTORY=str(output / "state/tracking"),
    )
    node_modules = ROOT / ".cache/seg-pipeline/node_modules"
    if node_modules.exists():
        env["NODE_PATH"] = str(node_modules)
    return env


def wait_service(process, url, timeout=60):
    deadline = time.monotonic() + timeout
    with httpx.Client(trust_env=False, timeout=2) as client:
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise RuntimeError("Harness exited; inspect service.log")
            try:
                response = client.get(url + "/benchmark/config")
                response.raise_for_status()
                return
            except httpx.HTTPError:
                time.sleep(0.2)
    raise RuntimeError("Harness startup timed out; inspect service.log")


def stop(process):
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


def trial(job, args, *, preflight=False):
    directory = Path(job["output"])
    directory.mkdir(parents=True)
    save(directory / "job.json", job)
    env = child_environment(directory)
    url = f"http://127.0.0.1:{args.port}"
    with (directory / "service.log").open("w") as log:
        service = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "tools.seg_pipeline.service",
                "--job",
                str(directory / "job.json"),
                "--port",
                str(args.port),
            ],
            cwd=ROOT,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        try:
            wait_service(service, url)
            command = [
                args.node,
                str(ROOT / "tools/evaluate_pipeline_latency.cjs"),
                "--url",
                url,
                "--video",
                job["video"],
                "--video-url",
                url + "/experiment-video/original",
                "--output",
                str(directory),
            ]
            if args.chrome:
                command += ["--chrome", args.chrome]
            if preflight:
                command += ["--preflight", "true"]
            with (directory / "browser.log").open("w") as browser_log:
                subprocess.run(
                    command,
                    cwd=ROOT,
                    env=env,
                    check=True,
                    stdout=browser_log,
                    stderr=subprocess.STDOUT,
                    timeout=360,
                )
            if not preflight:
                with httpx.Client(trust_env=False, timeout=10) as client:
                    response = client.get(job["endpoint"] + "/benchmark/timings")
                    response.raise_for_status()
                    save(directory / "worker-timings.json", response.json())
        finally:
            stop(service)
    return None if preflight else summarize_trial(directory)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODELS, default="yolov8-n")
    parser.add_argument("--endpoint", default="http://127.0.0.1:8092")
    parser.add_argument(
        "--video", type=Path, default=ROOT / "video_samples/sample_video.mp4"
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument(
        "--display-delay", type=display_delay,
        help="Explicit buffer experiment in seconds (0..8); otherwise keep main",
    )
    parser.add_argument("--port", type=int, default=18094)
    parser.add_argument("--node", default=shutil.which("node") or "node")
    parser.add_argument("--chrome")
    parser.add_argument(
        "--face-model-root", type=Path, default=ROOT / ".models/insightface"
    )
    parser.add_argument("--allow-face-download", action="store_true")
    parser.add_argument(
        "--preflight",
        action="store_true",
        help="Check original HEVC playback and contour startup; no analysis/API calls",
    )
    args = parser.parse_args(argv)
    if not 1 <= args.repeats <= 10:
        parser.error("Repeats must be between 1 and 10")
    parsed = urlsplit(args.endpoint)
    if (
        parsed.scheme != "http"
        or parsed.hostname not in {"127.0.0.1", "localhost"}
        or parsed.username
        or parsed.password
        or parsed.path not in {"", "/"}
        or parsed.query
        or parsed.fragment
    ):
        parser.error("Use a loopback worker origin, directly or through SSH forwarding")
    args.endpoint = args.endpoint.rstrip("/")
    if sha256(args.video) != SOURCE_HASH:
        parser.error(
            "Video does not match the recorded SAM baseline; do not transcode it"
        )
    output = (
        args.output
        or ROOT
        / "outputs/seg-pipeline"
        / datetime.now(timezone.utc).strftime(f"{args.model}-%Y%m%dT%H%M%S%fZ")
    ).resolve()
    common = {
        "video": str(args.video.resolve()),
        "video_sha256": SOURCE_HASH,
        "model": args.model,
        "endpoint": args.endpoint,
        "face_model_root": str(args.face_model_root.resolve()),
        "allow_face_download": args.allow_face_download,
        "display_delay_seconds": args.display_delay,
    }
    config = benchmark_config(common | {"output": str(output)})
    protocol = fixed_protocol(config)
    common.update(
        fixed_main=protocol,
        delay=protocol["effective_display_delay_seconds"],
        sample_fps=protocol["tracking"]["max_fps"],
        queue_policy=protocol["tracking"]["video_queue_policy"],
    )
    key_name = config.model_api.api_key_env
    if not args.preflight and not os.environ.get(key_name):
        parser.error(f"Full main pipeline requires {key_name}; preflight needs no key")
    output.mkdir(parents=True, exist_ok=False)
    baseline = json.loads(BASELINE.read_text())
    metadata = None
    if not args.preflight:
        with httpx.Client(trust_env=False, timeout=10) as client:
            response = client.get(args.endpoint + "/benchmark/info")
            response.raise_for_status()
            metadata = response.json()
        if (
            metadata.get("model") != args.model
            or metadata.get("ready") is not True
            or "4090" not in metadata.get("gpu", "")
            or not metadata.get("weight_sha256")
        ):
            raise RuntimeError(
                "Ready RTX 4090 worker/model/weight identity does not match"
            )
    schedule = list(range(1, args.repeats + 1))
    manifest = {
        "model": args.model,
        "repeats": args.repeats,
        "fixed_main": protocol,
        "config_sha256": sha256(ROOT / "config.yaml"),
        "schedule": schedule,
        "worker": metadata,
        "video_sha256": SOURCE_HASH,
        "client_platform": platform.platform(),
        "client_python": sys.version,
        "baseline_sha256": sha256(BASELINE),
        "baseline_ref": "ccff0525abcfa51e8cdb9f56e602cfda909314f3",
        "source_sha256": {
            str(p.relative_to(ROOT)): sha256(p)
            for p in sorted(
                [
                    *Path(ROOT / "src/goodvision_harness").glob("*.py"),
                    *Path(ROOT / "src/goodvision_harness").glob("*.js"),
                    *Path(ROOT / "tools/seg_pipeline").glob("*.py"),
                    ROOT / "tools/evaluate_pipeline_latency.cjs",
                ]
            )
        },
        "scope": "full_pipeline",
        "sam_reference_is_historical": True,
        "display_delay_override_seconds": args.display_delay,
    }
    save(output / "manifest.json", manifest)
    save(output / "sam-historical-baseline.json", baseline)
    if args.preflight:
        trial(
            common | {"output": str(output / "preflight"), "repeat": 0},
            args,
            preflight=True,
        )
        print(f"Original-video and contour-startup preflight passed: {output}")
        return
    results = []
    for repeat in schedule:
        directory = output / f"repeat-{repeat}"
        print(
            f"{args.model}: display buffer {common['delay']:g} s, repeat {repeat}",
            flush=True,
        )
        try:
            result = trial(
                common | {"output": str(directory), "repeat": repeat},
                args,
            )
        except (
            RuntimeError,
            subprocess.SubprocessError,
            OSError,
            ValueError,
            httpx.HTTPError,
        ) as error:
            result = {
                "model": args.model,
                "configured_delay_seconds": common["delay"],
                "repeat": repeat,
                "valid": False,
                "coverage_pass": False,
                "problems": [f"Trial failed ({type(error).__name__}); inspect logs"],
                "path": str(directory),
            }
        results.append(result)
        write_report(output, manifest, results, baseline)
        if not result["valid"]:
            print(
                f"INVALID trial; stopping repeats. Inspect {directory / 'browser.log'} "
                f"and {directory / 'service.log'}",
                file=sys.stderr,
                flush=True,
            )
            break
    print(output / "report.html")
    if any(not row["valid"] for row in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
