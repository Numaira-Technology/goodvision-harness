"""Run either meeting experiment locally against two SSH-forwarded cloud lanes."""

import argparse
import hashlib
import json
import math
import os
import platform
import re
import shutil
import signal
import socket
import subprocess
import time
from pathlib import Path

import httpx
import psutil

from .report import summarize, write_report

ROOT = Path(__file__).resolve().parents[2]
PLAN_DIR = Path(__file__).resolve().parent


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def load_plan(path):
    plan = json.loads(path.read_text())
    if plan["name"] not in {"test_1_recording", "test_2_dual_stream"}:
        raise ValueError("Unknown experiment")
    delays = plan["delays_seconds"]
    if (
        not delays
        or len(delays) != len(set(delays))
        or not all(
            type(d) in (float, int) and math.isfinite(d) and 0 <= d <= 30
            for d in delays
        )
    ):
        raise ValueError("Delays must be unique finite numbers in 0..30")
    if type(plan["repeats"]) is not int or not 1 <= plan["repeats"] <= 10:
        raise ValueError("Repeats must be within 1..10")
    expected = (
        [("no_recording", 1, False), ("recording", 1, True)]
        if plan["name"] == "test_1_recording"
        else [("single", 1, False), ("dual", 2, False)]
    )
    if [
        (c["name"], c["streams"], c["recording"]) for c in plan["conditions"]
    ] != expected:
        raise ValueError("Conditions must isolate recording or stream count")
    upload_windows = [condition.get("upload_window", 1) for condition in plan["conditions"]]
    if any(type(window) is not int or not 1 <= window <= 4 for window in upload_windows):
        raise ValueError("Upload window must be an integer in 1..4")
    if len(set(upload_windows)) != 1:
        raise ValueError("Both conditions must use the same upload window")
    for key, minimum, maximum in (("ack_timeout_ms", 5000, 30000), ("drain_timeout_ms", 10000, 60000)):
        values = [condition.get(key, minimum) for condition in plan["conditions"]]
        if any(type(value) is not int or not minimum <= value <= maximum for value in values):
            raise ValueError(f"{key} must be an integer in {minimum}..{maximum}")
        if len(set(values)) != 1:
            raise ValueError(f"Both conditions must use the same {key}")
    if plan["coverage_target"] != 0.7:
        raise ValueError("This acceptance test requires the agreed 70% coverage target")
    for key in (
        "coverage_drop_tolerance_pp",
        "sam_p95_increase_tolerance_ratio",
        "display_fps_retention_min",
        "minimum_overlap_ratio",
        "maximum_start_skew_seconds",
    ):
        if key in plan and (
            type(plan[key]) not in (int, float)
            or not math.isfinite(plan[key])
            or plan[key] < 0
        ):
            raise ValueError(f"Invalid tolerance: {key}")
    return plan


def schedule(plan):
    for delay in plan["delays_seconds"]:
        for repeat in range(1, plan["repeats"] + 1):
            conditions = plan["conditions"] if repeat % 2 else plan["conditions"][::-1]
            for condition in conditions:
                yield {"delay": delay, "repeat": repeat, **condition}


def stop(process):
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


def child_memory(process):
    result = []
    for child in process.children(recursive=True):
        try:
            result.append(
                {
                    "pid": child.pid,
                    "name": child.name(),
                    "rss_bytes": child.memory_info().rss,
                }
            )
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return result


def recording_probe(path):
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration:stream=codec_type,nb_frames",
            "-of",
            "json",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    value = json.loads(result.stdout)
    if not any(s.get("codec_type") == "video" for s in value.get("streams", [])):
        raise ValueError("Recording contains no video stream")
    try:
        duration = float(value["format"]["duration"])
    except (KeyError, ValueError):
        # MediaRecorder's live WebM may omit the container duration tag.
        packets = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_packets",
                "-show_entries",
                "packet=pts_time,duration_time",
                "-of",
                "json",
                str(path),
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        timestamps = [
            (float(p["pts_time"]), float(p.get("duration_time", 0)))
            for p in json.loads(packets.stdout)["packets"]
            if "pts_time" in p
        ]
        if not timestamps:
            raise ValueError("Recording has no timed video packets") from None
        duration = max(pts + length for pts, length in timestamps) - min(
            pts for pts, _ in timestamps
        )
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("Recording duration is invalid")
    return {
        "path": str(path),
        "duration_seconds": duration,
        "bytes": path.stat().st_size,
    }


def execute(job, args, client):
    directory = Path(job["output"])
    directory.mkdir()
    save(directory / "job.json", job)
    cloud = []
    for lane, url in enumerate(args.urls, 1):
        response = client.post(
            url + "/benchmark/trial", json={"id": job["id"], "delay": job["delay"]}
        )
        response.raise_for_status()
        data = response.json()
        if data.get("kind") != "sam_cloud_cpu_tests" or data["host"]["lane"] != lane:
            raise ValueError("Wrong cloud test lane")
        if (
            "Linux" not in data["host"]["platform"]
            or data["host"]["hostname"] == socket.gethostname()
        ):
            raise ValueError("Harness must run on the separate cloud Linux host")
        if "4090" not in data["host"]["gpu_at_start"]:
            raise ValueError("Cloud host does not report an RTX 4090")
        cloud.append(data)
    if cloud[0]["host"]["sam_endpoint"] == cloud[1]["host"]["sam_endpoint"]:
        raise ValueError("Two lanes must use different SAM sessions/endpoints")
    save(directory / "cloud-configs.json", cloud)
    env = os.environ.copy()
    modules = ROOT / ".cache/seg-pipeline/node_modules"
    if modules.exists():
        env["NODE_PATH"] = str(modules)
    children, logs = [], []
    recorder = None
    local_samples = []
    recording = None
    try:
        for lane in range(1, job["streams"] + 1):
            output = directory / f"lane-{lane}"
            output.mkdir()
            log = (output / "browser.log").open("w")
            logs.append(log)
            command = [
                args.node,
                str(ROOT / "tools/evaluate_pipeline_latency.cjs"),
                "--url",
                args.urls[lane - 1],
                "--video",
                str(args.video),
                "--video-url",
                args.urls[lane - 1] + "/experiment-video/original",
                "--local-video",
                "true",
                "--output",
                str(output),
                "--headful",
                "true",
                "--screenshots",
                "false",
                "--sync-dir",
                str(directory),
                "--lane",
                str(lane),
                "--window-position",
                "0,30" if lane == 1 else "100,100",
            ]
            if args.chrome:
                command += ["--chrome", args.chrome]
            if job.get("analysis_features"):
                command += [
                    "--analysis-features", ",".join(job["analysis_features"])
                ]
            if job.get("warmup_motion"):
                command += ["--warmup-motion", "true"]
            if job.get("upload_window"):
                command += ["--upload-window", str(job["upload_window"])]
            if job.get("ack_timeout_ms"):
                command += ["--ack-timeout-ms", str(job["ack_timeout_ms"])]
            if job.get("drain_timeout_ms"):
                command += ["--drain-timeout-ms", str(job["drain_timeout_ms"])]
            if job["recording"] and args.recording_method == "canvas":
                command += ["--record-canvas", "true"]
            children.append(
                subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
            )
        deadline = time.monotonic() + 150
        while not all(
            (directory / f"ready-{lane}.json").exists()
            for lane in range(1, job["streams"] + 1)
        ):
            if any(p.poll() is not None for p in children):
                raise RuntimeError(
                    "Browser failed before playback; inspect browser.log"
                )
            if time.monotonic() > deadline:
                raise RuntimeError("Playback preparation timed out")
            time.sleep(0.2)
        duration = max(
            json.loads(p.read_text())["duration"]
            for p in directory.glob("ready-*.json")
        )
        # Both conditions wait the same amount before playback. System recording
        # captures only the rectangle explicitly selected by the user.
        if job["recording"] and args.recording_method == "macos":
            recording = directory / "screen-recording.mov"
            log = (directory / "recording.log").open("w")
            logs.append(log)
            recorder = subprocess.Popen(
                [
                    "/usr/sbin/screencapture",
                    "-v",
                    "-R" + args.record_region,
                    "-V" + str(math.ceil(duration + job["delay"] + 12)),
                    str(recording),
                ],
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        time.sleep(3)
        if recorder and recorder.poll() is not None:
            raise RuntimeError(
                "System recording did not start; check Screen Recording permission"
            )
        save(directory / "start.json", {"start_at": int(time.time() * 1000) + 1000})
        deadline = time.monotonic() + duration + job["delay"] + 220
        client_cpu = psutil.Process()
        while any(p.poll() is None for p in children):
            if time.monotonic() > deadline:
                raise RuntimeError("Trial timed out")
            local_samples.append(
                {
                    "wall_time": time.time(),
                    "system_cpu_percent": psutil.cpu_percent(),
                    "available_ram_bytes": psutil.virtual_memory().available,
                    "swap_used_bytes": psutil.swap_memory().used,
                    "processes": child_memory(client_cpu),
                }
            )
            time.sleep(1)
        if any(p.returncode != 0 for p in children):
            raise RuntimeError("Browser trial failed; inspect per-lane logs")
        if recorder:
            recorder.wait(timeout=30)
            if recorder.returncode:
                raise RuntimeError("System recording failed")
        if job["recording"]:
            recording = recording or directory / "lane-1/canvas-recording.webm"
            save(
                directory / "recording.json",
                {
                    "method": args.recording_method,
                    **recording_probe(recording),
                    "expected_source_seconds": duration,
                    "region": args.record_region
                    if args.recording_method == "macos"
                    else None,
                },
            )
    finally:
        for child in children:
            stop(child)
        if recorder and recorder.poll() is None:
            recorder.send_signal(signal.SIGINT)
            try:
                recorder.wait(timeout=15)
            except subprocess.TimeoutExpired:
                stop(recorder)
        for log in logs:
            log.close()
        save(directory / "local-telemetry.json", local_samples)
        for lane in range(1, job["streams"] + 1):
            try:
                response = client.get(args.urls[lane - 1] + "/benchmark/evidence")
                response.raise_for_status()
                save(directory / f"cloud-evidence-{lane}.json", response.json())
            except httpx.HTTPError:
                pass
    return summarize(directory)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test", choices=("1", "2"), required=True)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--base-port", type=int, default=18094)
    parser.add_argument(
        "--video", type=Path, default=ROOT / "video_samples/sample_video.mp4"
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--recording-method", choices=("macos", "canvas"), default="macos"
    )
    parser.add_argument(
        "--record-region", help="System recording rectangle x,y,width,height"
    )
    parser.add_argument("--chrome")
    parser.add_argument("--node", default=shutil.which("node") or "node")
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="Print schedule; no model, browser or API calls",
    )
    args = parser.parse_args()
    name = "test_1_recording" if args.test == "1" else "test_2_dual_stream"
    plan_path = args.plan or PLAN_DIR / (name + ".json")
    plan = load_plan(plan_path)
    if plan["name"] != name:
        parser.error("Plan and test number disagree")
    jobs = list(schedule(plan))
    if args.plan_only:
        print(json.dumps(jobs, ensure_ascii=False, indent=2))
        return
    if not 1024 <= args.base_port <= 65534:
        parser.error("Choose two adjacent unprivileged forwarded ports")
    if args.test == "1":
        if not shutil.which("ffprobe"):
            parser.error("ffprobe is required to verify the recording file")
        if args.recording_method == "macos":
            if platform.system() != "Darwin":
                parser.error("macos recording requires a Mac")
            if not args.record_region or not re.fullmatch(
                r"-?\d+,-?\d+,[1-9]\d*,[1-9]\d*", args.record_region
            ):
                parser.error(
                    "Specify --record-region x,y,width,height for the SAM window"
                )
    args.video = args.video.resolve(strict=True)
    with args.video.open("rb") as source:
        video_hash = hashlib.file_digest(source, "sha256").hexdigest()
    output = (
        args.output
        or ROOT / "outputs/cloud-cpu-tests" / (name + time.strftime("-%Y%m%dT%H%M%S"))
    ).resolve()
    output.mkdir(parents=True, exist_ok=False)
    args.urls = [f"http://127.0.0.1:{args.base_port + lane}" for lane in (0, 1)]
    manifest = {
        "plan": plan,
        "video_sha256": video_hash,
        "local_host": socket.gethostname(),
        "local_platform": platform.platform(),
        "recording_method": args.recording_method,
        "record_region": args.record_region,
        "source_sha256": {
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [
                *PLAN_DIR.glob("*.py"),
                ROOT / "tools/evaluate_pipeline_latency.cjs",
            ]
        },
        "scope": "cloud_cpu_with_real_sam",
    }
    save(output / "manifest.json", manifest)
    results = []
    with httpx.Client(trust_env=False, timeout=20) as client:
        for index, entry in enumerate(jobs, 1):
            job = entry | {
                "id": f"trial-{index:02d}",
                "video_sha256": video_hash,
                "output": str(output / f"trial-{index:02d}"),
                "test": plan["name"],
                "recording_method": args.recording_method,
            }
            print(
                f"{job['id']}: buffer={job['delay']}s "
                f"{job['name']} repeat={job['repeat']}",
                flush=True,
            )
            try:
                results.append(execute(job, args, client))
            except (
                RuntimeError,
                OSError,
                ValueError,
                subprocess.SubprocessError,
                httpx.HTTPError,
            ) as error:
                results.append(
                    {
                        "job": job,
                        "valid": False,
                        "problems": [f"{type(error).__name__}: {error}"],
                        "lanes": [],
                    }
                )
                write_report(output, manifest, results)
                raise
            write_report(output, manifest, results)
    print(output / "report.html")
    if any(not row["valid"] for row in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
