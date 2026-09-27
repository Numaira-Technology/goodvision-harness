"""Compare Mac and cloud browser decode against one cloud Harness/SAM service."""
# ruff: noqa: RUF001 -- Chinese report prose

import argparse
import hashlib
import json
import os
import platform
import shutil
import socket
import subprocess
import time
from pathlib import Path
from statistics import mean

import httpx
import psutil

from .report import distribution, lane_summary
from .run import stop

ROOT = Path(__file__).resolve().parents[2]


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def browser_cpu_sample(process):
    try:
        family = [process, *process.children(recursive=True)]
        return {
            "at": time.time(),
            "browser_process_cpu_percent": sum(
                child.cpu_percent(interval=None) for child in family
            ),
            "browser_process_rss_bytes": sum(
                child.memory_info().rss for child in family
            ),
            "host_system_cpu_percent": psutil.cpu_percent(interval=None),
        }
    except psutil.Error:
        return None


def run(args):
    expected_system = "Darwin" if args.location == "mac" else "Linux"
    if platform.system() != expected_system:
        raise ValueError(f"{args.location} trial must run on {expected_system}")
    video = args.video.resolve(strict=True)
    with video.open("rb") as source:
        video_sha256 = hashlib.file_digest(source, "sha256").hexdigest()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    lane = output / "lane-1"
    lane.mkdir()
    trial_id = f"decode-{args.location}-{int(time.time() * 1000)}"
    with httpx.Client(trust_env=False, timeout=30) as client:
        response = client.post(
            args.url.rstrip("/") + "/benchmark/trial",
            json={"id": trial_id, "delay": args.delay},
        )
        response.raise_for_status()
        config = response.json()
        host = config["host"]
        if (
            config.get("kind") != "sam_cloud_cpu_tests"
            or host.get("lane") != 1
            or "Linux" not in host.get("platform", "")
            or "4090" not in host.get("gpu_at_start", "")
        ):
            raise ValueError("Expected the first cloud Harness lane on RTX 4090")
        if host.get("video_sha256") != video_sha256:
            raise ValueError(
                "Cloud video and local video differ, or --video was omitted "
                "on the cloud server"
            )
        if args.location == "cloud" and host["hostname"] != socket.gethostname():
            raise ValueError("Cloud browser must run on the Harness host")
        if args.location == "mac" and host["hostname"] == socket.gethostname():
            raise ValueError("Mac browser must run on a separate host")
        if config["fixed_main"]["effective_display_delay_seconds"] != args.delay:
            raise ValueError("Cloud display delay differs from the requested trial")
        save(output / "cloud-config.json", config)
        job = {"id": trial_id, "delay": args.delay, "video_sha256": video_sha256}
        save(output / "job.json", job)
        command = [
            args.node,
            str(ROOT / "tools/evaluate_pipeline_latency.cjs"),
            "--url",
            args.url.rstrip("/"),
            "--video",
            str(video),
            "--video-url",
            args.url.rstrip("/") + "/experiment-video/original",
            "--output",
            str(lane),
            "--headful",
            "false",
            "--screenshots",
            "false",
        ]
        if args.location == "mac":
            command += ["--local-video", "true"]
        if args.chrome:
            command += ["--chrome", args.chrome]
        env = os.environ.copy()
        env["NODE_PATH"] = str(ROOT / ".cache/seg-pipeline/node_modules")
        samples = []
        with (lane / "browser.log").open("w") as log:
            child = subprocess.Popen(
                command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT
            )
            process = psutil.Process(child.pid)
            deadline = time.monotonic() + 360
            try:
                while child.poll() is None:
                    if time.monotonic() > deadline:
                        raise TimeoutError("Cloud decode trial exceeded six minutes")
                    sample = browser_cpu_sample(process)
                    if sample is not None:
                        samples.append(sample)
                    time.sleep(1)
            finally:
                stop(child)
        save(output / "browser-host-telemetry.json", samples)
        response = client.get(args.url.rstrip("/") + "/benchmark/evidence")
        response.raise_for_status()
        save(output / "cloud-evidence-1.json", response.json())
    if child.returncode:
        raise RuntimeError(f"Browser trial failed; inspect {lane / 'browser.log'}")
    metrics = lane_summary(output, 1, job)
    browser = json.loads((lane / "browser-report.json").read_text())
    summary = {
        "location": args.location,
        "controller_host": socket.gethostname(),
        "video_sha256": video_sha256,
        "delay_seconds": args.delay,
        "video_duration_seconds": browser["config"]["video_duration"],
        "browser_version": browser["browser_version"],
        "browser_process_cpu_percent": distribution(
            [row["browser_process_cpu_percent"] for row in samples]
        ),
        "browser_host_system_cpu_percent": distribution(
            [row["host_system_cpu_percent"] for row in samples]
        ),
        "lane": metrics,
    }
    save(output / "summary.json", summary)
    print(output / "summary.json")
    if not metrics["valid"]:
        raise RuntimeError(f"Invalid trial: {metrics['problems']}")


METRICS = (
    ("连续描边率", "coverage", "%"),
    ("严格描边率", "strict_coverage", "%"),
    ("呈现 FPS", "presentation_fps", ""),
    ("云端收帧到结果 P95", "cloud_source_to_result_seconds.p95", "s"),
    ("采集到呈现 P95", "source_to_present_seconds.p95", "s"),
    ("云端 Harness CPU P95", "cloud_process_cpu_percent.p95", "%cpu"),
    ("浏览器进程 CPU P95", "browser_process_cpu_percent.p95", "%cpu"),
)


def metric(summary, key):
    value = summary
    for part in ("lane." + key if not key.startswith("browser_") else key).split("."):
        value = value[part]
    return value


def compare(args):
    if len(args.mac) != len(args.cloud):
        raise ValueError("Provide the same number of Mac and cloud trials")
    pairs = []
    for mac_path, cloud_path in zip(args.mac, args.cloud, strict=True):
        mac = json.loads((mac_path / "summary.json").read_text())
        cloud = json.loads((cloud_path / "summary.json").read_text())
        if mac["location"] != "mac" or cloud["location"] != "cloud":
            raise ValueError("Trial locations do not match --mac and --cloud")
        for key in ("video_sha256", "delay_seconds", "video_duration_seconds"):
            if mac[key] != cloud[key]:
                raise ValueError(f"Different {key} between paired trials")
        if mac["lane"]["invariant"] != cloud["lane"]["invariant"]:
            raise ValueError("Cloud model, server or hardware changed between trials")
        if not mac["lane"]["valid"] or not cloud["lane"]["valid"]:
            raise ValueError("Invalid browser trial; inspect both summaries")
        pairs.append((mac, cloud))
    lines = [
        "# Mac 与云端浏览器解码对照",
        "",
        (
            f"同一视频、同一云端 Harness/SAM、同一 "
            f"{pairs[0][0]['delay_seconds']} 秒显示缓冲；每侧 {len(pairs)} 轮。"
        ),
        (
            "Mac 条件在 Mac 浏览器解码并经 SSH 隧道上传；"
            "云端条件在云端无头浏览器解码、上传并呈现。"
        ),
        (
            "云端条件未将预览画面传回 Mac，因此这是定位解码与上传瓶颈的诊断，"
            "不是最终本机观看延迟。"
        ),
        "",
        "| 指标 | Mac 浏览器 | 云端浏览器 |",
        "| --- | ---: | ---: |",
    ]
    for label, key, unit in METRICS:
        values = [mean(metric(pair[index], key) for pair in pairs) for index in (0, 1)]
        if unit == "%":
            cells = [f"{value:.1%}" for value in values]
        else:
            cells = [f"{value:.2f}{unit}" for value in values]
        lines.append(f"| {label} | {cells[0]} | {cells[1]} |")
    lines.extend(
        [
            "",
            (
                "这些数字是同一条件各轮指标的平均值；请同时检查各轮 summary.json，"
                "不能把一次波动解释为稳定提速。"
            ),
            "浏览器版本和主机 CPU 不同；云端浏览器会与 Harness/SAM 争用云端资源。",
        ]
    )
    args.output.write_text("\n".join(lines) + "\n")
    print(args.output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="action", required=True)
    trial = subparsers.add_parser("run")
    trial.add_argument("--location", choices=("mac", "cloud"), required=True)
    trial.add_argument("--video", type=Path, required=True)
    trial.add_argument("--output", type=Path, required=True)
    trial.add_argument("--url", default="http://127.0.0.1:18094")
    trial.add_argument("--delay", type=float, default=1.0)
    trial.add_argument("--node", default=shutil.which("node") or "node")
    trial.add_argument("--chrome")
    comparison = subparsers.add_parser("compare")
    comparison.add_argument("--mac", type=Path, action="append", required=True)
    comparison.add_argument("--cloud", type=Path, action="append", required=True)
    comparison.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.action == "run":
        run(args)
    else:
        compare(args)


if __name__ == "__main__":
    main()
