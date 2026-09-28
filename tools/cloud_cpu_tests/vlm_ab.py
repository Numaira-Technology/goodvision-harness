"""Diagnose whether active VLM work affects single-stream SAM coverage.

Runs the same video on the same cloud instance at a 1 s display buffer in
VLM-on / VLM-off / VLM-off / VLM-on order. VLM-off requests only ``stranger``:
the SAM and face paths remain active, but no safety-event VLM is requested.
This is a diagnostic, not the Test 2 acceptance comparison.
"""
# ruff: noqa: E501

import argparse
import hashlib
import json
import shutil
import socket
import time
from collections import Counter
from pathlib import Path

import httpx

from .run import ROOT, execute, save

ALL_FEATURES = ["fight", "smoking", "fall", "weapon", "stranger"]
OFF_FEATURES = ["stranger"]


def vlm_states(path):
    states = Counter()
    if path.exists():
        for line in path.read_text().splitlines():
            payload = json.loads(line).get("payload", {})
            if payload.get("type") == "vlm_report":
                states[payload.get("state", "unknown")] += 1
    return dict(states)


def number(value, spec, suffix=""):
    return "—" if value is None else format(value, spec) + suffix


def require_valid_results(results):
    if any(not result["valid"] for result in results):
        raise SystemExit(1)


def write_comparison(output, results):
    lines = [
        "# Same-host single-stream VLM diagnostic",
        "",
        "All rounds use the same video, cloud instance, SAM service, and 1 s display buffer.",
        "VLM-off asks only for `stranger`; SAM contours and face processing remain active.",
        "This changes the requested analysis features and is not a formal Test 2 result.",
        "",
        "| Round | Condition | Valid | Contour coverage | SAM append P95 | Cloud source-to-result P95 | VLM reports | Upload backlog |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for result in results:
        job = result["job"]
        lane = result["lanes"][0] if result["lanes"] else None
        states = vlm_states(Path(job["output"]) / "lane-1/vlm.jsonl")
        if lane:
            coverage = number(lane.get("coverage"), ".1%")
            append_p95 = number(
                lane.get("sam_append_seconds", {}).get("p95"), ".3f", " s"
            )
            source_p95 = number(
                lane.get("cloud_source_to_result_seconds", {}).get("p95"),
                ".3f",
                " s",
            )
            lines.append(
                f"| {job['id']} | {job['name']} | {result['valid']} | "
                f"{coverage} | {append_p95} | {source_p95} | "
                f"{states} | {lane.get('analysis_backpressure_warnings', '—')} |"
            )
        else:
            lines.append(
                f"| {job['id']} | {job['name']} | False | — | — | — | {states} | — |"
            )
    lines += [
        "",
        "Interpretation requires valid rounds, completed VLM reports when on,"
        " and zero VLM reports when off. Compare both order-matched pairs; do not"
        " treat an isolated difference as proof of causation.",
    ]
    (output / "comparison.md").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-port", type=int, default=18094)
    parser.add_argument(
        "--video", type=Path, default=ROOT / "video_samples/sample_video.mp4"
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--chrome")
    parser.add_argument("--node", default=shutil.which("node") or "node")
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--upload-window", type=int, choices=(1, 2, 3, 4), default=1)
    args = parser.parse_args()
    if not 1024 <= args.base_port <= 65534:
        parser.error("Choose two adjacent forwarded ports")
    schedule = [
        ("vlm_on", ALL_FEATURES),
        ("vlm_off", OFF_FEATURES),
        ("vlm_off", OFF_FEATURES),
        ("vlm_on", ALL_FEATURES),
    ]
    if args.plan_only:
        print(json.dumps(schedule, ensure_ascii=False, indent=2))
        return
    args.video = args.video.resolve(strict=True)
    args.urls = [f"http://127.0.0.1:{args.base_port + lane}" for lane in (0, 1)]
    args.record_region = None
    args.recording_method = "macos"
    with args.video.open("rb") as video:
        video_hash = hashlib.file_digest(video, "sha256").hexdigest()
    output = (
        args.output
        or ROOT
        / "outputs/cloud-cpu-tests"
        / ("vlm-ab-" + time.strftime("%Y%m%dT%H%M%S"))
    ).resolve()
    output.mkdir(parents=True, exist_ok=False)
    manifest = {
        "scope": "same_host_vlm_diagnostic",
        "video_sha256": video_hash,
        "local_host": socket.gethostname(),
        "buffer_seconds": 1.0,
        "source_sha256": {
            str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (
                Path(__file__).resolve(),
                ROOT / "tools/cloud_cpu_tests/run.py",
                ROOT / "tools/evaluate_pipeline_latency.cjs",
            )
        },
    }
    save(output / "manifest.json", manifest)
    results = []
    reference_invariant = None
    with httpx.Client(trust_env=False, timeout=20) as client:
        configs = []
        for url in args.urls:
            response = client.get(url + "/benchmark/config")
            response.raise_for_status()
            configs.append(response.json())
        if any(c.get("kind") != "sam_cloud_cpu_tests" for c in configs):
            raise RuntimeError("Forwarded ports do not point to the cloud test service")
        if configs[0]["host"]["hostname"] != configs[1]["host"]["hostname"]:
            raise RuntimeError("The two forwarded ports point to different cloud hosts")
        if configs[0]["fixed_main"] != configs[1]["fixed_main"]:
            raise RuntimeError("Cloud lanes have different fixed configurations")
        save(output / "cloud-preflight.json", configs)
        for index, (name, features) in enumerate(schedule, 1):
            job = {
                "id": f"trial-{index:02d}",
                "name": name,
                "analysis_features": features,
                "warmup_motion": True,
                "upload_window": args.upload_window,
                "delay": 1.0,
                "repeat": 1 if index <= 2 else 2,
                "streams": 1,
                "recording": False,
                "recording_method": "macos",
                "video_sha256": video_hash,
                "output": str(output / f"trial-{index:02d}"),
                "test": "same_host_vlm_diagnostic",
            }
            print(f"{job['id']}: {name}, buffer=1.0s", flush=True)
            try:
                result = execute(job, args, client)
            except Exception as error:
                result = {
                    "job": job,
                    "valid": False,
                    "problems": [f"{type(error).__name__}: {error}"],
                    "lanes": [],
                }
                results.append(result)
                save(output / "summary.json", results)
                write_comparison(output, results)
                raise
            browser = json.loads(
                (Path(job["output"]) / "lane-1/browser-report.json").read_text()
            )
            states = vlm_states(Path(job["output"]) / "lane-1/vlm.jsonl")
            if browser.get("analysis_features") != features:
                result["problems"].append("Browser used unexpected analysis features")
            if name == "vlm_on" and not states.get("completed"):
                result["problems"].append("VLM-on produced no completed reports")
            if name == "vlm_off" and states:
                result["problems"].append("VLM-off still produced VLM reports")
            if result["lanes"]:
                invariant = result["lanes"][0]["invariant"]
                if reference_invariant is None:
                    reference_invariant = invariant
                elif invariant != reference_invariant:
                    result["problems"].append(
                        "Cloud host or fixed configuration changed"
                    )
            result["valid"] = result["valid"] and not result["problems"]
            results.append(result)
            save(output / "summary.json", results)
            write_comparison(output, results)
    print(output / "comparison.md")
    require_valid_results(results)


if __name__ == "__main__":
    main()
