# ruff: noqa: RUF001
"""Run an immutable suite on one instance, independent of the SSH connection."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import multiprocessing as mp
import os
import platform
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import replace
from datetime import datetime, timezone
from importlib.metadata import distributions
from pathlib import Path
from threading import Event, Thread
from types import SimpleNamespace

from tools.vlm_benchmark.lock import acquire
from tools.vlm_benchmark.plan import (
    availability_gate,
    concurrency_comparison,
    gate,
    ramp_jobs,
    request_budget,
    soak_allowed,
    validate,
)
from tools.vlm_benchmark.report import render, write_json


def configure_event_json(base, mode):
    if mode == "prompt":
        return base
    if mode != "schema" or base.model_api.provider != "openrouter":
        raise ValueError("unsupported event JSON mode/provider")
    from goodvision_harness.vlm_compact_model import event_response_format
    from tools.vlm_multistream_benchmark import LABELS

    parameters = copy.deepcopy(dict(base.inference.request_parameters))
    parameters["response_format"] = event_response_format(LABELS)
    parameters["provider"] = {
        **parameters.get("provider", {}),
        "require_parameters": True,
    }
    return replace(
        base, inference=replace(base.inference, request_parameters=parameters)
    )


def configure_provider_only(base, provider):
    if not provider:
        return base
    if base.model_api.provider != "openrouter" or provider not in {
        "deepinfra",
        "parasail",
        "venice",
    }:
        raise ValueError("unsupported fixed provider")
    parameters = copy.deepcopy(dict(base.inference.request_parameters))
    parameters["provider"] = {
        **parameters.get("provider", {}),
        "only": [provider],
        "allow_fallbacks": False,
    }
    return replace(
        base, inference=replace(base.inference, request_parameters=parameters)
    )


def configure_provider_sort(base, mode):
    if not mode:
        return base
    if (
        mode not in {"throughput", "latency"}
        or base.model_api.provider != "openrouter"
        or ":" in base.inference.model
    ):
        raise ValueError("unsupported provider sort or routing suffix")
    parameters = copy.deepcopy(dict(base.inference.request_parameters))
    parameters["provider"] = {**parameters.get("provider", {}), "sort": mode}
    return replace(
        base, inference=replace(base.inference, request_parameters=parameters)
    )


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_bundle(root):
    manifest = json.loads((root / "bundle-manifest.json").read_text(encoding="utf-8"))
    for name, expected in manifest["files"].items():
        relative = Path(name)
        path = root / relative
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or path.is_symlink()
            or not path.resolve().is_relative_to(root.resolve())
            or not path.is_file()
            or sha256(path) != expected
        ):
            raise ValueError("bundle_integrity_failed")
    return manifest


def machine_info():
    result = {
        "platform": platform.platform(),
        "python": sys.version,
        "cpu_count": os.cpu_count(),
        "hostname": platform.node(),
        "pid": os.getpid(),
        "disk_free_bytes": shutil.disk_usage(Path.cwd()).free,
        "packages": {d.metadata["Name"]: d.version for d in distributions()},
    }
    if hasattr(os, "sched_getaffinity"):
        result["affinity_cpu_count"] = len(os.sched_getaffinity(0))
    for name in (
        "/proc/meminfo",
        "/sys/fs/cgroup/memory.max",
        "/sys/fs/cgroup/cpu.max",
    ):
        try:
            result[name] = Path(name).read_text(encoding="utf-8")[:4000]
        except OSError:
            pass
    if shutil.which("nvidia-smi"):
        try:
            result["gpu_inventory_only"] = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=name,memory.total,driver_version,uuid",
                    "--format=csv,noheader",
                ],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            ).stdout.strip()
        except subprocess.TimeoutExpired:
            result["gpu_inventory_only"] = "unavailable"
    return result


def telemetry(directory, stop):
    start = time.monotonic()
    with (directory / "telemetry.jsonl").open("w") as output:
        while not stop.is_set():
            row = {
                "elapsed": time.monotonic() - start,
                "load_average": list(os.getloadavg()),
                "processes": [],
            }
            try:
                pids = [os.getpid(), *[p.pid for p in mp.active_children() if p.pid]]
                for pid in pids:
                    stat = (
                        Path(f"/proc/{pid}/stat")
                        .read_text(encoding="utf-8")
                        .rsplit(")", 1)[1]
                        .split()
                    )
                    row["processes"].append(
                        {
                            "pid": pid,
                            "cpu_seconds": (int(stat[11]) + int(stat[12]))
                            / os.sysconf("SC_CLK_TCK"),
                            "rss_bytes": int(stat[21]) * os.sysconf("SC_PAGE_SIZE"),
                        }
                    )
            except (OSError, IndexError, ValueError, RuntimeError):
                row["process_telemetry_partial"] = True
            output.write(json.dumps(row) + "\n")
            output.flush()
            stop.wait(2)


def run(root, plan, directory, secret_path=None):
    root, directory = Path(root).resolve(), Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    document = {
        "state": "preflight",
        "plan": plan,
        "jobs": [],
        "started_at": datetime.now(timezone.utc).isoformat(),
        "reason": "",
        "conclusion": "尚未完成；不构成十二路验收",
    }
    status_path = directory.parent / "status.json"
    stop = Event()
    monitor = None
    active_job = "preflight"
    lock_fd = None

    def save(state, reason=""):
        document["state"], document["reason"] = state, reason
        document["concurrency_comparison"] = concurrency_comparison(
            document["jobs"], plan
        )
        render(directory, document)
        write_json(
            status_path,
            {
                "state": state,
                "job": active_job,
                "reason": reason,
                "pid": os.getpid(),
                "completed_jobs": len(document["jobs"]),
                "updated_at": datetime.now(timezone.utc).isoformat(),
            },
        )

    def cancelled(_signum, _frame):
        raise KeyboardInterrupt

    original = signal.signal(signal.SIGTERM, cancelled)
    try:
        validate(plan)
        save("preflight")
        lock_fd = acquire(root)
        # Ignore all inherited route/profile overrides; this plan is the source
        # of truth. Never package a developer's .env or deployment-settings file.
        for name in list(os.environ):
            if name.startswith(("INFERENCE_", "SCENE_", "VLM_")) or name in {
                "HARNESS_INFERENCE_PROFILE",
                "MODEL_API_KEY",
                "AICLUB_API_KEY",
                "OPENROUTER_API_KEY",
                "DASHSCOPE_API_KEY",
            }:
                os.environ.pop(name)
        os.environ["PYTHON_DOTENV_DISABLED"] = "1"
        os.environ["INFERENCE_PROVIDER"] = plan["provider"]
        if plan["model"]:
            os.environ["INFERENCE_MODEL"] = plan["model"]
        if not plan["simulate"]:
            if not secret_path:
                raise ValueError("missing_runtime_key")
            secret = json.loads(Path(secret_path).read_text(encoding="utf-8"))
            Path(secret_path).unlink()
            if not isinstance(secret.get("key"), str) or not secret["key"].strip():
                raise ValueError("missing_runtime_key")
            os.environ["INFERENCE_API_KEY"] = secret.pop("key").strip()
        bundle = verify_bundle(root)
        manifest = {
            "environment": machine_info(),
            "bundle": bundle,
            "plan": plan,
            "request_budget": request_budget(plan),
            "scope": "External VLM API; no local GPU inference. "
            "Predecoded realtime replay; no camera networking or SAM.",
        }
        write_json(directory / "manifest.json", manifest)
        monitor = Thread(target=telemetry, args=(directory, stop), daemon=True)
        monitor.start()
        from tools import vlm_multistream_benchmark as bench

        base = bench.load_config(root / "tools/vlm_benchmark/config.json")
        base = configure_event_json(base, plan.get("event_json_mode", "prompt"))
        base = configure_provider_sort(base, plan.get("provider_sort", ""))
        base = configure_provider_only(base, plan.get("provider_only", ""))
        document["model"] = "SIMULATED" if plan["simulate"] else base.inference.model
        document["event_labels"] = list(bench.LABELS)
        manifest["effective_inference"] = {
            "model": document["model"],
            "provider": plan["provider"],
            "output_mode": "events",
            "event_labels": list(bench.LABELS),
            "request_parameters": dict(base.inference.request_parameters),
            "max_output_tokens": base.runtime.scene_max_tokens,
            "http_timeout_seconds": base.runtime.scene_timeout_seconds,
            "max_retries": 0,
            "max_image_side": base.runtime.scene_max_image_side,
            "jpeg_quality": base.runtime.scene_jpeg_quality,
            "sample_fps": max(2, plan["observation_fps"] * 2),
            "local_concurrency": 4,
            "event_json_mode": plan.get("event_json_mode", "prompt"),
            "parser_policy": "strict_json_or_single_outer_fence_no_content_repair",
        }
        write_json(directory / "manifest.json", manifest)
        framesets = [
            bench.decode(
                root / name,
                max(2, plan["observation_fps"] * 2),
                base.runtime.scene_max_image_side,
                base.runtime.scene_jpeg_quality,
            )
            for name in bundle["videos"]
        ]

        def execute(job):
            nonlocal active_job
            active_job = job["id"]
            save("running")
            print(
                f"Running {active_job}: {job['streams']} streams, {job['duration']} s",
                flush=True,
            )
            target = directory / active_job
            target.mkdir()
            args = SimpleNamespace(
                **{
                    **plan,
                    "real": not plan["simulate"],
                    "output_mode": "events",
                    "simulated_latency": 1,
                    "local_concurrency": 4,
                    "duration": job["duration"],
                    "observation_fps": job.get(
                        "observation_fps", plan["observation_fps"]
                    ),
                    "global_limit": 0
                    if job["variant"] == "unlimited"
                    else plan["global_limit"],
                    "per_stream_limit": job.get(
                        "per_stream_limit", plan["per_stream_limit"]
                    ),
                    "arrival_mode": job.get(
                        "arrival_mode", plan.get("arrival_mode", "aligned")
                    ),
                }
            )
            trial_base = configure_provider_sort(base, job.get("provider_sort", ""))
            stage = bench.run_stage(
                args, trial_base, job["streams"], framesets, target, job["repeat"]
            )
            completed = {
                **job,
                "effective_request_parameters": dict(
                    trial_base.inference.request_parameters
                ),
                "result": stage,
                "gate": (
                    availability_gate(stage)
                    if job["phase"] == "probe"
                    else gate(stage, plan)
                ),
            }
            document["jobs"].append(completed)
            save("running")
            return completed

        probe = execute(
            {
                "id": "probe",
                "streams": 1,
                "duration": 3,
                "observation_fps": 0.25,
                "repeat": 1,
                "variant": "shared",
                "phase": "probe",
            }
        )
        if not probe["gate"]["passed"]:
            save(
                "failed",
                "preflight_model_probe_failed: inspect probe errors/HTTP status",
            )
            return 2
        for job in ramp_jobs(plan):
            completed = execute(job)
            if any(
                row.get("http_status_counts", {}).get("429", 0)
                for row in completed["result"]["per_stream"]
            ):
                save("failed", f"{active_job}: HTTP 429; no further load stages")
                return 2
            if (
                not completed["result"]["input_timing_valid"]
                or completed["result"].get("result_records_missing", 0)
                or not completed["result"]["all_drained"]
            ):
                save("failed", f"{active_job}: input timing/drain invalid")
                return 2
            if any(
                any(r.get("http_status_counts", {}).get(str(c), 0) for c in (401, 403))
                or (
                    r["complete_valid_results"] == 0
                    and not plan.get("compare_concurrency")
                    and not plan.get("compare_arrival")
                    and not plan.get("compare_routing")
                )
                for r in completed["result"]["per_stream"]
            ):
                save(
                    "failed",
                    f"{active_job}: authorization failure or zero usable results",
                )
                return 2
            # The bounded single-stream A/B must observe both settings even if
            # one yields zero usable results. A failed arm is never a pass, but
            # should not prevent measuring the other arm. The wider ramp still
            # stops on zero usable results before increasing paid load.
        if plan["soak_seconds"] and soak_allowed(document["jobs"], plan):
            execute(
                {
                    "id": "soak-12-shared",
                    "streams": 12,
                    "duration": plan["soak_seconds"],
                    "repeat": 1,
                    "variant": "shared",
                    "phase": "soak",
                }
            )
        elif plan["soak_seconds"]:
            document["soak_skipped"] = (
                "12-stream ramp did not meet the frozen engineering gates"
            )
        soak = [j for j in document["jobs"] if j["phase"] == "soak"]
        qualified = (
            soak_allowed(document["jobs"], plan)
            and plan["repeats"] >= 2
            and plan["duration"] >= 30
            and bool(soak)
            and soak[0]["duration"] >= 900
            and soak[0]["gate"]["passed"]
        )
        document["engineering_capacity_gate_passed"] = (
            qualified and not plan["simulate"]
        )
        document["conclusion"] = (
            "模拟完成；未验证真实模型容量或事件准确率"
            if plan["simulate"]
            else "十二路达到本次工程容量门槛；事件准确率仍未验证"
            if qualified
            else "尚未满足完整十二路容量测试条件；请查看逐路结果和跳过原因"
        )
        if plan.get("compare_concurrency") and not plan["simulate"]:
            document["conclusion"] = (
                "单路并发 2/4 对照完成；请比较两轮结果，本次未测试十二路"
            )
        if plan.get("compare_arrival") and not plan["simulate"]:
            document["conclusion"] = (
                "同步/错峰输入对照完成；请逐路比较两轮结果，不构成十二路验收"
            )
        if plan.get("compare_routing") and not plan["simulate"]:
            document["conclusion"] = (
                "吞吐优先/低延迟优先路由对照完成；请逐路比较，不构成十二路验收"
            )
        save("completed", document.get("soak_skipped", ""))
        return 0
    except KeyboardInterrupt:
        save(
            "cancelled", f"{active_job}: cancelled; partial results are not acceptance"
        )
        return 130
    except Exception as exc:
        # Provider exceptions may contain private response bodies or URLs.
        save("failed", f"{active_job}: {type(exc).__name__}; inspect stage artifacts")
        return 2
    finally:
        stop.set()
        if monitor:
            monitor.join(3)
        if secret_path:
            Path(secret_path).unlink(missing_ok=True)
        os.environ.pop("INFERENCE_API_KEY", None)
        if lock_fd is not None:
            os.close(lock_fd)
        signal.signal(signal.SIGTERM, original)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--secret", type=Path)
    args = parser.parse_args()
    raise SystemExit(
        run(
            args.root,
            json.loads(args.plan.read_text(encoding="utf-8")),
            args.output,
            args.secret,
        )
    )


if __name__ == "__main__":
    main()
