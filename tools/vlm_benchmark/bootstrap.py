# ruff: noqa: RUF001
"""Isolated Linux installation, followed by exec into the cloud test process."""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import signal
import subprocess
import sys
from pathlib import Path

from tools.vlm_benchmark.lock import LOCK_ENV, acquire
from tools.vlm_benchmark.report import render, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    run = root.parent
    output, secret = run / "results", run / "runtime-key.json"
    plan = json.loads((run / "plan.json").read_text(encoding="utf-8"))

    def interrupted(_sig, _frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupted)
    lock_fd = None
    try:
        lock_fd = acquire(root)
        write_json(
            run / "status.json",
            {
                "state": "installing",
                "job": "environment",
                "pid": os.getpid(),
                "completed_jobs": 0,
            },
        )
        if platform.system() != "Linux" or platform.machine() != "x86_64":
            raise RuntimeError("requires Linux x86_64")
        if shutil.disk_usage(root).free < 3 * 1024**3:
            raise RuntimeError("requires at least 3 GiB free disk")
        from tools.vlm_benchmark.cloud import verify_bundle

        verify_bundle(root)
        candidates = [
            sys.executable,
            "/venv/main/bin/python",
            shutil.which("python3.12"),
        ]
        python = None
        for candidate in dict.fromkeys(c for c in candidates if c):
            try:
                check = subprocess.run(
                    [
                        candidate,
                        "-c",
                        "import sys; sys.exit(sys.version_info[:2] != (3,12))",
                    ],
                    timeout=10,
                    capture_output=True,
                    check=False,
                )
                if check.returncode == 0:
                    python = candidate
                    break
            except (OSError, subprocess.TimeoutExpired):
                pass
        if python is None:
            raise RuntimeError("Python 3.12 missing; use a template with Python 3.12")
        environment = run / ".venv"
        subprocess.run(
            [python, "-m", "venv", str(environment)], check=True, timeout=120
        )
        executable = str(environment / "bin/python")
        subprocess.run(
            [
                executable,
                "-m",
                "pip",
                "install",
                "-q",
                "-r",
                str(root / "tools/vlm_benchmark/requirements.txt"),
            ],
            check=True,
            timeout=1200,
        )
        subprocess.run(
            [executable, "-m", "pip", "install", "-q", "--no-deps", "-e", str(root)],
            check=True,
            timeout=180,
        )
        subprocess.run([executable, "-m", "pip", "check"], check=True, timeout=60)
        mode_suite = plan.get("suite") == "modes"
        if mode_suite and plan["settings"]["record"]:
            subprocess.run(
                [
                    executable,
                    "-m",
                    "pip",
                    "install",
                    "-q",
                    "-r",
                    str(root / "tools/vlm_benchmark/monitor-requirements.txt"),
                ],
                check=True,
                timeout=600,
            )
            subprocess.run(
                [executable, "-m", "playwright", "install", "--with-deps", "chromium"],
                check=True,
                timeout=1200,
            )
            subprocess.run(
                ["apt-get", "install", "-y", "fonts-noto-cjk"],
                check=True,
                timeout=600,
                env={**os.environ, "DEBIAN_FRONTEND": "noninteractive"},
            )
        os.chdir(root)
        command = [
            executable,
            "-m",
            "tools.vlm_benchmark.cloud_modes"
            if mode_suite
            else "tools.vlm_benchmark.cloud",
            "--root",
            str(root),
            "--plan",
            str(run / "plan.json"),
            "--output",
            str(output),
        ]
        if not plan["simulate"]:
            command += ["--secret", str(secret)]
        os.set_inheritable(lock_fd, True)
        os.environ[LOCK_ENV] = str(lock_fd)
        os.execv(executable, command)
    except (Exception, KeyboardInterrupt) as exc:
        state = "cancelled" if isinstance(exc, KeyboardInterrupt) else "failed"
        reason = str(exc) if isinstance(exc, RuntimeError) else type(exc).__name__
        secret.unlink(missing_ok=True)
        output.mkdir(exist_ok=True)
        if plan.get("suite") == "modes":
            from tools.vlm_benchmark.cloud_modes import failure_report

            failure_report(output, plan, state, reason)
        else:
            render(
                output,
                {
                    "state": state,
                    "plan": plan,
                    "jobs": [],
                    "reason": reason,
                    "conclusion": "环境尚未准备完成；没有容量测试结论",
                },
            )
        write_json(
            run / "status.json",
            {
                "state": state,
                "job": "environment",
                "reason": reason,
                "pid": os.getpid(),
                "completed_jobs": 0,
            },
        )
        print(f"Environment {state}: {reason}", flush=True)
        return 130 if state == "cancelled" else 2
    finally:
        if lock_fd is not None:
            os.close(lock_fd)


if __name__ == "__main__":
    raise SystemExit(main())
