"""Cloud lifecycle for the same three-mode runner used by local tests."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import signal
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from threading import Event, Thread

from tools.vlm_benchmark.mode_report import render_modes
from tools.vlm_benchmark.report import write_json
from tools.vlm_benchmark.routes import KEY_ENV


def launch_managed(run):
    """A bounded Supervisor job, never auto-restarted into duplicate paid tests."""
    run = Path(run).resolve()
    app = run / "app"
    name = "gv-vlm-" + run.name
    command = shlex.join(
        [sys.executable, "-m", "tools.vlm_benchmark.bootstrap", "--root", str(app)]
    )
    plan_file = run / "plan.json"
    plan = json.loads(plan_file.read_text()) if plan_file.exists() else {}
    dns = plan.get("dns_transport", "system")
    if dns not in {"system", "tcp"}:
        raise ValueError("unsupported DNS transport")
    environment = 'environment=RES_OPTIONS="use-vc"\n' if dns == "tcp" else ""
    config = Path("/etc/supervisor/conf.d") / (name + ".conf")
    config.write_text(
        f"[program:{name}]\ncommand={command}\ndirectory={app}\n"
        f"{environment}"
        "autostart=false\nautorestart=false\nstartsecs=0\nstartretries=0\n"
        "stopasgroup=true\nkillasgroup=true\nstopwaitsecs=30\n"
        f"stdout_logfile={run / 'setup-and-run.log'}\n"
        "stdout_logfile_maxbytes=0\nredirect_stderr=true\n",
        encoding="utf-8",
    )
    for operation in (["reread"], ["update", name], ["start", name]):
        if (run / "launch-failure.json").exists():
            raise RuntimeError("cloud launch was cancelled")
        subprocess.run(["supervisorctl", *operation], check=True, timeout=45)
    pid = int(
        subprocess.check_output(["supervisorctl", "pid", name], text=True, timeout=15)
    )
    if pid <= 0:
        raise RuntimeError("managed cloud job exited at startup; inspect its log")
    if (run / "launch-failure.json").exists():
        raise RuntimeError("cloud launch was cancelled")
    return {"pid": pid, "root": str(app), "service": name}


def make_plan(options):
    # Imported only on the prepared client, or after cloud dependency installation.
    from tools.vlm_benchmark import modes

    args = modes.parser().parse_args([])
    args.backend = getattr(options, "provider", None) or "openrouter"
    for name in (
        "real",
        "model",
        "endpoint",
        "streams",
        "duration",
        "repeats",
        "warmup_seconds",
        "soak_seconds",
        "global_limit",
        "per_stream_limit",
        "request_rate",
        "arrival_mode",
        "record",
        "monitor_port",
        "modes",
        "grounding_format",
        "grounding_concurrency_sweep",
        "parallel_concurrency_sweep",
        "parallel_scheduling",
        "local_concurrency",
        "provider_only",
        "provider_prefer",
        "require_route_match",
        "monitor_layout",
        "http_timeout",
    ):
        setattr(args, name, getattr(options, name, getattr(args, name)))
    args.monitor = True
    args.video = options.video or [Path.home() / "Downloads/output 3.mp4"]
    modes.validate(args)
    settings = vars(args).copy()
    # The server resolves the verified bundle's video names and result directory.
    settings.update(video=[], output=None)
    return {
        "suite": "modes",
        "simulate": not args.real,
        "settings": settings,
        "event_labels": list(modes.LABELS),
        "request_budget": modes.request_budget(args),
        "execution": f"cloud Harness and browser; {args.backend} model inference",
        "dns_transport": getattr(options, "dns_transport", "system"),
    }


def failure_report(output, plan, state, reason):
    """Also usable by the system Python before optional dependencies exist."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    path = output / "summary.json"
    document = (
        json.loads(path.read_text())
        if path.exists()
        else {
            "settings": plan["settings"],
            "event_labels": plan["event_labels"],
            "jobs": [],
        }
    )
    document.update(state=state, reason=reason)
    render_modes(output, document)


def run(root, plan, output, secret):
    from tools.vlm_benchmark import modes
    from tools.vlm_benchmark.cloud import verify_bundle
    from tools.vlm_benchmark.lock import acquire

    root, output = Path(root).resolve(), Path(output).resolve()
    secret = Path(secret) if secret else root.parent / "runtime-key.json"
    status = root.parent / "status.json"
    stop = Event()
    reporter = None
    lock_fd = None

    def progress():
        while not stop.is_set():
            try:
                summary = json.loads((output / "summary.json").read_text())
                data = {
                    "state": "running",
                    "job": summary.get("active_stage", "preflight"),
                    "pid": os.getpid(),
                    "completed_jobs": len(summary["jobs"]),
                }
                monitor = output / "monitor.json"
                if monitor.exists():
                    data["monitor_url"] = json.loads(monitor.read_text())["url"]
                write_json(status, data)
            except (OSError, ValueError):
                pass  # A report may not exist yet while preparing the browser.
            stop.wait(1)

    state, reason = "failed", "cloud runner did not complete"
    try:
        lock_fd = acquire(root)
        bundle = verify_bundle(root)
        args = modes.parser().parse_args([])
        args.__dict__.update(plan["settings"])
        if args.backend not in KEY_ENV or args.headed or plan["simulate"] == args.real:
            raise ValueError("invalid cloud mode plan")
        args.video = [root / name for name in bundle["videos"]]
        args.output = output
        modes.validate(args)
        # Do not inherit credentials from the instance or load an unrelated .env.
        os.environ["PYTHON_DOTENV_DISABLED"] = "1"
        for name in ("INFERENCE_API_KEY", "MODEL_API_KEY", *KEY_ENV.values()):
            os.environ.pop(name, None)
        if args.real and args.record:
            # Fail before paid inference if twelve panes, playback or encoding
            # cannot be captured on this host. Keep this clearly simulated.
            write_json(
                status,
                {"state": "preflight", "job": "recording-check", "pid": os.getpid()},
            )
            check = deepcopy(args)
            check.real, check.duration, check.repeats = False, 3, 1
            check.warmup_seconds, check.soak_seconds = 3, 0
            check.streams = [max(args.streams)]
            check.output = root.parent / "recording-check"
            modes.run(check)
        if args.real:
            key = json.loads(Path(secret).read_text())["key"].strip()
            if not key:
                raise ValueError("missing selected API credential")
            os.environ[KEY_ENV[args.backend]] = key
        if secret:
            Path(secret).unlink(missing_ok=True)
        write_json(status, {"state": "running", "job": "preflight", "pid": os.getpid()})
        reporter = Thread(target=progress, daemon=True)
        reporter.start()
        modes.run(args)
        state, reason = "completed", "cloud measurements finished; inspect report"
    except (Exception, KeyboardInterrupt) as exc:
        state = "cancelled" if isinstance(exc, KeyboardInterrupt) else "failed"
        # Exceptions can contain provider data; preserve detailed sanitized stage
        # diagnostics from modes, but never echo arbitrary exceptions with keys.
        reason = type(exc).__name__
        failure_report(output, plan, state, reason)
    finally:
        stop.set()
        if reporter:
            reporter.join(timeout=5)
        if secret:
            Path(secret).unlink(missing_ok=True)
        for name in KEY_ENV.values():
            os.environ.pop(name, None)
        if lock_fd is not None:
            os.close(lock_fd)
        write_json(status, {"state": state, "reason": reason, "pid": os.getpid()})
    return 0 if state == "completed" else 130 if state == "cancelled" else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--secret", type=Path)
    args = parser.parse_args()

    def interrupt(_signum, _frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupt)
    return run(args.root, json.loads(args.plan.read_text()), args.output, args.secret)


if __name__ == "__main__":
    raise SystemExit(main())
