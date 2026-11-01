"""Standard-library recovery helpers, also sent over SSH before bootstrap runs."""

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path


def save_control_json(path, data):
    path = Path(path)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(data), encoding="utf-8")
    os.replace(temporary, path)


def stop_control(run, control):
    """Stop only this run, including a managed launch with no PID returned yet."""
    if service := control.get("service"):
        if service != "gv-vlm-" + run.name:
            raise RuntimeError("unexpected managed service")
        result = subprocess.run(
            ["supervisorctl", "stop", service],
            capture_output=True,
            text=True,
            check=False,
            timeout=45,
        )
        # Supervisor returns nonzero for an already stopped or unregistered job.
        # A connection error/timeout is NOT evidence that the worker stopped.
        harmless = {
            f"{service}: ERROR (not running)",
            f"{service}: ERROR (no such process)",
        }
        if result.returncode and result.stdout.strip() not in harmless:
            raise RuntimeError("could not confirm managed job stopped")
        return
    pid = control.get("pid")
    if not pid:
        if control.get("launch_state") == "preparing":
            return
        raise RuntimeError("missing process identity; cannot confirm job stopped")
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        if control.get("process_start_ticks"):
            if fields[19] != control["process_start_ticks"] or fields[0] == "Z":
                return  # The original process exited; never signal a reused PID.
        else:
            raw = Path(f"/proc/{pid}/cmdline").read_bytes()
            if (
                b"tools.vlm_benchmark." not in raw
                or str(run / "app").encode() not in raw
            ):
                raise RuntimeError("refusing to stop an unrelated process")
        if os.getpgid(pid) != pid:
            raise RuntimeError("refusing to stop an unrelated group")
        os.killpg(pid, signal.SIGTERM)
        until = time.monotonic() + 35
        while time.monotonic() < until:
            current = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
            if current[0] == "Z" or current[19] != fields[19]:
                return
            time.sleep(0.1)
        raise RuntimeError("could not confirm process stopped")
    except (FileNotFoundError, ProcessLookupError):
        pass


def recovery_control(run):
    path = run / "control.json"
    if path.exists():
        return json.loads(path.read_text())
    # Older managed launches did not persist control.json until after start.
    # Their unique service name is enough to stop them without guessing a PID.
    plan = json.loads((run / "plan.json").read_text())
    if plan.get("suite") == "modes":
        return {"root": str(run / "app"), "service": "gv-vlm-" + run.name}
    return {}


def record_launch_failure(run, control, reason):
    """Preserve failure evidence even when stopping the partial launch fails."""
    (run / "runtime-key.json").unlink(missing_ok=True)
    data = {
        "state": "failed",
        "job": "launch",
        "reason": reason,
        "cleanup_required": True,
    }
    # Publish cancellation before stopping: a slow registration RPC must not
    # proceed to start the worker after a concurrent recovery request stopped it.
    save_control_json(run / "launch-failure.json", data)
    try:
        stop_control(run, control)
        data["cleanup_required"] = False
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
        data.update(cleanup_required=True, cleanup_error=type(exc).__name__)
    # Separate marker survives a partially started worker updating status.json.
    save_control_json(run / "launch-failure.json", data)
    save_control_json(run / "status.json", data)
    try:
        plan = json.loads((run / "plan.json").read_text())
        sys.path.insert(0, str(run / "app"))
        output = run / "results"
        output.mkdir(exist_ok=True)
        report_reason = reason + (
            "; stop not confirmed; retry --stop" if data["cleanup_required"] else ""
        )
        if plan.get("suite") == "modes":
            from tools.vlm_benchmark.cloud_modes import failure_report

            failure_report(output, plan, "failed", report_reason)
        else:
            from tools.vlm_benchmark.report import render

            summary = output / "summary.json"
            doc = (
                json.loads(summary.read_text())
                if summary.exists()
                else {"plan": plan, "jobs": []}
            )
            doc.update(
                state="failed",
                reason=report_reason,
                engineering_capacity_gate_passed=False,
                conclusion=(
                    "Launch failed; unfinished stages are not acceptance evidence."
                ),
            )
            render(output, doc)
    except Exception as exc:
        # Recovery must still work when installation/report imports failed.
        data["report_error"] = type(exc).__name__
        save_control_json(run / "launch-failure.json", data)
        save_control_json(run / "status.json", data)
    return data
