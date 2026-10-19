"""Own one CPU-only saved-answer phase using exact experiment resource handles.

The source-round resource adapter is reused only for its established Docker
transport and read-only provider inventory. No GPU creation or release is called.
"""
from __future__ import annotations

import argparse
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from saved_streaming_recovery import load, require_finished, save


def resource_adapter(source_round):
    spec = importlib.util.spec_from_file_location("saved_answer_resources", source_round / "control.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def state(control, name, experiment):
    names = control.docker("ps", "-a", "--filter", "name=^/" + name + "$", "--filter",
                           "label=experiment=" + experiment, "--format", "{{.Names}}", timeout=20)
    if names.stdout.strip() != name:
        raise RuntimeError("The exact owned container is not available")
    result = control.docker("inspect", name, "--format",
        '{"running":{{json .State.Running}},"exit_code":{{json .State.ExitCode}},'
        '"oom_killed":{{json .State.OOMKilled}},'
        '"experiment":{{json (index .Config.Labels "experiment")}}}', timeout=20)
    value = json.loads(result.stdout)
    if value.get("experiment") != experiment:
        raise RuntimeError("Container ownership differs from the prepared experiment")
    return value


def source_mounts(plan):
    """Mount only capture directories and local scoring files, not whole workspaces."""
    sources = {str(Path(group["capture"]).parent) for group in plan["groups"]}
    sources.update(group["partial_record_directory"] for group in plan["groups"]
                   if group.get("partial_record_directory"))
    # Seed rows are copied locally; original cumulative ledgers need no runtime mount.
    return sorted(sources)


def driver(root, phase):
    locks = [os.fdopen(int(value), "a") for value in os.environ["ANSWER_LOCK_FDS"].split(",")]
    plan = load(root / "plan.json")
    control = resource_adapter(Path(plan["source_round"]))
    name = root.name + "-" + phase
    stop_requested = False

    def stop(signum, frame):
        nonlocal stop_requested
        stop_requested = True

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    mounts = [(root, False), (root / "checkout", True), (root / "assets", True),
              (root / "seed-ledger", True), (root / "public", True), (root / "indexes", True),
              (root / "plan.json", True), (root / "ready.json", True), (root / "ledger-seed.json", True)]
    mounts.extend((Path(path), True) for path in source_mounts(plan))
    args = ["run", "-d", "--name", name, "--label", "experiment=" + root.name,
        "--network", "none" if phase == "inputs" else control.NETWORK, "--read-only",
        "--no-healthcheck", "--init", "--restart", "no", "--tmpfs", "/tmp:size=256m",
        "--tmpfs", "/opt/goodvision-eval-deps:rw,exec,nosuid,nodev,size=128m",
        "-w", str(root), "-e", "PYTHONDONTWRITEBYTECODE=1"]
    if phase == "answer":
        from dotenv import dotenv_values
        credentials = {**dotenv_values(control.WORKSPACE / "goodvision-harness/.env"),
                       **dotenv_values(Path.home() / ".config/goodvision/openrouter.env")}
        control.ENV["OPENROUTER_API_KEY"] = (credentials.get("OPENROUTER_API_KEY")
                                            or os.environ["OPENROUTER_API_KEY"])
        args += ["-e", "OPENROUTER_API_KEY"]
        mounts.extend((Path(plan["scoring"][key]), True) for key in ("truth", "scorer"))
        mounts.append((root / "output/cohorts", True))
        mounts.append((root / "inputs-completion.json", True))
        mounts.append((root / "inputs-run-outcome.json", True))
    for path, readonly in mounts:
        args += ["-v", str(path) + ":" + str(path) + (":ro" if readonly else "")]
    args += ["--entrypoint", "python", control.IMAGE, "-B", "-u",
        str(root / "checkout/tools/saved_streaming_recovery.py"), phase, "--root", str(root),
        "--dependencies", "/opt/goodvision-eval-deps"]
    terminal, error, stop_sent = None, None, False
    try:
        control.docker(*args)
        consecutive_read_errors = 0
        while True:
            try:
                current = state(control, name, root.name)
                consecutive_read_errors = 0
            except (OSError, RuntimeError, ValueError, subprocess.SubprocessError):
                consecutive_read_errors += 1
                if consecutive_read_errors >= 3:
                    raise RuntimeError("Container state remains unknown; preserve the container for reconciliation") from None
                time.sleep(5)
                continue
            if not current["running"]:
                terminal = current
                break
            if (stop_requested or (root / "cancel-requested.json").exists()) and not stop_sent:
                control.docker("stop", "--time", "30", name, timeout=45)
                stop_sent = True
            time.sleep(5)
    except BaseException as exc:
        error = {"type": type(exc).__name__, "message": "Inspect private controller logs and exact container state"}
        raise
    finally:
        cleanup_error = None
        if terminal is not None:
            try:
                # Logs/results are retained before removing only this exited owned container.
                logs = control.docker("logs", name, timeout=60)
                with (root / (phase + ".log")).open("x", encoding="utf-8") as handle:
                    handle.write(logs.stdout)
                    handle.write(logs.stderr)
                    handle.flush()
                    os.fsync(handle.fileno())
                current = state(control, name, root.name)
                if current["running"]:
                    raise RuntimeError("An exited container became running; do not remove it")
                control.docker("rm", name)
            except BaseException as exc:
                cleanup_error = {"type": type(exc).__name__, "container_preserved": True}
        save(root / (phase + "-run-outcome.json"), {"finished_epoch": time.time(),
            "exit_code": terminal["exit_code"] if terminal else None,
            "oom_killed": terminal["oom_killed"] if terminal else None,
            "completion_saved": (root / (phase + "-completion.json")).is_file(),
            "resource_state": "exited_removed" if terminal and cleanup_error is None else "requires_reconciliation",
            "error": error, "cleanup_error": cleanup_error, "source_commit": plan["source_commit"],
            "gpu_creations": 0})
        for lock in locks:
            lock.close()


def launch(root, phase):
    from dotenv import dotenv_values
    from vastai import VastAI

    locks = []
    try:
        for filename in ("benchmark200-recurrent-active.lock", "ashley-core116-active.lock"):
            handle = (root.parent / filename).open("a")
            locks.append(handle)
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        plan, ready = load(root / "plan.json"), load(root / "ready.json")
        if ready["plan_snapshot"] != plan:
            raise ValueError("The recovery plan changed after preparation")
        source_round, last_answer = Path(plan["source_round"]), Path(plan["last_answer"])
        if source_round.parent != root.parent or last_answer.parent != root.parent:
            raise ValueError("The recovery must use the same task experiment directory")
        require_finished(source_round, last_answer)
        control = resource_adapter(source_round)
        if any((root / (phase + suffix)).exists() for suffix in ("-launch.json", "-controller.log", "-intent.json")):
            raise ValueError("This phase already has a launch or execution record")
        if phase == "answer":
            outcome = load(root / "inputs-run-outcome.json")
            if (outcome.get("exit_code") != 0 or outcome.get("resource_state") != "exited_removed"
                    or load(root / "inputs-completion.json").get("ready_questions", 0) < 1):
                raise ValueError("Finish input preparation, release its container and inspect capacity first")
        for experiment in (root.name, source_round.name, last_answer.name):
            if control.docker("ps", "--filter", "label=experiment=" + experiment,
                              "--format", "{{.Names}}", timeout=20).stdout.strip():
                raise RuntimeError("A relevant experiment already has a running container")
        if control.docker("ps", "-a", "--filter", "label=experiment=" + root.name,
                          "--format", "{{.Names}}", timeout=20).stdout.strip():
            raise RuntimeError("Reconcile the existing recovery container before any launch")
        if control.docker("network", "inspect", control.NETWORK, "--format",
                          "{{json .Containers}}", timeout=20).stdout.strip() != "{}":
            raise RuntimeError("The dedicated task network is not empty")
        credentials = dotenv_values(control.WORKSPACE / "goodvision-harness/.env")
        vast = VastAI(api_key=credentials.get("VAST_API_KEY") or os.environ["VAST_API_KEY"], retry=1)
        vast.client.timeout = (5, 20)
        inventory = control.inventory(vast)
        release = plan["prior_release_record"]
        expected_ids = set(release["release_confirmed_instance_ids"])
        if not expected_ids or release["gpu_label"] != "goodvision-" + source_round.name:
            raise RuntimeError("The previous exact GPU release identity is unavailable")
        if any(row.get("id") in expected_ids or row.get("label") == release["gpu_label"] for row in inventory):
            raise RuntimeError("The exact previous GPU is still present")
        environment = dict(os.environ, ANSWER_LOCK_FDS=",".join(str(handle.fileno()) for handle in locks))
        with (root / (phase + "-controller.log")).open("xb", buffering=0) as log:
            child = subprocess.Popen(["/usr/bin/caffeinate", "-i", sys.executable, "-B", "-u",
                str(Path(__file__).resolve()), phase, "--root", str(root), "--driver"], cwd=root,
                env=environment, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                start_new_session=True, pass_fds=tuple(handle.fileno() for handle in locks))
        record = {"epoch": time.time(), "pid": child.pid, "phase": phase,
            "source_commit": plan["source_commit"], "paid_dispatch_enabled": phase == "answer",
            "gpu_creations": 0, "prior_exact_gpu_absent": True}
        save(root / (phase + "-launch.json"), record)
        print(json.dumps(record), flush=True)
    finally:
        for handle in locks:
            handle.close()


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("phase", choices=("inputs", "answer"))
    cli.add_argument("--root", type=Path, required=True)
    cli.add_argument("--driver", action="store_true", help=argparse.SUPPRESS)
    args = cli.parse_args()
    os.umask(0o077)
    root = args.root.resolve()
    if Path(__file__).resolve() != root / "checkout/tools/saved_streaming_controller.py":
        raise ValueError("Launch from the frozen committed controller")
    (driver if args.driver else launch)(root, args.phase)


if __name__ == "__main__":
    main()
