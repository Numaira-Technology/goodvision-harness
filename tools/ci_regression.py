"""Bound Linux CI tests below the job deadline and verify complete shard coverage."""

from __future__ import annotations

import argparse
import ctypes
import faulthandler
import hashlib
import json
import math
import os
import platform
import signal
import subprocess
import sys
import time
from pathlib import Path


def shard_for(nodeid: str, count: int) -> int:
    """Keep every module's fixtures and test order together; never use hash()."""
    if count < 1:
        raise ValueError("shard count must be positive")
    module = nodeid.split("::", 1)[0]
    return int.from_bytes(hashlib.sha256(module.encode()).digest()[:8], "big") % count


def _verify_execution(directory, selected):
    stages = {}
    finished = False
    for line in (directory / "progress.jsonl").read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if finished:
            raise ValueError("progress continued after session finish")
        if row["event"] == "result":
            node, stage = row["nodeid"], row["stage"]
            if node not in selected or stage not in {"setup", "call", "teardown"}:
                raise ValueError("unexpected test execution")
            result = stages.setdefault(node, {})
            if stage in result or row["outcome"] not in {"passed", "skipped"}:
                raise ValueError("duplicate or unsuccessful test stage")
            result[stage] = row["outcome"]
        elif row["event"] == "finished":
            if row["exit_code"] != 0:
                raise ValueError("pytest session did not succeed")
            finished = True
    if not finished or set(stages) != set(selected):
        raise ValueError("missing test execution or session finish evidence")
    for outcome in stages.values():
        if "setup" not in outcome or outcome.get("teardown") != "passed":
            raise ValueError("test did not complete setup/teardown")
        if outcome["setup"] == "passed" and "call" not in outcome:
            raise ValueError("test body never completed")


def verify_coverage(directory: Path, count: int, result: str) -> int:
    if result != "success":
        raise ValueError(f"Linux test jobs did not succeed: {result}")
    manifests = list(directory.rglob("collection.json"))
    if len(manifests) != count:
        raise ValueError("missing or extra shard collection manifests")
    universe = None
    covered = set()
    indexes = set()
    for path in manifests:
        data = json.loads(path.read_text(encoding="utf-8"))
        index = data["shard"]
        if data["schema"] != 1 or data["shards"] != count or index in indexes:
            raise ValueError("invalid or duplicate shard identity")
        if type(index) is not int or not 0 <= index < count:
            raise ValueError("invalid shard index")
        indexes.add(index)
        collected, selected = data["collected"], data["selected"]
        if not collected or any(not isinstance(node, str) for node in collected):
            raise ValueError("empty or malformed collection")
        if len(set(collected)) != len(collected):
            raise ValueError("duplicate collected tests")
        if universe is None:
            universe = collected
        if collected != universe:
            raise ValueError("shards collected different tests or order")
        expected = [node for node in collected if shard_for(node, count) == index]
        if selected != expected or covered.intersection(selected):
            raise ValueError("shard selection is incomplete, duplicated or incorrect")
        run = json.loads((path.parent / "run.json").read_text(encoding="utf-8"))
        if run.get("exit_code") != 0 or run.get("timed_out") is not False:
            raise ValueError("shard did not finish successfully")
        _verify_execution(path.parent, selected)
        covered.update(selected)
    if covered != set(universe) or indexes != set(range(count)):
        raise ValueError("not every collected test was covered exactly once")
    print(
        f"Verified {len(covered)} collected tests exactly once across {count} shards."
    )
    return len(covered)


def _snapshot(stream):
    stream.write(f"\nTIME {time.time()}\n")
    stream.flush()
    # No command arguments or environment values: these can contain credentials.
    subprocess.run(
        ["ps", "-e", "--forest", "-o", "pid,ppid,pgid,etimes,pcpu,rss,stat,comm"],
        stdout=stream,
        stderr=subprocess.STDOUT,
        check=False,
        timeout=5,
    )


class RunCancelled(BaseException):
    def __init__(self, signum):
        self.signum = signum


def _become_subreaper():
    # Process-local ownership, not a host security/resource setting. Linux adopts
    # our orphaned test grandchildren here, including services that call setsid.
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(36, 1, 0, 0, 0) != 0:  # PR_SET_CHILD_SUBREAPER
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))


def _owned_processes():
    table = {}
    for path in Path("/proc").glob("[0-9]*/stat"):
        try:
            fields = path.read_text().rsplit(") ", 1)[1].split()
            table[int(path.parent.name)] = (int(fields[1]), fields[19], fields[0])
        except (OSError, ValueError, IndexError):
            continue
    owned = {}
    parents = {os.getpid()}
    while True:
        added = {
            pid: row
            for pid, row in table.items()
            if row[0] in parents and pid not in parents
        }
        if not added:
            return owned
        owned.update(added)
        parents.update(added)


def _signal_owned(processes, signum):
    for pid, (_, started, _) in processes.items():
        try:
            fields = Path(f"/proc/{pid}/stat").read_text().rsplit(") ", 1)[1].split()
            if fields[19] == started:  # Never signal a recycled PID.
                os.kill(pid, signum)
        except (ProcessLookupError, FileNotFoundError):
            pass


def _signal_group(pid, signum):
    try:
        os.killpg(pid, signum)
    except ProcessLookupError:
        pass


def _cleanup(child):
    _signal_group(child.pid, signal.SIGTERM)
    _signal_owned(_owned_processes(), signal.SIGTERM)
    try:
        child.wait(timeout=5)
    except subprocess.TimeoutExpired:
        pass
    _signal_group(child.pid, signal.SIGKILL)
    # Detached grandchildren become our children even if pytest exited first.
    deadline = time.monotonic() + 5
    while True:
        owned = _owned_processes()
        _signal_owned(owned, signal.SIGKILL)
        child.poll()
        for pid in owned:
            if pid != child.pid:
                try:
                    os.waitpid(pid, os.WNOHANG)
                except (ChildProcessError, ProcessLookupError):
                    pass
        if not _owned_processes():
            break
        if time.monotonic() >= deadline:
            raise RuntimeError("test descendants survived cleanup")
        time.sleep(0.05)
    child.wait()


def run_shard(directory: Path, index: int, count: int, seconds: float, tests=()):
    if sys.platform != "linux":
        raise ValueError("bounded process-group runner is for Linux CI")
    if not 0 <= index < count or not math.isfinite(seconds) or not 0 < seconds <= 960:
        raise ValueError("invalid shard or timeout (maximum 960 seconds)")
    _become_subreaper()
    directory = directory.resolve()
    directory.mkdir(parents=True, exist_ok=False)
    command = [
        sys.executable,
        "-u",
        "-m",
        "tools.ci_regression",
        "_pytest",
        str(directory),
        "-vv",
        "-o",
        "faulthandler_timeout=0",
        "--durations=20",
        "-p",
        "tools.ci_pytest",
        "--ci-shard",
        str(index),
        "--ci-shards",
        str(count),
        "--ci-evidence",
        str(directory),
        "--junitxml=" + str(directory / "junit.xml"),
        *tests,
    ]
    record = {
        "command": command,
        "started_unix": time.time(),
        "timeout_seconds": seconds,
        "shard": index,
        "shards": count,
        "timed_out": False,
        "cancelled_signal": None,
        "platform": platform.platform(),
        "python": sys.version,
        "cpu_count": os.cpu_count(),
        "affinity": sorted(os.sched_getaffinity(0)),
    }
    record_path = directory / "run.json"
    record_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    started = time.monotonic()
    with (
        (directory / "stdout.log").open("wb") as stdout,
        (directory / "stderr.log").open("wb") as stderr,
        (directory / "processes.log").open("w") as processes,
    ):
        child = subprocess.Popen(
            command,
            # Durable files are the child's direct sinks. Console backpressure
            # must never block draining or Python's final stdout lock/flush.
            stdout=stdout,
            stderr=stderr,
            start_new_session=True,
        )

        def cancel(signum, frame):
            raise RunCancelled(signum)

        handlers = {
            sig: signal.signal(sig, cancel) for sig in (signal.SIGINT, signal.SIGTERM)
        }
        try:
            while child.poll() is None:
                _snapshot(processes)
                # Do not poke SIGUSR1 on a timer. A dump pauses every thread and
                # a second signal while the handler runs uses the default
                # terminate action, so a finished successful session can die
                # with status -SIGUSR1. The timeout path below still asks for
                # one stack before the hard stop.
                remaining = seconds - (time.monotonic() - started)
                if remaining <= 0:
                    # A failed diagnostic signal is not a completed timeout.
                    # Bootstrap keeps timeout stacks outside pytest capture.
                    try:
                        os.kill(child.pid, signal.SIGUSR1)
                    except ProcessLookupError:
                        pass
                    record["timed_out"] = True
                    time.sleep(1)
                    break
                try:
                    child.wait(timeout=min(10, remaining))
                except subprocess.TimeoutExpired:
                    pass
        except RunCancelled as error:
            record["cancelled_signal"] = error.signum
        finally:
            # A repeated cancellation must not interrupt ownership cleanup.
            for sig in handlers:
                signal.signal(sig, signal.SIG_IGN)
            try:
                _cleanup(child)
            except Exception as error:
                record["cleanup_error"] = str(error)
            _snapshot(processes)
            code = (
                128 + record["cancelled_signal"]
                if record["cancelled_signal"]
                else 124
                if record["timed_out"]
                else child.returncode
            )
            record.update(
                exit_code=code,
                elapsed_seconds=time.monotonic() - started,
                finished_unix=time.time(),
            )
            if "cleanup_error" in record:
                record["exit_code"] = 1
            record_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
            for sig, handler in handlers.items():
                signal.signal(sig, handler)
    return record["exit_code"]


def _session_exit(directory: Path):
    path = directory / "progress.jsonl"
    if not path.exists():
        return None
    finished = None
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("event") == "finished":
            finished = row.get("exit_code")
    return finished


def _install_stack_dump(path: Path) -> int:
    # A raw faulthandler SIGUSR1 handler can traverse changing Python frames
    # without the GIL and segfault (CPython issue 116008). Defer the dump to a
    # Python signal handler, which runs at a safe point with the GIL held.
    stack_fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    dump = faulthandler.dump_traceback

    def requested_dump(_signum, _frame):
        dump(file=stack_fd, all_threads=True)

    try:
        signal.signal(signal.SIGUSR1, requested_dump)
    except BaseException:
        os.close(stack_fd)
        raise
    # Keep the closure and FD until OS exit: pytest can return before Python
    # blocks joining a leaked non-daemon thread during interpreter shutdown.
    # A fully blocked interpreter may not service this handler; the parent
    # still retains progress/process evidence and enforces its hard deadline.
    return stack_fd


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == "_pytest":
        # Register before importing pytest/collecting tests. Its normal capture
        # can otherwise hide a timeout stack forever when the child is killed.
        _install_stack_dump(Path(argv[1]) / "stacks.log")
        import pytest

        # pytest's separate C watchdog also walks changing frames without the
        # GIL. Keep every ordinary dump on the safe periodic/deadline path,
        # even if a config or direct bootstrap caller requests that watchdog.
        args = list(argv[2:])
        end = args.index("--") if "--" in args else len(args)
        args[end:end] = ["-o", "faulthandler_timeout=0"]
        return pytest.main(args)
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run")
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--shard", type=int, required=True)
    run.add_argument("--shards", type=int, required=True)
    run.add_argument("--seconds", type=float, default=960)
    run.add_argument("tests", nargs="*")
    verify = commands.add_parser("verify")
    verify.add_argument("--shards", type=int, required=True)
    verify.add_argument("--result", required=True)
    verify.add_argument("directory", type=Path)
    args = parser.parse_args(argv)
    if args.command == "run":
        return run_shard(args.output, args.shard, args.shards, args.seconds, args.tests)
    verify_coverage(args.directory, args.shards, args.result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
