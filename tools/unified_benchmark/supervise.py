"""Bounded POSIX worker supervision, including controller death via a pipe."""
import argparse
import os
import select
import signal
import subprocess
import time


def cleanup_group(pid, grace):
    try:
        os.killpg(pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    deadline = time.monotonic() + grace
    while time.monotonic() < deadline:
        try:
            os.killpg(pid, 0)
        except ProcessLookupError:
            return
        time.sleep(.1)
    try:
        os.killpg(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def supervise(command, lifetime_fd, lock_fd, timeout, cleanup_timeout):
    stopped = False
    def stop(_signum, _frame):
        nonlocal stopped
        stopped = True
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, stop)
    env = dict(os.environ, GOODVISION_SUPERVISED_WORKER="1")
    child = subprocess.Popen(command, env=env, start_new_session=True,
                             stdin=subprocess.DEVNULL, pass_fds=(lock_fd,))
    deadline = time.monotonic() + timeout
    reason = None
    cleanup_deadline = None
    try:
        while child.poll() is None:
            if stopped or time.monotonic() >= deadline:
                reason = "supervisor stop or deadline"
                break
            ready, _, _ = select.select([lifetime_fd], [], [], .25)
            if ready and os.read(lifetime_fd, 1) == b"":
                reason = "controller disconnected"
                break
        if reason:
            print(reason, flush=True)
            cleanup_deadline = time.monotonic() + cleanup_timeout
            try:
                os.killpg(child.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                child.wait(timeout=cleanup_timeout)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait(timeout=5)
            return 124
        return child.returncode
    finally:
        # Remove native service descendants even if their worker crashed first.
        grace = cleanup_timeout if cleanup_deadline is None else max(0, cleanup_deadline - time.monotonic())
        cleanup_group(child.pid, grace)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--lifetime-fd", type=int, required=True)
    p.add_argument("--lock-fd", type=int, required=True)
    p.add_argument("--timeout", type=float, required=True)
    p.add_argument("--cleanup-timeout", type=float, required=True)
    p.add_argument("command", nargs=argparse.REMAINDER)
    a = p.parse_args()
    return supervise(a.command, a.lifetime_fd, a.lock_fd, a.timeout, a.cleanup_timeout)


if __name__ == "__main__":
    raise SystemExit(main())
