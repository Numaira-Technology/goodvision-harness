"""Serialize these benchmark suites on one host, including installation."""

import os
from pathlib import Path

try:
    import fcntl
except ImportError:  # Planning/reporting can still be imported on Windows.
    fcntl = None

LOCK_ENV = "GOODVISION_BENCH_LOCK_FD"


def acquire(root):
    if fcntl is None:
        raise RuntimeError("cloud execution requires POSIX file locking")
    path = Path(root).resolve().parent.parent / ".suite.lock"
    inherited = os.environ.pop(LOCK_ENV, None)
    if inherited is not None:
        fd = int(inherited)
        actual, expected = os.fstat(fd), path.stat()
        if (actual.st_dev, actual.st_ino) != (expected.st_dev, expected.st_ino):
            raise RuntimeError("invalid inherited suite lock")
    else:
        fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            os.close(fd)
            raise RuntimeError(
                "another VLM benchmark is running on this instance"
            ) from None
    os.set_inheritable(fd, False)
    return fd
