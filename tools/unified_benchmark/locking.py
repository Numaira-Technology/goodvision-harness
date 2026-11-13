"""Nonblocking process exclusion, held until the owning file is closed.

POSIX keeps the existing flock contract. Windows uses an exclusive CRT byte
lock at offset zero, including on empty files. Every cooperating writer and
reader must acquire this same lock before touching the protected state.
"""

import importlib
import os
import sys
import threading


def _load_backend(name):
    """A blocked import must fail closed. importlib treats a None entry as unloaded."""
    if name in sys.modules and sys.modules[name] is None:
        raise ImportError(name)
    return importlib.import_module(name)


def lock_exclusive(file):
    """Lock an open file or raise; never continue with an unlocked fallback.

    The caller owns the file's lifetime and must close it on any failure. Do
    not explicitly unlock a POSIX descriptor inherited across fork: closing a
    child's copy must not release the parent's shared open-file-description
    lock. Closing the last owning descriptor releases the OS lock on either
    platform, including after process termination.
    """
    if os.name == "posix":
        backend = _load_backend("fcntl")
        backend.flock(file, backend.LOCK_EX | backend.LOCK_NB)
    elif os.name == "nt":
        backend = _load_backend("msvcrt")
        # a/a+ opens at EOF; locking there would let a second opener lock a
        # different byte after the first append. Do not write a sentinel into
        # JSONL evidence. CRT locking permits a region beyond the current EOF.
        # Same-process LK_NBLCK can sit inside the CRT instead of returning,
        # which stalls the Windows CI shard. Bound the call and fail closed.
        error = []
        done = threading.Event()

        def acquire():
            position = file.tell()
            try:
                file.seek(0)
                backend.locking(file.fileno(), backend.LK_NBLCK, 1)
            except BaseException as exc:
                error.append(exc)
            finally:
                try:
                    file.seek(position)
                finally:
                    done.set()

        threading.Thread(target=acquire, name="benchmark-file-lock", daemon=True).start()
        if not done.wait(1):
            raise OSError("exclusive lock is held")
        if error:
            raise error[0]
    else:
        raise RuntimeError(
            "exclusive benchmark locking is unsupported on this platform"
        )
