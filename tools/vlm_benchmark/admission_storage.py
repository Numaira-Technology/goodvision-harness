"""Stage-owned shared SQLite on disk or Linux tmpfs; never per-process :memory:."""

import json
import re
import shutil
import sys
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path


def filesystem_type(path, mountinfo):
    """Resolve the most specific visible mount, including escaped mount paths."""
    path = Path(path).resolve()
    matches = []
    for line in mountinfo.splitlines():
        fields = line.split()
        try:
            separator = fields.index("-")
            mount = Path(
                re.sub(
                    r"\\([0-7]{3})",
                    lambda m: chr(int(m[1], 8)),
                    fields[4],
                )
            )
            if path.is_relative_to(mount):
                matches.append((len(mount.parts), fields[separator + 1]))
        except (ValueError, IndexError):
            continue
    return max(matches, default=(0, None), key=lambda item: item[0])[1]


def memory_root():
    root = Path("/dev/shm").resolve()
    if sys.platform != "linux":
        raise ValueError("memory admission requires Linux tmpfs at /dev/shm")
    try:
        kind = filesystem_type(root, Path("/proc/self/mountinfo").read_text())
    except OSError as exc:
        raise ValueError("cannot verify admission tmpfs mount") from exc
    if not root.is_dir() or kind != "tmpfs":
        raise ValueError("memory admission requires verified tmpfs at /dev/shm")
    return root


@dataclass
class StageAdmissionDatabase:
    path: Path | None
    metadata: dict
    # A forced/unconfirmed worker shutdown must not remove its live database.
    workers_may_be_alive: bool = False


@contextmanager
def admission_database(stage, storage="workspace", *, enabled=True):
    if storage not in {"workspace", "memory"}:
        raise ValueError("invalid admission storage")
    stage = Path(stage).resolve()
    directory = None
    if not enabled:
        path, effective = None, "disabled"
    elif storage == "memory":
        # A private unique directory per stage, shared verbatim by all workers.
        directory = Path(
            tempfile.mkdtemp(prefix="goodvision-admission-", dir=memory_root())
        )
        path, effective = directory / "admission.sqlite", "memory"
    else:
        path, effective = stage / "admission.sqlite", "workspace"
    database = StageAdmissionDatabase(
        path,
        {
            "requested": storage,
            "effective": effective,
            "filesystem": "tmpfs" if directory else None,
            "database_path": str(path) if path else None,
            "ephemeral": bool(directory),
            "cleanup": "pending" if directory else "not_required",
        },
    )
    manifest = stage / "admission-storage.json"
    try:
        manifest.write_text(json.dumps(database.metadata, indent=2) + "\n")
        yield database
    finally:
        try:
            if directory:
                if database.workers_may_be_alive:
                    database.metadata["cleanup"] = (
                        "retained_unconfirmed_worker_shutdown"
                    )
                else:
                    try:
                        shutil.rmtree(directory)
                        database.metadata["cleanup"] = "removed"
                    except OSError:
                        database.metadata["cleanup"] = "failed"
                        raise
        finally:
            manifest.write_text(json.dumps(database.metadata, indent=2) + "\n")
