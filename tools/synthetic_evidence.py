"""Bounded, failure-only evidence for explicitly opted-in synthetic load tests."""

from __future__ import annotations

import hashlib
import json
import os
import stat
from contextlib import contextmanager
from pathlib import Path

TESTS = frozenset(
    {
        "tests/test_tracked_grounding.py::test_twelve_sources_through_http_broker_and_spawned_workers",
        "tests/test_tracked_decisions_flow.py::test_twelve_sources_native_decisions_and_person_reports",
    }
)
FILES = (
    "streams-12-repeat-1/summary.json",
    "streams-12-repeat-1/reports.jsonl",
    "streams-12-repeat-1/status.jsonl",
    "detector.jsonl",
)
MAX_FILE_BYTES = 2 * 1024 * 1024
MAX_TOTAL_BYTES = 8 * 1024 * 1024
MAX_FILES = 5  # Four exact disk paths plus the in-memory diagnostic.


def register_synthetic_evidence(item, root, diagnostic):
    if item.nodeid.split("[", 1)[0] in TESTS:
        item._synthetic_evidence = (Path(root), diagnostic)


def _read(root, relative):
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("unsafe relative path")
    # Never follow symlinks, including ancestors of the registered tmp_path.
    target = root / path
    for part in (target, *target.parents):
        if part.is_symlink():
            raise ValueError("symlink")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    if os.open in os.supports_dir_fd:
        # Hold each directory descriptor while walking: a concurrent rename or
        # symlink replacement cannot redirect a subsequent open outside it.
        absolute = target.absolute()
        parent_fd = os.open(absolute.anchor, os.O_RDONLY | os.O_DIRECTORY)
        try:
            for component in absolute.parts[1:-1]:
                child_fd = os.open(
                    component,
                    os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                    dir_fd=parent_fd,
                )
                os.close(parent_fd)
                parent_fd = child_fd
            fd = os.open(absolute.name, flags, dir_fd=parent_fd)
        finally:
            os.close(parent_fd)
    else:
        fd = os.open(target, flags)
    with os.fdopen(fd, "rb") as stream:
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode):
            raise ValueError("not a regular file")
        if info.st_size > MAX_FILE_BYTES:
            raise ValueError("file byte limit")
        data = stream.read(MAX_FILE_BYTES + 1)
        if len(data) > MAX_FILE_BYTES:
            raise ValueError("file byte limit")
        return data


def _valid_json(data, name):
    try:
        text = data.decode("utf-8")
        if name.endswith(".jsonl"):
            for line in text.splitlines():
                json.loads(line)
            return "valid" if not text or text.endswith("\n") else "unterminated_jsonl"
        json.loads(text)
        return "valid"
    except (ValueError, UnicodeError):
        return "invalid_or_truncated_json"
    except RecursionError:
        return "unvalidated_json_depth"


@contextmanager
def _directory_fd(path):
    # The CI plugin is Linux-only. Fail closed on platforms without descriptor
    # traversal rather than claim equivalent protection from path-only checks.
    if os.open not in os.supports_dir_fd or os.mkdir not in os.supports_dir_fd:
        raise OSError("safe evidence directory traversal unavailable")
    path = path.absolute()
    if ".." in path.parts:
        raise ValueError("unsafe evidence path")
    fd = os.open(path.anchor, os.O_RDONLY | os.O_DIRECTORY)
    try:
        for component in path.parts[1:]:
            try:
                os.mkdir(component, dir_fd=fd)
            except FileExistsError:
                pass
            child = os.open(
                component, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd
            )
            os.close(fd)
            fd = child
        yield fd
    finally:
        os.close(fd)


class _RetainedWriteError(OSError):
    """Cleanup failed: charge the entire bounded write, never assume omission."""

    def __init__(self, size, cleanup_error):
        super().__init__("failed write could not be removed")
        self.retained_bytes_upper_bound = size
        self.cleanup_error_type = type(cleanup_error).__name__


def _write(directory_fd, name, data):
    path = Path(name)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("unsafe output path")
    fd = os.dup(directory_fd)
    try:
        for component in path.parts[:-1]:
            try:
                os.mkdir(component, dir_fd=fd)
            except FileExistsError:
                pass
            child = os.open(
                component, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd
            )
            os.close(fd)
            fd = child
        output = os.open(
            path.name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            0o600,
            dir_fd=fd,
        )
        try:
            with os.fdopen(output, "wb") as stream:
                if stream.write(data) != len(data):
                    raise OSError("short evidence write")
        except Exception:
            try:
                os.unlink(path.name, dir_fd=fd)
            except FileNotFoundError:
                pass
            except OSError as cleanup_error:
                raise _RetainedWriteError(len(data), cleanup_error) from cleanup_error
            raise
    finally:
        os.close(fd)


def _bounded_json(value):
    # Bound traversal and individual encoder tokens before iterencode can
    # allocate a large escaped string. This conservative budget may omit a
    # diagnostic that would narrowly fit; omission is explicit in the manifest.
    budget = MAX_FILE_BYTES

    def check(part, depth=0):
        nonlocal budget
        budget -= 2
        if budget < 0 or depth > 64:
            raise ValueError("diagnostic byte limit")
        if isinstance(part, str):
            budget -= len(part) * 6
        elif isinstance(part, dict):
            if len(part) > budget:
                raise ValueError("diagnostic byte limit")
            for key, child in part.items():
                check(key, depth + 1)
                check(child, depth + 1)
        elif isinstance(part, (list, tuple)):
            if len(part) > budget:
                raise ValueError("diagnostic byte limit")
            for child in part:
                check(child, depth + 1)
        elif isinstance(part, int) and part.bit_length() > 1024:
            raise ValueError("diagnostic byte limit")
        if budget < 0:
            raise ValueError("diagnostic byte limit")

    check(value)
    data = bytearray()
    for chunk in json.JSONEncoder(ensure_ascii=True).iterencode(value):
        # Check character length before encoding to bound each encoded chunk.
        if len(chunk) > MAX_FILE_BYTES - len(data) - 1:
            raise ValueError("diagnostic byte limit")
        data.extend(chunk.encode("ascii"))
    data.extend(b"\n")
    return bytes(data)


def retain_synthetic_evidence(item, stage):
    """Caller catches errors: retention can never replace the original failure."""
    registration = getattr(item, "_synthetic_evidence", None)
    if registration is None or item.nodeid.split("[", 1)[0] not in TESTS:
        return None
    if stage not in {"setup", "call", "teardown"}:
        raise ValueError("invalid failure stage")
    root, diagnostic = registration
    directory = Path(item.config.option.ci_evidence) / "synthetic-failures"
    name = hashlib.sha256(item.nodeid.encode()).hexdigest() + "-" + stage
    with _directory_fd(directory) as parent_fd:
        os.mkdir(name, dir_fd=parent_fd)  # Exclusive, never reuse an existing path.
        fd = os.open(
            name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent_fd
        )
        try:
            return _retain(item, stage, root, diagnostic, fd)
        finally:
            os.close(fd)


def _retain(item, stage, root, diagnostic, directory_fd):
    manifest = {
        "schema": 1,
        "nodeid": item.nodeid,
        "stage": stage,
        "scope": "allowlisted synthetic evidence only",
        "files": [],
    }
    total = 0
    for index, name in enumerate((*FILES, "sampling-diagnostic.json")):
        record = {"path": name}
        manifest["files"].append(record)
        try:
            if index >= MAX_FILES:
                raise ValueError("file count limit")
            data = (
                _bounded_json(diagnostic)
                if name == "sampling-diagnostic.json"
                else _read(root, name)
            )
            if len(data) > MAX_FILE_BYTES or total + len(data) > MAX_TOTAL_BYTES:
                raise ValueError("byte limit")
            # Validate before writing, so a parser failure cannot relabel an
            # already-copied payload as omitted or hide its hash/length.
            content_status = _valid_json(data, name)
            digest = hashlib.sha256(data).hexdigest()
            _write(directory_fd, name, data)
            total += len(data)
            record.update(
                status="copied",
                bytes=len(data),
                sha256=digest,
                content_status=content_status,
            )
        except _RetainedWriteError as exc:
            # The partial file may remain. Reserve its full attempted length so
            # subsequent writes cannot exceed the physical payload-byte cap.
            total += exc.retained_bytes_upper_bound
            record.update(
                status="write_failed_cleanup_failed",
                retained_bytes_upper_bound=exc.retained_bytes_upper_bound,
                cleanup_error_type=exc.cleanup_error_type,
            )
        except Exception as exc:
            record.update(
                status="omitted",
                error_type=type(exc).__name__,
                reason=str(exc)
                if isinstance(exc, ValueError)
                else "read or write failed",
            )
    manifest["complete"] = all(
        row.get("status") == "copied" and row.get("content_status") == "valid"
        for row in manifest["files"]
    )
    manifest["copied_bytes"] = sum(row.get("bytes", 0) for row in manifest["files"])
    manifest["retained_bytes_upper_bound"] = total
    _write(directory_fd, "manifest.json", _bounded_json(manifest))
    return manifest
