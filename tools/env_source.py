"""Restore SHA-pinned conda packs on compatible Linux x86_64 GPU hosts.

Only a complete registry pin is trusted. Remote metadata must agree with it.
Archives are verified and safely staged before any downloaded executable runs.
An existing prefix is either a matching completed restore or is left untouched;
operators must move an old/damaged prefix aside before restoring a different pin.
"""

import hashlib
import json
import os
import platform
import re
import shutil
import signal
import subprocess
import sys
import tarfile
import tempfile
import time
from collections import deque
from contextlib import contextmanager
from pathlib import Path, PurePosixPath

from tools import model_source, registry

IDENTITY_FIELDS = ("r2_key", "sha256", "size", "unpack_to", "activate_rel", "compat")
MARKER = ".r2_env.json"


def _log(*a):
    print(f"[env_source {time.strftime('%H:%M:%S')}]", *a, flush=True, file=sys.stderr)


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _marker(unpack_to):
    return Path(unpack_to) / MARKER


def _activate_path(env):
    return str(Path(env["unpack_to"]) / env["activate_rel"])


def _identity(env):
    return {key: env[key] for key in IDENTITY_FIELDS}


def subprocess_env(prefix=None):
    """Do not pass R2/HF/cloud credentials or Python injection vars to pack code."""
    result = {
        key: os.environ[key] for key in ("HOME", "LANG", "TMPDIR") if key in os.environ
    }
    result["PATH"] = (
        str(Path(prefix) / "bin") + os.pathsep if prefix else ""
    ) + os.defpath
    return result


def check_compatibility(env):
    """Check the host with host tools, before executing anything from the archive."""
    if platform.system() != "Linux" or platform.machine() != "x86_64":
        raise RuntimeError("packed GPU environments require Linux x86_64")
    try:
        caps = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=30,
            env=subprocess_env(),
        ).splitlines()
        driver = subprocess.check_output(
            ["nvidia-smi"],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=30,
            env=subprocess_env(),
        )
    except (OSError, subprocess.SubprocessError):
        raise RuntimeError("cannot verify the GPU/driver with nvidia-smi") from None
    supported = registry.GPU_ARCHES[env["compat"]["gpu_family"]]
    if not caps or any(cap.strip() not in supported for cap in caps):
        raise RuntimeError(
            "GPU compute capability does not match env.compat.gpu_family"
        )
    match = re.search(r"CUDA Version:\s*(\d+)\.(\d+)", driver)
    required = tuple(int(part) for part in env["compat"]["cuda"].split(".")[:2])
    if not match or tuple(map(int, match.groups())) < required:
        raise RuntimeError("NVIDIA driver CUDA support is below env.compat.cuda")


def read_remote_manifest(env, *, client=None, bucket=None):
    """Read bounded metadata, close its body, and require agreement with the pin."""
    registry.validate_env(env)
    client = client if client is not None else model_source.r2_client()
    if client is None:
        raise RuntimeError("R2 credentials not configured")
    bucket = bucket or os.environ["R2_BUCKET"]
    try:
        body = client.get_object(Bucket=bucket, Key=env["manifest_key"])["Body"]
        try:
            raw = body.read(65537)
        finally:
            body.close()
    except Exception:
        # SDK error messages may contain endpoint credentials; never echo them.
        raise RuntimeError("cannot read R2 environment manifest") from None
    if len(raw) > 65536:
        raise ValueError("environment manifest exceeds 64 KiB")
    try:
        man = json.loads(raw)
    except (ValueError, UnicodeError):
        raise ValueError("invalid environment manifest JSON") from None
    if not isinstance(man, dict) or man.get("format") != "conda-pack":
        raise ValueError("environment manifest must describe a conda-pack object")
    if type(man.get("size")) is not int or any(
        man.get(key) != env[key] for key in IDENTITY_FIELDS
    ):
        raise ValueError("environment manifest does not match registry pin/metadata")
    return man


def _check_prefix(prefix):
    # Do not follow a symlink in the prefix or its ancestors (including dangling ones).
    for path in (prefix, *prefix.parents):
        if path.is_symlink():
            raise ValueError("environment prefix must not contain symlinks")


def _critical_files(prefix):
    for rel in ("bin/python", "bin/conda-unpack", "bin/activate"):
        file = prefix / rel
        if not file.is_file() or not file.resolve().is_relative_to(prefix.resolve()):
            raise ValueError(f"missing or unsafe conda-pack file: {rel}")
    if not os.access(prefix / "bin/python", os.X_OK):
        raise ValueError("packed bin/python is not executable")


def _already_restored(env):
    registry.validate_env(env)
    prefix = Path(env["unpack_to"])
    try:
        _check_prefix(prefix)
        _critical_files(prefix)
        marker = _marker(prefix)
        if marker.is_symlink():
            return False
        man = json.loads(marker.read_text(encoding="utf-8"))
        return isinstance(man, dict) and man.get("identity") == _identity(env)
    except (OSError, ValueError):
        return False


@contextmanager
def prefix_lock(prefix):
    """Serialize cooperating restore/build processes; never unlink the lock inode."""
    import fcntl  # packed environments are Linux-only; imported lazily for readers

    prefix = Path(prefix)
    _check_prefix(prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    lock = prefix.with_name(f".{prefix.name}.r2-env.lock")
    fd = os.open(lock, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError(
                "another environment restore/build holds the prefix lock"
            ) from None
        _check_prefix(prefix)
        yield
    finally:
        os.close(fd)


def _validate_archive_links(paths):
    """Resolve original symlink targets in the archive before filters rewrite them."""
    for name, member in paths.items():
        if not member.issym():
            continue
        pending = deque(name.split("/"))
        resolved = []
        hops = 0
        while pending:
            part = pending.popleft()
            if part in {"", "."}:
                continue
            if part == "..":
                if not resolved:
                    raise ValueError("archive link escapes the environment")
                resolved.pop()
                continue
            target = paths.get("/".join([*resolved, part]))
            if target is not None and target.issym():
                hops += 1
                if hops > 40:
                    raise ValueError("archive link cycle or excessive indirection")
                pending.extendleft(reversed(target.linkname.split("/")))
            else:
                resolved.append(part)


def _safe_extract(tarball, dest):
    """Validate every header, extract with the data filter, then validate link targets.

    Links may point within the environment, but are never extraction parents.
    This also rejects devices, FIFOs, absolute paths, duplicates and forged markers.
    Requires a patched Python with tarfile.data_filter (3.10.12/3.11.4 or later).
    """
    if not hasattr(tarfile, "data_filter"):
        raise RuntimeError(
            "update Python: safe tar extraction requires tarfile.data_filter"
        )
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=False)
    with tarfile.open(tarball, "r:gz") as tf:
        members = tf.getmembers()
        paths = {}
        for member in members:
            if member.name in {".", "./"} and member.isdir():
                continue
            name = member.name.rstrip("/") if member.isdir() else member.name
            path = registry.relative_path(name)
            if path.parts[0] == MARKER or name in paths:
                raise ValueError(
                    "archive contains a duplicate path or completion marker"
                )
            if not (
                member.isfile() or member.isdir() or member.issym() or member.islnk()
            ):
                raise ValueError("archive contains a special file")
            paths[name] = member
        for name, member in paths.items():
            for parent in PurePosixPath(name).parents:
                if str(parent) in paths and not paths[str(parent)].isdir():
                    raise ValueError("archive entry has a non-directory parent")
            if member.issym() or member.islnk():
                target = member.linkname
                if (
                    not target
                    or target.startswith("/")
                    or any(c in target for c in "\\:\x00\r\n")
                ):
                    raise ValueError("archive contains an unsafe link")
                if member.islnk():
                    registry.relative_path(target)
                    if target not in paths or not paths[target].isfile():
                        raise ValueError(
                            "archive hardlink must target a regular member"
                        )
        _validate_archive_links(paths)
        ordered = sorted(paths.values(), key=lambda m: m.issym() or m.islnk())

        def data_filter(member, destination):
            filtered = tarfile.data_filter(member, destination)
            # chmod on a hardlink would also change the original file's mode.
            if filtered.islnk():
                return filtered.replace(mode=None)
            if filtered.issym():
                # Retain the validated target: normalization can change semantics
                # when a target traverses another symlink before '..'.
                return filtered.replace(linkname=member.linkname)
            return filtered

        tf.extractall(dest, members=ordered, filter=data_filter)
        for name, member in paths.items():
            if member.issym() or member.islnk():
                try:
                    safe = (dest / name).resolve().is_relative_to(dest.resolve())
                except (OSError, RuntimeError):
                    safe = False
                if not safe:
                    raise ValueError(
                        "archive link escapes the environment or forms a cycle"
                    )
    _critical_files(dest)


def _run_unpack(prefix):
    # conda-unpack's shebang may name the build host; always use packed Python.
    subprocess.run(
        [str(prefix / "bin/python"), "-I", str(prefix / "bin/conda-unpack")],
        check=True,
        stdin=subprocess.DEVNULL,
        stdout=sys.stderr,
        env=subprocess_env(prefix),
        timeout=300,
    )


def _restore(env, *, prefer_local, client=None, bucket=None, scratch=None):
    registry.validate_env(env)
    check_compatibility(env)
    prefix = Path(env["unpack_to"])
    with prefix_lock(prefix):
        if prefer_local and _already_restored(env):
            return _activate_path(env), "local"
        if prefix.exists():
            raise FileExistsError(
                f"preserving existing environment: {prefix}; "
                "inspect and move it aside first"
            )
        client = client if client is not None else model_source.r2_client()
        if client is None:
            raise RuntimeError("R2 credentials not configured")
        bucket = bucket or os.environ["R2_BUCKET"]
        read_remote_manifest(env, client=client, bucket=bucket)
        if scratch is not None:
            Path(scratch).mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            prefix="r2-env-download-", dir=scratch
        ) as download:
            tarball = Path(download) / "env.tar.gz"
            try:
                client.download_file(bucket, env["r2_key"], str(tarball))
            except Exception:
                raise RuntimeError("R2 environment download failed") from None
            if tarball.stat().st_size != env["size"]:
                raise ValueError("env tarball size mismatch")
            if _sha256(tarball) != env["sha256"]:
                raise ValueError("env tarball sha256 mismatch")
            with tempfile.TemporaryDirectory(
                prefix=".r2-env-stage-", dir=prefix.parent
            ) as stage:
                staged = Path(stage) / "env"
                _safe_extract(tarball, staged)
                # Reserve without replacing even an empty directory created
                # by a noncooperating writer.
                owned = staged.stat()
                prefix.mkdir()
                reserved = prefix.stat()
                try:
                    staged.rename(prefix)
                    _run_unpack(prefix)  # relocation must run at the final prefix
                    _critical_files(prefix)
                    marker = {"identity": _identity(env), "restored_utc": time.time()}
                    _marker(prefix).write_text(json.dumps(marker), encoding="utf-8")
                except BaseException:
                    # Never remove files from a competing writer if publication
                    # failed before our staging directory became the prefix.
                    if not prefix.is_symlink() and prefix.exists():
                        current = prefix.stat()
                        if (current.st_dev, current.st_ino) == (
                            owned.st_dev,
                            owned.st_ino,
                        ):
                            shutil.rmtree(prefix)
                        elif (current.st_dev, current.st_ino) == (
                            reserved.st_dev,
                            reserved.st_ino,
                        ):
                            try:
                                prefix.rmdir()  # preserves a now nonempty reservation
                            except OSError:
                                pass
                    raise
    return _activate_path(env), "r2"


def pull_env(env, *, client=None, bucket=None, scratch=None):
    """Restore into an absent prefix; refuse to replace an existing environment."""
    return _restore(
        env, prefer_local=False, client=client, bucket=bucket, scratch=scratch
    )[0]


def ensure_env(name, *, prefer_local=True, scratch=None, registry_path=None):
    env = registry.env_of(name, registry_path)
    if env is None:
        raise registry.RegistryError(f"model {name!r} has no packed env registered")
    activate, source = _restore(env, prefer_local=prefer_local, scratch=scratch)
    return {
        "name": name,
        "activate": activate,
        "unpack_to": env["unpack_to"],
        "compat": env["compat"],
        "source": source,
    }


@contextmanager
def interruptible():
    """Let SSH hangup/termination take the normal cleanup path (SIGKILL cannot)."""

    def interrupted(signum, frame):
        raise KeyboardInterrupt

    signals = (signal.SIGTERM, signal.SIGHUP) if os.name == "posix" else ()
    previous = {s: signal.signal(s, interrupted) for s in signals}
    try:
        yield
    finally:
        for s, handler in previous.items():
            signal.signal(s, handler)


def main(argv=None):
    import argparse

    ap = argparse.ArgumentParser(
        description="Restore a pinned conda-pack environment from R2."
    )
    ap.add_argument("--model", required=True)
    ap.add_argument("--no-local", action="store_true", help="require an absent prefix")
    ap.add_argument("--scratch", help="parent directory for private download staging")
    args = ap.parse_args(argv)
    try:
        with interruptible():
            info = ensure_env(
                args.model, prefer_local=not args.no_local, scratch=args.scratch
            )
    except Exception as exc:
        # Do not print SDK exception text/tracebacks containing credentials.
        ap.exit(1, f"environment restore failed ({type(exc).__name__})\n")
    print(json.dumps(info))


if __name__ == "__main__":
    main()
