"""Restore SHA-pinned conda packs on compatible Linux x86_64 GPU hosts.

Only a complete registry pin is trusted. Remote metadata must agree with it.
Archives are verified and safely staged before any downloaded executable runs.
An existing prefix is either a matching completed restore or is left untouched;
operators must move an old/damaged prefix aside before restoring a different pin.
"""

import ctypes
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


def packed_python(env):
    """Absolute packed interpreter. Never PATH python, conda activate, or user site.

    Fail-closed: missing, non-executable, or escaped ``bin/python`` aborts.
    """
    registry.validate_env(env)
    prefix = Path(env["unpack_to"])
    _check_prefix(prefix)
    python = prefix / "bin/python"
    if not python.is_file() or not os.access(python, os.X_OK):
        raise RuntimeError(
            f"packed interpreter missing or not executable: {python}; "
            "refusing PATH python/python3"
        )
    return str(python)


def eval_subprocess_env(prefix=None):
    """Child env for eval/serve: packed bin first, no user site or PYTHONPATH."""
    result = subprocess_env(prefix)
    result["PYTHONNOUSERSITE"] = "1"
    return result


_PREFIX_PROBE = "\n".join(
    [
        "import os, sys, site",
        "want = os.path.realpath(os.environ['GOODVISION_ENV_PREFIX'])",
        "got = os.path.realpath(sys.prefix)",
        "if got != want:",
        "    raise SystemExit(f'sys.prefix {got!r} != restored env {want!r}')",
        "exe = os.path.realpath(sys.executable)",
        "if not (exe == os.path.realpath(os.path.join(want, 'bin', 'python'))",
        "        or exe.startswith(want + os.sep)):",
        "    raise SystemExit(f'sys.executable {exe!r} is outside {want!r}')",
        "if site.ENABLE_USER_SITE:",
        "    raise SystemExit('user site must be disabled (PYTHONNOUSERSITE)')",
    ]
)


_LAUNCH_WRAPPER = (
    _PREFIX_PROBE
    + "\n"
    + """
def _launch():
    import sys
    # Load launcher dependencies without letting the caller's cwd shadow stdlib.
    path0 = sys.path.pop(0)
    try:
        import builtins, importlib.machinery, io, os, pkgutil, runpy
    finally:
        sys.path.insert(0, path0)
    argv = sys.argv[1:]
    if not argv:
        raise SystemExit('missing eval/serve arguments')
    if argv[0] in ('-m', '-c') and len(argv) < 2:
        raise SystemExit('missing module name or -c command')

    # Keep the real __main__ for pickling, spawn and atexit callbacks. Public
    # run_module/run_path use a temporary module and restore it on return.
    namespace = sys.modules['__main__'].__dict__
    namespace.clear()
    namespace.update(__name__='__main__', __doc__=None, __package__=None,
                     __spec__=None, __annotations__={}, __builtins__=builtins)
    if argv[0] == '-m':
        sys.argv = argv[1:]
        sys.path[0] = os.getcwd()
        # This is CPython's own -m entrypoint, including package resolution.
        runpy._run_module_as_main(argv[1])
    elif argv[0] == '-c':
        sys.argv = ['-c', *argv[2:]]
        namespace['__loader__'] = importlib.machinery.BuiltinImporter
        exec(compile(argv[1], '<string>', 'exec'), namespace)
    else:
        sys.argv = argv
        filename = (argv[0] if os.path.isabs(argv[0])
                    else os.path.join(os.getcwd(), argv[0]))
        if pkgutil.get_importer(argv[0]) is not None:
            sys.path[0] = argv[0]
            runpy._run_module_as_main('__main__', alter_argv=False)
        else:
            sys.path[0] = os.path.dirname(os.path.realpath(filename))
            with io.open_code(filename) as source:
                code = pkgutil.read_code(source)
            loader = importlib.machinery.SourcelessFileLoader
            if code is None:
                with io.open_code(filename) as source:
                    code = compile(source.read(), filename, 'exec')
                loader = importlib.machinery.SourceFileLoader
            namespace.update(__file__=filename, __cached__=None,
                             __loader__=loader('__main__', filename))
            try:
                exec(code, namespace)
            finally:
                # CPython removes these file-only attributes before shutdown.
                namespace.pop('__file__', None)
                namespace.pop('__cached__', None)

_launch()
"""
)


def verify_packed_interpreter(env, *, runner=subprocess.run):
    """Run the packed interpreter with -s and require sys.prefix == unpack_to."""
    python = packed_python(env)
    prefix = str(Path(env["unpack_to"]).resolve())
    child = eval_subprocess_env(prefix)
    child["GOODVISION_ENV_PREFIX"] = prefix
    proc = runner(
        [python, "-s", "-c", _PREFIX_PROBE],
        env=child,
        capture_output=True,
        text=True,
        timeout=30,
        stdin=subprocess.DEVNULL,
    )
    if proc.returncode:
        detail = (proc.stderr or proc.stdout or "").strip() or f"exit {proc.returncode}"
        raise RuntimeError(
            f"packed interpreter {python} did not match restored env {prefix}: {detail}"
        )
    return python


def launch_argv(env, *args):
    """Eval/serve argv whose argv[0] is the restored env's absolute python.

    Isolation is PYTHONNOUSERSITE=1 plus python -s. The launched process itself
    checks sys.prefix before running the eval/serve module. Callers must not
    prepend ``python``/``python3`` from PATH or ``source`` the activate script.
    """
    python = verify_packed_interpreter(env)
    child = eval_subprocess_env(env["unpack_to"])
    child["GOODVISION_ENV_PREFIX"] = str(Path(env["unpack_to"]).resolve())
    return [python, "-s", "-c", _LAUNCH_WRAPPER, *args], child


def _identity(env):
    return {key: env[key] for key in IDENTITY_FIELDS}


def restore_identity(env):
    """Bind the SSH acknowledgement to both the artifact and selected manifest.

    Completion markers keep their existing format; the handoff additionally
    identifies the manifest location so a changed uploaded registry cannot be
    acknowledged merely because it restores into the same absolute prefix.
    """
    registry.validate_env(env)
    return {**_identity(env), "manifest_key": env["manifest_key"]}


def _packed_bin(prefix):
    """Keep a registry POSIX prefix intact. pathlib rewrites ``/opt/...`` on Windows."""
    text = str(prefix)
    if text.startswith("/") and not text.startswith("//"):
        return text.rstrip("/") + "/bin"
    return str(Path(text) / "bin")


def subprocess_env(prefix=None):
    """Do not pass R2/HF/cloud credentials or Python injection vars to pack code."""
    result = {
        key: os.environ[key] for key in ("HOME", "LANG", "TMPDIR") if key in os.environ
    }
    result["PATH"] = (_packed_bin(prefix) + os.pathsep if prefix else "") + os.defpath
    result["PYTHONNOUSERSITE"] = "1"
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
    """Serialize cooperating restore/build processes; never unlink the lock inode.

    Uses the shared nonblocking lock: flock on POSIX, a one-byte CRT lock on
    Windows. There is no unlocked fallback.
    """
    from tools.unified_benchmark.locking import lock_exclusive

    prefix = Path(prefix)
    _check_prefix(prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    lock = prefix.with_name(f".{prefix.name}.r2-env.lock")
    flags = os.O_CREAT | os.O_RDWR
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd = os.open(lock, flags, 0o600)
    file = os.fdopen(fd, "r+b")
    try:
        try:
            lock_exclusive(file)
        except OSError as exc:
            # flock raises BlockingIOError. CRT locking raises OSError
            # (lock violation / deadlock) and must not be treated as success.
            if not _lock_contended(exc):
                raise
            raise RuntimeError(
                "another environment restore/build holds the prefix lock"
            ) from None
        _check_prefix(prefix)
        yield
    finally:
        file.close()


def _lock_contended(exc):
    if isinstance(exc, BlockingIOError):
        return True
    winerror = getattr(exc, "winerror", None)
    if winerror in (33, 36):  # ERROR_LOCK_VIOLATION, ERROR_LOCK_FAILED
        return True
    return exc.errno in (11, 13, 35, 36)


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


# conda-pack's conda-unpack imports its sibling bin/conda_unpack_progress.py.
# Python 3.11+ -I also implies -P (no script directory on sys.path), so run
# the script via runpy with only the archive's own, already verified bin/
# directory added; -I still excludes PYTHONPATH, user site and env vars.
_UNPACK_PROGRAM = (
    "import os, runpy, sys; script = sys.argv[1]; sys.argv = [script]; "
    "sys.path.insert(0, os.path.dirname(script)); "
    "runpy.run_path(script, run_name='__main__')"
)


def _run_unpack(prefix):
    # conda-unpack's shebang may name the build host; always use packed Python.
    subprocess.run(
        [
            str(prefix / "bin/python"),
            "-I",
            "-c",
            _UNPACK_PROGRAM,
            str(prefix / "bin/conda-unpack"),
        ],
        check=True,
        stdin=subprocess.DEVNULL,
        stdout=sys.stderr,
        env=subprocess_env(prefix),
        timeout=300,
    )


def _publish_prefix(staged, prefix):
    """Atomically move the owned stage without replacing any existing prefix.

    Linux restores require libc/kernel/filesystem RENAME_NOREPLACE support.
    macOS's equivalent permits offline synthetic tests. Never fall back to
    ordinary rename: it can replace another writer's empty directory.
    """
    library = ctypes.CDLL(None, use_errno=True)
    source, target = os.fsencode(staged), os.fsencode(prefix)
    if sys.platform == "linux":
        rename = getattr(library, "renameat2", None)
        arguments = [-100, source, -100, target, 1]  # AT_FDCWD, RENAME_NOREPLACE
        types = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
    elif sys.platform == "darwin":
        rename = getattr(library, "renamex_np", None)
        arguments = [source, target, 4]  # RENAME_EXCL
        types = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint]
    else:
        rename = None
    if rename is None:
        raise RuntimeError("atomic no-replace environment publication unavailable")
    rename.argtypes, rename.restype = types, ctypes.c_int
    if rename(*arguments) != 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error), str(staged), None, str(prefix))


def _restore(env, *, prefer_local, client=None, bucket=None, scratch=None):
    registry.validate_env(env)
    check_compatibility(env)
    prefix = Path(env["unpack_to"])
    with prefix_lock(prefix):
        if prefer_local and _already_restored(env):
            packed_python(env)
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
                if model_source._is_real_client(client):
                    # Parallel ranged parts (processes x streams), like weights.
                    model_source.download_object(
                        client, bucket, env["r2_key"], tarball, env["size"]
                    )
                else:
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
                # Capture ownership before publication. No empty public reservation
                # is left unowned if a signal arrives between mkdir and stat.
                owned = staged.stat()
                try:
                    _publish_prefix(staged, prefix)
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
                    raise
    packed_python(env)
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
    python = packed_python(env)
    return {
        "name": name,
        "python": python,
        "activate": activate,
        "unpack_to": env["unpack_to"],
        "compat": env["compat"],
        "identity": restore_identity(env),
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
