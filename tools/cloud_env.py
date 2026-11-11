"""Restore a pinned environment over SSH; return its path on the remote host.

Only reader modules are uploaded. R2 credentials travel on SSH stdin and are
removed from the environment before any packed executable runs. The caller must
supply a read-only bucket token. The returned activation path is remote, not local.
"""

import copy
import io
import json
import os
import shlex
import subprocess
import tarfile
import uuid
from pathlib import Path

from tools import registry
from tools.cloud_prefetch import R2_KEYS, credential_blob
from tools.env_source import restore_identity

TOOLS = Path(__file__).resolve().parent
UPLOAD = ("model_source.py", "registry.py", "env_source.py", "models.yaml")


def _default_runner(command, *, input=None, timeout=1800):
    proc = subprocess.Popen(
        command,
        stdin=subprocess.PIPE if input is not None else subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        out, _ = proc.communicate(input=input, timeout=timeout)
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()
    if proc.returncode:
        raise RuntimeError(f"{command[0]} exited with status {proc.returncode}")
    return out


def remote_command(name, *, python="python3", remote_dir="."):
    # Host python3 is only the restore bootstrap. Eval uses {unpack_to}/bin/python.
    # -I excludes cwd/site PYTHONPATH injection. Insert only the uploaded reader root.
    program = (
        f"import runpy,sys; sys.path.insert(0, {remote_dir!r}); "
        "runpy.run_module('tools.env_source', run_name='__main__')"
    )
    inner = shlex.join([python, "-I", "-c", program, "--model", name])
    return (
        f"set +x; set +v; unset {' '.join(R2_KEYS)} HF_TOKEN; "
        f". /dev/stdin || exit; exec {inner}"
    )


def _default_uploader(ssh_options, host, port, remote_dir):
    # Avoid scp's different legacy-shell and SFTP path-quoting rules entirely.
    payload = io.BytesIO()
    with tarfile.open(fileobj=payload, mode="w") as tf:
        for name in UPLOAD:
            tf.add(TOOLS / name, arcname=name)
        tf.addfile(tarfile.TarInfo("__init__.py"))
    target = shlex.quote(f"{remote_dir}/tools")
    subprocess.run(
        [
            "ssh",
            *ssh_options,
            "-p",
            str(port),
            f"root@{host}",
            f"mkdir -p -- {target} && tar -xf - --no-same-owner -C {target}",
        ],
        input=payload.getvalue(),
        check=True,
        timeout=180,
    )


def _default_cleaner(ssh_options, host, port, remote_dir):
    subprocess.run(
        [
            "ssh",
            *ssh_options,
            "-p",
            str(port),
            f"root@{host}",
            f"rm -rf -- {shlex.quote(remote_dir)}",
        ],
        check=True,
        stdin=subprocess.DEVNULL,
        timeout=60,
    )


def restore_env(
    *,
    ssh_options,
    host,
    port,
    name,
    remote_dir="/workspace/goodvision-models",
    r2_env=None,
    python="python3",
    runner=_default_runner,
    uploader=_default_uploader,
    cleaner=_default_cleaner,
    timeout=1800,
    expected_env=None,
):
    """Return packed Python only for the caller's captured artifact, if supplied."""
    env = registry.env_of(name)
    if env is None:
        raise registry.RegistryError(f"model {name!r} has no packed environment")
    expected_identity = copy.deepcopy(restore_identity(env))
    if expected_env is not None:
        caller_identity = copy.deepcopy(restore_identity(expected_env))
        if caller_identity != expected_identity:
            raise ValueError(
                "registry environment differs from caller's expected identity"
            )
        expected_identity = caller_identity
    if not isinstance(remote_dir, str) or not remote_dir.startswith("/"):
        raise ValueError("remote_dir must be an absolute POSIX path")
    registry.relative_path(remote_dir[1:])
    if not all((r2_env or {}).get(key) for key in R2_KEYS):
        raise ValueError("all four read-only R2 environment values are required")
    blob = credential_blob(r2_env)
    # Each invocation owns just this unpredictable child, never the caller's directory.
    reader_dir = f"{remote_dir}/.env-reader-{uuid.uuid4().hex}"
    try:
        uploader(ssh_options, host, port, reader_dir)
        cmd = remote_command(name, python=python, remote_dir=reader_dir)
        ssh = ["ssh", *ssh_options, "-p", str(port), f"root@{host}", cmd]
        out = runner(ssh, input=blob, timeout=timeout)
        expected_activate = f"{env['unpack_to']}/{env['activate_rel']}"
        expected_python = f"{env['unpack_to']}/bin/python"
        # Only the final acknowledgement is authoritative. An earlier success
        # must not hide a subsequent failure or a different selected prefix.
        line = next(
            (line for line in reversed((out or "").splitlines()) if line.strip()), ""
        )
        try:
            info = json.loads(line)
        except ValueError:
            raise RuntimeError(
                "env_source final reply did not report the expected packed python path"
            ) from None
        if not (
            isinstance(info, dict)
            and info.get("name") == name
            and info.get("source") in ("local", "r2")
            and info.get("activate") == expected_activate
            and info.get("python") == expected_python
        ):
            raise RuntimeError(
                "env_source did not report the expected packed python path"
            )
        try:
            reported_identity = registry.validate_env(info.get("identity"))
        except registry.RegistryError:
            raise RuntimeError(
                "env_source did not report a valid packed environment identity"
            ) from None
        if reported_identity != expected_identity:
            raise RuntimeError(
                "env_source reported a different packed environment identity"
            )
        return expected_python
    finally:
        try:
            cleaner(ssh_options, host, port, reader_dir)
        except Exception:
            # Preserve a restore/upload failure; cleanup contains no credentials.
            import sys

            print(
                "warning: could not remove temporary remote env readers",
                file=sys.stderr,
            )


def main(argv=None):
    import argparse

    ap = argparse.ArgumentParser(
        description="Restore a pinned env over SSH (path is remote)."
    )
    ap.add_argument("--host", required=True)
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--ssh-key", type=Path)
    ap.add_argument("--model", required=True)
    ap.add_argument("--remote-dir", default="/workspace/goodvision-models")
    ap.add_argument(
        "--python",
        default="python3",
        help="host Python used only to run env_source before the packed env exists",
    )
    args = ap.parse_args(argv)
    opts = ["-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=accept-new"]
    if args.ssh_key:
        opts += ["-i", str(args.ssh_key), "-o", "IdentitiesOnly=yes"]
    python = restore_env(
        ssh_options=opts,
        host=args.host,
        port=args.port,
        name=args.model,
        remote_dir=args.remote_dir,
        python=args.python,
        r2_env={k: os.environ.get(k) for k in R2_KEYS},
    )
    print(python)


if __name__ == "__main__":
    main()
