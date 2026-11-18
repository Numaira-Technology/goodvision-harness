"""Prefetch pinned models onto a rented GPU from R2 (HF fallback), over SSH.

Generic, worker-agnostic: any SAM-style renter (start_sam / start_detector / ...)
calls prefetch_models() right after SSH is up and before starting its worker, so
weights land from the R2 fixed store (egress-free) instead of a slow HF pull.

Credentials reach the instance the same way start_sam carries HF_TOKEN: over SSH
stdin, never in argv or an image. Only the prefetch child inherits them in its
environment. The remote runs
the uploaded tools/model_source.py, which verifies against the bucket MANIFEST and
falls back to a normal HF download when R2 is not reachable.
"""

import os
import shlex
import subprocess
from pathlib import Path

MODEL_SOURCE = Path(__file__).resolve().parent / "model_source.py"
R2_KEYS = ("R2_ENDPOINT", "R2_BUCKET", "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY")


def credential_blob(r2_env, hf_token=None):
    """Shell `export` lines fed to the remote over stdin (so secrets stay off argv).

    r2_env maps the R2_* names to values; missing/empty entries are dropped, which
    lets the remote fall back to HF. hf_token, when given, lifts HF rate limits for
    the fallback path.
    """
    lines = []
    for key in R2_KEYS:
        value = (r2_env or {}).get(key)
        if value:
            lines.append(f"export {key}={shlex.quote(str(value))}")
    if hf_token:
        lines.append(f"export HF_TOKEN={shlex.quote(str(hf_token))}")
    return "\n".join(lines) + "\n"


# Non-secret download tuning knobs forwarded from the controller's environment
# (see model_source.py): integers only, so they are safe on the command line.
PULL_TUNING = ("R2_PULL_FILE_WORKERS", "R2_PULL_STREAMS", "R2_PULL_PROCESSES")


def pull_tuning(environ=None):
    environ = os.environ if environ is None else environ
    parts = [
        f"{name}={environ[name]} "
        for name in PULL_TUNING
        if str(environ.get(name, "")).isdigit()
    ]
    return "".join(parts)


def remote_command(
    cache_dir,
    repo,
    revision,
    *,
    python="python3",
    module="model_source.py",
    remote_dir=".",
):
    """One remote shell command: source creds from stdin, then run the prefetch.

    `. /dev/stdin` consumes the credential blob piped in over SSH; the Python
    process itself reads no stdin. Paths/args are quoted; no secret appears here.
    """
    script = f"{remote_dir}/{module}"
    inner = (
        f"{pull_tuning()}{shlex.quote(python)} {shlex.quote(script)} "
        f"--repo {shlex.quote(repo)} --revision {shlex.quote(revision)} "
        f"--cache-dir {shlex.quote(str(cache_dir))}"
    )
    # set -a so the sourced exports apply to the child python; set +a after.
    return f"set -a; . /dev/stdin; set +a; {inner}"


def _default_runner(command, *, input=None, timeout=3600):
    proc = subprocess.Popen(
        command,
        stdin=subprocess.PIPE if input is not None else subprocess.DEVNULL,
        start_new_session=True,
    )
    try:
        proc.communicate(
            input=input.encode() if input is not None else None, timeout=timeout
        )
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()
    if proc.returncode:
        raise RuntimeError(f"{command[0]} exited with status {proc.returncode}")


def _default_uploader(ssh_options, host, port, remote_dir):
    subprocess.run(
        [
            "ssh",
            *ssh_options,
            "-p",
            str(port),
            f"root@{host}",
            f"mkdir -p {shlex.quote(remote_dir)}",
        ],
        check=True,
        stdin=subprocess.DEVNULL,
        timeout=60,
    )
    subprocess.run(
        [
            "scp",
            *ssh_options,
            "-P",
            str(port),
            str(MODEL_SOURCE),
            f"root@{host}:{remote_dir}/model_source.py",
        ],
        check=True,
        stdin=subprocess.DEVNULL,
        timeout=120,
    )


def prefetch_models(
    *,
    ssh_options,
    host,
    port,
    remote_dir,
    models,
    cache_dir=None,
    r2_env=None,
    hf_token=None,
    python="python3",
    runner=_default_runner,
    uploader=_default_uploader,
    timeout=3600,
):
    """Upload model_source.py and prefetch each (repo, revision) on the instance.

    models: iterable of (repo, revision). cache_dir defaults to remote_dir/cache.
    Returns the remote HF cache directory the worker should load from.
    Credentials travel over SSH stdin; nothing secret is placed in argv.
    """
    cache_dir = cache_dir or f"{remote_dir}/cache"
    blob = credential_blob(r2_env, hf_token)
    uploader(ssh_options, host, port, remote_dir)
    for repo, revision in models:
        cmd = remote_command(
            cache_dir, repo, revision, python=python, remote_dir=remote_dir
        )
        ssh = ["ssh", *ssh_options, "-p", str(port), f"root@{host}", cmd]
        runner(ssh, input=blob, timeout=timeout)
    return cache_dir


def main(argv=None):
    import argparse
    import os

    ap = argparse.ArgumentParser(
        description="Prefetch pinned models from R2 (HF fallback) over SSH."
    )
    ap.add_argument("--host", required=True)
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--ssh-key", type=Path, help="private key (adds -i/-o options)")
    ap.add_argument("--remote-dir", default="/workspace/goodvision-models")
    ap.add_argument(
        "--cache-dir", help="remote HF cache dir (default: <remote-dir>/cache)"
    )
    ap.add_argument(
        "--model",
        action="append",
        metavar="REPO@REVISION",
        required=True,
        help="repeatable; e.g. ZhuYuhan/VideoJev-4B@1c1a253...",
    )
    ap.add_argument("--python", default="python3")
    args = ap.parse_args(argv)

    options = ["-o", "StrictHostKeyChecking=accept-new", "-o", "BatchMode=yes"]
    if args.ssh_key:
        options += ["-i", str(args.ssh_key)]
    models = []
    for item in args.model:
        if "@" not in item:
            ap.error(f"--model must be REPO@REVISION: {item!r}")
        repo, revision = item.rsplit("@", 1)
        models.append((repo, revision))

    r2_env = {k: os.environ.get(k) for k in R2_KEYS}
    cache = prefetch_models(
        ssh_options=options,
        host=args.host,
        port=args.port,
        remote_dir=args.remote_dir,
        models=models,
        cache_dir=args.cache_dir,
        r2_env=r2_env,
        hf_token=os.environ.get("HF_TOKEN"),
        python=args.python,
    )
    print(cache)


if __name__ == "__main__":
    main()
