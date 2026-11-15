"""Restore a model's packed environment onto a rented GPU over SSH.

The remote analogue of ``tools/cloud_env.ensure``: where cloud_prefetch fills the
weight cache on the box, this restores the pre-built conda-pack'd env there, so a
launcher can `source` it instead of compiling torch/flash-attn.

It uploads the small reader modules (model_source.py, registry.py, env_source.py,
models.yaml) into ``<remote_dir>/tools`` and runs ``python -m tools.env_source``
on the box, carrying the R2 credentials over SSH **stdin** only. The box needs the
**read-only** R2 key -- restore never uploads -- plus ``boto3`` and ``PyYAML``.

Returns the remote path to the env's ``activate`` script, e.g.::

    source "$(restore result)"  # then run the model in its own env
"""

import json
import shlex
import subprocess
from pathlib import Path

from tools.cloud_prefetch import credential_blob

TOOLS = Path(__file__).resolve().parent
# Everything env_source needs to resolve + pull an env (no weights, no HF).
UPLOAD = ("model_source.py", "registry.py", "env_source.py", "models.yaml")


def _default_runner(command, *, input=None, timeout=1800):
    """Run ``command``, feeding ``input`` on stdin, returning captured stdout.

    env_source logs to stderr (streamed through) and prints its JSON result to
    stdout, which the caller parses for the activate path.
    """
    proc = subprocess.Popen(
        command, stdin=subprocess.PIPE if input is not None else subprocess.DEVNULL,
        stdout=subprocess.PIPE, text=True, start_new_session=True,
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
    """One remote shell command: source creds from stdin, then restore the env.

    Prints the env_source JSON; nothing secret appears in the command itself.
    """
    inner = (
        f"cd {shlex.quote(remote_dir)} && PYTHONPATH={shlex.quote(remote_dir)} "
        f"{shlex.quote(python)} -m tools.env_source --model {shlex.quote(name)}"
    )
    return f"set -a; . /dev/stdin; set +a; {inner}"


def _default_uploader(ssh_options, host, port, remote_dir):
    subprocess.run(
        ["ssh", *ssh_options, "-p", str(port), f"root@{host}",
         f"mkdir -p {shlex.quote(remote_dir)}/tools"],
        check=True, stdin=subprocess.DEVNULL, timeout=60,
    )
    srcs = [str(TOOLS / f) for f in UPLOAD]
    subprocess.run(
        ["scp", *ssh_options, "-P", str(port), *srcs,
         f"root@{host}:{remote_dir}/tools/"],
        check=True, stdin=subprocess.DEVNULL, timeout=180,
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
    timeout=1800,
):
    """Upload the readers and restore model ``name``'s env on the box.

    Returns the remote activate-script path. Credentials travel over SSH stdin;
    nothing secret is placed in argv. The restore uses the read-only R2 key.
    """
    blob = credential_blob(r2_env)
    uploader(ssh_options, host, port, remote_dir)
    cmd = remote_command(name, python=python, remote_dir=remote_dir)
    ssh = ["ssh", *ssh_options, "-p", str(port), f"root@{host}", cmd]
    out = runner(ssh, input=blob, timeout=timeout)
    # env_source prints one JSON object; take the last such line.
    for line in reversed((out or "").splitlines()):
        line = line.strip()
        if line.startswith("{") and "activate" in line:
            return json.loads(line)["activate"]
    raise RuntimeError("env_source did not report an activate path")


def main(argv=None):
    import argparse
    import os

    ap = argparse.ArgumentParser(
        description="Restore a pinned packed env onto a rented GPU over SSH (R2).")
    ap.add_argument("--host", required=True)
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--ssh-key", type=Path, help="private key (adds -i/-o options)")
    ap.add_argument("--model", required=True, help="model name in tools/models.yaml")
    ap.add_argument("--remote-dir", default="/workspace/goodvision-models")
    args = ap.parse_args(argv)

    opts = []
    if args.ssh_key:
        opts = ["-i", str(args.ssh_key), "-o", "IdentitiesOnly=yes",
                "-o", "StrictHostKeyChecking=accept-new"]
    r2_env = {k: os.environ.get(k) for k in
              ("R2_ENDPOINT", "R2_BUCKET", "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY")}
    activate = restore_env(ssh_options=opts, host=args.host, port=args.port,
                           name=args.model, remote_dir=args.remote_dir, r2_env=r2_env)
    print(activate)


if __name__ == "__main__":
    main()
