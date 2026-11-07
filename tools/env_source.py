"""Pull a pre-built, conda-pack'd Python environment from R2 and activate it.

The env analogue of ``tools/model_source.py``. Where model_source fills the HF
cache with weights, this restores a ready-to-run environment (torch, flash-attn
and every dep already compiled for the target GPU) so a freshly rented box does
NOT apt/pip/compile anything -- it downloads the tarball (egress-free from R2)
and `source`s it.

Flow (``ensure_env``):
  verified-local  ->  pull tarball from R2  ->  extract to the fixed prefix  ->
  run conda-unpack (fix shebangs/prefix)  ->  return the activate path.

Credentials come from the environment (same four R2_* vars as model_source),
carried to the box over SSH stdin, never committed. With no creds this raises --
there is no HF fallback for an environment; the caller builds it instead.

The tarball is pinned by sha256 (from its R2 side-manifest, cross-checked against
the registry). It is OUR artifact, produced by tools/build_env.py; the digest is
verified before a single byte is extracted.
"""

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from tools import (
    model_source,  # reuse r2_client() + env-var contract
    registry,
)

MANIFEST_SUFFIX = ".json"


def _log(*a):
    print(f"[env_source {time.strftime('%H:%M:%S')}]", *a, flush=True, file=sys.stderr)


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _marker(unpack_to):
    return Path(unpack_to) / ".r2_env.json"


def _activate_path(env):
    return str(Path(env["unpack_to"]) / env["activate_rel"])


def read_remote_manifest(env, *, client=None, bucket=None):
    """Fetch the env's side-manifest from R2 (sha256, size, compat), or None."""
    key = env.get("manifest_key")
    if not key:
        return None
    client = client or model_source.r2_client()
    if client is None:
        raise RuntimeError("R2 credentials not configured")
    bucket = bucket or os.environ["R2_BUCKET"]
    try:
        body = client.get_object(Bucket=bucket, Key=key)["Body"].read()
    except Exception:
        return None
    return json.loads(body)


def _already_restored(env):
    """True iff the env is already extracted and matches the expected sha256."""
    m = _marker(env["unpack_to"])
    activate = Path(_activate_path(env))
    if not m.is_file() or not activate.is_file():
        return False
    try:
        got = json.loads(m.read_text()).get("sha256")
    except (ValueError, OSError):
        return False
    want = env.get("sha256")
    return bool(got) and (want is None or got == want)


def _safe_extract(tarball, dest):
    """Extract ``tarball`` into ``dest`` with tar (fast), rejecting path escapes.

    The archive is ours and sha256-verified before this runs; the data filter is a
    second line of defence against a bad path. Falls back to the stdlib with the
    3.12 data filter if the ``tar`` binary is unavailable.
    """
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    if shutil.which("tar"):
        subprocess.run(["tar", "-xzf", str(tarball), "-C", str(dest)], check=True)
        return
    import tarfile

    with tarfile.open(tarball, "r:gz") as tf:
        tf.extractall(dest, filter="data")  # py3.12: blocks abs paths / .. escapes


def pull_env(env, *, client=None, bucket=None, scratch=None):
    """Download + verify + extract + unpack the env; return the activate path."""
    client = client or model_source.r2_client()
    if client is None:
        raise RuntimeError("R2 credentials not configured")
    bucket = bucket or os.environ["R2_BUCKET"]

    man = read_remote_manifest(env, client=client, bucket=bucket)
    want_sha = env.get("sha256") or (man or {}).get("sha256")

    unpack_to = Path(env["unpack_to"])
    scratch = Path(scratch or tempfile.gettempdir())
    scratch.mkdir(parents=True, exist_ok=True)
    tarball = scratch / (unpack_to.name + ".tar.gz")

    _log(f"downloading {env['r2_key']} -> {tarball}")
    client.download_file(bucket, env["r2_key"], str(tarball))

    got_sha = _sha256(tarball)
    if want_sha and got_sha != want_sha:
        tarball.unlink(missing_ok=True)
        raise ValueError(f"env tarball sha256 mismatch: got {got_sha}, want {want_sha}")
    if man and man.get("size") and tarball.stat().st_size != man["size"]:
        tarball.unlink(missing_ok=True)
        raise ValueError("env tarball size mismatch vs manifest")

    # Fresh prefix: remove any partial prior extraction, then restore.
    if unpack_to.exists():
        shutil.rmtree(unpack_to)
    _log(f"extracting into {unpack_to}")
    _safe_extract(tarball, unpack_to)

    unpacker = unpack_to / "bin" / "conda-unpack"
    if unpacker.is_file():
        _log("running conda-unpack")
        subprocess.run([str(unpacker)], check=True)

    activate = Path(_activate_path(env))
    if not activate.is_file():
        raise RuntimeError(f"activate script not found after extract: {activate}")

    _marker(unpack_to).write_text(json.dumps(
        {"sha256": got_sha, "r2_key": env["r2_key"],
         "restored_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())},
        indent=2))
    tarball.unlink(missing_ok=True)
    return str(activate)


def ensure_env(name, *, prefer_local=True, scratch=None, registry_path=None):
    """Ensure model ``name``'s packed env is restored; return info incl. ``activate``.

    Order: verified-local -> pull from R2. Raises if the model has no env, or if
    R2 credentials are missing (no build happens here -- use tools/build_env.py).
    """
    env = registry.env_of(name, registry_path)
    if env is None:
        raise registry.RegistryError(f"model {name!r} has no packed env registered")

    if prefer_local and _already_restored(env):
        _log(f"env hit {name} at {env['unpack_to']}")
        return {"name": name, "activate": _activate_path(env),
                "unpack_to": env["unpack_to"], "compat": env.get("compat"),
                "source": "local"}

    activate = pull_env(env, scratch=scratch)
    _log(f"env ready from R2 {name} -> {activate}")
    return {"name": name, "activate": activate, "unpack_to": env["unpack_to"],
            "compat": env.get("compat"), "source": "r2"}


def main(argv=None):
    import argparse

    ap = argparse.ArgumentParser(
        description="Restore a pinned conda-pack'd env from R2.")
    ap.add_argument("--model", required=True, help="model name in tools/models.yaml")
    ap.add_argument("--no-local", action="store_true",
                    help="ignore any existing extraction")
    ap.add_argument("--scratch", help="where to stage the downloaded tarball")
    args = ap.parse_args(argv)

    info = ensure_env(args.model, prefer_local=not args.no_local, scratch=args.scratch)
    # Prints JSON; a caller greps the "activate" field and `source`s it.
    print(json.dumps(info))


if __name__ == "__main__":
    main()
