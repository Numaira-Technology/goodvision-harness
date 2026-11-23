"""Pinned, prebuilt CUDA extension wheels (flash-attn, causal-conv1d) in R2.

Compiling flash-attn and causal-conv1d takes most of an env build (~70 min of
~80 on an RTX 4090). ``build_env`` keeps the wheels it compiles; this module
uploads them create-only to content-addressed keys and, on the next build,
fetches the ones pinned in ``tools/models.yaml`` (``wheels:``) for the same
Python / CUDA / torch / GPU family, verifying SHA-256 and size before pip sees
them. The trusted hash is the one in the repository, never the remote sidecar.

    # after a build that compiled them (read-write R2_* from the environment):
    python -m tools.wheel_cache upload --model videojev-fp8 WHEEL [WHEEL ...]
    # prints the pins to add under `wheels:` in tools/models.yaml after review
"""

import argparse
import hashlib
import json
import os
import re
import sys
import tempfile
import time
from pathlib import Path

from tools import model_source, registry

# The CUDA extensions build_env compiles, in install order.
COMPILED = (("flash-attn", "2.8.3.post2"), ("causal-conv1d", "1.7.0"))
_NAME = re.compile(r"[a-z0-9][a-z0-9-]*")
_FILENAME = re.compile(r"[A-Za-z0-9_.+-]+-[^-]+-cp\d+-cp\d+[a-z]*-linux_x86_64\.whl")
COMPAT_KEYS = ("python", "cuda", "torch", "gpu_family")


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_pin(pin):
    if not isinstance(pin, dict):
        raise registry.RegistryError("wheel pin must be a mapping")
    if not isinstance(pin.get("name"), str) or not _NAME.fullmatch(pin["name"]):
        raise registry.RegistryError("wheel pin name must be a lowercase project name")
    if not isinstance(pin.get("version"), str) or not pin["version"]:
        raise registry.RegistryError("wheel pin needs a version")
    if not isinstance(pin.get("filename"), str) or not _FILENAME.fullmatch(
        pin["filename"]
    ):
        raise registry.RegistryError(
            "wheel pin filename must be a linux_x86_64 CPython wheel"
        )
    sha = pin.get("sha256")
    if not isinstance(sha, str) or not re.fullmatch(r"[0-9a-f]{64}", sha):
        raise registry.RegistryError("wheel pin sha256 must be a 64-character digest")
    if pin.get("r2_key") != f"wheels/{pin['name']}/{sha}.whl":
        raise registry.RegistryError(
            "wheel pin r2_key must be wheels/<name>/<sha256>.whl"
        )
    if type(pin.get("size")) is not int or pin["size"] <= 0:
        raise registry.RegistryError("wheel pin size must be a positive integer")
    compat = pin.get("compat")
    if not isinstance(compat, dict) or set(compat) != set(COMPAT_KEYS):
        raise registry.RegistryError(
            f"wheel pin compat must have exactly {COMPAT_KEYS}"
        )
    return pin


def pins_for(compat, path=None):
    """Pinned wheels for exactly this compat, keyed by project name."""
    wheels = registry._load(path).get("wheels") or []
    if not isinstance(wheels, list):
        raise registry.RegistryError("registry 'wheels' must be a list")
    wanted = {k: compat[k] for k in COMPAT_KEYS}
    found = {}
    for pin in map(validate_pin, wheels):
        if pin["compat"] == wanted and (pin["name"], pin["version"]) in COMPILED:
            if pin["name"] in found:
                raise registry.RegistryError(f"two wheel pins for {pin['name']}")
            found[pin["name"]] = pin
    return found


def fetch(pin, dest_dir, *, client=None, bucket=None):
    """Download one pinned wheel into dest_dir; refuse any size or hash mismatch."""
    validate_pin(pin)
    client = client if client is not None else model_source.r2_client()
    if client is None:
        raise RuntimeError("R2 credentials not configured")
    bucket = bucket or os.environ["R2_BUCKET"]
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    final = dest_dir / pin["filename"]
    with tempfile.NamedTemporaryFile(dir=dest_dir, delete=False, suffix=".part") as tmp:
        part = Path(tmp.name)
        try:
            body = client.get_object(Bucket=bucket, Key=pin["r2_key"])["Body"]
            digest, size = hashlib.sha256(), 0
            try:
                for chunk in iter(lambda: body.read(1 << 20), b""):
                    digest.update(chunk)
                    size += len(chunk)
                    if size > pin["size"]:
                        raise RuntimeError(f"{pin['filename']}: larger than pinned")
                    tmp.write(chunk)
            finally:
                body.close()
            if size != pin["size"] or digest.hexdigest() != pin["sha256"]:
                raise RuntimeError(
                    f"{pin['filename']}: size or SHA-256 differs from the pin"
                )
        except BaseException:
            part.unlink(missing_ok=True)
            raise
    part.replace(final)
    return final


def upload(wheel, name, version, compat, *, client=None, bucket=None):
    """Create-only upload to wheels/<name>/<sha256>.whl (+ .json); returns the pin.

    An identical object already at the content-addressed key (same size) counts
    as present; nothing is ever overwritten or deleted.
    """
    wheel = Path(wheel)
    if (name, version) not in COMPILED:
        raise ValueError(f"{name}=={version} is not a wheel build_env compiles")
    sha, size = _sha256(wheel), wheel.stat().st_size
    pin = validate_pin(
        {
            "name": name,
            "version": version,
            "filename": wheel.name,
            "compat": {k: compat[k] for k in COMPAT_KEYS},
            "r2_key": f"wheels/{name}/{sha}.whl",
            "sha256": sha,
            "size": size,
        }
    )
    client = client if client is not None else model_source.r2_client()
    if client is None:
        raise RuntimeError("R2 credentials not configured")
    bucket = bucket or os.environ["R2_BUCKET"]
    sidecar = {
        **pin,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "built_by": "goodvision-harness build_env",
    }
    for key, body in ((pin["r2_key"], None), (pin["r2_key"][:-4] + ".json", sidecar)):
        try:
            if body is None:
                with wheel.open("rb") as f:
                    client.put_object(
                        Bucket=bucket,
                        Key=key,
                        Body=f,
                        ContentLength=size,
                        IfNoneMatch="*",
                    )
            else:
                client.put_object(
                    Bucket=bucket,
                    Key=key,
                    IfNoneMatch="*",
                    Body=json.dumps(body, indent=2).encode(),
                )
        except Exception as error:
            code = getattr(error, "response", {}).get("Error", {}).get("Code", "")
            if code == "PreconditionFailed" and body is None:
                head = client.head_object(Bucket=bucket, Key=key)
                if head.get("ContentLength") == size:
                    continue  # content-addressed and same size: already present
            if code == "PreconditionFailed" and body is not None:
                continue
            raise RuntimeError(
                f"create-only wheel upload failed ({type(error).__name__}"
                f"{': ' + code if code else ''}); inspect {key}; "
                "nothing was overwritten"
            ) from None
    return pin


def name_version(wheel):
    """(project name, version) from a wheel filename."""
    dist, version = Path(wheel).name.split("-")[:2]
    return dist.replace("_", "-").lower(), version


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="command", required=True)
    up = sub.add_parser("upload", help="create-only upload of compiled wheels")
    up.add_argument(
        "--model", required=True, help="registry model whose env compat applies"
    )
    up.add_argument("wheels", nargs="+", type=Path)
    args = ap.parse_args(argv)
    compat = registry.env_of(args.model)["compat"]
    pins = [upload(w, *name_version(w), compat) for w in args.wheels]
    json.dump({"wheels": pins}, sys.stdout, indent=2)
    print()


if __name__ == "__main__":
    main()
