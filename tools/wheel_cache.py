"""Pinned, prebuilt CUDA extension wheels (flash-attn, causal-conv1d) in R2.

``build_env`` keeps the wheels it compiles; this module
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
import zipfile
from email.parser import Parser
from pathlib import Path

from tools import model_source, registry
from tools.artifact_errors import upload_failure_category

# The CUDA extensions build_env compiles, in install order.
COMPILED = (("flash-attn", "2.8.3.post2"), ("causal-conv1d", "1.7.0"))
_FILENAME = re.compile(
    r"(?P<name>flash_attn|causal_conv1d)-"
    r"(?P<version>\d+(?:\.\d+)+(?:\.post\d+)?)-"
    r"(?P<python>cp3\d+)-(?P<abi>cp3\d+)-linux_x86_64\.whl", re.ASCII
)
COMPAT_KEYS = ("python", "cuda", "torch", "gpu_family")
MAX_WHEEL_SIZE = 5 * 1024**3 - 5 * 1024**2  # single create-only PUT, no multipart


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_pin(pin):
    if not isinstance(pin, dict):
        raise registry.RegistryError("wheel pin must be a mapping")
    match = (
        _FILENAME.fullmatch(pin["filename"])
        if isinstance(pin.get("filename"), str) else None
    )
    if not match:
        raise registry.RegistryError(
            "wheel filename must be a supported native CPython Linux basename"
        )
    if (pin.get("name"), pin.get("version")) != (
        match["name"].replace("_", "-"), match["version"]
    ):
        raise registry.RegistryError("wheel filename and pin identity differ")
    sha = pin.get("sha256")
    if not isinstance(sha, str) or not re.fullmatch(r"[0-9a-f]{64}", sha):
        raise registry.RegistryError("wheel pin sha256 must be a 64-character digest")
    if pin.get("r2_key") != f"wheels/{pin['name']}/{sha}.whl":
        raise registry.RegistryError(
            "wheel pin r2_key must be wheels/<name>/<sha256>.whl"
        )
    if type(pin.get("size")) is not int or not 0 < pin["size"] <= MAX_WHEEL_SIZE:
        raise registry.RegistryError("wheel size must fit one create-only PUT")
    compat = pin.get("compat")
    if not isinstance(compat, dict) or set(compat) != set(COMPAT_KEYS):
        raise registry.RegistryError(
            f"wheel pin compat must have exactly {COMPAT_KEYS}"
        )
    for key in ("python", "cuda", "torch"):
        if not isinstance(compat[key], str) or not re.fullmatch(
            r"\d+\.\d+(?:\.\d+)?", compat[key], re.ASCII
        ):
            raise registry.RegistryError("wheel compatibility versions must be numeric")
    if not re.fullmatch(r"3\.\d+", compat["python"]):
        raise registry.RegistryError("wheel Python must be a CPython 3 major.minor")
    if (
        not isinstance(compat["gpu_family"], str)
        or compat["gpu_family"] not in registry.GPU_ARCHES
    ):
        raise registry.RegistryError("unsupported wheel GPU family")
    tag = "cp" + compat["python"].replace(".", "")
    if match["python"] != tag or match["abi"] != tag:
        raise registry.RegistryError("wheel Python/ABI tags differ from compatibility")
    return pin


def inspect_wheel(path, filename):
    """Check bounded wheel metadata without importing or executing wheel code.

    Tags establish Python/platform identity only. CUDA, torch ABI and GPU-family
    values remain producer declarations requiring separate runtime evidence.
    """
    match = _FILENAME.fullmatch(filename)
    if not match:
        raise registry.RegistryError("unsupported wheel filename")
    info = f"{match['name']}-{match['version']}.dist-info/"
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        identities = {n.split("/", 1)[0] for n in names if ".dist-info/" in n}
        if len(names) != len(set(names)) or identities != {info[:-1]}:
            raise ValueError("wheel must contain one matching distribution identity")

        def metadata(name):
            member = archive.getinfo(info + name)
            if member.file_size > 65536:
                raise ValueError("wheel metadata exceeds 64 KiB")
            with archive.open(member) as stream:
                raw = stream.read(65537)
            if len(raw) > 65536:
                raise ValueError("wheel metadata exceeds 64 KiB")
            return Parser().parsestr(raw.decode("utf-8"), headersonly=True)

        package, wheel = metadata("METADATA"), metadata("WHEEL")
        expected_name = match["name"].replace("_", "-")
        if (
            package.get_all("Name") is None
            or len(package.get_all("Name")) != 1
            or re.sub(r"[-_.]+", "-", package["Name"]).lower() != expected_name
            or package.get_all("Version") != [match["version"]]
        ):
            raise ValueError("wheel METADATA differs from filename identity")
        expected_tag = f"{match['python']}-{match['abi']}-linux_x86_64"
        if (
            wheel.get_all("Wheel-Version") != ["1.0"]
            or wheel.get_all("Root-Is-Purelib") != ["false"]
            or wheel.get_all("Tag") != [expected_tag]
        ):
            raise ValueError("wheel WHEEL metadata differs from native filename tags")


def verify_file(path, pin):
    """Recheck the pinned bytes immediately before installation or cache reuse."""
    validate_pin(pin)
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError("wheel must be a regular, non-symlink file")
    if path.stat().st_size != pin["size"] or _sha256(path) != pin["sha256"]:
        raise ValueError("wheel size or SHA-256 differs from the pin")
    inspect_wheel(path, pin["filename"])
    return path


def pin_for_file(path, compat):
    path = Path(path)
    name, version = name_version(path)
    sha = _sha256(path)
    return validate_pin({
        "name": name, "version": version, "filename": path.name,
        "compat": dict(compat), "sha256": sha, "size": path.stat().st_size,
        "r2_key": f"wheels/{name}/{sha}.whl",
    })


def pins_for(compat, path=None):
    """Pinned wheels for exactly this compat, keyed by project name."""
    wheels = registry._load(path).get("wheels", [])
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
    """Publish verified bytes by digest without replacing an existing cache entry."""
    validate_pin(pin)
    client = client if client is not None else model_source.r2_client()
    if client is None:
        raise RuntimeError("R2 credentials not configured")
    bucket = bucket or os.environ["R2_BUCKET"]
    dest_dir = Path(dest_dir) / pin["sha256"]
    dest_dir.mkdir(parents=True, exist_ok=True)
    final = dest_dir / pin["filename"]
    if final.exists() or final.is_symlink():
        return verify_file(final, pin)
    part = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=dest_dir, delete=False, suffix=".part"
        ) as tmp:
            part = Path(tmp.name)
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
        verify_file(part, pin)
        try:
            os.link(part, final)  # atomic no-replace; never ordinary replace/rename
        except FileExistsError:
            verify_file(final, pin)
        return final
    except Exception as error:
        raise RuntimeError(
            f"pinned wheel fetch failed ({upload_failure_category(error)})"
        ) from None
    finally:
        # The temporary handle is closed first, including on Windows/signals.
        if part is not None:
            part.unlink(missing_ok=True)


def _verify_remote(client, bucket, key, pin, *, sidecar=False):
    body = client.get_object(Bucket=bucket, Key=key)["Body"]
    try:
        if sidecar:
            raw = body.read(65537)
            if len(raw) > 65536:
                raise ValueError("wheel sidecar exceeds 64 KiB")
            stored = validate_pin(json.loads(raw))
            if any(stored.get(k) != v for k, v in pin.items()):
                raise ValueError("immutable wheel sidecar identity differs")
        else:
            digest, size = hashlib.sha256(), 0
            for chunk in iter(lambda: body.read(1 << 20), b""):
                size += len(chunk)
                if size > pin["size"]:
                    raise ValueError("immutable wheel is larger than pinned")
                digest.update(chunk)
            if size != pin["size"] or digest.hexdigest() != pin["sha256"]:
                raise ValueError("immutable wheel bytes differ")
    finally:
        body.close()


def upload(wheel, name, version, compat, *, client=None, bucket=None):
    """Create-only upload to wheels/<name>/<sha256>.whl (+ .json); returns the pin.

    Existing objects count as present only after full byte/sidecar verification;
    neither a content-addressed name nor equal length alone establishes identity.
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
    verify_file(wheel, pin)
    client = client if client is not None else model_source.r2_client()
    if client is None:
        raise RuntimeError("R2 credentials not configured")
    bucket = bucket or os.environ["R2_BUCKET"]
    sidecar = {
        **pin,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "uploaded_by": "goodvision-harness wheel_cache",
        "compatibility_evidence": "producer-declared; requires runtime validation",
    }
    for key, body in ((pin["r2_key"], None), (pin["r2_key"][:-4] + ".json", sidecar)):
        try:
            try:
                if body is None:
                    # Check the same open descriptor that supplies the upload.
                    with wheel.open("rb") as f:
                        digest, read = hashlib.sha256(), 0
                        for chunk in iter(lambda: f.read(1 << 20), b""):
                            digest.update(chunk)
                            read += len(chunk)
                        if read != size or digest.hexdigest() != sha:
                            raise ValueError("wheel changed before upload")
                        f.seek(0)
                        client.put_object(
                            Bucket=bucket, Key=key, Body=f,
                            ContentLength=size, IfNoneMatch="*",
                        )
                else:
                    client.put_object(
                        Bucket=bucket, Key=key, IfNoneMatch="*",
                        Body=json.dumps(body, indent=2).encode(),
                    )
            except Exception as error:
                if upload_failure_category(error) != "PreconditionFailed":
                    raise
                _verify_remote(client, bucket, key, pin, sidecar=body is not None)
        except Exception as error:
            raise RuntimeError(
                "create-only wheel upload failed or is uncertain "
                f"({upload_failure_category(error)}); inspect {key}; "
                "existing objects were not deleted or overwritten"
            ) from None
    return pin


def name_version(wheel):
    """(project name, version) from a wheel filename."""
    match = _FILENAME.fullmatch(Path(wheel).name)
    if not match:
        raise registry.RegistryError("unsupported CUDA wheel filename")
    return match["name"].replace("_", "-"), match["version"]


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
