"""Build once on a compatible GPU and publish NEW immutable environment objects.

No weights are built or uploaded. Existing prefixes, packs and R2 objects are
never replaced. The output is a proposed registry pin; review it before use.
A trusted installed Miniforge, or a local installer plus its independently
verified SHA256, is required. Nothing is fetched from a floating installer URL.
"""

import argparse
import base64
import csv
import hashlib
import io
import json
import os
import posixpath
import re
import subprocess
import sys
import tarfile
import tempfile
import time
from email.parser import Parser
from pathlib import Path

from tools import env_source, model_source, registry

# R2 documents create-once PutObject; do not assume multipart completion supports it.
# https://developers.cloudflare.com/r2/api/s3/extensions/#conditional-operations-in-putobject
MAX_PUT_SIZE = 5 * 1024**3 - 5 * 1024**2
# Published objects. A new pack needs a new key; these must not be replaced.
FROZEN_R2_KEYS = (
    "envs/qwen8b/cu128-py312-torch280-ada.tar.gz",
    "envs/qwen8b-vllm/cu128-py312-vllm0310-ada.tar.gz",
    "envs/qwen8b-vllm/cu130-py312-vllm0310-ada.tar.gz",
    "envs/qwen8b-vllm/cu130-py312-vllm0310-ada2.tar.gz",
    "envs/videojev/cu128-py312-torch280-ada.tar.gz",
    "envs/clip/cu128-py312-torch280-ada.tar.gz",
)
_sha256 = env_source._sha256


def _refuse_frozen_key(r2_key: str) -> None:
    if r2_key in FROZEN_R2_KEYS:
        sys.exit(f"refusing to upload over frozen env key {r2_key}")


def _log(*a):
    print(f"[build_env {time.strftime('%H:%M:%S')}]", *a, flush=True, file=sys.stderr)


def _run(cmd, **kw):
    # Package installers must not inherit storage credentials from the builder.
    child_env = env_source.subprocess_env()
    for key in (
        "PATH",
        "CUDA_HOME",
        "CUDA_PATH",
        "LD_LIBRARY_PATH",
        "CC",
        "CXX",
        "MAX_JOBS",
    ):
        if key in os.environ:
            child_env[key] = os.environ[key]
    child_env.update(kw.pop("env", {}))
    subprocess.run(
        cmd,
        check=True,
        stdin=subprocess.DEVNULL,
        stdout=sys.stderr,
        env=child_env,
        **kw,
    )


def _ensure_miniconda(home, installer=None, installer_sha256=None):
    home = Path(home).expanduser().resolve()
    conda = home / "bin" / "conda"
    if conda.is_file():
        return conda
    if (
        installer is None
        or not isinstance(installer_sha256, str)
        or not re.fullmatch(r"[0-9a-f]{64}", installer_sha256)
    ):
        raise ValueError(
            "install trusted Miniforge first, or supply a local installer and SHA256"
        )
    installer = Path(installer).resolve()
    if _sha256(installer) != installer_sha256:
        raise ValueError("Miniforge installer sha256 mismatch")
    _run(["bash", str(installer), "-b", "-p", str(home)])
    if not conda.is_file():
        raise RuntimeError("Miniforge did not install conda")
    return conda


# conda owns these; pip must never upgrade them. conda-pack copies a noarch
# package's files from the conda package cache, not from the prefix, so a pip
# upgrade of conda's pip packed old pip source under new pip metadata/launchers
# (PR #94 audit of envs/videojev/cu128-py312-torch280-ada.tar.gz).
CONDA_BASE_PACKAGES = ("pip", "setuptools", "wheel")
# Declared, same-version distributions that intentionally share files.
ALLOWED_SHARED_OWNERS = frozenset({"opencv-python", "opencv-python-headless"})
SITE_PACKAGES = re.compile(r"lib/python3\.\d+/site-packages/")
_CONDA_PLACEHOLDER = b"/opt/anaconda1anaconda2" + b"anaconda3"
_CONDA_INSTALLER = None  # set below, after _record_digest


def _canonical(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _conda_records(prefix):
    meta = Path(prefix) / "conda-meta"
    if not meta.is_dir():
        raise RuntimeError(f"{prefix} is not a conda environment")
    return [json.loads(p.read_text()) for p in sorted(meta.glob("*.json"))]


def _distribution_identity(path):
    with path.open(encoding="utf-8") as stream:
        text = stream.read(1024 * 1024 + 1)
    if len(text) > 1024 * 1024:
        raise RuntimeError("distribution metadata exceeds 1 MiB")
    metadata = Parser().parsestr(text, headersonly=True)
    name, version = metadata.get("Name", ""), metadata.get("Version", "")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", name) or not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9.!+_-]*", version
    ):
        raise RuntimeError(f"invalid distribution identity: {path.name}")
    return _canonical(name), version


def _conda_python_dists(prefix):
    """Include conda-owned dist-info and both forms of legacy egg-info."""
    owned = {}
    for rec in _conda_records(prefix):
        for f in rec.get("files", []):
            if re.fullmatch(
                r"lib/python3\.\d+/site-packages/[^/]+"
                r"(?:\.dist-info/METADATA|\.egg-info/PKG-INFO|\.egg-info)",
                f,
            ):
                name, version = _distribution_identity(Path(prefix) / f)
                if name in owned and owned[name] != version:
                    raise RuntimeError(f"conflicting conda versions of {name}")
                owned[name] = version
    return owned


def _write_constraints(prefix, path):
    owned = _conda_python_dists(prefix)
    path.write_text("".join(f"{n}=={v}\n" for n, v in sorted(owned.items())))
    return owned


def check_conda_ownership(prefix):
    """Fail if pip removed or replaced any file of a conda-installed package."""
    prefix = Path(prefix)
    missing = {}
    for rec in _conda_records(prefix):
        gone = [
            f
            for f in rec.get("files", [])
            if not (prefix / f).exists()
            and not (prefix / f).is_symlink()
            and not f.endswith(".pyc")
        ]
        if gone:
            missing[f"{rec.get('name')}=={rec.get('version')}"] = gone[:3]
    if missing:
        raise RuntimeError(f"pip clobbered conda-owned packages: {missing}")
    seen = {}
    for site in sorted(prefix.glob("lib/python3.*/site-packages")):
        for info in sorted([*site.glob("*.dist-info"), *site.glob("*.egg-info")]):
            metadata = (
                info / ("METADATA" if info.suffix == ".dist-info" else "PKG-INFO")
                if info.is_dir()
                else info
            )
            name, _ = _distribution_identity(metadata)
            if name in seen:
                raise RuntimeError(
                    f"two installed versions of {name}: {seen[name]}, {info.name}"
                )
            seen[name] = info.name


def _record_digest(data: bytes) -> str:
    return base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()


_CONDA_INSTALLER = _record_digest(b"conda\n")


def verify_pack(tarball):
    """Hash packed files in bounded chunks; parse installed RECORDs as CSV.

    No packed code executes. Launcher hashes outside site-packages are reported
    separately because conda-pack rewrites their shebangs. A placeholder string
    never exempts a file from its digest. Only same-version OpenCV distributions
    may declare differing shared bytes; an actual owner's digest must match.
    """
    algorithms = ("sha256", "sha384", "sha512")
    files, records, infos, sites = {}, {}, set(), set()
    with tarfile.open(tarball, "r|gz") as tar:
        for member in tar:
            hit = SITE_PACKAGES.match(member.name)
            if not hit:
                continue
            sites.add(hit.group())
            rel = member.name[hit.end() :]
            if not rel or member.isdir():
                continue
            if not member.isfile() or rel in files:
                raise RuntimeError(f"nonregular or duplicate packed file: {rel}")
            if rel != posixpath.normpath(rel) or rel.startswith(("/", "../")):
                raise RuntimeError(f"unsafe packed path: {rel}")
            top = rel.split("/", 1)[0]
            if top.endswith(".dist-info"):
                infos.add(top)
            is_record = re.fullmatch(r"[^/]+\.dist-info/RECORD", rel)
            if is_record and member.size > 16 * 1024**2:
                raise RuntimeError("RECORD exceeds 16 MiB")
            hashes = {algorithm: hashlib.new(algorithm) for algorithm in algorithms}
            chunks, tail, placeholder = [], b"", False
            stream = tar.extractfile(member)
            while data := stream.read(1024 * 1024):
                for digest in hashes.values():
                    digest.update(data)
                placeholder |= _CONDA_PLACEHOLDER in tail + data
                tail = data[-len(_CONDA_PLACEHOLDER) :]
                if is_record:
                    chunks.append(data)
            files[rel] = (
                {
                    key: base64.urlsafe_b64encode(value.digest()).rstrip(b"=").decode()
                    for key, value in hashes.items()
                },
                member.size,
                placeholder,
            )
            if is_record:
                records[rel] = b"".join(chunks).decode("utf-8")
    if len(sites) != 1 or not records:
        raise RuntimeError("pack must contain one site-packages tree with RECORDs")
    for info in sorted(infos):
        if f"{info}/RECORD" not in records or f"{info}/METADATA" not in files:
            raise RuntimeError(f"{info}: missing RECORD or METADATA")
    claims, problems, outside, placeholder, conda_installer = {}, [], 0, 0, 0
    unhashed, dists = 0, {}
    for rec, text in sorted(records.items()):
        dist = rec.split("/", 1)[0][: -len(".dist-info")]
        name, version = dist.rsplit("-", 1)
        name = _canonical(name)
        if name in dists:
            problems.append(f"{name}: two dist-info directories")
        dists[name] = version
        seen, checked = set(), 0
        for row in csv.reader(io.StringIO(text, newline=""), strict=True):
            if len(row) != 3 or not row[0]:
                raise RuntimeError(f"{rec}: malformed RECORD row")
            path, digest, size = row
            path = posixpath.normpath(path)
            if path.startswith("/") or path in seen:
                raise RuntimeError(f"{rec}: absolute or duplicate RECORD path")
            seen.add(path)
            if path == ".." or path.startswith("../"):
                outside += 1
                continue
            if not digest:
                unhashed += 1
                continue
            algorithm, separator, value = digest.partition("=")
            if (
                not separator
                or algorithm not in algorithms
                or not re.fullmatch(r"[A-Za-z0-9_-]+", value)
            ):
                raise RuntimeError(f"{rec}: unsupported or malformed RECORD digest")
            if size and not size.isdecimal():
                raise RuntimeError(f"{rec}: invalid RECORD size")
            claims.setdefault(path, []).append(
                (name, algorithm, value, int(size) if size else None)
            )
            checked += 1
        if not checked:
            raise RuntimeError(f"{rec}: no verifiable RECORD entries")
    shared = {}
    for path, owners in sorted(claims.items()):
        names = {owner[0] for owner in owners}
        if path not in files:
            problems.append(f"{path}: listed by {sorted(names)} but not packed")
            continue
        got, size, has_placeholder = files[path]
        matches = [
            got[a] == digest and (expected is None or size == expected)
            for _, a, digest, expected in owners
        ]
        if len(names) > 1 and not all(matches):
            if names != ALLOWED_SHARED_OWNERS or len({dists[n] for n in names}) != 1:
                problems.append(f"{path}: claimed by {sorted(names)}")
                continue
            key = ",".join(sorted(names))
            shared[key] = shared.get(key, 0) + 1
        if any(matches):
            placeholder += bool(has_placeholder)
            continue
        if path.endswith(".dist-info/INSTALLER") and got["sha256"] == _CONDA_INSTALLER:
            conda_installer += 1
            continue
        problems.append(f"{path}: bytes differ from {sorted(names)} RECORD")
    if problems:
        raise RuntimeError(
            f"pack RECORD check failed ({len(problems)}): " + "; ".join(problems[:10])
        )
    return {
        "dists": len(dists),
        "pip": dists.get("pip"),
        "files_checked": len(claims),
        "unhashed_record_entries": unhashed,
        "outside_site_packages_skipped": outside,
        "conda_placeholder_files": placeholder,
        "conda_installer_rewrites": conda_installer,
        "declared_shared_files": shared,
    }


def _build_prefix(conda, prefix, requirements, python_ver, arch, *, flash_attn=True):
    """Use argv throughout: paths containing shell syntax remain literal paths."""
    _run(
        [
            str(conda),
            "create",
            "-y",
            "-p",
            str(prefix),
            "-c",
            "conda-forge",
            "--override-channels",
            f"python={python_ver}",
            *CONDA_BASE_PACKAGES,
        ]
    )
    with tempfile.TemporaryDirectory() as tmp:
        constraints = Path(tmp) / "conda-owned.txt"
        _write_constraints(prefix, constraints)
        _install_python_packages(prefix, requirements, arch, constraints, flash_attn)


def _install_python_packages(prefix, requirements, arch, constraints, flash_attn):
    python = str(Path(prefix) / "bin/python")
    pip = [
        python,
        "-I",
        "-m",
        "pip",
        "install",
        "--no-cache-dir",
        "--constraint",
        str(constraints),
    ]
    _run([*pip, "packaging", "ninja"])
    _run(
        [
            *pip,
            "torch==2.8.0",
            "torchvision==0.23.0",
            "--index-url",
            "https://download.pytorch.org/whl/cu128",
        ]
    )
    _run([*pip, "-r", str(requirements)])
    if flash_attn:
        _run(
            [
                *pip,
                "--no-build-isolation",
                "flash-attn==2.8.3.post2",
                "causal-conv1d==1.7.0",
            ],
            env={
                "TORCH_CUDA_ARCH_LIST": arch,
                "MAX_JOBS": os.environ.get("MAX_JOBS", "4"),
                "FLASH_ATTENTION_FORCE_BUILD": "TRUE",
                "CAUSAL_CONV1D_FORCE_BUILD": "TRUE",
            },
        )
    checks = [
        "import sys, torch",
        "assert sys.version_info[:2] == (3, 12), 'unexpected Python'",
        "assert torch.__version__ == '2.8.0+cu128', 'unexpected torch'",
        "assert torch.version.cuda == '12.8', 'unexpected CUDA runtime'",
        "assert torch.cuda.is_available(), 'CUDA unavailable'",
    ]
    if flash_attn:
        checks += [
            "import flash_attn, causal_conv1d",
            "assert flash_attn.__version__ == '2.8.3.post2', 'unexpected flash-attn'",
        ]
    _run(
        [
            python,
            "-I",
            "-c",
            "\n".join(checks),
        ]
    )


def _pack(conda, prefix, out):
    base_python = str(conda.parent / "python")
    _run(
        [
            base_python,
            "-I",
            "-m",
            "pip",
            "install",
            "--no-cache-dir",
            "conda-pack==0.9.2",
        ]
    )
    _run([base_python, "-I", "-m", "conda_pack.cli", "-p", str(prefix), "-o", str(out)])


def _upload(env_spec, name, tarball, compat, repo_rev):
    """Create content-addressed objects conditionally; never delete after a timeout.

    A failed/uncertain second write can leave an unreferenced tarball. Preserve it
    for author inspection rather than risking deletion of a committed artifact.
    """
    registry.validate_env(env_spec)
    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", name):
        raise ValueError("invalid model name")
    if compat != env_spec["compat"]:
        raise ValueError("build compatibility differs from registry")
    tarball = Path(tarball)
    sha = _sha256(tarball)
    size = tarball.stat().st_size
    if not 0 < size <= MAX_PUT_SIZE:
        raise ValueError(
            "create-only upload supports packs below 5 GiB; no multipart fallback"
        )
    key = f"envs/{name}/{sha}.tar.gz"
    # Refuse only the object about to be written. A new sha is a successor,
    # not an overwrite of the model's current pin.
    _refuse_frozen_key(key)
    manifest_key = f"envs/{name}/{sha}.json"
    manifest = {
        "name": name,
        "r2_key": key,
        "sha256": sha,
        "size": size,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "unpack_to": env_spec["unpack_to"],
        "activate_rel": env_spec["activate_rel"],
        "compat": compat,
        "built_by": "goodvision-harness build_env",
        "for_model": repo_rev,
        "format": "conda-pack",
    }
    client = model_source.r2_client()
    if client is None:
        raise RuntimeError("R2 credentials not configured")
    bucket = os.environ["R2_BUCKET"]
    try:
        with tarball.open("rb") as body:
            client.put_object(
                Bucket=bucket, Key=key, Body=body, ContentLength=size, IfNoneMatch="*"
            )
        client.put_object(
            Bucket=bucket,
            Key=manifest_key,
            IfNoneMatch="*",
            Body=json.dumps(manifest, indent=2).encode(),
        )
    except Exception:
        raise RuntimeError(
            f"create-only upload failed or is uncertain; inspect {key} "
            f"and {manifest_key}; "
            "existing objects were not deleted or overwritten"
        ) from None
    return {**manifest, "manifest_key": manifest_key}


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Build a new immutable conda-pack environment."
    )
    ap.add_argument("--model", required=True)
    ap.add_argument("--conda-home", default=str(Path.home() / "miniconda3"))
    ap.add_argument("--miniforge-installer", type=Path)
    ap.add_argument("--miniforge-sha256")
    ap.add_argument(
        "--out", type=Path, required=True, help="new tarball path (must not exist)"
    )
    ap.add_argument(
        "--requirements",
        type=Path,
        default=Path(__file__).parent / "videojev_probe/requirements.txt",
    )
    ap.add_argument("--skip-upload", action="store_true")
    args = ap.parse_args(argv)
    env_spec = registry.env_of(args.model)
    if env_spec is None:
        ap.error("model has no env block")
    compat = env_spec["compat"]
    build = env_spec.get("build", {})
    if not isinstance(build, dict) or type(build.get("flash_attn", True)) is not bool:
        ap.error("env.build.flash_attn must be a boolean")
    if any(
        compat[k] != v
        for k, v in {"python": "3.12", "torch": "2.8.0", "cuda": "12.8"}.items()
    ):
        ap.error("this builder supports only Python 3.12 / torch 2.8.0 / CUDA 12.8")
    env_source.check_compatibility(env_spec)
    repo, rev = registry.weights_of(args.model)
    prefix = Path(env_spec["unpack_to"])
    out = args.out.absolute()
    if out.exists() or out.is_symlink():
        ap.error("output already exists; choose a new path")
    if not args.requirements.is_file():
        ap.error("requirements file does not exist")
    if out.is_relative_to(prefix):
        ap.error("output must be outside the environment prefix")
    arch = ";".join(registry.GPU_ARCHES[compat["gpu_family"]])
    with env_source.prefix_lock(prefix), env_source.interruptible():
        if prefix.exists():
            ap.error("preserving existing environment; inspect and move it aside first")
        conda = _ensure_miniconda(
            args.conda_home, args.miniforge_installer, args.miniforge_sha256
        )
        _build_prefix(
            conda,
            prefix,
            args.requirements.resolve(),
            compat["python"],
            arch,
            flash_attn=build.get("flash_attn", True),
        )
        check_conda_ownership(prefix)
        _pack(conda, prefix, out)
    pack_check = verify_pack(out)
    sha = _sha256(out)
    result = {
        "model": args.model,
        "tarball": str(out),
        "sha256": sha,
        "size": out.stat().st_size,
        "arch": arch,
        "pack_check": pack_check,
        "uploaded": False,
    }
    if not args.skip_upload:
        result.update(_upload(env_spec, args.model, out, compat, f"{repo}@{rev}"))
        result["uploaded"] = True
    print(json.dumps(result))


if __name__ == "__main__":
    main()
