"""Build once on a compatible GPU and publish NEW immutable environment objects.

No weights are built or uploaded. Existing prefixes, packs and R2 objects are
never replaced. The output is a proposed registry pin; review it before use.
A trusted installed Miniforge, or a local installer plus its independently
verified SHA256, is required. Nothing is fetched from a floating installer URL.
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

from tools import env_source, model_source, registry

# R2 documents create-once PutObject; do not assume multipart completion supports it.
# https://developers.cloudflare.com/r2/api/s3/extensions/#conditional-operations-in-putobject
MAX_PUT_SIZE = 5 * 1024**3 - 5 * 1024**2
_sha256 = env_source._sha256


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
        "HF_HUB_DISABLE_XET",
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


def _prefix_python(prefix):
    return str(Path(prefix) / "bin/python")


def _verify_code(flash_attn):
    lines = [
        "import sys, torch",
        "assert sys.version_info[:2] == (3, 12), 'unexpected Python'",
        "assert torch.__version__ == '2.8.0+cu128', 'unexpected torch'",
        "assert torch.version.cuda == '12.8', 'unexpected CUDA runtime'",
        "assert torch.cuda.is_available(), 'CUDA unavailable'",
    ]
    if flash_attn:
        lines[0] = "import sys, torch, flash_attn, causal_conv1d"
        lines.insert(
            4,
            "assert flash_attn.__version__ == '2.8.3.post2', 'unexpected flash-attn'",
        )
    return "\n".join(lines)


def _build_prefix(conda, prefix, requirements, python_ver, arch, flash_attn=True):
    """Use argv throughout: paths containing shell syntax remain literal paths.

    ``flash_attn`` gates the slow compile-once step: when False, flash-attn and
    causal-conv1d are NOT installed (and the readiness check does not import
    flash_attn). Models whose runtime never touches flash-attn (e.g. a CLIP/SigLIP
    ViT) set ``env.build.flash_attn: false`` to skip minutes of nvcc compile.
    Every pip/verify step uses the new prefix's ``bin/python -I``, never PATH
    python or ``conda activate``.
    """
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
        ]
    )
    python = _prefix_python(prefix)
    pip = [python, "-I", "-m", "pip", "install", "--no-cache-dir"]
    _run([*pip, "--upgrade", "pip", "setuptools", "wheel", "packaging", "ninja"])
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
    _run([python, "-I", "-c", _verify_code(flash_attn)])


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
        "--out",
        type=Path,
        required=True,
        help="tarball path (must not exist for build; must exist for --upload-only)",
    )
    ap.add_argument(
        "--requirements",
        type=Path,
        default=Path(__file__).parent / "videojev_probe/requirements.txt",
    )
    ap.add_argument(
        "--skip-upload",
        action="store_true",
        help="build + pack only; keep the tarball at --out",
    )
    ap.add_argument(
        "--upload-only",
        action="store_true",
        help="upload an already-built tarball at --out; do not rebuild",
    )
    args = ap.parse_args(argv)
    if args.skip_upload and args.upload_only:
        ap.error("--skip-upload and --upload-only are mutually exclusive")
    env_spec = registry.env_of(args.model)
    if env_spec is None:
        ap.error("model has no env block")
    compat = env_spec["compat"]
    if any(
        compat[k] != v
        for k, v in {"python": "3.12", "torch": "2.8.0", "cuda": "12.8"}.items()
    ):
        ap.error("this builder supports only Python 3.12 / torch 2.8.0 / CUDA 12.8")
    repo, rev = registry.weights_of(args.model)
    prefix = Path(env_spec["unpack_to"])
    out = args.out.absolute()
    arch = ";".join(registry.GPU_ARCHES[compat["gpu_family"]])
    flash_attn = bool((env_spec.get("build") or {}).get("flash_attn", True))
    if args.upload_only:
        if not out.is_file():
            ap.error(
                f"--upload-only: no tarball at {out}; build it first "
                "(e.g. with --skip-upload)"
            )
        _log(f"upload-only {args.model}: reusing existing {out}")
    else:
        env_source.check_compatibility(env_spec)
        if out.exists() or out.is_symlink():
            ap.error("output already exists; choose a new path")
        if not args.requirements.is_file():
            ap.error("requirements file does not exist")
        if out.is_relative_to(prefix):
            ap.error("output must be outside the environment prefix")
        with env_source.prefix_lock(prefix), env_source.interruptible():
            if prefix.exists():
                ap.error(
                    "preserving existing environment; inspect and move it aside first"
                )
            conda = _ensure_miniconda(
                args.conda_home, args.miniforge_installer, args.miniforge_sha256
            )
            _build_prefix(
                conda,
                prefix,
                args.requirements.resolve(),
                compat["python"],
                arch,
                flash_attn,
            )
            _pack(conda, prefix, out)
    sha = _sha256(out)
    result = {
        "model": args.model,
        "tarball": str(out),
        "sha256": sha,
        "size": out.stat().st_size,
        "arch": arch,
        "flash_attn": flash_attn,
        "uploaded": False,
    }
    if not args.skip_upload:
        result.update(_upload(env_spec, args.model, out, compat, f"{repo}@{rev}"))
        result["uploaded"] = True
    print(json.dumps(result))


if __name__ == "__main__":
    main()
