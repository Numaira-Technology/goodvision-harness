"""Build a model's environment ONCE on a matching GPU, conda-pack it, upload to R2.

Runs ON a rented box whose GPU matches the model's ``env.compat`` (e.g. an RTX
4090 / Ada for CUDA 12.8). It:

  1. installs miniconda if absent (each model env is a self-contained conda env),
  2. creates the env at the registry's fixed ``unpack_to`` prefix,
  3. pip-installs torch (cu128), the model's requirements, and flash-attn --
     the slow, compile-once step,
  4. conda-pack's the prefix into a single tarball,
  5. uploads the tarball + a side-manifest (sha256, size, compat) to R2 at the
     registry's ``env.r2_key`` / ``env.manifest_key``.

After this, tools/env_source.py restores the env on any matching box with a plain
download -- no apt, no pip, no flash-attn compile. Run build_env again only to
refresh a pin or target a new GPU family (a second tarball, not a rebuild of the
first).

Credentials (R2_* read-write) come from the environment, carried over SSH stdin,
never committed and never echoed.
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from tools import model_source, registry

# Miniforge: a miniconda-family distribution defaulting to conda-forge, so the
# build is not blocked by the Anaconda defaults' Terms-of-Service gate.
MINICONDA_URL = ("https://github.com/conda-forge/miniforge/releases/latest/"
                 "download/Miniforge3-Linux-x86_64.sh")
# flash-attn is compiled for one SM; map the registry gpu_family to an arch list.
_ARCH = {"ada": "8.9", "ampere": "8.0;8.6", "hopper": "9.0"}


def _log(*a):
    print(f"[build_env {time.strftime('%H:%M:%S')}]", *a, flush=True, file=sys.stderr)


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _run(cmd, **kw):
    _log("$", cmd if isinstance(cmd, str) else " ".join(cmd))
    subprocess.run(cmd, check=True, **kw)


def _ensure_miniconda(home):
    conda = Path(home) / "bin" / "conda"
    if conda.is_file():
        return conda
    installer = Path("/tmp/miniconda.sh")
    _run(["curl", "-fsSL", "-o", str(installer), MINICONDA_URL])
    _run(["bash", str(installer), "-b", "-p", str(home)])
    return conda


def _build_script(conda, prefix, requirements, python_ver, arch, flash_attn=True):
    """The bash recipe that creates + fills the env -- a pure, testable string.

    ``flash_attn`` gates the slow compile-once step: when False, flash-attn and
    causal-conv1d are NOT installed (and the readiness check does not import
    flash_attn). Models whose runtime never touches flash-attn (e.g. a CLIP/SigLIP
    ViT) set ``env.build.flash_attn: false`` to skip minutes of nvcc compile.
    """
    conda_sh = f'source "{Path(conda).parent.parent}/etc/profile.d/conda.sh"'
    if flash_attn:
        flash_step = (
            f'export TORCH_CUDA_ARCH_LIST="{arch}"\n'
            'export MAX_JOBS="${MAX_JOBS:-4}"\n'
            'python -m pip install --no-cache-dir --no-build-isolation \\\n'
            '  flash-attn==2.8.3.post2 causal-conv1d==1.7.0'
        )
        verify = (
            "import torch, flash_attn\n"
            'print("torch", torch.__version__, "cuda", torch.version.cuda)\n'
            'print("flash_attn", flash_attn.__version__)\n'
            'assert torch.cuda.is_available(), "CUDA not available in built env"'
        )
    else:
        flash_step = "# flash-attn/causal-conv1d skipped (env.build.flash_attn: false)"
        verify = (
            "import torch\n"
            'print("torch", torch.__version__, "cuda", torch.version.cuda)\n'
            'assert torch.cuda.is_available(), "CUDA not available in built env"'
        )
    # One bash block so `conda activate` sticks for the pip steps.
    return f"""
set -euo pipefail
{conda_sh}
conda create -y -p "{prefix}" -c conda-forge --override-channels python={python_ver}
conda activate "{prefix}"
python -m pip install --no-cache-dir --upgrade pip setuptools wheel packaging ninja
python -m pip install --no-cache-dir torch==2.8.0 torchvision==0.23.0 \\
  --index-url https://download.pytorch.org/whl/cu128
python -m pip install --no-cache-dir -r "{requirements}"
{flash_step}
python - <<'PY'
{verify}
PY
"""


def _build_prefix(conda, prefix, requirements, python_ver, arch, flash_attn=True):
    """Create the conda env at ``prefix`` and install the full stack into it."""
    script = _build_script(conda, prefix, requirements, python_ver, arch, flash_attn)
    _run(["bash", "-c", script], env={**os.environ, "HF_HUB_DISABLE_XET": "1"})


def _pack(conda, prefix, out):
    """conda-pack the prefix into a single tarball at ``out``.

    Use the standalone ``conda-pack`` entry point rather than the ``conda pack``
    subcommand: recent conda (current Miniforge) does not always register the
    conda-pack plugin, so ``conda pack`` fails with "invalid choice: 'pack'" while
    the ``conda-pack`` executable (same tool, same flags) works directly.
    """
    script = f"""
set -euo pipefail
source "{conda.parent.parent}/etc/profile.d/conda.sh"
conda activate base
python -m pip install --no-cache-dir conda-pack
conda-pack -p "{prefix}" -o "{out}" --ignore-missing-files --force
"""
    _run(["bash", "-c", script])


def _upload(env_spec, name, tarball, compat, repo_rev):
    client = model_source.r2_client()
    if client is None:
        sys.exit("no R2 client: export read-write R2_* first")
    bucket = os.environ["R2_BUCKET"]
    sha = _sha256(tarball)
    size = Path(tarball).stat().st_size
    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    manifest = {
        "name": name, "r2_key": env_spec["r2_key"], "sha256": sha, "size": size,
        "created_utc": now, "unpack_to": env_spec["unpack_to"],
        "activate_rel": env_spec["activate_rel"], "compat": compat,
        "built_by": "goodvision-harness build_env", "for_model": repo_rev,
        "format": "conda-pack",
    }
    _log(f"uploading {tarball} ({size/1e9:.2f} GB) -> {env_spec['r2_key']}")
    client.upload_file(str(tarball), bucket, env_spec["r2_key"])
    client.put_object(Bucket=bucket, Key=env_spec["manifest_key"],
                      Body=json.dumps(manifest, indent=2).encode())
    return manifest


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Build + conda-pack + upload a model env to R2.")
    ap.add_argument("--model", required=True, help="model name in tools/models.yaml")
    ap.add_argument("--conda-home", default=str(Path.home() / "miniconda3"))
    ap.add_argument("--out", help="tarball path (default: /tmp/<name>-env.tar.gz)")
    ap.add_argument("--requirements",
                    default="tools/videojev_probe/requirements.txt",
                    help="pip requirements installed into the env")
    ap.add_argument("--skip-upload", action="store_true",
                    help="build + pack only; print the tarball path and sha256")
    args = ap.parse_args(argv)

    env_spec = registry.env_of(args.model)
    if env_spec is None:
        sys.exit(f"model {args.model!r} has no env block in the registry")
    compat = env_spec.get("compat") or {}
    gpu_family = compat.get("gpu_family", "ada")
    arch = _ARCH.get(gpu_family)
    if not arch:
        sys.exit(f"unknown gpu_family {gpu_family!r}; known: {sorted(_ARCH)}")
    python_ver = compat.get("python", "3.12")
    # env.build.flash_attn gates the flash-attn/causal-conv1d compile (default on,
    # so videojev and anything omitting the field keep the original behaviour).
    build = env_spec.get("build") or {}
    flash_attn = bool(build.get("flash_attn", True))
    prefix = env_spec["unpack_to"]
    out = Path(args.out or f"/tmp/{args.model}-env.tar.gz")
    repo, rev = registry.weights_of(args.model)

    _log(f"building {args.model} (python={python_ver} arch={arch} "
         f"flash_attn={flash_attn})")
    conda = _ensure_miniconda(args.conda_home)
    _build_prefix(conda, prefix, args.requirements, python_ver, arch, flash_attn)
    _pack(conda, prefix, out)
    sha = _sha256(out)
    _log(f"packed {out} sha256={sha} size={Path(out).stat().st_size/1e9:.2f}GB")

    if args.skip_upload:
        print(json.dumps({"model": args.model, "tarball": str(out), "sha256": sha,
                          "arch": arch, "uploaded": False}))
        return
    manifest = _upload(env_spec, args.model, out, compat, f"{repo}@{rev}")
    print(json.dumps({"model": args.model, "r2_key": manifest["r2_key"],
                      "sha256": manifest["sha256"], "size": manifest["size"],
                      "arch": arch, "uploaded": True}))


if __name__ == "__main__":
    main()
