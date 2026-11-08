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

# Never overwrite these published objects. New packs get a new r2_key.
FROZEN_R2_KEYS = (
    "envs/qwen8b/cu128-py312-torch280-ada.tar.gz",
    "envs/qwen8b-vllm/cu128-py312-vllm0310-ada.tar.gz",
)

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


def _build_script(conda, prefix, requirements, python_ver, arch, flash_attn=True,
                  install_default_torch=True, require_cuda=True,
                  extra_imports=()):
    """The bash recipe that creates + fills the env -- a pure, testable string.

    ``flash_attn`` gates the slow compile-once step: when False, flash-attn and
    causal-conv1d are NOT installed (and the readiness check does not import
    flash_attn). Models whose runtime never touches flash-attn (e.g. a CLIP/SigLIP
    ViT) set ``env.build.flash_attn: false`` to skip minutes of nvcc compile.

    ``install_default_torch`` installs the HF-pack pin ``torch==2.8.0`` (cu128).
    vLLM envs set it false so ``vllm==0.31.0`` can pull its own torch. Never
    reuse this to overwrite the existing qwen8b HF tarball key.
    """
    conda_sh = f'source "{Path(conda).parent.parent}/etc/profile.d/conda.sh"'
    if flash_attn:
        flash_step = (
            f'export TORCH_CUDA_ARCH_LIST="{arch}"\n'
            'export MAX_JOBS="${MAX_JOBS:-4}"\n'
            'python -s -m pip install --no-user --no-cache-dir --no-build-isolation \\\n'
            '  flash-attn==2.8.3.post2 causal-conv1d==1.7.0'
        )
        cuda_check = (
            'assert torch.cuda.is_available(), "CUDA not available in built env"'
            if require_cuda else
            'assert torch.version.cuda, "packed torch is not a CUDA wheel"\n'
            'print("cpu-pack: CUDA wheels installed; device check deferred to GPU restore")'
        )
        extra = (", " + ", ".join(extra_imports)) if extra_imports else ""
        verify = (
            f"import torch, flash_attn, typing_extensions{extra}\n"
            'print("torch", torch.__version__, "cuda", torch.version.cuda)\n'
            'print("flash_attn", flash_attn.__version__)\n'
            f"{cuda_check}"
        )
    else:
        flash_step = "# flash-attn/causal-conv1d skipped (env.build.flash_attn: false)"
        cuda_check = (
            'assert torch.cuda.is_available(), "CUDA not available in built env"'
            if require_cuda else
            'assert torch.version.cuda, "packed torch is not a CUDA wheel"\n'
            'print("cpu-pack: CUDA wheels installed; device check deferred to GPU restore")'
        )
        extra = (", " + ", ".join(extra_imports)) if extra_imports else ""
        verify = (
            f"import torch, typing_extensions{extra}\n"
            'print("torch", torch.__version__, "cuda", torch.version.cuda)\n'
            f"{cuda_check}"
        )
    if install_default_torch:
        torch_step = (
            "python -s -m pip install --no-user --no-cache-dir torch==2.8.0 torchvision==0.23.0 \\\n"
            "  --index-url https://download.pytorch.org/whl/cu128"
        )
    else:
        torch_step = (
            "# default torch==2.8.0 pin skipped "
            "(env.build.install_default_torch: false)"
        )
    # One bash block so `conda activate` sticks for the pip steps.
    return f"""
set -euo pipefail
{conda_sh}
rm -rf "{prefix}"
conda create -y -p "{prefix}" -c conda-forge --override-channels python={python_ver}
conda activate "{prefix}"
export PYTHONNOUSERSITE=1
python -s -m pip install --no-user --no-cache-dir --upgrade pip setuptools wheel packaging ninja
{torch_step}
python -s -m pip install --no-user --no-cache-dir -r "{requirements}"
{flash_step}
python -s - <<'PY'
import sys
assert sys.flags.no_user_site, "verify must run with python -s"
{verify}
for _name, _mod in list(sys.modules.items()):
    _f = getattr(_mod, "__file__", None)
    if not _f:
        continue
    if ".local/lib/python" in _f:
        raise SystemExit(f"user-site leak: {{_name}} from {{_f}}")
print("isolated prefix ok", sys.executable)
PY
"""


def _build_prefix(conda, prefix, requirements, python_ver, arch, flash_attn=True,
                  install_default_torch=True, require_cuda=True, extra_imports=()):
    """Create the conda env at ``prefix`` and install the full stack into it."""
    script = _build_script(conda, prefix, requirements, python_ver, arch, flash_attn,
                           install_default_torch, require_cuda, extra_imports)
    _run(["bash", "-c", script], env={**os.environ, "HF_HUB_DISABLE_XET": "1",
                                      "PYTHONNOUSERSITE": "1"})


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


def _isolated_smoke(python: Path, prefix: Path, modules: tuple[str, ...]) -> dict:
    """Import modules with python -s and require every file to live under prefix."""
    names = tuple(modules) or ("torch", "typing_extensions")
    script = (
        "import importlib, os, sys\n"
        "assert sys.flags.no_user_site, 'need python -s'\n"
        f"prefix = os.path.realpath({str(prefix)!r})\n"
        "loaded = {}\n"
        f"for name in {list(names)!r}:\n"
        "    mod = importlib.import_module(name)\n"
        "    path = os.path.realpath(getattr(mod, '__file__', '') or '')\n"
        "    if not path.startswith(prefix + os.sep):\n"
        "        raise SystemExit(f'import {name} resolved outside prefix: {path}')\n"
        "    loaded[name] = {'file': path, 'version': getattr(mod, '__version__', None)}\n"
        "import torch\n"
        "print('torch', torch.__version__, 'cuda', torch.version.cuda)\n"
        "print('loaded', loaded)\n"
    )
    env = {**os.environ, "PYTHONNOUSERSITE": "1", "PYTHONPATH": ""}
    env.pop("PYTHONHOME", None)
    proc = subprocess.run(
        [str(python), "-s", "-c", script],
        check=True, env=env, capture_output=True, text=True,
    )
    _log(proc.stdout.strip())
    if proc.stderr.strip():
        _log(proc.stderr.strip())
    return {"stdout": proc.stdout, "modules": list(names)}


def _restore_tarball_isolated(tarball: Path, dest: Path, modules: tuple[str, ...]) -> dict:
    """Extract + conda-unpack into a fresh directory and smoke-import in isolation."""
    import shutil
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    _run(["tar", "-xzf", str(tarball), "-C", str(dest)])
    unpacker = dest / "bin" / "conda-unpack"
    python = dest / "bin" / "python"
    if unpacker.is_file():
        _run([str(python), str(unpacker)] if python.is_file() else [str(unpacker)])
    if not python.is_file():
        raise RuntimeError(f"packed python missing after restore: {python}")
    return _isolated_smoke(python, dest, modules)


def _refuse_frozen_key(r2_key: str) -> None:
    if r2_key in FROZEN_R2_KEYS:
        sys.exit(f"refusing to upload over frozen env key {r2_key}")


def _upload(env_spec, name, tarball, compat, repo_rev):
    _refuse_frozen_key(env_spec["r2_key"])
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
                    help="build + pack only; keep the tarball at --out and print its "
                         "path + sha256 + size (upload it later with --upload-only)")
    ap.add_argument("--skip-pack", action="store_true",
                    help="build the prefix only; skip conda-pack (eval first, pack later)")
    ap.add_argument("--pack-only", action="store_true",
                    help="conda-pack an already-built prefix; do not recreate the env")
    ap.add_argument("--upload-only", action="store_true",
                    help="skip build + pack; upload an already-built tarball at --out "
                         "to R2 (so an interrupted/credential-blocked upload needs no "
                         "rebuild). Fails if the tarball is missing")
    args = ap.parse_args(argv)
    if args.skip_upload and args.upload_only:
        ap.error("--skip-upload and --upload-only are mutually exclusive")
    if args.skip_pack and args.upload_only:
        ap.error("--skip-pack and --upload-only are mutually exclusive")
    if args.pack_only and args.skip_pack:
        ap.error("--pack-only and --skip-pack are mutually exclusive")
    if args.pack_only and args.upload_only:
        ap.error("--pack-only and --upload-only are mutually exclusive")

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
    install_default_torch = bool(build.get("install_default_torch", True))
    # Wheel-only stacks (vLLM) may be conda-packed on a CPU VM; GPU restore
    # still uses the CUDA wheels. flash-attn compiles keep require_cuda.
    require_cuda = bool(build.get("require_cuda", True)) or flash_attn
    extra_imports = tuple(
        name for name in (build.get("verify_imports") or ())
        if str(name).isidentifier()
    )
    prefix = env_spec["unpack_to"]
    out = Path(args.out or f"/tmp/{args.model}-env.tar.gz")
    repo, rev = registry.weights_of(args.model)
    if not args.skip_upload:
        _refuse_frozen_key(env_spec["r2_key"])

    if args.upload_only:
        if not out.is_file():
            sys.exit(f"--upload-only: no tarball at {out}; build it first "
                     f"(e.g. with --skip-upload)")
        _log(f"upload-only {args.model}: reusing existing {out}")
    elif args.pack_only:
        if not Path(prefix).is_dir():
            sys.exit(f"--pack-only: missing prefix {prefix}; build it first")
        conda = _ensure_miniconda(args.conda_home)
        _log(f"pack-only {args.model} from {prefix}")
        _pack(conda, prefix, out)
    else:
        _log(f"building {args.model} (python={python_ver} arch={arch} "
             f"flash_attn={flash_attn} install_default_torch={install_default_torch} "
             f"require_cuda={require_cuda})")
        conda = _ensure_miniconda(args.conda_home)
        _build_prefix(conda, prefix, args.requirements, python_ver, arch, flash_attn,
                      install_default_torch, require_cuda, extra_imports)
        if args.skip_pack:
            print(json.dumps({"model": args.model, "prefix": prefix, "packed": False,
                              "uploaded": False, "arch": arch}))
            return
        _pack(conda, prefix, out)
    sha = _sha256(out)
    size = out.stat().st_size
    _log(f"packed {out} sha256={sha} size={size/1e9:.2f}GB")
    isolate_modules = ("torch", "typing_extensions") + extra_imports
    if not args.upload_only and out.is_file():
        _log(f"isolated restore check -> /tmp/{args.model}-isolated-restore")
        _restore_tarball_isolated(
            out, Path(f"/tmp/{args.model}-isolated-restore"), isolate_modules,
        )
        prefix_py = Path(prefix) / "bin" / "python"
        if prefix_py.is_file():
            _isolated_smoke(prefix_py, Path(prefix), isolate_modules)

    if args.skip_upload:
        print(json.dumps({"model": args.model, "tarball": str(out), "sha256": sha,
                          "size": size, "arch": arch, "uploaded": False}))
        return
    manifest = _upload(env_spec, args.model, out, compat, f"{repo}@{rev}")
    print(json.dumps({"model": args.model, "r2_key": manifest["r2_key"],
                      "sha256": manifest["sha256"], "size": manifest["size"],
                      "arch": arch, "uploaded": True}))


if __name__ == "__main__":
    main()
