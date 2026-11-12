"""Prepare pinned native environments/checkpoints; no model inference.

Without --execute this prints a plan only. Historical service source is reconstructed
from public Git objects plus the small, already-used 1 FPS patches.
"""
import argparse
import fnmatch
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from .contract import file_hash, write_new

ROOT = Path(__file__).resolve().parents[2]
ASSETS = Path(__file__).with_name("native_assets")
ALIASES = ("clip", "siglip2", "videojev", "decision3")
PATTERNS = ("*.json", "*.safetensors", "*.txt", "*.model", "*.tiktoken", "*.py", "*.jinja")
COMMON = ["numpy==2.2.6", "pillow==11.3.0", "safetensors==0.8.0",
          "opencv-python-headless==4.12.0.88", "certifi"]
PINS = {
    "clip": COMMON + ["transformers==5.14.1", "tokenizers==0.22.2"],
    "videojev": COMMON + ["transformers==5.14.1", "tokenizers==0.22.2", "accelerate==1.10.1",
        "flash-linear-attention==0.4.2", "fla-core==0.4.2", "av==18.1.0",
        "httpx==0.28.1", "PyYAML==6.0.2", "python-dotenv==1.1.1"],
    "decision3": COMMON + ["transformers==5.17.0", "tokenizers==0.23.3", "accelerate==1.10.1",
        "flash-linear-attention==0.5.2", "fla-core==0.5.2", "av==18.1.0",
        "httpx==0.28.1", "PyYAML==6.0.2", "python-dotenv==1.1.1"],
}
WHEELS = {
    "flash-attn": "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3.post2/flash_attn-2.8.3.post2%2Bcu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl",
    "causal-conv1d": "https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.7.0/causal_conv1d-1.7.0%2Bcu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl",
}


def environment_for(alias):
    return "clip" if alias in ("clip", "siglip2") else alias


def profile(alias):
    if alias not in ALIASES:
        raise ValueError("unsupported native model")
    return json.loads((ROOT / "tools/unified_benchmark/profiles" / (alias + ".json")).read_text())


def safe_path(root, name):
    path = (Path(root) / name).resolve()
    if not path.is_relative_to(Path(root).resolve()):
        raise ValueError("path escapes its root")
    return path


def check_files(root, hashes):
    for name, expected in hashes.items():
        if file_hash(safe_path(root, name)) != expected:
            raise ValueError("deployment file differs: " + name)


def export_service(alias, output, repository=ROOT):
    """Recreate, then verify the exact historical service bytes; no CUDA imports."""
    spec = json.loads((ASSETS / "sources.json").read_text())[alias]
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    for name in spec["files"]:
        path = safe_path(output, name)
        path.parent.mkdir(parents=True, exist_ok=True)
        if name in spec["extra_files"]:
            path.write_bytes((ASSETS / spec["extra_files"][name]).read_bytes())
        else:
            path.write_bytes(subprocess.check_output(
                ["git", "-C", str(repository), "show", spec["commit"] + ":" + name]))
    # Windows builds of GNU patch open files in text mode and fail on the
    # LF-only patches and sources; --binary keeps the exact bytes. POSIX is
    # unchanged (the flag is a no-op there, and production runs on Linux).
    binary = ["--binary"] if os.name == "nt" else []
    try:
        subprocess.run(["patch", "-p1", *binary, "--batch", "--forward",
                        "--input", str(ASSETS / spec["patch"])],
                       cwd=output, check=True, capture_output=True)
    except subprocess.CalledProcessError as exc:
        # Keep patch's own diagnosis visible; the patch and sources are public.
        sys.stderr.write((exc.stdout or b"").decode(errors="replace")[-2000:])
        sys.stderr.write((exc.stderr or b"").decode(errors="replace")[-2000:])
        raise
    check_files(output, spec["files"])
    return spec["files"]


def verify_checkpoint(record):
    if record.get("passed") is not True or not record.get("files"):
        raise ValueError("verified checkpoint inventory required")
    root = Path(record["snapshot"])
    names = [item["name"] for item in record["files"]]
    if len(set(names)) != len(names) or not any(n.endswith(".safetensors") for n in names):
        raise ValueError("checkpoint inventory incomplete")
    for item in record["files"]:
        # HF snapshots contain symlinks into the cache's blobs directory.
        name = Path(item["name"])
        if name.is_absolute() or ".." in name.parts:
            raise ValueError("invalid checkpoint filename")
        path = root / name
        if path.stat().st_size != item["bytes"] or file_hash(path) != item["sha256"]:
            raise ValueError("checkpoint differs: " + str(name))


def download_checkpoint(alias, cache, output):
    from huggingface_hub import HfApi, snapshot_download
    cfg = profile(alias)
    info = HfApi().model_info(cfg["model"], revision=cfg["revision"], files_metadata=True)
    if info.sha != cfg["revision"]:
        raise ValueError("official checkpoint revision differs")
    snapshot = Path(snapshot_download(cfg["model"], revision=cfg["revision"],
                    cache_dir=str(cache), allow_patterns=list(PATTERNS), max_workers=4))
    rows = []
    for item in info.siblings:
        if not any(fnmatch.fnmatch(item.rfilename, pat) for pat in PATTERNS):
            continue
        path = snapshot / item.rfilename
        size, sha = path.stat().st_size, file_hash(path)
        if item.size is not None and size != item.size:
            raise ValueError("official checkpoint size differs")
        lfs = item.lfs
        expected = lfs.get("sha256") if isinstance(lfs, dict) else getattr(lfs, "sha256", None)
        if expected:
            if sha != expected:
                raise ValueError("official LFS checksum differs")
        elif item.blob_id:
            h = hashlib.sha1(f"blob {size}\0".encode())
            with path.open("rb") as f:
                for chunk in iter(lambda: f.read(1024 * 1024), b""):
                    h.update(chunk)
            if h.hexdigest() != item.blob_id:
                raise ValueError("official Git checksum differs")
        else:
            raise ValueError("official checkpoint checksum unavailable")
        rows.append({"name": item.rfilename, "bytes": size, "sha256": sha})
    record = {"passed": True, "model": cfg["model"], "revision": cfg["revision"],
              "snapshot": str(snapshot.resolve()), "files": rows}
    verify_checkpoint(record)
    write_new(output, record)


def verify_deployment(root, expected_hash, alias):
    root = Path(root).resolve()
    if file_hash(root / "deployment.json") != expected_hash:
        raise ValueError("deployment SHA256 differs")
    record = json.loads((root / "deployment.json").read_text())
    cfg = profile(alias)
    if record.get("complete") is not True or alias not in record["models"]:
        raise ValueError("deployment incomplete for model")
    check_files(root, record["files"])
    checkpoint = json.loads((root / "checkpoints" / (alias + ".json")).read_text())
    if any(checkpoint[k] != cfg[k] for k in ("model", "revision")):
        raise ValueError("deployment model/revision mismatch")
    verify_checkpoint(checkpoint)
    return record


def prepare(output, models, clip_python, native_python):
    if sys.platform != "linux":
        raise ValueError("native deployment requires Linux x86_64 and CUDA 12.8")
    root = Path(output).resolve()
    root.mkdir(parents=True, exist_ok=False)
    logs = root / "logs"
    logs.mkdir()
    steps = []
    env = dict(os.environ, PYTHONNOUSERSITE="1", PYTHONDONTWRITEBYTECODE="1",
               HF_HUB_DISABLE_XET="1", HF_HUB_DOWNLOAD_TIMEOUT="120", PIP_DEFAULT_TIMEOUT="120")
    env.pop("PYTHONPATH", None)
    def run(name, command):
        with (logs / (name + ".log")).open("xb") as log:
            subprocess.run(command, env=env, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
        steps.append(name)
    try:
        gpu = subprocess.check_output(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"], text=True).strip().splitlines()
        if len(gpu) != 1 or "4090" not in gpu[0] or not 23000 <= float(gpu[0].rsplit(",", 1)[1]) <= 25000:
            raise ValueError("one RTX 4090 24GB required")
        specs = json.loads((ASSETS / "sources.json").read_text())
        files = {}
        for group in sorted({environment_for(a) for a in models}):
            destination = root / group
            destination.mkdir()
            if group != "clip":
                # Fetch only the required public source commit if this checkout is shallow.
                commit = specs[group]["commit"]
                found = subprocess.run(["git", "cat-file", "-e", commit + "^{commit}"], cwd=ROOT, capture_output=True)
                if found.returncode:
                    run(group + "-source-fetch", ["git", "fetch", "origin", commit])
                hashes = export_service(group, destination / "app")
                files.update({f"{group}/app/{name}": h for name, h in hashes.items()})
            interpreter = clip_python if group == "clip" else native_python
            minor = subprocess.check_output([interpreter, "-c", "import sys; print('%d.%d' % sys.version_info[:2])"], text=True).strip()
            if minor != ("3.11" if group == "clip" else "3.12"):
                raise ValueError("use Python3.11 for CLIP/SigLIP2 and Python3.12 for native services")
            run(group + "-venv", [interpreter, "-m", "venv", str(destination / "env")])
            py = str(destination / "env/bin/python")
            run(group + "-bootstrap", [py, "-m", "pip", "install", "pip", "setuptools", "wheel", "packaging", "ninja"])
            run(group + "-torch", [py, "-m", "pip", "install", "torch==2.8.0", "torchvision==0.23.0", "--index-url", "https://download.pytorch.org/whl/cu128"])
            run(group + "-dependencies", [py, "-m", "pip", "install", *PINS[group]])
            if group != "clip":
                run(group + "-abi", [py, "-c", "import torch; assert torch._C._GLIBCXX_USE_CXX11_ABI and torch.version.cuda == '12.8'"])
                wheels = [WHEELS["causal-conv1d"]] + ([WHEELS["flash-attn"]] if group == "videojev" else [])
                run(group + "-native-wheels", [py, "-m", "pip", "install", "--no-deps", *wheels])
            run(group + "-pip-check", [py, "-m", "pip", "check"])
            packages = json.loads(subprocess.check_output([py, "-m", "pip", "list", "--format=json"], env=env, text=True))
            path = destination / "packages.json"
            write_new(path, packages)
            files[str(path.relative_to(root))] = file_hash(path)
        for alias in models:
            group = environment_for(alias)
            command = [str(root / group / "env/bin/python"), "-m", "tools.unified_benchmark.native_deploy",
                       "checkpoint", "--model", alias, "--root", str(root), "--execute"]
            run(alias + "-checkpoint", command)
            name = f"checkpoints/{alias}.json"
            files[name] = file_hash(root / name)
        write_new(root / "deployment.json", {"complete": True, "models": models, "files": files,
            "gpu": gpu, "inference_calls": 0, "source_commits": {a: specs[a]["commit"] for a in models if a in specs}})
        return file_hash(root / "deployment.json")
    except BaseException as e:
        write_new(root / "failure.json", {"completed_steps": steps, "error": str(e), "inference_calls": 0})
        raise


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("prepare", "checkpoint", "verify"))
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--models", nargs="+", choices=ALIASES, default=list(ALIASES))
    p.add_argument("--model", choices=ALIASES)
    p.add_argument("--clip-python", default="python3.11")
    p.add_argument("--native-python", default="python3.12")
    p.add_argument("--deployment-sha256")
    p.add_argument("--execute", action="store_true")
    args = p.parse_args(argv)
    if args.action == "verify":
        if not args.model or not args.deployment_sha256:
            p.error("verify requires --model and --deployment-sha256")
        verify_deployment(args.root, args.deployment_sha256, args.model)
        print("Deployment verified; no model loaded.")
    elif not args.execute:
        print(json.dumps({"plan_only": True, "models": args.models, "root": str(args.root),
                          "installs_and_weight_downloads_require_execute": True, "inference_calls": 0}))
    elif args.action == "checkpoint":
        if not args.model:
            p.error("checkpoint requires --model")
        download_checkpoint(args.model, args.root / environment_for(args.model) / "cache",
                            args.root / "checkpoints" / (args.model + ".json"))
    else:
        if len(set(args.models)) != len(args.models):
            p.error("duplicate models")
        print(json.dumps({"deployment_sha256": prepare(args.root, args.models, args.clip_python, args.native_python)}))


if __name__ == "__main__":
    main()
