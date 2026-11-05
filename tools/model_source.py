"""Populate the HuggingFace cache from R2 (fixed model store), HF as fallback.

Runner-level prefetch only: the model services (videojev_core, neural_service)
are unchanged and keep loading by repo id + revision from ``cache_dir`` with
``local_files_only``. This module just fills that cache first, preferring the
team R2 bucket (egress-free, revision-pinned) over a slow HF download.

R2 is read from the environment, mirroring how the controller already carries
HF_TOKEN to the GPU over SSH (never committed, never baked into an image):

    R2_ENDPOINT, R2_BUCKET, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY

With no R2 credentials present this falls back to a normal HF snapshot_download,
so nothing breaks for models not yet mirrored to R2.
"""

import hashlib
import json
import os
import sys
import time
from pathlib import Path

# R2 keys mirror the bucket layout written by the upload step:
#   models/<repo>/<revision>/<original HF relative path>  (+ MANIFEST.json)
R2_PREFIX = "models"
MANIFEST_NAME = "MANIFEST.json"


def repo_cache_folder(repo):
    """HF cache folder name, e.g. ZhuYuhan/VideoJev-4B -> models--ZhuYuhan--VideoJev-4B."""
    if "/" not in repo or repo.startswith("/") or ".." in repo.split("/"):
        raise ValueError(f"unexpected repo id: {repo!r}")
    return "models--" + repo.replace("/", "--")


def snapshot_dir(cache_dir, repo, revision):
    """The directory from_pretrained(repo, revision, cache_dir) resolves to."""
    if len(revision) != 40 or not all(c in "0123456789abcdef" for c in revision):
        raise ValueError("revision must be a full 40-char commit sha")
    return Path(cache_dir) / repo_cache_folder(repo) / "snapshots" / revision


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_manifest(snap):
    p = Path(snap) / MANIFEST_NAME
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text())
    except (ValueError, OSError):
        return None


def verify_snapshot(snap, manifest):
    """True iff every file in the manifest is present with matching size + sha256."""
    snap = Path(snap)
    files = (manifest or {}).get("files") or {}
    if not files:
        return False
    for rel, meta in files.items():
        p = snap / rel
        if p.is_symlink() or not p.is_file():
            return False
        if not p.resolve().is_relative_to(snap.resolve()):
            return False
        if p.stat().st_size != meta["size"]:
            return False
        if _sha256(p) != meta["sha256"]:
            return False
    return True


def r2_client():
    """boto3 S3 client for R2, or None when credentials are not configured."""
    need = ("R2_ENDPOINT", "R2_BUCKET", "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY")
    if not all(os.environ.get(k) for k in need):
        return None
    import boto3
    from botocore.config import Config

    return boto3.client(
        "s3",
        endpoint_url=os.environ["R2_ENDPOINT"],
        aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
        config=Config(signature_version="s3v4", retries={"max_attempts": 8}),
    )


def _log(*a):
    print(f"[model_source {time.strftime('%H:%M:%S')}]", *a, flush=True, file=sys.stderr)


def pull_from_r2(repo, revision, snap, *, client=None, bucket=None):
    """Download models/<repo>/<revision>/ from R2 into ``snap``; return its manifest.

    Verifies every downloaded file against the bucket's MANIFEST.json before
    returning. Raises on any missing credential, missing manifest, or mismatch.
    """
    client = client or r2_client()
    if client is None:
        raise RuntimeError("R2 credentials not configured")
    bucket = bucket or os.environ["R2_BUCKET"]
    snap = Path(snap)
    prefix = f"{R2_PREFIX}/{repo}/{revision}/"

    manifest_key = prefix + MANIFEST_NAME
    manifest = json.loads(client.get_object(Bucket=bucket, Key=manifest_key)["Body"].read())
    if (manifest.get("model"), manifest.get("revision")) != (repo, revision):
        raise ValueError("R2 manifest repo/revision mismatch")

    snap.mkdir(parents=True, exist_ok=True)
    for rel, meta in manifest["files"].items():
        dest = snap / rel
        if dest.is_symlink() or not dest.resolve().is_relative_to(snap.resolve()):
            raise ValueError(f"unsafe path in manifest: {rel!r}")
        if dest.is_file() and dest.stat().st_size == meta["size"] and _sha256(dest) == meta["sha256"]:
            continue  # resume: keep already-correct files
        dest.parent.mkdir(parents=True, exist_ok=True)
        client.download_file(bucket, prefix + rel, str(dest))
        if dest.stat().st_size != meta["size"] or _sha256(dest) != meta["sha256"]:
            raise ValueError(f"checksum/size mismatch after R2 download: {rel}")
        _log(f"pulled {rel} ({meta['size'] / 1e6:.1f} MB)")

    (snap / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2, sort_keys=True))
    # Belt-and-suspenders for tools that resolve a branch ref instead of a sha.
    refs = snap.parent.parent / "refs"
    refs.mkdir(parents=True, exist_ok=True)
    (refs / "main").write_text(revision)
    return manifest


def ensure_cached(repo, revision, cache_dir, *, prefer_r2=True):
    """Ensure repo@revision is in the HF cache at ``cache_dir``; return its path.

    Order: verified-local -> R2 (if credentials present) -> HF snapshot_download.
    Returns the snapshot directory that from_pretrained(repo, revision, cache_dir)
    will resolve to.
    """
    cache_dir = Path(cache_dir)
    snap = snapshot_dir(cache_dir, repo, revision)

    local = read_manifest(snap)
    if local and verify_snapshot(snap, local):
        _log(f"cache hit {repo}@{revision[:10]}")
        return snap

    if prefer_r2 and r2_client() is not None:
        try:
            pull_from_r2(repo, revision, snap)
            _log(f"ready from R2 {repo}@{revision[:10]}")
            return snap
        except Exception as e:  # noqa: BLE001 - fall back to HF on any R2 failure
            _log(f"R2 pull failed ({type(e).__name__}: {e}); falling back to HF")

    from huggingface_hub import snapshot_download

    _log(f"downloading from HF {repo}@{revision[:10]}")
    path = snapshot_download(repo, revision=revision, cache_dir=str(cache_dir))
    return Path(path)


# Each entry is either a module path exposing MODEL/REVISION, or a direct
# (repo, revision) tuple for a model not yet wired into any runtime module.
_KNOWN = {
    "videojev": "tools.videojev_probe.contract",
    "clip": "goodvision_harness.neural_service",
    # SigLIP2 is pinned here (no service imports it yet); move to a runtime
    # module's MODEL/REVISION once it is wired in.
    "siglip2": ("google/siglip2-base-patch16-224", "75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2"),
}


def _resolve_known(name):
    import importlib

    entry = _KNOWN[name]
    if isinstance(entry, tuple):
        return entry
    mod = importlib.import_module(entry)
    return mod.MODEL, mod.REVISION


def main(argv=None):
    import argparse

    ap = argparse.ArgumentParser(description="Prefetch a pinned model into the HF cache (R2 first, HF fallback).")
    ap.add_argument("--cache-dir", type=Path, required=True)
    ap.add_argument("--model", choices=sorted(_KNOWN), help="use the pinned repo+revision from code")
    ap.add_argument("--repo", help="explicit HF repo id (with --revision)")
    ap.add_argument("--revision", help="explicit 40-char commit sha (with --repo)")
    ap.add_argument("--no-r2", action="store_true", help="skip R2, use HF only")
    args = ap.parse_args(argv)

    if args.model:
        repo, revision = _resolve_known(args.model)
    elif args.repo and args.revision:
        repo, revision = args.repo, args.revision
    else:
        ap.error("pass --model, or both --repo and --revision")

    path = ensure_cached(repo, revision, args.cache_dir, prefer_r2=not args.no_r2)
    print(json.dumps({"model": repo, "revision": revision, "path": str(path)}))


if __name__ == "__main__":
    main()
