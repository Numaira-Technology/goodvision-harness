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
import multiprocessing
import os
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from contextlib import ExitStack
from pathlib import Path
from tempfile import TemporaryDirectory, mkdtemp

# R2 keys mirror the bucket layout written by the upload step:
#   models/<repo>/<revision>/<original HF relative path>  (+ MANIFEST.json)
R2_PREFIX = "models"
MANIFEST_NAME = "MANIFEST.json"

# Download parallelism. A single HTTPS stream to R2 tops out far below a GPU
# box's NIC, so we open several at once: FILE_WORKERS whole files in parallel,
# and within each large file STREAMS byte-range parts in parallel. The HTTP
# connection pool must hold every stream that can be open at the same time.
# Overridable per run (e.g. a slow box): R2_PULL_FILE_WORKERS, R2_PULL_STREAMS.
# By default each file gets its own Python process (R2_PULL_PROCESSES) so TLS
# decryption and response copying are not serialized by one interpreter lock.
# Measured 2026-11-12 on Qwen3-VL-8B (17.55 GB, one Ontario box): one process
# with 4 files x 16 streams took 93 s (190 MB/s); 4 processes x 8 streams took
# 45 s (391 MB/s), on par with s5cmd. R2_PULL_PROCESSES=1 restores threads.
DEFAULT_PROCESSES = 4
DEFAULT_FILE_WORKERS = 4
DEFAULT_STREAMS = 8
PART_SIZE = 64 * 1024 * 1024


def _env_int(name, default, lo=1, hi=128):
    try:
        value = int(os.environ.get(name, default))
    except ValueError:
        return default
    return max(lo, min(hi, value))


def _parallelism():
    return (
        _env_int("R2_PULL_FILE_WORKERS", DEFAULT_FILE_WORKERS),
        _env_int("R2_PULL_STREAMS", DEFAULT_STREAMS),
    )


def _processes():
    return _env_int("R2_PULL_PROCESSES", DEFAULT_PROCESSES, hi=32)


def _transfer_config(streams):
    """boto3 TransferConfig for ranged parallel GETs, or None without boto3."""
    try:
        from boto3.s3.transfer import TransferConfig
    except ImportError:
        return None
    settings = dict(
        multipart_threshold=PART_SIZE,
        multipart_chunksize=PART_SIZE,
        max_concurrency=streams,
        use_threads=True,
    )
    try:
        # R2 is not AWS; keep the pure-Python transfer path (no CRT client).
        return TransferConfig(preferred_transfer_client="classic", **settings)
    except TypeError:  # boto3 older than the CRT option
        return TransferConfig(**settings)


def repo_cache_folder(repo):
    """HF cache folder name: org/name becomes models--org--name."""
    if "/" not in repo or repo.startswith("/") or ".." in repo.split("/"):
        raise ValueError(f"unexpected repo id: {repo!r}")
    return "models--" + repo.replace("/", "--")


def snapshot_dir(cache_dir, repo, revision):
    """The directory from_pretrained(repo, revision, cache_dir) resolves to."""
    if len(revision) != 40 or not all(c in "0123456789abcdef" for c in revision):
        raise ValueError("revision must be a full 40-char commit sha")
    return (
        Path(cache_dir).expanduser().resolve()
        / repo_cache_folder(repo)
        / "snapshots"
        / revision
    )


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
        return json.loads(p.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return None


def _manifest_files(manifest):
    """Validate the complete manifest before touching any of its paths."""
    if not isinstance(manifest, dict):
        raise ValueError("manifest must be an object")
    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError("manifest must list at least one file")
    for rel, meta in files.items():
        if (
            not isinstance(rel, str)
            or any(part in {"", ".", ".."} for part in rel.split("/"))
            or "\\" in rel
            or ":" in rel
            or rel.split("/")[0] == MANIFEST_NAME
        ):
            raise ValueError(f"unsafe path in manifest: {rel!r}")
        if (
            not isinstance(meta, dict)
            or type(meta.get("size")) is not int
            or meta["size"] < 0
            or not isinstance(meta.get("sha256"), str)
            or len(meta["sha256"]) != 64
            or any(c not in "0123456789abcdef" for c in meta["sha256"])
        ):
            raise ValueError(f"invalid file metadata: {rel!r}")
    return files


def verify_snapshot(snap, manifest):
    """True iff every file in the manifest is present with matching size + sha256."""
    snap = Path(snap)
    try:
        files = _manifest_files(manifest)
    except ValueError:
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


def _r2_env_ready():
    need = ("R2_ENDPOINT", "R2_BUCKET", "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY")
    return all(os.environ.get(k) for k in need)


def r2_client():
    """boto3 S3 client for R2, or None when credentials are not configured."""
    if not _r2_env_ready():
        return None
    import boto3
    from botocore.config import Config

    file_workers, streams = _parallelism()
    return boto3.client(
        "s3",
        endpoint_url=os.environ["R2_ENDPOINT"],
        aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
        config=Config(
            signature_version="s3v4",
            retries={"max_attempts": 8, "mode": "standard"},
            # boto3's default pool is 10; extra streams would just queue.
            max_pool_connections=file_workers * streams + 4,
        ),
    )


def _log(*a):
    print(
        f"[model_source {time.strftime('%H:%M:%S')}]", *a, flush=True, file=sys.stderr
    )


def _fetch_one(client, bucket, key, dest, rel, meta, extra):
    """Download one file (unless already correct) and verify it; bytes pulled."""
    dest = Path(dest)
    if (
        dest.is_file()
        and dest.stat().st_size == meta["size"]
        and _sha256(dest) == meta["sha256"]
    ):
        return 0  # resume: keep already-correct files
    dest.parent.mkdir(parents=True, exist_ok=True)
    client.download_file(bucket, key, str(dest), **extra)
    # Hashing one file overlaps with other files still on the wire.
    if dest.stat().st_size != meta["size"] or _sha256(dest) != meta["sha256"]:
        raise ValueError(f"checksum/size mismatch after R2 download: {rel}")
    _log(f"pulled {rel} ({meta['size'] / 1e6:.1f} MB)")
    return meta["size"]


def _fetch_in_child(bucket, key, dest, rel, meta, streams):
    """Process-pool entry point: own client, own connection pool."""
    client = r2_client()
    if client is None:
        raise RuntimeError("R2 credentials not configured")
    extra = {}
    config = _transfer_config(streams)
    if config is not None:
        extra["Config"] = config
    return _fetch_one(client, bucket, key, dest, rel, meta, extra)


def _run_pool(pool, order, submit):
    """Run every file through ``pool``; return bytes downloaded.

    On the first failure queued files are cancelled, and leaving the
    with-block waits for running ones, so no worker writes into staging after
    the caller cleans it up.
    """
    pulled = 0
    with pool:
        futures = [submit(pool, rel, meta) for rel, meta in order]
        try:
            for future in as_completed(futures):
                pulled += future.result()
        except BaseException:
            for future in futures:
                future.cancel()
            raise
    return pulled


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
    manifest = json.loads(
        client.get_object(Bucket=bucket, Key=manifest_key)["Body"].read()
    )
    files = _manifest_files(manifest)
    if (manifest.get("model"), manifest.get("revision")) != (repo, revision):
        raise ValueError("R2 manifest repo/revision mismatch")

    if snap.is_symlink():
        raise ValueError("snapshot directory must not be a symlink")
    snap.mkdir(parents=True, exist_ok=True)
    marker = snap / MANIFEST_NAME
    if marker.is_symlink():
        raise ValueError("manifest must not be a symlink")
    # Check every destination before any byte is written or any thread starts.
    root = snap.resolve()
    for rel in files:
        dest = snap / rel
        if dest.is_symlink() or not dest.resolve().is_relative_to(root):
            raise ValueError(f"unsafe path in manifest: {rel!r}")

    file_workers, streams = _parallelism()
    processes = _processes()
    # Largest first so the long shards start early and small files fill gaps.
    order = sorted(files.items(), key=lambda kv: kv[1]["size"], reverse=True)
    # Children log in with their own client from the inherited environment, so
    # only a real boto3 client with R2_* set qualifies; injected fakes and
    # explicit clients without env credentials keep the in-process threads.
    use_processes = (
        processes > 1
        and _r2_env_ready()
        and type(client).__module__.startswith("botocore")
    )
    started = time.monotonic()
    if use_processes:
        try:
            pulled = _run_pool(
                ProcessPoolExecutor(
                    max_workers=processes,
                    mp_context=multiprocessing.get_context("spawn"),
                ),
                order,
                lambda pool, rel, meta: pool.submit(
                    _fetch_in_child,
                    bucket,
                    prefix + rel,
                    str(snap / rel),
                    rel,
                    meta,
                    streams,
                ),
            )
            workers, unit = processes, "processes"
        except BrokenProcessPool:
            # e.g. the interpreter cannot spawn children from this entry point.
            # Already-correct files are kept; the rest are fetched in threads.
            _log("process pool unavailable; continuing with threads")
            use_processes = False
    if not use_processes:
        extra = {}
        config = _transfer_config(streams)
        if config is not None:
            extra["Config"] = config
        pulled = _run_pool(
            ThreadPoolExecutor(max_workers=file_workers),
            order,
            lambda pool, rel, meta: pool.submit(
                _fetch_one, client, bucket, prefix + rel, snap / rel, rel, meta, extra
            ),
        )
        workers, unit = file_workers, "files"
    elapsed = max(time.monotonic() - started, 1e-6)
    if pulled:
        _log(
            f"R2 pull {pulled / 1e6:.1f} MB in {elapsed:.1f}s "
            f"({pulled / 1e6 / elapsed:.1f} MB/s, "
            f"{workers} {unit} x {streams} streams)"
        )

    marker.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    # A pinned historical revision does not establish the current HF main ref.
    return manifest


def ensure_cached(repo, revision, cache_dir, *, prefer_r2=True):
    """Ensure repo@revision is in the HF cache at ``cache_dir``; return its path.

    Order: verified-local -> R2 (if credentials present) -> HF snapshot_download.
    Returns the snapshot directory that from_pretrained(repo, revision, cache_dir)
    will resolve to.
    """
    cache_dir = Path(cache_dir).expanduser().resolve()
    snap = snapshot_dir(cache_dir, repo, revision)

    local = read_manifest(snap)
    marker = snap / MANIFEST_NAME
    if marker.exists() or marker.is_symlink():
        if (
            not marker.is_symlink()
            and not snap.is_symlink()
            and isinstance(local, dict)
            and (local.get("model"), local.get("revision")) == (repo, revision)
            and verify_snapshot(snap, local)
        ):
            _log(f"cache hit {repo}@{revision[:10]}")
            return snap
        # HF can reuse snapshot files without hashing them, even on fallback.
        # Do not silently load a snapshot we already know has lost integrity.
        raise ValueError(f"cached R2 snapshot failed verification: {snap}")

    if prefer_r2:
        # Never expose partial R2 downloads to HF's existing-file fast path.
        # Publish only after every object and the full manifest are verified.
        with ExitStack() as staging:
            ready = False
            try:
                client = r2_client()
                if client is not None:
                    cache_dir.mkdir(parents=True, exist_ok=True)
                    temporary = staging.enter_context(
                        TemporaryDirectory(prefix=".r2-", dir=cache_dir)
                    )
                    staged = Path(temporary) / "snapshot"
                    pull_from_r2(repo, revision, staged, client=client)
                    ready = True
            except Exception as e:
                # SDK exception text can contain endpoint credentials or tokens.
                _log(f"R2 pull failed ({type(e).__name__}); falling back to HF")
            if ready:
                # Keep publication errors visible rather than falling back to a
                # possibly incomplete snapshot after a filesystem failure.
                snap.parent.mkdir(parents=True, exist_ok=True)
                if snap.is_symlink():
                    raise ValueError("snapshot directory must not be a symlink")
                backup = None
                if snap.exists():
                    # The old cache must survive even if publication and its
                    # rollback both fail. Keep it outside staging's cleanup.
                    backup = Path(mkdtemp(prefix=".r2-previous-", dir=snap.parent))
                    previous = backup / "snapshot"
                    snap.rename(previous)
                try:
                    staged.rename(snap)
                except BaseException:
                    if backup is not None and not snap.exists():
                        previous.rename(snap)
                        backup.rmdir()
                    raise
                if backup is not None:
                    shutil.rmtree(backup)
                _log(f"ready from R2 {repo}@{revision[:10]}")
                return snap

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
    "siglip2": (
        "google/siglip2-base-patch16-224",
        "75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2",
    ),
}


def _known_names():
    names = set(_KNOWN)
    try:
        from tools import registry

        names.update(registry.model_names())
    except Exception:
        pass
    return sorted(names)


def _resolve_known(name):
    import importlib

    try:
        from tools import registry

        return registry.weights_of(name)
    except Exception:
        pass
    entry = _KNOWN[name]
    if isinstance(entry, tuple):
        return entry
    mod = importlib.import_module(entry)
    return mod.MODEL, mod.REVISION


def main(argv=None):
    import argparse

    ap = argparse.ArgumentParser(
        description="Prefetch a pinned model into the HF cache (R2 first, HF fallback)."
    )
    ap.add_argument("--cache-dir", type=Path, required=True)
    ap.add_argument(
        "--model",
        choices=_known_names(),
        help="use the pinned repo+revision from the registry",
    )
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
