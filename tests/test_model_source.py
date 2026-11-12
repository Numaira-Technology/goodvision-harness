"""R2-first model prefetch: cache layout, manifest verify, source selection.

No network, no boto3, no torch: the S3 client is a small in-memory fake.
"""

import hashlib
import io
import json
from pathlib import Path

import pytest

from tools import model_source as ms


def _sha(b):
    return hashlib.sha256(b).hexdigest()


class FakeS3:
    """Minimal S3 surface used by pull_from_r2: get_object + download_file."""

    def __init__(self, objects):
        self.objects = objects  # key -> bytes
        self.gets = 0

    def get_object(self, Bucket, Key):
        return {"Body": io.BytesIO(self.objects[Key])}

    def download_file(self, Bucket, Key, Filename, Config=None):
        self.gets += 1
        with open(Filename, "wb") as f:
            f.write(self.objects[Key])


def _bucket(repo, rev, files):
    prefix = f"{ms.R2_PREFIX}/{repo}/{rev}/"
    objects = {prefix + rel: data for rel, data in files.items()}
    manifest = {
        "model": repo,
        "revision": rev,
        "files": {
            rel: {"size": len(data), "sha256": _sha(data)}
            for rel, data in files.items()
        },
    }
    objects[prefix + ms.MANIFEST_NAME] = json.dumps(manifest).encode()
    return objects, manifest


REPO = "openai/clip-vit-large-patch14"
REV = "32bd64288804d66eefd0ccbe215aa642df71cc41"
FILES = {"config.json": b'{"ok": true}', "sub/model.safetensors": b"weights-bytes"}


def test_repo_cache_folder_and_snapshot_path(tmp_path):
    assert ms.repo_cache_folder(REPO) == "models--openai--clip-vit-large-patch14"
    snap = ms.snapshot_dir(tmp_path, REPO, REV)
    assert (
        snap == tmp_path / "models--openai--clip-vit-large-patch14" / "snapshots" / REV
    )


def test_snapshot_dir_rejects_non_sha(tmp_path):
    with pytest.raises(ValueError):
        ms.snapshot_dir(tmp_path, REPO, "main")
    with pytest.raises(ValueError):
        ms.repo_cache_folder("no-slash-here")


def test_pull_from_r2_populates_and_verifies(tmp_path):
    objects, manifest = _bucket(REPO, REV, FILES)
    snap = ms.snapshot_dir(tmp_path, REPO, REV)
    got = ms.pull_from_r2(REPO, REV, snap, client=FakeS3(objects), bucket="b")
    assert got["files"].keys() == manifest["files"].keys()
    assert (snap / "config.json").read_bytes() == FILES["config.json"]
    assert (snap / "sub/model.safetensors").read_bytes() == FILES[
        "sub/model.safetensors"
    ]
    assert ms.verify_snapshot(snap, manifest)
    # Fetching a pinned historical revision must not move a cached branch ref.
    assert not (snap.parent.parent / "refs" / "main").exists()


def test_pull_from_r2_resumes_without_redownloading(tmp_path):
    objects, _ = _bucket(REPO, REV, FILES)
    snap = ms.snapshot_dir(tmp_path, REPO, REV)
    first = FakeS3(objects)
    ms.pull_from_r2(REPO, REV, snap, client=first, bucket="b")
    # Second pull with everything already correct downloads nothing.
    second = FakeS3(objects)
    ms.pull_from_r2(REPO, REV, snap, client=second, bucket="b")
    assert second.gets == 0


def test_verify_snapshot_detects_tamper(tmp_path):
    objects, manifest = _bucket(REPO, REV, FILES)
    snap = ms.snapshot_dir(tmp_path, REPO, REV)
    ms.pull_from_r2(REPO, REV, snap, client=FakeS3(objects), bucket="b")
    assert ms.verify_snapshot(snap, manifest)
    (snap / "config.json").write_bytes(b"tampered")
    assert not ms.verify_snapshot(snap, manifest)


def test_pull_rejects_mismatched_manifest(tmp_path):
    objects, _ = _bucket(REPO, REV, FILES)
    # Corrupt the manifest's declared revision.
    key = f"{ms.R2_PREFIX}/{REPO}/{REV}/{ms.MANIFEST_NAME}"
    bad = json.loads(objects[key])
    bad["revision"] = "0" * 40
    objects[key] = json.dumps(bad).encode()
    snap = ms.snapshot_dir(tmp_path, REPO, REV)
    with pytest.raises(ValueError):
        ms.pull_from_r2(REPO, REV, snap, client=FakeS3(objects), bucket="b")


def _stub_snapshot_download(monkeypatch, snapshot_download):
    """Replace the lazy HF import. Regression CI does not install the package."""
    import sys
    from types import SimpleNamespace

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(snapshot_download=snapshot_download),
    )


def test_ensure_cached_hits_local_without_r2_or_hf(tmp_path, monkeypatch):
    objects, _ = _bucket(REPO, REV, FILES)
    snap = ms.snapshot_dir(tmp_path, REPO, REV)
    ms.pull_from_r2(REPO, REV, snap, client=FakeS3(objects), bucket="b")
    for k in ("R2_ENDPOINT", "R2_BUCKET", "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY"):
        monkeypatch.delenv(k, raising=False)
    _stub_snapshot_download(
        monkeypatch,
        lambda *a, **k: pytest.fail("should not touch HF on cache hit"),
    )
    assert ms.ensure_cached(REPO, REV, tmp_path) == snap


def test_ensure_cached_falls_back_to_hf_without_r2(tmp_path, monkeypatch):
    for k in ("R2_ENDPOINT", "R2_BUCKET", "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY"):
        monkeypatch.delenv(k, raising=False)
    snap = ms.snapshot_dir(tmp_path, REPO, REV)

    def fake_download(repo, revision, cache_dir):
        snap.mkdir(parents=True, exist_ok=True)
        (snap / "config.json").write_bytes(FILES["config.json"])
        return str(snap)

    _stub_snapshot_download(monkeypatch, fake_download)
    assert ms.ensure_cached(REPO, REV, tmp_path) == snap


@pytest.mark.parametrize("error", [ImportError("no boto3"), ValueError("bad endpoint")])
def test_client_initialization_failure_falls_back(tmp_path, monkeypatch, error):
    def unavailable():
        raise error

    monkeypatch.setattr(ms, "r2_client", unavailable)
    calls = []
    snap = ms.snapshot_dir(tmp_path, REPO, REV)
    _stub_snapshot_download(
        monkeypatch, lambda *a, **kw: calls.append((a, kw)) or str(snap)
    )
    assert ms.ensure_cached(REPO, REV, tmp_path) == snap
    assert calls == [((REPO,), {"revision": REV, "cache_dir": str(tmp_path)})]


@pytest.mark.parametrize("failure", ["checksum", "interrupted", "missing"])
def test_failed_r2_pull_does_not_contaminate_hf_cache(
    tmp_path, monkeypatch, failure, capsys
):
    objects, _ = _bucket(REPO, REV, FILES)
    key = f"{ms.R2_PREFIX}/{REPO}/{REV}/sub/model.safetensors"
    if failure == "checksum":
        objects[key] = b"bad weights"
    elif failure == "missing":
        del objects[key]

    class BrokenS3(FakeS3):
        def download_file(self, Bucket, Key, Filename, Config=None):
            if failure == "interrupted" and Key == key:
                from pathlib import Path

                Path(Filename).write_bytes(b"partial")
                raise OSError("secret-that-must-not-appear-in-logs")
            super().download_file(Bucket, Key, Filename, Config)

    monkeypatch.setenv("R2_BUCKET", "b")
    monkeypatch.setattr(ms, "r2_client", lambda: BrokenS3(objects))
    snap = ms.snapshot_dir(tmp_path, REPO, REV)

    def hf_download(*a, **kw):
        # HF's pinned-revision fast path would reuse any existing files here.
        assert not snap.exists()
        snap.mkdir(parents=True)
        for rel, data in FILES.items():
            dest = snap / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(data)
        return str(snap)

    _stub_snapshot_download(monkeypatch, hf_download)
    assert ms.ensure_cached(REPO, REV, tmp_path) == snap
    assert (snap / "sub/model.safetensors").read_bytes() == FILES[
        "sub/model.safetensors"
    ]
    assert not list(tmp_path.glob(".r2-*"))
    assert "secret-that-must-not-appear-in-logs" not in capsys.readouterr().err


def test_r2_publication_replaces_existing_hf_snapshot_and_preserves_main(
    tmp_path, monkeypatch
):
    objects, manifest = _bucket(REPO, REV, FILES)
    snap = ms.snapshot_dir(tmp_path, REPO, REV)
    snap.mkdir(parents=True)
    (snap / "obsolete.py").write_text("old snapshot contents")
    refs = snap.parent.parent / "refs"
    refs.mkdir()
    (refs / "main").write_text("a" * 40)
    monkeypatch.setenv("R2_BUCKET", "b")
    monkeypatch.setattr(ms, "r2_client", lambda: FakeS3(objects))
    _stub_snapshot_download(monkeypatch, lambda *a, **kw: pytest.fail("unexpected HF"))

    assert ms.ensure_cached(REPO, REV, tmp_path) == snap
    assert ms.verify_snapshot(snap, manifest)
    assert not (snap / "obsolete.py").exists()
    assert (refs / "main").read_text() == "a" * 40
    assert not list(tmp_path.glob(".r2-*"))


@pytest.mark.parametrize("damage", ["file", "identity", "json", "schema"])
def test_invalid_local_r2_snapshot_fails_closed(tmp_path, monkeypatch, damage):
    objects, _ = _bucket(REPO, REV, FILES)
    snap = ms.snapshot_dir(tmp_path, REPO, REV)
    ms.pull_from_r2(REPO, REV, snap, client=FakeS3(objects), bucket="b")
    marker = snap / ms.MANIFEST_NAME
    if damage == "file":
        (snap / "config.json").write_bytes(b"corrupted")
    elif damage == "identity":
        manifest = ms.read_manifest(snap)
        manifest["revision"] = "a" * 40
        marker.write_text(json.dumps(manifest))
    elif damage == "json":
        marker.write_text("{")
    else:
        marker.write_text("[]")
    _stub_snapshot_download(
        monkeypatch, lambda *a, **kw: pytest.fail("HF must not reuse invalid files")
    )
    with pytest.raises(ValueError, match="failed verification"):
        ms.ensure_cached(REPO, REV, tmp_path)


@pytest.mark.parametrize(
    "files",
    [
        {},
        [],
        {"../outside": {"size": 1, "sha256": "0" * 64}},
        {"/outside": {"size": 1, "sha256": "0" * 64}},
        {"..\\outside": {"size": 1, "sha256": "0" * 64}},
        {"MANIFEST.json": {"size": 1, "sha256": "0" * 64}},
        {"config.json": {"size": -1, "sha256": "0" * 64}},
        {"config.json": {"size": 1, "sha256": "invalid"}},
    ],
)
def test_reject_invalid_manifest_before_writing(tmp_path, files):
    objects, manifest = _bucket(REPO, REV, FILES)
    manifest["files"] = files
    objects[f"{ms.R2_PREFIX}/{REPO}/{REV}/{ms.MANIFEST_NAME}"] = json.dumps(
        manifest
    ).encode()
    client = FakeS3(objects)
    snap = ms.snapshot_dir(tmp_path, REPO, REV)
    with pytest.raises(ValueError):
        ms.pull_from_r2(REPO, REV, snap, client=client, bucket="b")
    assert client.gets == 0
    assert not snap.exists()
    assert not ms.verify_snapshot(snap, manifest)


def test_no_r2_skips_client(tmp_path, monkeypatch):
    monkeypatch.setattr(ms, "r2_client", lambda: pytest.fail("R2 disabled"))
    snap = ms.snapshot_dir(tmp_path, REPO, REV)
    _stub_snapshot_download(monkeypatch, lambda *a, **kw: str(snap))
    assert ms.ensure_cached(REPO, REV, tmp_path, prefer_r2=False) == snap


@pytest.mark.parametrize("configured", [False, True])
def test_read_only_hf_cache_still_usable(tmp_path, monkeypatch, configured):
    snap = ms.snapshot_dir(tmp_path, REPO, REV)
    snap.mkdir(parents=True)
    (snap / "config.json").write_bytes(FILES["config.json"])
    monkeypatch.setattr(ms, "r2_client", lambda: object() if configured else None)

    def no_staging(**kw):
        if not configured:
            pytest.fail("no R2 credentials must not require cache write access")
        raise PermissionError("read-only cache")

    monkeypatch.setattr(ms, "TemporaryDirectory", no_staging)
    _stub_snapshot_download(monkeypatch, lambda *a, **kw: str(snap))
    assert ms.ensure_cached(REPO, REV, tmp_path) == snap
    assert (snap / "config.json").read_bytes() == FILES["config.json"]


@pytest.mark.parametrize("use_r2", [False, True])
def test_cache_path_matches_hf_tilde_expansion(tmp_path, monkeypatch, use_r2):
    cache = tmp_path / "expanded-cache"
    expanduser = Path.expanduser
    monkeypatch.setattr(
        Path,
        "expanduser",
        lambda path: cache if path == Path("~/cache") else expanduser(path),
    )
    objects, manifest = _bucket(REPO, REV, FILES)
    monkeypatch.setattr(ms, "r2_client", lambda: FakeS3(objects) if use_r2 else None)
    monkeypatch.setenv("R2_BUCKET", "b")

    def hf_download(*a, **kw):
        assert kw["cache_dir"] == str(cache)
        return str(ms.snapshot_dir(cache, REPO, REV))

    _stub_snapshot_download(monkeypatch, hf_download)
    snap = ms.ensure_cached(REPO, REV, "~/cache")
    assert snap == ms.snapshot_dir(cache, REPO, REV)
    if use_r2:
        assert ms.verify_snapshot(snap, manifest)


@pytest.mark.parametrize("error", [OSError("rename failed"), KeyboardInterrupt()])
@pytest.mark.parametrize("rollback_fails", [False, True])
def test_publication_failure_preserves_previous_snapshot(
    tmp_path, monkeypatch, error, rollback_fails
):
    objects, _ = _bucket(REPO, REV, FILES)
    snap = ms.snapshot_dir(tmp_path, REPO, REV)
    snap.mkdir(parents=True)
    (snap / "existing").write_bytes(b"previous HF snapshot")
    monkeypatch.setenv("R2_BUCKET", "b")
    monkeypatch.setattr(ms, "r2_client", lambda: FakeS3(objects))
    rename = Path.rename

    def interrupted(path, target):
        if path.name == "snapshot" and path.parent.parent == tmp_path:
            raise error
        if rollback_fails and path.parent.name.startswith(".r2-previous-"):
            raise OSError("rollback failed")
        return rename(path, target)

    monkeypatch.setattr(Path, "rename", interrupted)
    _stub_snapshot_download(monkeypatch, lambda *a, **kw: pytest.fail("no fallback"))
    with pytest.raises((OSError, KeyboardInterrupt)):
        ms.ensure_cached(REPO, REV, tmp_path)
    if rollback_fails:
        backups = list(snap.parent.glob(".r2-previous-*/snapshot"))
        assert len(backups) == 1
        assert (backups[0] / "existing").read_bytes() == b"previous HF snapshot"
    else:
        assert (snap / "existing").read_bytes() == b"previous HF snapshot"
        assert not list(snap.parent.glob(".r2-previous-*"))


def test_pull_downloads_files_concurrently(tmp_path, monkeypatch):
    """Several files must be in flight at once, not one after another."""
    import threading

    many = {f"model-{i:05d}.safetensors": bytes([i]) * (100 + i) for i in range(6)}
    many["config.json"] = b"{}"
    objects, manifest = _bucket(REPO, REV, many)
    monkeypatch.setenv("R2_PULL_FILE_WORKERS", "3")
    barrier = threading.Barrier(3, timeout=5)

    class ConcurrentS3(FakeS3):
        def __init__(self, objects):
            super().__init__(objects)
            self.lock = threading.Lock()
            self.active = self.peak = 0

        def download_file(self, Bucket, Key, Filename, Config=None):
            with self.lock:
                self.active += 1
                self.peak = max(self.peak, self.active)
            try:
                if Key.endswith("safetensors"):
                    barrier.wait()  # deadlocks (times out) if run serially
                super().download_file(Bucket, Key, Filename, Config)
            finally:
                with self.lock:
                    self.active -= 1

    client = ConcurrentS3(objects)
    snap = ms.snapshot_dir(tmp_path, REPO, REV)
    ms.pull_from_r2(REPO, REV, snap, client=client, bucket="b")
    assert client.peak == 3
    assert client.gets == len(many)
    assert ms.verify_snapshot(snap, manifest)


def test_parallel_failure_stops_queue_and_waits_for_running(tmp_path, monkeypatch):
    """A bad file aborts the pull; no worker keeps writing after it returns."""
    import threading
    import time

    many = {f"f{i}.bin": bytes([i]) * 50 for i in range(8)}
    objects, _ = _bucket(REPO, REV, many)
    objects[f"{ms.R2_PREFIX}/{REPO}/{REV}/f0.bin"] = b"corrupt"
    monkeypatch.setenv("R2_PULL_FILE_WORKERS", "2")
    finished = []

    class SlowS3(FakeS3):
        def download_file(self, Bucket, Key, Filename, Config=None):
            if not Key.endswith("f0.bin"):
                time.sleep(0.05)
            super().download_file(Bucket, Key, Filename, Config)
            finished.append(threading.current_thread().name)

    client = SlowS3(objects)
    snap = ms.snapshot_dir(tmp_path, REPO, REV)
    with pytest.raises(ValueError, match="mismatch"):
        ms.pull_from_r2(REPO, REV, snap, client=client, bucket="b")
    count = len(finished)
    time.sleep(0.2)
    assert len(finished) == count  # nothing still running in the background
    assert client.gets < len(many)  # queued files were cancelled
    assert not (snap / ms.MANIFEST_NAME).exists()


def test_parallelism_env_is_clamped(monkeypatch):
    monkeypatch.setenv("R2_PULL_FILE_WORKERS", "0")
    monkeypatch.setenv("R2_PULL_STREAMS", "nonsense")
    assert ms._parallelism() == (1, ms.DEFAULT_STREAMS)
    monkeypatch.setenv("R2_PULL_FILE_WORKERS", "100000")
    assert ms._parallelism()[0] == 128
