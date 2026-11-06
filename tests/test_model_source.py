"""R2-first model prefetch: cache layout, manifest verify, source selection.

No network, no boto3, no torch: the S3 client is a small in-memory fake.
"""

import hashlib
import io
import json

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

    def download_file(self, Bucket, Key, Filename):
        self.gets += 1
        with open(Filename, "wb") as f:
            f.write(self.objects[Key])


def _bucket(repo, rev, files):
    prefix = f"{ms.R2_PREFIX}/{repo}/{rev}/"
    objects = {prefix + rel: data for rel, data in files.items()}
    manifest = {
        "model": repo,
        "revision": rev,
        "files": {rel: {"size": len(data), "sha256": _sha(data)} for rel, data in files.items()},
    }
    objects[prefix + ms.MANIFEST_NAME] = json.dumps(manifest).encode()
    return objects, manifest


REPO = "openai/clip-vit-large-patch14"
REV = "32bd64288804d66eefd0ccbe215aa642df71cc41"
FILES = {"config.json": b'{"ok": true}', "sub/model.safetensors": b"weights-bytes"}


def test_repo_cache_folder_and_snapshot_path(tmp_path):
    assert ms.repo_cache_folder(REPO) == "models--openai--clip-vit-large-patch14"
    snap = ms.snapshot_dir(tmp_path, REPO, REV)
    assert snap == tmp_path / "models--openai--clip-vit-large-patch14" / "snapshots" / REV


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
    assert (snap / "sub/model.safetensors").read_bytes() == FILES["sub/model.safetensors"]
    assert ms.verify_snapshot(snap, manifest)
    # refs/main written for branch-based resolvers
    assert (snap.parent.parent / "refs" / "main").read_text() == REV


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
    bad = json.loads(objects[key]); bad["revision"] = "0" * 40
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
