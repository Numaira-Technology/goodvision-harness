"""No-GPU tests for the R2 env registry + restore path (registry.py, env_source.py)."""

import io
import json
import tarfile
from pathlib import Path

import pytest

from tools import env_source, registry

VIDEOJEV_REV = "1c1a253bf90dfa66176124d25a48ffb791d517d5"


# --- registry.py -----------------------------------------------------------

def test_registry_ships_and_parses():
    assert "videojev" in registry.model_names()
    repo, rev = registry.weights_of("videojev")
    assert repo == "ZhuYuhan/VideoJev-4B"
    assert rev == VIDEOJEV_REV


def test_env_of_videojev_has_required_keys():
    env = registry.env_of("videojev")
    assert env["r2_key"].startswith("envs/videojev/")
    assert env["unpack_to"] == "/opt/env/videojev"
    assert env["activate_rel"] == "bin/activate"
    assert env["compat"]["gpu_family"] == "ada"


def test_env_of_clip_is_none():
    assert registry.env_of("clip") is None


def test_unknown_model_raises():
    with pytest.raises(registry.RegistryError):
        registry.weights_of("does-not-exist")


def test_bad_revision_rejected(tmp_path):
    bad = tmp_path / "models.yaml"
    bad.write_text("version: 1\nmodels:\n  x:\n    repo: a/b\n    revision: short\n")
    with pytest.raises(registry.RegistryError):
        registry.weights_of("x", path=bad)


# --- a fake env tarball + fake R2 client -----------------------------------

def _make_env_tarball(dest_tar: Path):
    """Build a tiny conda-pack-shaped env: bin/activate + a no-op bin/conda-unpack."""
    staging = dest_tar.parent / "staging"
    (staging / "bin").mkdir(parents=True, exist_ok=True)
    (staging / "bin" / "activate").write_text("# fake activate\n")
    unpack = staging / "bin" / "conda-unpack"
    unpack.write_text("#!/bin/sh\nexit 0\n")
    unpack.chmod(0o755)
    with tarfile.open(dest_tar, "w:gz") as tf:
        for p in sorted(staging.rglob("*")):
            tf.add(p, arcname=p.relative_to(staging).as_posix())
    return dest_tar


class _FakeR2:
    """Minimal stand-in for the boto3 S3 client env_source uses."""

    def __init__(self, tarball: Path, manifest: dict):
        self._tarball = tarball
        self._manifest = manifest

    def download_file(self, bucket, key, dest):
        Path(dest).write_bytes(self._tarball.read_bytes())

    def get_object(self, Bucket, Key):
        return {"Body": io.BytesIO(json.dumps(self._manifest).encode())}


def _sha256(p):
    import hashlib

    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


# --- env_source.pull_env ---------------------------------------------------

def test_pull_env_extracts_and_marks(tmp_path, monkeypatch):
    tarball = _make_env_tarball(tmp_path / "env.tar.gz")
    sha = _sha256(tarball)
    unpack_to = tmp_path / "opt" / "videojev"
    env = {"r2_key": "envs/videojev/x.tar.gz", "manifest_key": "envs/videojev/x.json",
           "unpack_to": str(unpack_to), "activate_rel": "bin/activate", "sha256": sha}
    manifest = {"sha256": sha, "size": tarball.stat().st_size}
    monkeypatch.setenv("R2_BUCKET", "goodvision-models")

    activate = env_source.pull_env(env, client=_FakeR2(tarball, manifest),
                                   scratch=str(tmp_path / "scratch"))
    assert Path(activate).is_file()
    assert activate == str(unpack_to / "bin" / "activate")
    marker = json.loads((unpack_to / ".r2_env.json").read_text())
    assert marker["sha256"] == sha


def test_pull_env_rejects_sha_mismatch(tmp_path, monkeypatch):
    tarball = _make_env_tarball(tmp_path / "env.tar.gz")
    unpack_to = tmp_path / "opt" / "videojev"
    env = {"r2_key": "k", "manifest_key": "m", "unpack_to": str(unpack_to),
           "activate_rel": "bin/activate", "sha256": "0" * 64}
    manifest = {"sha256": "0" * 64, "size": tarball.stat().st_size}
    monkeypatch.setenv("R2_BUCKET", "goodvision-models")
    with pytest.raises(ValueError, match="sha256 mismatch"):
        env_source.pull_env(env, client=_FakeR2(tarball, manifest),
                            scratch=str(tmp_path / "scratch"))
    assert not unpack_to.exists()


def test_already_restored_short_circuits(tmp_path):
    unpack_to = tmp_path / "opt" / "videojev"
    (unpack_to / "bin").mkdir(parents=True)
    (unpack_to / "bin" / "activate").write_text("x")
    (unpack_to / ".r2_env.json").write_text(json.dumps({"sha256": "abc"}))
    env = {"unpack_to": str(unpack_to), "activate_rel": "bin/activate", "sha256": "abc"}
    assert env_source._already_restored(env) is True
    env_wrong = dict(env, sha256="different")
    assert env_source._already_restored(env_wrong) is False


def test_ensure_env_errors_without_packed_env():
    with pytest.raises(registry.RegistryError):
        env_source.ensure_env("clip")  # clip has env: null
