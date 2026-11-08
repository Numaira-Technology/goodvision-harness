"""No-GPU tests for the R2 env registry + restore path (registry.py, env_source.py)."""

import io
import json
import tarfile
from pathlib import Path

import pytest

from tools import build_env, env_source, registry

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


def test_env_of_clip_has_env_without_flash_attn():
    env = registry.env_of("clip")
    assert env["r2_key"].startswith("envs/clip/")
    assert env["unpack_to"] == "/opt/env/clip"
    assert env["build"]["flash_attn"] is False


def test_new_models_registered_with_pinned_revisions():
    names = registry.model_names()
    for name in ("siglip2", "d3-nano", "qwen8b", "qwen8b-vllm"):
        assert name in names
        registry.weights_of(name)  # also validates the 40-char sha
        assert registry.env_of(name)["compat"]["gpu_family"] == "ada"
    assert registry.weights_of("qwen8b") == (
        "Qwen/Qwen3-VL-8B-Instruct",
        "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b",
    )
    assert registry.weights_of("qwen8b-vllm") == registry.weights_of("qwen8b")
    # flash-attn on for the VLMs, off for the ViTs.
    assert registry.env_of("qwen8b")["build"]["flash_attn"] is True
    assert registry.env_of("d3-nano")["build"]["flash_attn"] is True
    assert registry.env_of("siglip2")["build"]["flash_attn"] is False
    vllm_env = registry.env_of("qwen8b-vllm")
    hf_env = registry.env_of("qwen8b")
    assert vllm_env["r2_key"] != hf_env["r2_key"]
    assert vllm_env["r2_key"].startswith("envs/qwen8b-vllm/")
    assert vllm_env["unpack_to"] != hf_env["unpack_to"]
    assert vllm_env["build"]["flash_attn"] is False
    assert vllm_env["build"]["install_default_torch"] is False
    assert vllm_env["build"]["require_cuda"] is False
    assert len(vllm_env["sha256"]) == 64
    assert int(vllm_env["size"]) == 4236991211


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


def test_ensure_env_errors_without_packed_env(tmp_path):
    # A model whose env is null has no packed env to restore.
    reg = tmp_path / "models.yaml"
    reg.write_text(
        "version: 1\nmodels:\n  noenv:\n    repo: a/b\n"
        "    revision: " + ("a" * 40) + "\n    env: null\n"
    )
    with pytest.raises(registry.RegistryError):
        env_source.ensure_env("noenv", registry_path=reg)


# --- build_env flash-attn toggle -------------------------------------------

def _script(flash_attn):
    return build_env._build_script(
        Path("/opt/miniforge3/bin/conda"), "/opt/env/x",
        "tools/x/requirements.txt", "8.9", "3.12", flash_attn=flash_attn,
    )


def test_build_script_includes_flash_attn_by_default():
    script = _script(True)
    assert "flash-attn==2.8.3.post2 causal-conv1d==1.7.0" in script
    assert "import torch, flash_attn" in script


def test_build_script_skips_flash_attn_when_false():
    script = _script(False)
    # The compile-once pip step and the flash_attn import are both gone.
    assert "flash-attn==2.8.3.post2" not in script
    assert "--no-build-isolation" not in script
    assert "flash_attn.__version__" not in script
    assert "import torch, flash_attn" not in script
    # torch is still installed and still verified.
    assert "torch==2.8.0" in script
    assert "import torch\n" in script


def test_build_script_always_installs_torch_and_requirements():
    for flash_attn in (True, False):
        script = _script(flash_attn)
        assert "torch==2.8.0 torchvision==0.23.0" in script
        assert '-r "tools/x/requirements.txt"' in script


def test_build_script_skips_default_torch_when_disabled():
    script = build_env._build_script(
        Path("/opt/miniforge3/bin/conda"), "/opt/env/x",
        "tools/qwen8b_vllm/requirements.txt", "8.9", "3.12",
        flash_attn=False, install_default_torch=False,
    )
    assert "pip install --no-cache-dir torch==2.8.0" not in script
    assert "install_default_torch: false" in script
    assert '-r "tools/qwen8b_vllm/requirements.txt"' in script


def test_build_script_cpu_pack_skips_cuda_device_assert():
    script = build_env._build_script(
        Path("/opt/miniforge3/bin/conda"), "/opt/env/x",
        "tools/qwen8b_vllm/requirements.txt", "8.9", "3.12",
        flash_attn=False, install_default_torch=False, require_cuda=False,
    )
    assert "torch.cuda.is_available()" not in script
    assert "cpu-pack: CUDA wheels installed" in script
    gpu = build_env._build_script(
        Path("/opt/miniforge3/bin/conda"), "/opt/env/x",
        "tools/x/requirements.txt", "8.9", "3.12",
        flash_attn=True, require_cuda=True,
    )
    assert "torch.cuda.is_available()" in gpu


# --- build_env --upload-only / --skip-upload -------------------------------

class _FakeUploadClient:
    def __init__(self):
        self.uploaded = []
        self.objects = {}

    def upload_file(self, path, bucket, key):
        self.uploaded.append((path, bucket, key))

    def put_object(self, Bucket, Key, Body):
        self.objects[(Bucket, Key)] = Body


def test_skip_and_upload_only_are_mutually_exclusive():
    with pytest.raises(SystemExit):
        build_env.main(["--model", "videojev", "--skip-upload", "--upload-only"])


def test_upload_only_requires_existing_tarball(tmp_path):
    missing = tmp_path / "nope.tar.gz"
    with pytest.raises(SystemExit):
        build_env.main(["--model", "videojev", "--upload-only", "--out", str(missing)])


def test_upload_only_uploads_without_building(tmp_path, monkeypatch, capsys):
    tarball = tmp_path / "videojev-env.tar.gz"
    tarball.write_bytes(b"packed-env-bytes")
    fake = _FakeUploadClient()
    monkeypatch.setattr(build_env.model_source, "r2_client", lambda: fake)
    monkeypatch.setenv("R2_BUCKET", "goodvision-models")
    # Building must NOT happen in upload-only mode.
    def _boom(*a, **k):
        raise AssertionError("upload-only must not build/pack")
    monkeypatch.setattr(build_env, "_ensure_miniconda", _boom)
    monkeypatch.setattr(build_env, "_pack", _boom)

    build_env.main(["--model", "videojev", "--upload-only", "--out", str(tarball)])

    result = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert result["uploaded"] is True
    assert result["r2_key"].startswith("envs/videojev/")
    assert len(fake.uploaded) == 1
    assert fake.uploaded[0][0] == str(tarball)
    # the side-manifest was written too
    assert any(k.endswith(".json") for (_b, k) in fake.objects)
