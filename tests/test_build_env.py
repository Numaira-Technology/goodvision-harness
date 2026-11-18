"""Builder safety tests with subprocess and R2 mocks; no package builds/uploads."""

import copy
import os

import pytest

from tools import build_env as be
from tools import registry


class FakeR2:
    def __init__(self, fail_at=None):
        self.calls = []
        self.fail_at = fail_at

    def put_object(self, **kw):
        self.calls.append(kw)
        assert kw["IfNoneMatch"] == "*"
        if len(self.calls) == self.fail_at:
            raise RuntimeError("SECRET in SDK error")
        if hasattr(kw["Body"], "read"):
            kw["Body"].read()


@pytest.fixture
def upload(tmp_path, monkeypatch):
    tar = tmp_path / "synthetic.tar.gz"
    tar.write_bytes(b"mock pack content")
    env = copy.deepcopy(registry.env_of("videojev"))
    client = FakeR2()
    monkeypatch.setattr(be.model_source, "r2_client", lambda: client)
    monkeypatch.setenv("R2_BUCKET", "synthetic")
    return env, tar, client


def test_upload_creates_new_content_addressed_keys_and_preserves_registry(upload):
    env, tar, client = upload
    original = copy.deepcopy(env)
    result = be._upload(env, "videojev", tar, env["compat"], "a/b@revision")
    assert result["r2_key"] == f"envs/videojev/{be._sha256(tar)}.tar.gz"
    assert result["manifest_key"] == f"envs/videojev/{be._sha256(tar)}.json"
    assert [c["Key"] for c in client.calls] == [
        result["r2_key"],
        result["manifest_key"],
    ]
    assert env == original
    assert all(
        c["Key"] not in {env["r2_key"], env["manifest_key"]} for c in client.calls
    )


@pytest.mark.parametrize("fail_at", [1, 2])
def test_upload_conflict_or_uncertainty_never_deletes_or_overwrites(upload, fail_at):
    env, tar, client = upload
    client.fail_at = fail_at
    with pytest.raises(RuntimeError, match="uncertain") as exc:
        be._upload(env, "videojev", tar, env["compat"], "a/b@revision")
    assert "SECRET" not in str(exc.value)
    assert len(client.calls) == fail_at


def test_large_pack_fails_before_upload(upload, monkeypatch):
    env, tar, client = upload
    monkeypatch.setattr(be, "MAX_PUT_SIZE", 1)
    with pytest.raises(ValueError, match="5 GiB"):
        be._upload(env, "videojev", tar, env["compat"], "a/b@revision")
    assert not client.calls


def test_builder_invokes_literal_argv_and_strips_credentials(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setenv("R2_SECRET_ACCESS_KEY", "SECRET")
    monkeypatch.setenv("HF_TOKEN", "TOKEN")
    monkeypatch.setenv("PYTHONPATH", "/injected")
    monkeypatch.setattr(
        be.subprocess, "run", lambda argv, **kw: calls.append((argv, kw))
    )
    conda = tmp_path / "conda's $(literal)/bin/conda"
    prefix = tmp_path / "env $(literal)"
    requirements = tmp_path / "requirements;literal.txt"
    be._build_prefix(conda, prefix, requirements, "3.12", "8.9")
    be._pack(conda, prefix, tmp_path / "out;literal.tar.gz")
    assert calls[0][0][0] == str(conda)
    assert str(prefix) in calls[0][0]
    assert any(str(requirements) in c[0] for c in calls)
    for argv, kw in calls:
        assert "bash" not in argv
        assert not {"R2_SECRET_ACCESS_KEY", "HF_TOKEN", "PYTHONPATH"} & kw["env"].keys()
        assert "--ignore-missing-files" not in argv and "--force" not in argv


def test_installer_requires_trusted_pin_and_never_downloads(tmp_path, monkeypatch):
    monkeypatch.setattr(
        be, "_run", lambda *a, **k: pytest.fail("unverified installer executed")
    )
    with pytest.raises(ValueError, match="trusted Miniforge"):
        be._ensure_miniconda(tmp_path / "conda")
    installer = tmp_path / "installer.sh"
    installer.write_text("# synthetic")
    with pytest.raises(ValueError, match="sha256 mismatch"):
        be._ensure_miniconda(tmp_path / "conda", installer, "0" * 64)


@pytest.mark.parametrize("name", ["clip", "siglip2", "videojev", "d3-nano"])
def test_main_honors_model_attention_recipe(tmp_path, monkeypatch, name):
    env = copy.deepcopy(registry.env_of(name))
    env["unpack_to"] = str(tmp_path / "prefix")
    out = tmp_path / "new.tar.gz"
    calls = []
    monkeypatch.setattr(be.registry, "env_of", lambda _: env)
    monkeypatch.setattr(be.env_source, "check_compatibility", lambda _: None)
    monkeypatch.setattr(be, "_ensure_miniconda", lambda *a: tmp_path / "bin/conda")
    monkeypatch.setattr(be, "_run", lambda argv, **kw: calls.append((argv, kw)))
    monkeypatch.setattr(be, "_pack", lambda *a: out.write_bytes(b"synthetic pack"))
    be.main(["--model", name, "--out", str(out), "--skip-upload"])
    needs_attention = name in {"videojev", "d3-nano"}
    compiled = [argv for argv, _ in calls if "flash-attn==2.8.3.post2" in argv]
    assert bool(compiled) is needs_attention
    if needs_attention:
        assert "causal-conv1d==1.7.0" in compiled[0]
    probe = calls[-1][0][-1]
    assert ("import flash_attn, causal_conv1d" in probe) is needs_attention
    assert "torch.cuda.is_available()" in probe
    assert "torch.__version__ == '2.8.0+cu128'" in probe


@pytest.mark.parametrize(
    "build", [None, [], {"flash_attn": "false"}, {"flash_attn": 0}]
)
def test_malformed_attention_recipe_fails_before_host_checks(
    tmp_path, monkeypatch, build
):
    env = copy.deepcopy(registry.env_of("clip"))
    env["build"] = build
    monkeypatch.setattr(be.registry, "env_of", lambda _: env)
    monkeypatch.setattr(
        be.env_source, "check_compatibility", lambda _: pytest.fail("host probed")
    )
    with pytest.raises(SystemExit):
        be.main(["--model", "clip", "--out", str(tmp_path / "new.tar.gz")])


@pytest.mark.skipif(os.name != "posix", reason="Linux builder prefix locking")
def test_main_preserves_existing_prefix(tmp_path, monkeypatch):
    env = copy.deepcopy(registry.env_of("videojev"))
    prefix = tmp_path / "existing"
    prefix.mkdir()
    keep = prefix / "keep"
    keep.write_text("important")
    env["unpack_to"] = str(prefix)
    monkeypatch.setattr(be.registry, "env_of", lambda name: env)
    monkeypatch.setattr(be.env_source, "check_compatibility", lambda env: None)
    monkeypatch.setattr(
        be, "_ensure_miniconda", lambda *a: pytest.fail("build started")
    )
    with pytest.raises(SystemExit):
        be.main(["--model", "videojev", "--out", str(tmp_path / "new.tar.gz")])
    assert keep.read_text() == "important"
