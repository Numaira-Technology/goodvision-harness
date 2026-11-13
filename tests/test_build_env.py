"""Builder safety tests with subprocess and R2 mocks; no package builds/uploads."""

import copy
import os

import pytest
from env_build_helpers import completed_probe, runtime_facts, synthetic_pack

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
    synthetic_pack(tar)
    env = copy.deepcopy(registry.env_of("videojev"))
    client = FakeR2()
    monkeypatch.setattr(be.model_source, "r2_client", lambda: client)
    monkeypatch.setenv("R2_BUCKET", "synthetic")
    recipe_sha = be._sha256(be._requirements("videojev"))
    record = be._new_record(
        env, "videojev", tar, "a/b@revision", recipe_sha, runtime_facts()
    )
    return env, tar, client, {"record": record, "recipe_sha": recipe_sha}


def test_upload_creates_new_content_addressed_keys_and_preserves_registry(upload):
    env, tar, client, evidence = upload
    original = copy.deepcopy(env)
    result = be._upload(env, "videojev", tar, env["compat"], "a/b@revision", **evidence)
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
    env, tar, client, evidence = upload
    client.fail_at = fail_at
    with pytest.raises(RuntimeError, match="uncertain") as exc:
        be._upload(env, "videojev", tar, env["compat"], "a/b@revision", **evidence)
    assert "SECRET" not in str(exc.value)
    assert len(client.calls) == fail_at


def test_large_pack_fails_before_upload(upload, monkeypatch):
    env, tar, client, evidence = upload
    monkeypatch.setattr(be, "MAX_PUT_SIZE", 1)
    with pytest.raises(ValueError, match="5 GiB"):
        be._upload(env, "videojev", tar, env["compat"], "a/b@revision", **evidence)
    assert not client.calls


def test_builder_invokes_literal_argv_and_strips_credentials(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setenv("R2_SECRET_ACCESS_KEY", "SECRET")
    monkeypatch.setenv("HF_TOKEN", "TOKEN")
    monkeypatch.setenv("PYTHONPATH", "/injected")

    def run(argv, **kw):
        calls.append((argv, kw))
        return completed_probe()

    monkeypatch.setattr(be.subprocess, "run", run)
    conda = tmp_path / "conda's $(literal)/bin/conda"
    prefix = tmp_path / "env $(literal)"
    requirements = tmp_path / "requirements;literal.txt"
    be._build_prefix(conda, prefix, requirements, "3.12", "8.9")
    be._pack(conda, prefix, tmp_path / "out;literal.tar.gz")
    assert calls[0][0][0] == str(conda)
    assert str(prefix) in calls[0][0]
    packed_python = str(prefix / "bin/python")
    assert any(c[0][0] == packed_python and "-I" in c[0] for c in calls)
    assert any(str(requirements) in c[0] for c in calls)
    for argv, kw in calls:
        assert "bash" not in argv
        assert "python" not in argv[:1]
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


def test_build_prefix_uses_prefix_python_and_gates_flash_attn(tmp_path, monkeypatch):
    calls = []

    def run(argv, **kw):
        calls.append(argv)
        return completed_probe()

    monkeypatch.setattr(be, "_run", run)
    prefix = tmp_path / "packed env"
    be._build_prefix(
        tmp_path / "conda", prefix, tmp_path / "req.txt", "3.12", "8.9", False
    )
    python = str(prefix / "bin/python")
    assert all(c[0] in {str(tmp_path / "conda"), python} for c in calls)
    assert all("flash-attn==2.8.3.post2" not in c for c in calls)
    calls.clear()
    be._build_prefix(
        tmp_path / "conda", prefix, tmp_path / "req.txt", "3.12", "8.9", True
    )
    assert any("flash-attn==2.8.3.post2" in c for c in calls)
    assert any(c[:3] == [python, "-I", "-c"] for c in calls)


@pytest.mark.parametrize("stage", ["client", "put"])
def test_upload_uses_validated_snapshot_if_source_changes(upload, monkeypatch, stage):
    import hashlib
    import json
    from pathlib import Path

    env, tar, client, evidence = upload
    original = tar.read_bytes()
    bodies, snapshots = [], []

    def mutate():
        tar.write_bytes(b"X" + original[1:])

    def put(**kwargs):
        if hasattr(kwargs["Body"], "read"):
            snapshots.append(Path(kwargs["Body"].name))
            if stage == "put":
                mutate()
            bodies.append(kwargs["Body"].read())
        else:
            bodies.append(kwargs["Body"])
        assert kwargs["IfNoneMatch"] == "*"

    def connect():
        if stage == "client":
            mutate()
        return client

    monkeypatch.setattr(client, "put_object", put)
    monkeypatch.setattr(be.model_source, "r2_client", connect)
    result = be._upload(env, "videojev", tar, env["compat"], "a/b@revision", **evidence)
    assert bodies[0] == original != tar.read_bytes()
    assert hashlib.sha256(bodies[0]).hexdigest() == result["sha256"]
    assert len(bodies[0]) == result["size"]
    assert json.loads(bodies[1])["build"] == evidence["record"]
    assert snapshots and not any(p.exists() for p in snapshots)


@pytest.mark.parametrize("stage", ["validation", "client", "put"])
def test_upload_snapshot_is_removed_on_failure(upload, monkeypatch, stage):
    from pathlib import Path

    env, tar, client, evidence = upload
    snapshots = []
    validate = be._validate_record

    def capture(record, spec, name, path, *args):
        snapshots.append(Path(path))
        if stage == "validation":
            raise ValueError("synthetic validation failure")
        return validate(record, spec, name, path, *args)

    def fail(*args, **kwargs):
        raise RuntimeError("synthetic failure")

    monkeypatch.setattr(be, "_validate_record", capture)
    if stage == "client":
        monkeypatch.setattr(be.model_source, "r2_client", fail)
    elif stage == "put":
        monkeypatch.setattr(client, "put_object", fail)
    with pytest.raises((ValueError, RuntimeError)):
        be._upload(env, "videojev", tar, env["compat"], "a/b@revision", **evidence)
    assert snapshots and not any(p.exists() for p in snapshots)
    assert tar.exists()  # source and producer record remain the caller's artifacts


def test_upload_source_changed_before_snapshot_fails_before_client(upload, monkeypatch):
    env, tar, client, evidence = upload
    tar.write_bytes(b"X" + tar.read_bytes()[1:])
    monkeypatch.setattr(
        be.model_source,
        "r2_client",
        lambda: pytest.fail("invalid bytes reached storage"),
    )
    with pytest.raises(ValueError, match="digest/size"):
        be._upload(env, "videojev", tar, env["compat"], "a/b@revision", **evidence)
    assert not client.calls
