"""Builder safety tests with subprocess and R2 mocks; no package builds/uploads."""

import copy
import io
import json
import os
import tarfile

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


def test_upload_failure_names_the_error_code_but_not_its_message(upload):
    env, tar, client = upload

    class AccessDenied(Exception):
        def __init__(self, message):
            super().__init__(message)
            self.response = {"Error": {"Code": "AccessDenied", "Message": "SECRET"}}

    def deny(**kw):
        raise AccessDenied("SECRET in SDK error")

    client.put_object = deny
    with pytest.raises(RuntimeError, match="AccessDenied: AccessDenied") as exc:
        be._upload(env, "videojev", tar, env["compat"], "a/b@revision")
    assert "SECRET" not in str(exc.value)


def test_large_pack_fails_before_upload(upload, monkeypatch):
    env, tar, client = upload
    monkeypatch.setattr(be, "MAX_PUT_SIZE", 1)
    with pytest.raises(ValueError, match="5 GiB"):
        be._upload(env, "videojev", tar, env["compat"], "a/b@revision")
    assert not client.calls


def test_builder_invokes_literal_argv_and_strips_credentials(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(
        be, "_write_constraints", lambda prefix, path: path.write_text("pip==26.2.2\n")
    )
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
    monkeypatch.setattr(
        be, "_write_constraints", lambda prefix, path: path.write_text("pip==26.2.2\n")
    )
    monkeypatch.setattr(be.registry, "env_of", lambda _: env)
    monkeypatch.setattr(be.env_source, "check_compatibility", lambda _: None)
    monkeypatch.setattr(be, "_ensure_miniconda", lambda *a: tmp_path / "bin/conda")
    monkeypatch.setattr(be, "_run", lambda argv, **kw: calls.append((argv, kw)))
    monkeypatch.setattr(be, "_pack", lambda *a: out.write_bytes(b"synthetic pack"))
    monkeypatch.setattr(be, "check_conda_ownership", lambda prefix: None)
    monkeypatch.setattr(be, "verify_pack", lambda tarball: {"dists": 0})
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


# --- conda/pip ownership (PR #94 pip 26.3-metadata / 26.2.2-source pack) ---

SP = "lib/python3.12/site-packages/"


def _record(files):
    rows = [f"{p},sha256={be._record_digest(b)},{len(b)}" for p, b in files.items()]
    return "\n".join(rows) + "\n"


def _pack_tar(path, members):
    with tarfile.open(path, "w:gz") as tar:
        for name, data in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return path


def _dist(name, version, files):
    info = f"{name}-{version}.dist-info/"
    files = {**files, info + "METADATA": f"Name: {name}\n".encode()}
    out = {SP + p: b for p, b in files.items()}
    out[SP + info + "RECORD"] = (_record(files) + f"{info}RECORD,,\n").encode()
    return out


def test_pack_check_rejects_old_pip_source_under_new_pip_record(tmp_path):
    new = {
        "pip/__init__.py": b"__version__ = '26.3'\n",
        "pip/_internal/cli/main.py": b"def main(): ...\ncli = main\n",
    }
    members = _dist("pip", "26.3", new)
    # conda-pack copied the cached conda pip 26.2.2 source over the same paths
    members[SP + "pip/__init__.py"] = b"__version__ = '26.2.2'\n"
    members[SP + "pip/_internal/cli/main.py"] = b"def main(): ...\n"
    tar = _pack_tar(tmp_path / "broken.tar.gz", members)
    with pytest.raises(RuntimeError, match=r"pip/_internal/cli/main\.py: bytes differ"):
        be.verify_pack(tar)


def test_pack_check_accepts_consistent_pack_and_reports_skips(tmp_path):
    members = {
        **_dist("pip", "26.2.2", {"pip/__init__.py": b"__version__ = '26.2.2'\n"}),
        **_dist("demo", "1.0", {"demo.py": be._CONDA_PLACEHOLDER}),
    }
    rec = SP + "demo-1.0.dist-info/RECORD"
    members[rec] += b"../../../bin/demo,sha256=abc,1\n"
    # A placeholder is reported only after its packed bytes match the RECORD.
    members["bin/pip"] = b"#!/usr/bin/env python\n"
    report = be.verify_pack(_pack_tar(tmp_path / "ok.tar.gz", members))
    assert report["pip"] == "26.2.2"
    assert report["outside_site_packages_skipped"] == 1
    assert report["conda_placeholder_files"] == 1


def test_pack_check_allows_only_declared_shared_owners(tmp_path):
    cv2 = {"cv2/__init__.py": b"full\n"}
    members = {
        **_dist("opencv_python", "4.12.0.88", cv2),
        **_dist(
            "opencv_python_headless", "4.12.0.88", {"cv2/__init__.py": b"headless\n"}
        ),
    }
    members[SP + "cv2/__init__.py"] = b"full\n"  # one owner's bytes win
    report = be.verify_pack(_pack_tar(tmp_path / "cv.tar.gz", members))
    assert report["declared_shared_files"] == {
        "opencv-python,opencv-python-headless": 1
    }
    members = {**_dist("a", "1", {"x.py": b"1"}), **_dist("b", "1", {"x.py": b"2"})}
    with pytest.raises(RuntimeError, match="claimed by"):
        be.verify_pack(_pack_tar(tmp_path / "ab.tar.gz", members))


def test_pack_check_rejects_missing_file(tmp_path):
    members = _dist("demo", "1.0", {"demo.py": b"x\n"})
    del members[SP + "demo.py"]
    with pytest.raises(RuntimeError, match="not packed"):
        be.verify_pack(_pack_tar(tmp_path / "m.tar.gz", members))


def _conda_prefix(tmp_path, pip_version="26.2.2"):
    prefix = tmp_path / "prefix"
    meta = prefix / "conda-meta"
    meta.mkdir(parents=True)
    files = [SP + f"pip-{pip_version}.dist-info/METADATA", SP + "pip/__init__.py"]
    (meta / f"pip-{pip_version}-pyh.json").write_text(
        json.dumps({"name": "pip", "version": pip_version, "files": files})
    )
    for f in files:
        (prefix / f).parent.mkdir(parents=True, exist_ok=True)
        (prefix / f).write_text(f"Name: pip\nVersion: {pip_version}\n")
    return prefix


def test_constraints_pin_conda_owned_dists(tmp_path):
    prefix = _conda_prefix(tmp_path)
    out = tmp_path / "c.txt"
    assert be._write_constraints(prefix, out) == {"pip": "26.2.2"}
    assert out.read_text() == "pip==26.2.2\n"


def test_ownership_check_catches_pip_upgrade_of_conda_pip(tmp_path):
    prefix = _conda_prefix(tmp_path)
    be.check_conda_ownership(prefix)
    # what `pip install --upgrade pip` did: drop conda's dist-info, add its own
    (prefix / SP / "pip-26.2.2.dist-info/METADATA").unlink()
    (prefix / SP / "pip-26.2.2.dist-info").rmdir()
    (prefix / SP / "pip-26.3.dist-info").mkdir()
    with pytest.raises(RuntimeError, match="clobbered conda-owned"):
        be.check_conda_ownership(prefix)


def test_ownership_check_catches_duplicate_dist_info(tmp_path):
    prefix = _conda_prefix(tmp_path)
    (prefix / SP / "pip-26.3.dist-info").mkdir()
    (prefix / SP / "pip-26.3.dist-info/METADATA").write_text(
        "Name: pip\nVersion: 26.3\n"
    )
    with pytest.raises(RuntimeError, match="two installed versions of pip"):
        be.check_conda_ownership(prefix)


def test_ownership_check_scans_the_python31_compat_symlink_once(tmp_path):
    # conda's python 3.12 ships lib/python3.1 -> python3.12; one PyYAML is one dist.
    prefix = _conda_prefix(tmp_path)
    (prefix / "lib/python3.1").symlink_to("python3.12")
    info = prefix / SP / "PyYAML-6.0.2.dist-info"
    info.mkdir()
    (info / "METADATA").write_text("Name: PyYAML\nVersion: 6.0.2\n")
    be.check_conda_ownership(prefix)
    (prefix / SP / "pip-26.3.dist-info").mkdir()  # a real duplicate is still caught
    duplicate = prefix / SP / "pip-26.3.dist-info/METADATA"
    duplicate.write_text("Name: pip\nVersion: 26.3\n")
    with pytest.raises(RuntimeError, match="two installed versions of pip"):
        be.check_conda_ownership(prefix)


def test_builder_lets_conda_own_pip_and_constrains_every_pip_install(
    tmp_path, monkeypatch
):
    calls = []
    monkeypatch.setattr(
        be, "_write_constraints", lambda prefix, path: path.write_text("pip==26.2.2\n")
    )
    monkeypatch.setattr(be, "_run", lambda argv, **kw: calls.append(argv))
    be._build_prefix(
        tmp_path / "conda", tmp_path / "p", tmp_path / "r.txt", "3.12", "8.9"
    )
    assert {"pip", "setuptools", "wheel"} <= set(calls[0])
    installs = [a for a in calls if "install" in a and "-m" in a]
    assert installs
    for argv in installs:
        assert "--constraint" in argv
        assert "--upgrade" not in argv and "-U" not in argv
        requested = argv[argv.index("install") + 1 :]
        assert not {"pip", "setuptools", "wheel"} & set(requested)


def test_pack_check_accepts_identical_namespace_files_and_conda_installer(tmp_path):
    ns = {"nvidia/__init__.py": b""}
    members = {
        **_dist("nvidia_cublas_cu12", "12.8", ns),
        **_dist("nvidia_nvtx_cu12", "12.8", ns),
        **_dist("packaging", "26.3", {"packaging-26.3.dist-info/INSTALLER": b"pip\n"}),
    }
    members[SP + "packaging-26.3.dist-info/INSTALLER"] = b"conda\n"
    report = be.verify_pack(_pack_tar(tmp_path / "ns.tar.gz", members))
    assert report["conda_installer_rewrites"] == 1
    assert report["declared_shared_files"] == {}
    # conda-forge's noarch pip ships INSTALLER as b"conda" without a newline.
    members[SP + "packaging-26.3.dist-info/INSTALLER"] = b"conda"
    report = be.verify_pack(_pack_tar(tmp_path / "nonl.tar.gz", members))
    assert report["conda_installer_rewrites"] == 1
    for other in (b"other\n", b"conda \n", b"CONDA\n"):
        members[SP + "packaging-26.3.dist-info/INSTALLER"] = other
        with pytest.raises(RuntimeError, match="INSTALLER: bytes differ"):
            bad = tmp_path / f"bad-{len(other)}-{other[:1].decode()}.tar.gz"
            be.verify_pack(_pack_tar(bad, members))
