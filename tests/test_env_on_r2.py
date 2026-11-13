"""Offline synthetic conda-pack restore tests. No R2 objects or GPU execution."""

import copy
import io
import json
import os
import subprocess
import sys
import tarfile
import zlib
from pathlib import Path

import pytest
from env_build_helpers import completed_probe, runtime_facts

from tools import build_env, registry
from tools import env_source as es


def test_registry_ships_and_parses():
    assert registry.weights_of("videojev") == (
        "ZhuYuhan/VideoJev-4B",
        "1c1a253bf90dfa66176124d25a48ffb791d517d5",
    )
    assert registry.env_of("videojev")["compat"]["gpu_family"] == "ada"
    assert "videojev" in registry.model_names()
    with pytest.raises(registry.RegistryError):
        registry.weights_of("does-not-exist")


def test_env_of_clip_has_env_without_flash_attn():
    env = registry.get_model("clip")["env"]
    assert env["r2_key"].startswith("envs/clip/")
    assert env["unpack_to"] == "/opt/env/clip"
    assert env["build"]["flash_attn"] is False


def test_new_models_registered_with_pinned_revisions():
    names = registry.model_names()
    for name in ("siglip2", "d3-nano", "qwen8b"):
        assert name in names
        registry.weights_of(name)
        assert registry.get_model(name)["env"]["compat"]["gpu_family"] == "ada"
    assert registry.weights_of("qwen8b") == (
        "Qwen/Qwen3-VL-8B-Instruct",
        "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b",
    )
    assert registry.get_model("qwen8b")["env"]["build"]["flash_attn"] is True
    assert registry.get_model("d3-nano")["env"]["build"]["flash_attn"] is True
    assert registry.get_model("siglip2")["env"]["build"]["flash_attn"] is False


def test_ensure_env_errors_without_packed_env(tmp_path):
    # A model whose env is null has no packed env to restore.
    reg = tmp_path / "models.yaml"
    reg.write_text(
        "version: 1\nmodels:\n  noenv:\n    repo: a/b\n"
        "    revision: " + ("a" * 40) + "\n    env: null\n"
        "    weights:\n      r2_prefix: models/a/b/" + ("a" * 40) + "/\n"
    )
    with pytest.raises(registry.RegistryError):
        es.ensure_env("noenv", registry_path=reg)


@pytest.mark.parametrize(
    "document",
    [
        "[]",
        "null",
        "version: 2",
        "version: 1\nmodels: []",
        "version: 1\nmodels: {x: []}",
        "version: 1\nmodels: {x: {repo: a/b, revision: 123}}",
    ],
)
def test_invalid_registry_shapes(tmp_path, document):
    p = tmp_path / "models.yaml"
    p.write_text(document)
    with pytest.raises(registry.RegistryError):
        registry.weights_of("x", p)


@pytest.mark.parametrize(
    "field,value",
    [
        ("sha256", None),
        ("sha256", "abc"),
        ("sha256", "F" * 64),
        ("size", 0),
        ("size", True),
        ("size", "12"),
        ("size", -1),
        ("r2_key", "models/a/b/weights.tar.gz"),
        ("r2_key", "envs/../x.tar.gz"),
        ("manifest_key", "envs/videojev/other.json"),
        ("unpack_to", "/"),
        ("unpack_to", "relative"),
        ("unpack_to", "/opt/../env"),
        ("unpack_to", "/opt//env"),
        ("unpack_to", "/opt/env\n"),
        ("activate_rel", "/tmp/activate"),
        ("activate_rel", "../activate"),
        ("compat", {}),
        ("compat", []),
    ],
)
def test_invalid_env_specs_rejected_before_side_effects(field, value, monkeypatch):
    env = copy.deepcopy(registry.env_of("videojev"))
    env[field] = value
    monkeypatch.setattr(
        es.model_source, "r2_client", lambda: pytest.fail("client created")
    )
    with pytest.raises(registry.RegistryError):
        es.pull_env(env)


def archive(path, extra=(), omit=()):
    entries = [
        ("bin/python", b"synthetic Python", 0o755),
        ("bin/conda-unpack", b"# synthetic unpack", 0o644),
        ("bin/activate", b"# synthetic activate", 0o644),
    ]
    with tarfile.open(path, "w:gz") as tf:
        for name, data, mode in entries:
            if name in omit:
                continue
            item = tarfile.TarInfo(name)
            item.size, item.mode = len(data), mode
            tf.addfile(item, io.BytesIO(data))
        for item, data in extra:
            tf.addfile(item, io.BytesIO(data) if data is not None else None)
    return path


class FakeR2:
    def __init__(self, tarball, env):
        self.tarball = tarball
        self.manifest = {**es._identity(env), "format": "conda-pack"}
        self.downloads = []
        self.bodies = []

    def get_object(self, **kwargs):
        body = io.BytesIO(json.dumps(self.manifest).encode())
        self.bodies.append(body)
        return {"Body": body}

    def download_file(self, bucket, key, dest):
        self.downloads.append(dest)
        Path(dest).write_bytes(self.tarball.read_bytes())


@pytest.fixture
def pack(tmp_path, monkeypatch):
    if os.name != "posix":
        pytest.skip("restore targets POSIX/Linux hosts")
    tar = archive(tmp_path / "env.tar.gz")
    env = copy.deepcopy(registry.env_of("videojev"))
    env.update(
        unpack_to=str(tmp_path / "opt/env"),
        sha256=es._sha256(tar),
        size=tar.stat().st_size,
    )
    client = FakeR2(tar, env)
    monkeypatch.setattr(es, "check_compatibility", lambda env: None)
    monkeypatch.setattr(es, "_run_unpack", lambda prefix: None)
    return env, client


def pull(pack, tmp_path):
    env, client = pack
    return es.pull_env(
        env, client=client, bucket="synthetic", scratch=tmp_path / "scratch"
    )


def assert_clean(tmp_path, env):
    assert not Path(env["unpack_to"]).exists()
    assert not list((tmp_path / "scratch").glob("*"))
    assert not list(Path(env["unpack_to"]).parent.glob(".r2-env-stage-*"))


def test_pull_verifies_publishes_and_marks(pack, tmp_path):
    env, client = pack
    activate = pull(pack, tmp_path)
    assert activate == env["unpack_to"] + "/bin/activate"
    marker = json.loads(es._marker(env["unpack_to"]).read_text())
    assert marker["identity"] == es._identity(env)
    assert es._already_restored(env)
    assert all(b.closed for b in client.bodies)
    assert not list((tmp_path / "scratch").iterdir())
    assert not Path(client.downloads[0]).exists()


@pytest.mark.parametrize(
    "change",
    [
        {"sha256": "0" * 64},
        {"size": 0},
        {"size": True},
        {"size": "100"},
        {"r2_key": "envs/another/env.tar.gz"},
        {"format": "weights"},
        {"unpack_to": "/opt/wrong"},
        {"activate_rel": "elsewhere"},
        {"compat": {}},
    ],
)
def test_manifest_must_agree_with_trusted_pin(pack, tmp_path, change):
    env, client = pack
    client.manifest.update(change)
    with pytest.raises(ValueError, match="manifest"):
        pull(pack, tmp_path)
    assert client.downloads == []
    assert all(b.closed for b in client.bodies)
    assert_clean(tmp_path, env)


@pytest.mark.parametrize("manifest", [None, [], "oops", {}])
def test_invalid_manifest_shapes(pack, tmp_path, manifest):
    env, client = pack
    client.manifest = manifest
    with pytest.raises(ValueError, match="manifest"):
        pull(pack, tmp_path)
    assert_clean(tmp_path, env)


def test_manifest_read_failure_is_sanitized(pack, tmp_path, monkeypatch):
    _env, client = pack

    def fail(**kw):
        raise RuntimeError("URL with SECRET-TOKEN")

    monkeypatch.setattr(client, "get_object", fail)
    with pytest.raises(RuntimeError, match="cannot read") as exc:
        pull(pack, tmp_path)
    assert "SECRET" not in str(exc.value)
    assert client.downloads == []


@pytest.mark.parametrize("corrupt", ["size", "sha"])
def test_download_verifies_registry_size_and_hash(pack, tmp_path, corrupt):
    env, client = pack
    data = client.tarball.read_bytes()
    client.tarball.write_bytes(
        data + b"extra" if corrupt == "size" else b"x" * len(data)
    )
    with pytest.raises(
        ValueError, match=f"{corrupt if corrupt == 'size' else 'sha256'} mismatch"
    ):
        pull(pack, tmp_path)
    assert_clean(tmp_path, env)


@pytest.mark.parametrize("error", [RuntimeError("failed"), KeyboardInterrupt()])
@pytest.mark.parametrize("phase", ["download", "extract", "unpack", "marker"])
def test_failed_or_interrupted_restore_cleans_owned_files(
    pack, tmp_path, monkeypatch, error, phase
):
    env, client = pack

    def fail(*a, **kw):
        raise error

    if phase == "download":
        monkeypatch.setattr(client, "download_file", fail)
    elif phase == "extract":
        monkeypatch.setattr(es, "_safe_extract", fail)
    elif phase == "unpack":
        monkeypatch.setattr(es, "_run_unpack", fail)
    else:
        monkeypatch.setattr(Path, "write_text", fail)
    with pytest.raises(type(error)):
        pull(pack, tmp_path)
    assert_clean(tmp_path, env)


def test_preserves_existing_env_even_with_no_local(pack, tmp_path, monkeypatch):
    env, client = pack
    prefix = Path(env["unpack_to"])
    prefix.mkdir(parents=True)
    (prefix / "keep").write_bytes(b"existing unregistered work")
    with pytest.raises(FileExistsError, match="preserving"):
        pull(pack, tmp_path)
    assert (prefix / "keep").read_bytes() == b"existing unregistered work"
    assert not client.bodies and not client.downloads


def test_existing_empty_prefix_is_preserved(pack, tmp_path):
    env, _ = pack
    Path(env["unpack_to"]).mkdir(parents=True)
    with pytest.raises(FileExistsError):
        pull(pack, tmp_path)
    assert Path(env["unpack_to"]).is_dir()


def test_matching_local_env_needs_no_credentials(pack, tmp_path, monkeypatch):
    env, _ = pack
    pull(pack, tmp_path)
    monkeypatch.setattr(registry, "env_of", lambda *a: env)
    monkeypatch.setattr(
        es.model_source, "r2_client", lambda: pytest.fail("network client")
    )
    info = es.ensure_env("videojev")
    assert info["source"] == "local"
    assert info["python"] == env["unpack_to"] + "/bin/python"
    assert info["python"] == es.packed_python(env)
    for key, value in [
        ("r2_key", "envs/new/pack.tar.gz"),
        ("compat", {**env["compat"], "gpu_family": "hopper"}),
    ]:
        changed = {**env, key: value}
        if key == "r2_key":
            changed["manifest_key"] = "envs/new/pack.json"
        assert not es._already_restored(changed)


@pytest.mark.parametrize("marker", ["[]", "null", "broken", '{"sha256":"abc"}'])
def test_invalid_completion_marker_is_not_a_cache_hit(pack, tmp_path, marker):
    env, _ = pack
    pull(pack, tmp_path)
    es._marker(env["unpack_to"]).write_text(marker)
    assert not es._already_restored(env)


def test_scratch_symlink_cannot_redirect_download(pack, tmp_path):
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    untouched = tmp_path / "untouched"
    untouched.write_text("keep")
    (scratch / "env.tar.gz").symlink_to(untouched)
    pull(pack, tmp_path)
    assert untouched.read_text() == "keep"


@pytest.mark.parametrize("ancestor", [False, True])
def test_symlink_prefix_is_rejected_without_writes(pack, tmp_path, ancestor):
    env, _ = pack
    prefix = Path(env["unpack_to"])
    target = tmp_path / "untouched"
    target.mkdir()
    if ancestor:
        prefix.parent.symlink_to(target)
    else:
        prefix.parent.mkdir()
        prefix.symlink_to(target)
    with pytest.raises(ValueError, match="symlink"):
        pull(pack, tmp_path)
    assert list(target.iterdir()) == []


def test_prefix_lock_blocks_concurrent_writer(pack):
    env, _ = pack
    prefix = Path(env["unpack_to"])
    with es.prefix_lock(prefix), pytest.raises(RuntimeError, match="lock"):
        with es.prefix_lock(prefix):
            pytest.fail("concurrent writer acquired lock")
    with es.prefix_lock(prefix):
        pass  # lock released without unlinking/replacing its inode


def member(name, kind=tarfile.REGTYPE, link="", data=b"x"):
    item = tarfile.TarInfo(name)
    item.type, item.linkname = kind, link
    item.size = len(data) if kind == tarfile.REGTYPE else 0
    return item, data if kind == tarfile.REGTYPE else None


@pytest.mark.parametrize(
    "extra",
    [
        [member("../escape")],
        [member("/absolute")],
        [member("C:/escape")],
        [member("bin\\escape")],
        [member("bin/../escape")],
        [member("device", tarfile.CHRTYPE)],
        [member("fifo", tarfile.FIFOTYPE)],
        [member("bin/activate")],
        [member(es.MARKER)],
        [member("out", tarfile.SYMTYPE, "../../escape")],
        [member("out", tarfile.SYMTYPE, "/etc/passwd")],
        [member("out", tarfile.LNKTYPE, "../../escape")],
        [member("out", tarfile.LNKTYPE, "missing")],
        [member("alias", tarfile.SYMTYPE, "."), member("alias/file")],
        [member("a", tarfile.SYMTYPE, "b"), member("b", tarfile.SYMTYPE, "a")],
        [
            member("a", tarfile.SYMTYPE, "."),
            member("b", tarfile.SYMTYPE, "a/../escape"),
        ],
    ],
)
def test_unsafe_archive_rejected_without_execution(pack, tmp_path, monkeypatch, extra):
    env, client = pack
    archive(client.tarball, extra)
    env.update(sha256=es._sha256(client.tarball), size=client.tarball.stat().st_size)
    client.manifest = {**es._identity(env), "format": "conda-pack"}
    monkeypatch.setattr(
        es, "_run_unpack", lambda *a: pytest.fail("unsafe pack executed")
    )
    with pytest.raises((ValueError, tarfile.TarError)):
        pull(pack, tmp_path)
    assert_clean(tmp_path, env)
    assert not (tmp_path / "escape").exists()


def test_safe_relative_symlinks_and_hardlinks_extract(pack, tmp_path):
    env, client = pack
    archive(
        client.tarball,
        [
            member("lib/python", tarfile.SYMTYPE, "../bin/python"),
            member("bin/python-copy", tarfile.LNKTYPE, "bin/python"),
        ],
    )
    env.update(sha256=es._sha256(client.tarball), size=client.tarball.stat().st_size)
    client.manifest = {**es._identity(env), "format": "conda-pack"}
    pull(pack, tmp_path)
    prefix = Path(env["unpack_to"])
    assert (prefix / "lib/python").read_bytes() == b"synthetic Python"
    assert (prefix / "bin/python-copy").read_bytes() == b"synthetic Python"


@pytest.mark.parametrize("missing", ["bin/python", "bin/conda-unpack", "bin/activate"])
def test_incomplete_pack_never_runs(pack, tmp_path, monkeypatch, missing):
    env, client = pack
    archive(client.tarball, omit=(missing,))
    env.update(sha256=es._sha256(client.tarball), size=client.tarball.stat().st_size)
    client.manifest = {**es._identity(env), "format": "conda-pack"}
    monkeypatch.setattr(
        es, "_run_unpack", lambda *a: pytest.fail("incomplete pack executed")
    )
    with pytest.raises(ValueError, match="conda-pack file"):
        pull(pack, tmp_path)
    assert_clean(tmp_path, env)


def test_unpack_uses_packed_python_at_final_prefix_without_credentials(
    tmp_path, monkeypatch
):
    captured = {}
    monkeypatch.setenv("R2_SECRET_ACCESS_KEY", "SECRET")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "OTHER-SECRET")
    monkeypatch.setenv("HF_TOKEN", "TOKEN")
    monkeypatch.setenv("PYTHONPATH", "/untrusted")

    def run(argv, **kw):
        captured.update(argv=argv, **kw)

    monkeypatch.setattr(subprocess, "run", run)
    es._run_unpack(tmp_path)
    assert captured["argv"] == [
        str(tmp_path / "bin/python"),
        "-I",
        "-c",
        es._UNPACK_PROGRAM,
        str(tmp_path / "bin/conda-unpack"),
    ]
    assert (
        not {"R2_SECRET_ACCESS_KEY", "AWS_SECRET_ACCESS_KEY", "HF_TOKEN", "PYTHONPATH"}
        & captured["env"].keys()
    )
    assert captured["env"]["PYTHONNOUSERSITE"] == "1"
    assert captured["timeout"] == 300


@pytest.mark.skipif(os.name != "posix", reason="known synthetic POSIX interpreter shim")
def test_unpack_executes_script_with_stale_shebang(tmp_path):
    import shlex

    (tmp_path / "bin").mkdir()
    python = tmp_path / "bin/python"
    python.write_text(f'#!/bin/sh\nexec {shlex.quote(sys.executable)} "$@"\n')
    python.chmod(0o755)
    script = tmp_path / "bin/conda-unpack"
    script.write_text(
        "#!/missing/build/prefix/python\nfrom pathlib import Path\n"
        "Path(__file__).with_name('ran').write_text('ok')\n"
    )
    es._run_unpack(tmp_path)
    assert (tmp_path / "bin/ran").read_text() == "ok"


@pytest.mark.skipif(os.name != "posix", reason="known synthetic POSIX interpreter shim")
def test_unpack_can_import_its_sibling_progress_module(tmp_path, monkeypatch):
    """conda-pack's conda-unpack imports bin/conda_unpack_progress.py; -I alone
    (3.11+: no script dir on sys.path) made every real restore fail."""
    import shlex

    monkeypatch.setenv("PYTHONPATH", str(tmp_path / "untrusted"))
    (tmp_path / "untrusted").mkdir()
    (tmp_path / "untrusted/conda_unpack_progress.py").write_text("raise SystemExit(9)")
    (tmp_path / "bin").mkdir()
    python = tmp_path / "bin/python"
    python.write_text(f'#!/bin/sh\nexec {shlex.quote(sys.executable)} "$@"\n')
    python.chmod(0o755)
    (tmp_path / "bin/conda_unpack_progress.py").write_text(
        "def progressbar(x):\n    return x\n"
    )
    (tmp_path / "bin/conda-unpack").write_text(
        "#!/usr/bin/env python\nfrom conda_unpack_progress import progressbar\n"
        "import sys\nfrom pathlib import Path\n"
        "Path(__file__).with_name('ran').write_text(sys.argv[0])\n"
    )
    es._run_unpack(tmp_path)
    assert (tmp_path / "bin/ran").read_text() == str(tmp_path / "bin/conda-unpack")


@pytest.mark.parametrize(
    "system,machine,caps,driver,valid",
    [
        ("Linux", "x86_64", "8.9\n", "CUDA Version: 12.8", True),
        ("Linux", "x86_64", "8.9\n8.9\n", "CUDA Version: 13.0", True),
        ("Darwin", "arm64", "8.9", "CUDA Version: 12.8", False),
        ("Linux", "aarch64", "8.9", "CUDA Version: 12.8", False),
        ("Linux", "x86_64", "8.0", "CUDA Version: 12.8", False),
        ("Linux", "x86_64", "8.9\n9.0", "CUDA Version: 12.8", False),
        ("Linux", "x86_64", "", "CUDA Version: 12.8", False),
        ("Linux", "x86_64", "8.9", "CUDA Version: 12.7", False),
        ("Linux", "x86_64", "8.9", "no CUDA info", False),
    ],
)
def test_host_compatibility_is_enforced(
    monkeypatch, system, machine, caps, driver, valid
):
    monkeypatch.setattr(es.platform, "system", lambda: system)
    monkeypatch.setattr(es.platform, "machine", lambda: machine)
    monkeypatch.setattr(
        subprocess, "check_output", lambda cmd, **kw: caps if len(cmd) > 1 else driver
    )
    env = registry.env_of("videojev")
    if valid:
        es.check_compatibility(env)
    else:
        with pytest.raises(RuntimeError):
            es.check_compatibility(env)


@pytest.mark.parametrize("competing", [False, True])
def test_publication_failure_never_deletes_competing_files(
    pack, tmp_path, monkeypatch, competing
):
    env, _ = pack
    prefix = Path(env["unpack_to"])

    def fail_rename(source, target):
        if competing:
            Path(target).mkdir()
            (Path(target) / "someone-elses-file").write_text("keep")
        raise OSError("synthetic publication failure")

    monkeypatch.setattr(es, "_publish_prefix", fail_rename)
    with pytest.raises(OSError, match="publication"):
        pull(pack, tmp_path)
    if competing:
        assert (prefix / "someone-elses-file").read_text() == "keep"
        assert not es._already_restored(env)
    else:
        assert not prefix.exists()
    assert not list((tmp_path / "scratch").iterdir())


@pytest.mark.parametrize(
    "name,kind", [("/", tarfile.DIRTYPE), ("bin/python/", tarfile.REGTYPE)]
)
def test_absolute_root_or_trailing_file_separator_rejected(pack, tmp_path, name, kind):
    env, client = pack
    archive(client.tarball, [member(name, kind)])
    env.update(sha256=es._sha256(client.tarball), size=client.tarball.stat().st_size)
    client.manifest = {**es._identity(env), "format": "conda-pack"}
    with pytest.raises(registry.RegistryError):
        pull(pack, tmp_path)
    assert_clean(tmp_path, env)


def test_unknown_or_malformed_gpu_family_rejected():
    for family in (None, [], {}, "blackwell"):
        env = copy.deepcopy(registry.env_of("videojev"))
        env["compat"]["gpu_family"] = family
        with pytest.raises(registry.RegistryError, match="gpu_family"):
            registry.validate_env(env)


def test_manifest_is_bounded_and_body_closed(pack, tmp_path, monkeypatch):
    _env, client = pack
    body = io.BytesIO(b" " * 65537)
    monkeypatch.setattr(client, "get_object", lambda **kw: {"Body": body})
    with pytest.raises(ValueError, match="64 KiB"):
        pull(pack, tmp_path)
    assert body.closed
    assert not client.downloads


def test_archive_does_not_restore_setuid_bits(pack, tmp_path):
    env, client = pack
    item, data = member("bin/extra")
    item.mode = 0o6777
    archive(client.tarball, [(item, data)])
    env.update(sha256=es._sha256(client.tarball), size=client.tarball.stat().st_size)
    client.manifest = {**es._identity(env), "format": "conda-pack"}
    pull(pack, tmp_path)
    assert (Path(env["unpack_to"]) / "bin/extra").stat().st_mode & 0o6022 == 0


def test_main_never_prints_backend_secret(monkeypatch, capsys):
    def fail(*a, **kw):
        raise RuntimeError("SDK error with SECRET")

    monkeypatch.setattr(es, "ensure_env", fail)
    with pytest.raises(SystemExit) as exc:
        es.main(["--model", "videojev"])
    assert exc.value.code == 1
    assert "SECRET" not in capsys.readouterr().err


def test_link_escape_rejected_before_newer_tar_filter_normalization(
    pack, tmp_path, monkeypatch
):
    import posixpath

    env, client = pack
    archive(
        client.tarball,
        [
            member("a", tarfile.SYMTYPE, "."),
            member("b", tarfile.SYMTYPE, "a/../escape"),
        ],
    )
    env.update(sha256=es._sha256(client.tarball), size=client.tarball.stat().st_size)
    client.manifest = {**es._identity(env), "format": "conda-pack"}
    original_filter = tarfile.data_filter

    def normalizing_filter(item, dest):
        if item.issym() or item.islnk():
            item = item.replace(linkname=posixpath.normpath(item.linkname))
        return original_filter(item, dest)

    monkeypatch.setattr(tarfile, "data_filter", normalizing_filter)
    monkeypatch.setattr(es, "_run_unpack", lambda *a: pytest.fail("pack executed"))
    with pytest.raises(ValueError, match="link"):
        pull(pack, tmp_path)
    assert_clean(tmp_path, env)


def test_valid_link_target_semantics_survive_filter_normalization(
    pack, tmp_path, monkeypatch
):
    import posixpath

    env, client = pack
    archive(
        client.tarball,
        [
            member("deep/dir", tarfile.DIRTYPE),
            member("deep/file", data=b"correct"),
            member("file", data=b"wrong"),
            member("alias", tarfile.SYMTYPE, "deep/dir"),
            member("link", tarfile.SYMTYPE, "alias/../file"),
        ],
    )
    env.update(sha256=es._sha256(client.tarball), size=client.tarball.stat().st_size)
    client.manifest = {**es._identity(env), "format": "conda-pack"}
    original_filter = tarfile.data_filter

    def normalizing_filter(item, dest):
        if item.issym() or item.islnk():
            item = item.replace(linkname=posixpath.normpath(item.linkname))
        return original_filter(item, dest)

    monkeypatch.setattr(tarfile, "data_filter", normalizing_filter)
    pull(pack, tmp_path)
    assert (Path(env["unpack_to"]) / "link").read_bytes() == b"correct"


def _launch_spec(prefix):
    env = copy.deepcopy(registry.env_of("videojev"))
    env["unpack_to"] = str(prefix)
    return env


@pytest.mark.skipif(
    os.name != "posix", reason="packed env prefixes are absolute POSIX paths"
)
def test_packed_python_missing_aborts(tmp_path):
    prefix = tmp_path / "opt/env"
    prefix.mkdir(parents=True)
    env = _launch_spec(prefix)
    with pytest.raises(RuntimeError, match="packed interpreter missing"):
        es.packed_python(env)


@pytest.mark.skipif(
    os.name != "posix", reason="packed env prefixes are absolute POSIX paths"
)
def test_packed_python_path_stays_under_unpack_to_even_if_symlink(tmp_path):
    prefix = tmp_path / "opt/env"
    (prefix / "bin").mkdir(parents=True)
    (prefix / "bin/python").symlink_to(sys.executable)
    env = _launch_spec(prefix)
    assert es.packed_python(env) == str(prefix / "bin/python")


@pytest.mark.skipif(os.name != "posix", reason="venv layout is POSIX for packed envs")
def test_launch_argv_uses_restored_venv_python_and_matching_prefix(tmp_path):
    prefix = tmp_path / "opt/env"
    subprocess.run(
        [sys.executable, "-m", "venv", "--without-pip", str(prefix)],
        check=True,
    )
    env = _launch_spec(prefix)
    argv, child = es.launch_argv(env, "-c", "import sys; print(sys.prefix)")
    python = str(prefix / "bin/python")
    assert argv[0] == python
    assert argv[1:3] == ["-s", "-c"]
    assert "sys.prefix" in argv[3]
    assert "python3" not in argv[0]
    assert child["PYTHONNOUSERSITE"] == "1"
    assert child["GOODVISION_ENV_PREFIX"] == str(prefix.resolve())
    assert "PYTHONPATH" not in child
    proc = subprocess.run(argv, env=child, capture_output=True, text=True, check=True)
    assert Path(proc.stdout.strip()).resolve() == prefix.resolve()


@pytest.mark.skipif(os.name != "posix", reason="known synthetic POSIX interpreter shim")
def test_launch_argv_aborts_when_sys_prefix_does_not_match(tmp_path):
    import shlex

    prefix = tmp_path / "opt/env"
    (prefix / "bin").mkdir(parents=True)
    python = prefix / "bin/python"
    python.write_text(f'#!/bin/sh\nexec {shlex.quote(sys.executable)} "$@"\n')
    python.chmod(0o755)
    env = _launch_spec(prefix)
    with pytest.raises(RuntimeError, match=r"sys\.prefix"):
        es.launch_argv(env)


def test_eval_subprocess_env_disables_user_site_and_drops_injection():
    child = es.eval_subprocess_env("/opt/env/videojev")
    assert child["PYTHONNOUSERSITE"] == "1"
    assert "PYTHONPATH" not in child
    assert "PYTHONUSERBASE" not in child
    assert child["PATH"].startswith("/opt/env/videojev/bin" + os.pathsep)


def _captured_build_prefix(tmp_path, monkeypatch, flash_attn):
    calls = []

    def run(argv, **kw):
        calls.append(argv)
        return completed_probe(flash_attn)

    monkeypatch.setattr(build_env, "_run", run)
    prefix = tmp_path / "env"
    requirements = tmp_path / "requirements.txt"
    requirements.write_text("numpy\n")
    build_env._build_prefix(
        tmp_path / "conda",
        prefix,
        requirements,
        "3.12",
        "8.9",
        flash_attn=flash_attn,
    )
    return calls, str(prefix / "bin/python")


def test_build_prefix_includes_flash_attn_by_default(tmp_path, monkeypatch):
    calls, python = _captured_build_prefix(tmp_path, monkeypatch, True)
    assert all(call[0] != "python" for call in calls)
    assert any(call[0] == python and "-I" in call for call in calls)
    assert any("flash-attn==2.8.3.post2" in call for call in calls)
    assert any("causal-conv1d==1.7.0" in call for call in calls)
    assert any("flash_attn" in call[-1] for call in calls if call[-2:-1] == ["-c"])


def test_build_prefix_skips_flash_attn_when_false(tmp_path, monkeypatch):
    calls, python = _captured_build_prefix(tmp_path, monkeypatch, False)
    assert all("flash-attn==2.8.3.post2" not in call for call in calls)
    assert all("--no-build-isolation" not in call for call in calls)
    verify = next(call[-1] for call in calls if call[:3] == [python, "-I", "-c"])
    assert "flash_attn" not in verify
    assert "import sys, torch" in verify
    assert any("torch==2.8.0" in call for call in calls)


def test_build_prefix_always_installs_torch_and_requirements(tmp_path, monkeypatch):
    for flash_attn in (True, False):
        calls, python = _captured_build_prefix(tmp_path, monkeypatch, flash_attn)
        assert any(
            "torch==2.8.0" in call and "torchvision==0.23.0" in call for call in calls
        )
        assert any(
            "-r" in call and str(tmp_path / "requirements.txt") in call
            for call in calls
        )
        assert all(call[0] in {str(tmp_path / "conda"), python} for call in calls)


def test_skip_and_upload_only_are_mutually_exclusive(tmp_path):
    with pytest.raises(SystemExit):
        build_env.main(
            [
                "--model",
                "videojev",
                "--skip-upload",
                "--upload-only",
                "--out",
                str(tmp_path / "x.tar.gz"),
            ]
        )


def test_upload_only_requires_existing_tarball(tmp_path):
    missing = tmp_path / "nope.tar.gz"
    with pytest.raises(SystemExit):
        build_env.main(["--model", "videojev", "--upload-only", "--out", str(missing)])


def test_upload_only_uploads_without_building(tmp_path, monkeypatch, capsys):
    tarball = tmp_path / "videojev-env.tar.gz"
    archive(tarball)
    env = registry.env_of("videojev")
    build_env._new_record(
        env,
        "videojev",
        tarball,
        "@".join(registry.weights_of("videojev")),
        build_env._sha256(build_env._requirements("videojev")),
        runtime_facts(),
    )
    calls = []

    class FakeR2:
        def put_object(self, **kw):
            calls.append(kw)
            if hasattr(kw["Body"], "read"):
                kw["Body"].read()

    monkeypatch.setattr(build_env.model_source, "r2_client", lambda: FakeR2())
    monkeypatch.setenv("R2_BUCKET", "goodvision-models")

    def boom(*a, **k):
        raise AssertionError("upload-only must not build/pack")

    monkeypatch.setattr(build_env, "_ensure_miniconda", boom)
    monkeypatch.setattr(build_env, "_build_prefix", boom)
    monkeypatch.setattr(build_env, "_pack", boom)
    monkeypatch.setattr(build_env.env_source, "check_compatibility", boom)

    build_env.main(["--model", "videojev", "--upload-only", "--out", str(tarball)])

    result = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert result["uploaded"] is True
    assert result["r2_key"].startswith("envs/videojev/")
    assert [c["Key"] for c in calls] == [result["r2_key"], result["manifest_key"]]
    assert all(c["IfNoneMatch"] == "*" for c in calls)


@pytest.mark.skipif(
    os.name != "posix",
    reason="registry unpack_to must be an absolute POSIX prefix",
)
@pytest.mark.parametrize("name", ["videojev", "clip", "siglip2", "d3-nano", "qwen8b"])
def test_each_registry_env_launch_path_is_absolute_packed_python(name, tmp_path):
    env = copy.deepcopy(registry.env_of(name))
    prefix = tmp_path / "opt" / name
    (prefix / "bin").mkdir(parents=True)
    python = prefix / "bin/python"
    python.write_text("#!/bin/sh\nexit 0\n")
    python.chmod(0o755)
    env["unpack_to"] = str(prefix)
    assert es.packed_python(env) == str(python)
    absent = copy.deepcopy(env)
    absent["unpack_to"] = str(tmp_path / "missing" / name)
    Path(absent["unpack_to"]).mkdir(parents=True)
    with pytest.raises(RuntimeError, match="packed interpreter missing"):
        es.packed_python(absent)


def test_real_client_downloads_env_in_parallel_ranges(pack, tmp_path, monkeypatch):
    """A real boto3 client takes the ranged parallel path; checks are unchanged."""
    env, client = pack
    data = client.tarball.read_bytes()

    class BotoLikeR2(FakeR2):
        def get_object(self, Bucket=None, Key=None, Range=None, **kw):
            if Range is None:
                return super().get_object(Bucket=Bucket, Key=Key)
            a, b = map(int, Range.removeprefix("bytes=").split("-"))
            self.ranges.append((a, b))
            return {"Body": io.BytesIO(data[a : b + 1])}

        def download_file(self, *a, **kw):
            pytest.fail("real clients must use ranged parallel parts")

    BotoLikeR2.__module__ = "botocore.client"
    boto = BotoLikeR2(client.tarball, env)
    boto.ranges = []
    monkeypatch.setattr(es.model_source, "PART_SIZE", 97)
    monkeypatch.setenv("R2_PULL_PROCESSES", "1")  # threads in-process for the test
    es.pull_env(env, client=boto, bucket="synthetic", scratch=tmp_path / "scratch")
    assert len(boto.ranges) == -(-len(data) // 97)
    assert es._already_restored(env)


def _gz_tar(path, entries, *, members=1):
    """Write a tar of (TarInfo, data) entries, gzip-compressed in ``members`` parts."""
    import gzip

    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w", format=tarfile.PAX_FORMAT) as tf:
        for info, data in entries:
            tf.addfile(info, io.BytesIO(data) if data is not None else None)
    raw = buf.getvalue()
    step = -(-len(raw) // members)
    with open(path, "wb") as out:
        for i in range(0, len(raw), step):
            out.write(gzip.compress(raw[i : i + step]))
    return raw


def _file(name, data=b"x", mode=0o644):
    info = tarfile.TarInfo(name)
    info.size, info.mode = len(data), mode
    return info, data


def _core(extra=()):
    return [
        _file("bin/python", b"synthetic Python", 0o755),
        _file("bin/conda-unpack", b"#!/bin/sh\n", 0o755),
        _file("bin/activate", b"# activate\n"),
        *extra,
    ]


@pytest.mark.parametrize("members", [1, 3])
def test_inflater_streams_single_and_multi_member_gzip(tmp_path, members):
    raw = _gz_tar(
        tmp_path / "a.tar.gz",
        _core([_file("big", os.urandom(300_000))]),
        members=members,
    )
    with es._Inflater(tmp_path / "a.tar.gz") as reader:
        assert reader.read() == raw


@pytest.mark.parametrize("output_limit", [4096, 8 << 20])
def test_inflater_does_not_repeat_members_after_output_limit(
    tmp_path, monkeypatch, output_limit
):
    import gzip

    monkeypatch.setattr(es._Inflater, "_OUT", output_limit)
    parts = [b"a" * (output_limit + 1), b"b" * 31, b"c" * 47]
    path = tmp_path / "members.gz"
    path.write_bytes(b"".join(gzip.compress(part) for part in parts))
    with es._Inflater(path) as reader:
        assert reader.read() == b"".join(parts)


@pytest.mark.parametrize(
    "damage, error", [("truncate", EOFError), ("corrupt", zlib.error)]
)
def test_inflater_rejects_damaged_stream(tmp_path, damage, error):
    path = tmp_path / "a.tar.gz"
    _gz_tar(path, _core([_file("big", os.urandom(300_000))]))
    data = bytearray(path.read_bytes())
    if damage == "truncate":
        data = data[: len(data) // 2]
    else:
        data[len(data) // 2] ^= 0xFF
    path.write_bytes(bytes(data))
    with pytest.raises(error), es._Inflater(path) as reader:
        reader.read()
    # tarfile may see the damage first as a bad header; either way it fails.
    with pytest.raises((error, tarfile.ReadError)):
        es._safe_extract(path, tmp_path / "out")


def test_inflater_close_stops_helper_thread_early(tmp_path):
    path = tmp_path / "a.tar.gz"
    _gz_tar(path, _core([_file("big", os.urandom(4_000_000))]))
    reader = es._Inflater(path)
    reader.read(10)
    reader.close()
    assert not reader._thread.is_alive()


def test_child_written_before_its_non_directory_parent_is_rejected(tmp_path):
    path = tmp_path / "a.tar.gz"
    _gz_tar(path, _core([_file("lib/x/y"), _file("lib/x")]))
    with pytest.raises(ValueError, match="non-directory parent"):
        es._safe_extract(path, tmp_path / "out")


def test_directory_attributes_are_applied_after_contents_and_links(tmp_path):
    path = tmp_path / "a.tar.gz"
    share = tarfile.TarInfo("share")
    share.type, share.mtime = tarfile.DIRTYPE, 1_000_000
    alias = tarfile.TarInfo("share/alias")
    alias.type, alias.linkname = tarfile.SYMTYPE, "doc"
    _gz_tar(path, _core([(share, None), _file("share/doc"), (alias, None)]))
    out = tmp_path / "out"
    es._safe_extract(path, out)
    assert (out / "share/alias").read_bytes() == b"x"
    # Writing doc and alias would bump the mtime had it been set first.
    assert (out / "share").stat().st_mtime == 1_000_000


@pytest.mark.parametrize("chunk_size", [17, 4 << 20])
@pytest.mark.parametrize("padding", [1, 2, 51])
def test_inflater_accepts_zero_padding_between_and_after_members(
    tmp_path, monkeypatch, chunk_size, padding
):
    import gzip

    monkeypatch.setattr(es._Inflater, "_CHUNK", chunk_size)
    monkeypatch.setattr(es._Inflater, "_OUT", 4096)
    parts = [b"a" * 60000, b"", b"b" * 47]
    data = b"".join(gzip.compress(part) + b"\0" * padding for part in parts)
    path = tmp_path / "padded.gz"
    path.write_bytes(data)
    with es._Inflater(path) as reader:
        assert reader.read() == gzip.decompress(data) == b"".join(parts)


@pytest.mark.parametrize(
    "damage, error", [("junk", zlib.error), ("truncate", EOFError)]
)
def test_inflater_rejects_invalid_member_after_zero_padding(tmp_path, damage, error):
    import gzip

    tail = b"not gzip" if damage == "junk" else gzip.compress(b"second")[:-1]
    path = tmp_path / "bad-padded.gz"
    path.write_bytes(gzip.compress(b"first") + b"\0" * 51 + tail)
    with pytest.raises(error), es._Inflater(path) as reader:
        reader.read()


@pytest.mark.parametrize(
    "damage, error",
    [("trailer", EOFError), ("next-member", EOFError), ("next-member-crc", zlib.error)],
)
def test_safe_extract_rejects_late_gzip_errors(tmp_path, damage, error):
    import gzip

    path = tmp_path / "late-error.tar.gz"
    alias = tarfile.TarInfo("python-link")
    alias.type, alias.linkname = tarfile.SYMTYPE, "bin/python"
    _gz_tar(path, _core([(alias, None)]))
    data = path.read_bytes()
    if damage == "trailer":
        data = data[:-1]
    else:
        tail = bytearray(gzip.compress(b""))
        if damage == "next-member":
            tail = tail[:-1]
        else:
            tail[-8] ^= 1
        data += tail
    path.write_bytes(data)
    out = tmp_path / "out"
    with pytest.raises(error):
        es._safe_extract(path, out)
    assert not (out / "python-link").is_symlink()


def test_safe_extract_accepts_zero_padded_archive(tmp_path):
    path = tmp_path / "padded.tar.gz"
    _gz_tar(path, _core())
    path.write_bytes(path.read_bytes() + b"\0" * 51)
    out = tmp_path / "out"
    es._safe_extract(path, out)
    assert (out / "bin/python").read_bytes() == b"synthetic Python"
