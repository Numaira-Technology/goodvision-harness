"""Offline synthetic conda-pack restore tests. No R2 objects or GPU execution."""

import copy
import io
import json
import os
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest

from tools import env_source as es
from tools import registry


def test_registry_ships_and_parses():
    assert registry.weights_of("videojev") == (
        "ZhuYuhan/VideoJev-4B",
        "1c1a253bf90dfa66176124d25a48ffb791d517d5",
    )
    assert registry.env_of("videojev")["compat"]["gpu_family"] == "ada"
    assert registry.env_of("clip") is None
    assert "videojev" in registry.model_names()
    with pytest.raises(registry.RegistryError):
        registry.weights_of("does-not-exist")
    with pytest.raises(registry.RegistryError):
        es.ensure_env("clip")


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
            (Path(target) / "someone-elses-file").write_text("keep")
        raise OSError("synthetic publication failure")

    monkeypatch.setattr(Path, "rename", fail_rename)
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


def test_packed_python_missing_aborts(tmp_path):
    prefix = tmp_path / "opt/env"
    prefix.mkdir(parents=True)
    env = _launch_spec(prefix)
    with pytest.raises(RuntimeError, match="packed interpreter missing"):
        es.packed_python(env)


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
    python.write_text(f"#!/bin/sh\nexec {shlex.quote(sys.executable)} \"$@\"\n")
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
