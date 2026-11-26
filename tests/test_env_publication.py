"""Atomic restore publication and handled-interruption ownership boundaries."""

import ctypes
import errno
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_env_on_r2 import assert_clean, pull
from test_env_on_r2 import pack as restore_pack  # noqa: F401

from tools import env_source as es


@pytest.fixture
def pack(request):
    return request.getfixturevalue("restore_pack")


@pytest.mark.parametrize("after_publication", [False, True])
def test_interrupted_publication_cleans_owned_prefix_and_can_retry(
    pack, tmp_path, monkeypatch, after_publication,
):
    env, _ = pack
    publish = es._publish_prefix

    def interrupt(staged, prefix):
        if after_publication:
            publish(staged, prefix)
        raise KeyboardInterrupt("synthetic handled termination")

    monkeypatch.setattr(es, "_publish_prefix", interrupt)
    with pytest.raises(KeyboardInterrupt):
        pull(pack, tmp_path)
    assert_clean(tmp_path, env)
    monkeypatch.setattr(es, "_publish_prefix", publish)
    pull(pack, tmp_path)
    assert es._already_restored(env)


@pytest.mark.parametrize("competitor", ["empty", "nonempty", "file", "symlink"])
def test_publication_race_preserves_competitor(pack, tmp_path, monkeypatch, competitor):
    env, _ = pack
    prefix = Path(env["unpack_to"])
    extract = es._safe_extract
    original = []

    def race(tarball, stage):
        extract(tarball, stage)
        if competitor in {"empty", "nonempty"}:
            prefix.mkdir()
            if competitor == "nonempty":
                (prefix / "keep").write_text("competitor")
        elif competitor == "file":
            prefix.write_text("competitor")
        else:
            prefix.symlink_to(tmp_path / "absent-competitor")
        original.append(prefix.lstat())

    monkeypatch.setattr(es, "_safe_extract", race)
    with pytest.raises(FileExistsError):
        pull(pack, tmp_path)
    after = prefix.lstat()
    assert (after.st_dev, after.st_ino) == (original[0].st_dev, original[0].st_ino)
    if competitor == "nonempty":
        assert (prefix / "keep").read_text() == "competitor"
    elif competitor == "file":
        assert prefix.read_text() == "competitor"
    elif competitor == "symlink":
        assert os.readlink(prefix) == str(tmp_path / "absent-competitor")
    else:
        assert list(prefix.iterdir()) == []
    assert not list((tmp_path / "scratch").iterdir())
    assert not list(prefix.parent.glob(".r2-env-stage-*"))


def test_unsupported_atomic_publish_fails_closed(pack, tmp_path, monkeypatch):
    monkeypatch.setattr(es.ctypes, "CDLL", lambda *args, **kwargs: SimpleNamespace())
    with pytest.raises(RuntimeError, match="atomic no-replace"):
        pull(pack, tmp_path)
    assert_clean(tmp_path, pack[0])


@pytest.mark.parametrize("platform,function,flag", [
    ("linux", "renameat2", 1), ("darwin", "renamex_np", 4),
])
def test_atomic_publish_uses_exclusive_flag_and_preserves_errno(
    tmp_path, monkeypatch, platform, function, flag,
):
    calls = []

    def unavailable(*args):
        calls.append(args)
        ctypes.set_errno(errno.ENOTSUP)
        return -1

    monkeypatch.setattr(es.sys, "platform", platform)
    monkeypatch.setattr(
        es.ctypes, "CDLL",
        lambda *args, **kwargs: SimpleNamespace(**{function: unavailable}),
    )
    source, target = tmp_path / "stage", tmp_path / "prefix"
    source.mkdir()
    with pytest.raises(OSError) as caught:
        es._publish_prefix(source, target)
    assert caught.value.errno == errno.ENOTSUP
    assert calls[0][-1] == flag
    assert os.fsencode(source) in calls[0] and os.fsencode(target) in calls[0]
    assert source.is_dir() and not target.exists()
