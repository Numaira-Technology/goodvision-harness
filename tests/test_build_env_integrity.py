"""Offline regressions for package constraints and packed RECORD validation."""

import base64
import csv
import hashlib
import io
import json
import tarfile

import pytest

from tests.test_build_env import SP, _conda_prefix, _dist, _pack_tar
from tools import build_env as be


@pytest.mark.parametrize("kind", ["directory", "flat"])
def test_conda_constraints_include_egg_info_and_duplicate_ownership(tmp_path, kind):
    prefix = _conda_prefix(tmp_path)
    rel = SP + "setuptools-84.2.1-py3.12.egg-info"
    if kind == "directory":
        rel += "/PKG-INFO"
    path = prefix / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("Name: setuptools\nVersion: 84.2.1\n")
    (prefix / "conda-meta/setuptools.json").write_text(json.dumps({"files": [rel]}))
    assert be._conda_python_dists(prefix) == {"pip": "26.2.2", "setuptools": "84.2.1"}
    newer = prefix / SP / "setuptools-85.0.dist-info"
    newer.mkdir()
    (newer / "METADATA").write_text("Name: setuptools\nVersion: 85.0\n")
    with pytest.raises(RuntimeError, match="two installed versions of setuptools"):
        be.check_conda_ownership(prefix)


def test_missing_conda_metadata_never_runs_unconstrained_pip(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(be, "_run", lambda argv, **kw: calls.append(argv))
    with pytest.raises(RuntimeError, match="not a conda environment"):
        be._build_prefix(
            tmp_path / "conda", tmp_path / "p", tmp_path / "r", "3.12", "8.9"
        )
    assert len(calls) == 1 and "create" in calls[0]


def test_missing_record_is_not_a_verified_pack(tmp_path):
    members = _dist("demo", "1", {"demo.py": b"ok"})
    del members[SP + "demo-1.dist-info/RECORD"]
    with pytest.raises(RuntimeError, match="RECORD"):
        be.verify_pack(_pack_tar(tmp_path / "missing.tgz", members))


@pytest.mark.parametrize("algorithm", ["sha256", "sha384", "sha512"])
def test_csv_quoted_paths_and_all_supported_digests_are_checked(tmp_path, algorithm):
    name, original = "comma,and\nnewline.py", b"good"
    members = _dist("demo", "1", {name: original})
    output = io.StringIO(newline="")
    digest = base64.urlsafe_b64encode(hashlib.new(algorithm, original).digest())
    csv.writer(output).writerow(
        [name, algorithm + "=" + digest.rstrip(b"=").decode(), 4]
    )
    members[SP + "demo-1.dist-info/RECORD"] = output.getvalue().encode()
    assert (
        be.verify_pack(_pack_tar(tmp_path / "valid.tgz", members))["files_checked"] == 1
    )
    members[SP + name] = b"evil"
    with pytest.raises(RuntimeError, match="bytes differ"):
        be.verify_pack(_pack_tar(tmp_path / "bad.tgz", members))


@pytest.mark.parametrize("change", ["placeholder", "size", "weak_hash", "malformed"])
def test_mismatches_are_never_silently_exempted(tmp_path, change):
    members = _dist("demo", "1", {"demo.py": b"good"})
    record = SP + "demo-1.dist-info/RECORD"
    if change == "placeholder":
        members[SP + "demo.py"] = b"evil" + be._CONDA_PLACEHOLDER
    elif change == "size":
        members[record] = members[record].replace(b",4\n", b",999\n")
    elif change == "weak_hash":
        members[record] = members[record].replace(b"sha256=", b"md5=")
    else:
        members[record] = b"demo.py\n"
    with pytest.raises(RuntimeError):
        be.verify_pack(_pack_tar(tmp_path / "bad.tgz", members))


def test_shared_opencv_files_require_same_version(tmp_path):
    members = {
        **_dist("opencv_python", "4.12", {"cv2.py": b"full"}),
        **_dist("opencv_python_headless", "4.11", {"cv2.py": b"headless"}),
    }
    with pytest.raises(RuntimeError, match="claimed by"):
        be.verify_pack(_pack_tar(tmp_path / "bad.tgz", members))


def test_duplicate_archive_members_fail_instead_of_using_last_bytes(tmp_path):
    members = _dist("demo", "1", {"demo.py": b"good"})
    path = tmp_path / "duplicate.tgz"
    with tarfile.open(path, "w:gz") as tar:
        for name, data in [*members.items(), (SP + "demo.py", b"good")]:
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    with pytest.raises(RuntimeError, match="duplicate packed file"):
        be.verify_pack(path)


def test_archive_hashing_uses_bounded_reads(tmp_path, monkeypatch):
    members = _dist("demo", "1", {"demo.so": b"x" * (3 * 1024**2)})
    path = _pack_tar(tmp_path / "large.tgz", members)
    original = tarfile.ExFileObject.read
    sizes = []

    def bounded(self, size=-1):
        assert 0 < size <= 1024 * 1024
        sizes.append(size)
        return original(self, size)

    monkeypatch.setattr(tarfile.ExFileObject, "read", bounded)
    assert be.verify_pack(path)["files_checked"] == 2
    assert len(sizes) > 3


@pytest.mark.parametrize("record", [b"", b"demo.py,,\n"])
def test_empty_or_hashless_records_are_not_verified(tmp_path, record):
    members = _dist("demo", "1", {"demo.py": b"good"})
    members[SP + "demo-1.dist-info/RECORD"] = record
    with pytest.raises(RuntimeError, match="no verifiable RECORD"):
        be.verify_pack(_pack_tar(tmp_path / "empty.tgz", members))


def test_optional_record_size_does_not_skip_hash_check(tmp_path):
    members = _dist("demo", "1", {"demo.py": b"good"})
    rec = SP + "demo-1.dist-info/RECORD"
    members[rec] = members[rec].replace(b",4\n", b",\n")
    assert be.verify_pack(_pack_tar(tmp_path / "ok.tgz", members))["files_checked"] == 2
    members[SP + "demo.py"] = b"evil"
    with pytest.raises(RuntimeError, match="bytes differ"):
        be.verify_pack(_pack_tar(tmp_path / "bad.tgz", members))
