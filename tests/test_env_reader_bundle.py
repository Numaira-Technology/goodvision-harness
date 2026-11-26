"""Execute only the uploaded reader bundle with synthetic local lock paths."""

import io
import os
import subprocess
import sys
import tarfile

import pytest

from tools import cloud_env


@pytest.mark.skipif(os.name != "posix", reason="SSH restore targets POSIX hosts")
def test_uploaded_reader_can_acquire_contend_and_release_prefix_lock(
    tmp_path, monkeypatch,
):
    payloads = []
    with monkeypatch.context() as patch:
        patch.setattr(
            cloud_env.subprocess, "run",
            lambda argv, **kwargs: payloads.append(kwargs["input"]),
        )
        cloud_env._default_uploader([], "synthetic.invalid", 22, "/unused-reader")
    reader = tmp_path / "reader"
    package = reader / "tools"
    package.mkdir(parents=True)
    with tarfile.open(fileobj=io.BytesIO(payloads[0])) as archive:
        archive.extractall(package, filter="data")
    program = """
import sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from tools.env_source import prefix_lock
prefix = Path(sys.argv[2])
with prefix_lock(prefix):
    try:
        with prefix_lock(prefix):
            raise AssertionError('second opener acquired an owned lock')
    except RuntimeError as error:
        assert 'holds the prefix lock' in str(error)
with prefix_lock(prefix):
    pass
assert not prefix.exists()
print('uploaded reader lock acquired, contended, released')
"""
    completed = subprocess.run(
        [sys.executable, "-I", "-c", program, str(reader), str(tmp_path / "env")],
        cwd=tmp_path, capture_output=True, text=True, timeout=15,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == (
        "uploaded reader lock acquired, contended, released"
    )
