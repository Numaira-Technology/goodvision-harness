"""The Windows controller must send POSIX scripts without CRLF translation."""

import sys

import pytest

from tools import start_sam


def test_ssh_script_stdin_preserves_utf8_and_lf(tmp_path):
    target = tmp_path / "received.bin"
    script = "set -euo pipefail\nprintf '视觉测试\\n'\n"
    start_sam.run(
        [
            sys.executable,
            "-c",
            "import pathlib,sys; "
            "pathlib.Path(sys.argv[1]).write_bytes(sys.stdin.buffer.read())",
            str(target),
        ],
        input=script,
    )
    assert target.read_bytes() == script.encode("utf-8")


def test_nonzero_command_still_fails():
    with pytest.raises(RuntimeError, match="status 7"):
        start_sam.run([sys.executable, "-c", "raise SystemExit(7)"])
