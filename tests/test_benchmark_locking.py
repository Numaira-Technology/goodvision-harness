"""Offline process-lock regressions; no provider, model, camera or network calls."""

import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
UNIT = {"id": "one", "request_sha256": "frozen", "targets": ["fight"]}
INTENT = {"unit": "one", "request_sha256": "frozen", "attempt": 1, "type": "intent"}


def child(code, *args):
    return subprocess.run(
        [sys.executable, "-c", code, *map(str, args)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=20,
    )


def contender(path, read_only=False):
    result = child(
        """
import json, sys
from tools.unified_benchmark.retry import Ledger
try:
    ledger = Ledger(sys.argv[1], 'lock-test', read_only=sys.argv[2] == 'True')
except OSError:
    print('blocked')
else:
    try:
        unit = {'id':'one','request_sha256':'frozen','targets':['fight']}
        print(json.dumps(ledger.state(unit, 'choice')[0]))
    finally:
        ledger.close()
""",
        path,
        read_only,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return result.stdout.strip()


def test_offline_imports_do_not_require_a_lock_backend():
    result = child("""
import subprocess  # Initialize stdlib platform detection before hiding backends.
import sys
sys.modules['fcntl'] = None
sys.modules['msvcrt'] = None
from tools.unified_benchmark import retry, native_run, api_run, dataset202
from tools.qwen8b import eval_local
assert retry.Ledger and native_run.main
assert api_run.Transport and eval_local.LocalTransport
""")
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("read_only", [False, True])
def test_ledger_excludes_second_process_before_and_after_append(tmp_path, read_only):
    from tools.unified_benchmark.retry import Ledger

    path = tmp_path / "attempts.jsonl"
    first = Ledger(path, "lock-test")
    try:
        # Both empty-file and shifted-EOF attempts must lock the same region.
        assert first.state(UNIT, "choice")[0] == 0
        assert contender(path, read_only) == "blocked"
        first.append(INTENT)
        assert contender(path, read_only) == "blocked"
        assert first.state(UNIT, "choice")[0] == 1
    finally:
        first.close()
    assert contender(path, read_only) == "1"
    assert len(path.read_text().splitlines()) == 1


def test_read_only_owner_is_exclusive_and_does_not_modify_evidence(tmp_path):
    from tools.unified_benchmark.retry import Ledger

    path = tmp_path / "attempts.jsonl"
    writer = Ledger(path, "lock-test")
    writer.append(INTENT)
    writer.close()
    before = path.read_bytes()
    reader = Ledger(path, "lock-test", read_only=True)
    try:
        assert reader.state(UNIT, "choice")[0] == 1
        assert contender(path) == "blocked"
        assert contender(path, True) == "blocked"
    finally:
        reader.close()
    assert path.read_bytes() == before
    assert contender(path) == "1"


def test_ledger_closes_failed_owner_and_does_not_unlock_existing_owner(tmp_path):
    from tools.unified_benchmark.retry import Ledger

    path = tmp_path / "attempts.jsonl"
    first = Ledger(path, "lock-test")
    try:
        with pytest.raises(OSError):
            Ledger(path, "lock-test")
        assert contender(path) == "blocked"
    finally:
        first.close()
    path.write_text("not-json\n")
    with pytest.raises(ValueError):
        Ledger(path, "lock-test")
    # Failed parsing must have closed its descriptor and released its lock.
    path.write_text("")
    assert contender(path) == "0"


@pytest.mark.parametrize("platform,missing", [("posix", "fcntl"), ("nt", "msvcrt")])
def test_missing_backend_fails_closed(tmp_path, monkeypatch, platform, missing):
    from tools.unified_benchmark import locking, retry

    monkeypatch.setattr(locking, "os", SimpleNamespace(name=platform))
    monkeypatch.setitem(sys.modules, missing, None)
    path = tmp_path / "attempts.jsonl"
    with pytest.raises(ImportError):
        retry.Ledger(path, "lock-test")
    assert path.read_bytes() == b""


def test_unsupported_platform_fails_closed(tmp_path, monkeypatch):
    from tools.unified_benchmark import locking

    monkeypatch.setattr(locking, "os", SimpleNamespace(name="unsupported"))
    with (tmp_path / "lock").open("a+") as file:
        with pytest.raises(RuntimeError, match="unsupported"):
            locking.lock_exclusive(file)


@pytest.mark.parametrize("contents", ["", "first line\nsecond line\n"])
@pytest.mark.parametrize("fail", [False, True])
def test_windows_lock_anchors_byte_zero_and_restores_position(
    tmp_path, monkeypatch, contents, fail
):
    from tools.unified_benchmark import locking

    calls = []

    def acquire(fd, mode, length):
        calls.append((fd, mode, length, os.lseek(fd, 0, os.SEEK_CUR)))
        if fail:
            raise OSError("occupied")

    backend = SimpleNamespace(LK_NBLCK=7, locking=acquire)
    monkeypatch.setattr(locking, "os", SimpleNamespace(name="nt"))
    monkeypatch.setitem(sys.modules, "msvcrt", backend)
    path = tmp_path / "evidence.jsonl"
    path.write_text(contents)
    with path.open("a+", encoding="utf-8") as file:
        before = file.tell()
        if fail:
            with pytest.raises(OSError, match="occupied"):
                locking.lock_exclusive(file)
        else:
            locking.lock_exclusive(file)
        assert calls == [(file.fileno(), 7, 1, 0)]
        assert file.tell() == before
    assert path.read_text() == contents


def test_posix_backend_keeps_existing_flock_flags_and_position(tmp_path, monkeypatch):
    from tools.unified_benchmark import locking

    calls = []
    backend = SimpleNamespace(
        LOCK_EX=2, LOCK_NB=4, flock=lambda f, flags: calls.append((f, flags, f.tell()))
    )
    monkeypatch.setattr(locking, "os", SimpleNamespace(name="posix"))
    monkeypatch.setitem(sys.modules, "fcntl", backend)
    path = tmp_path / "lock"
    path.write_text("existing")
    with path.open("a+") as file:
        position = file.tell()
        locking.lock_exclusive(file)
        assert calls == [(file, 6, position)]
        assert file.tell() == position


def test_native_cli_uses_same_process_lock_before_execute(tmp_path, monkeypatch):
    from tools.unified_benchmark import locking, native_run

    lock_path = tmp_path / "evaluation.lock"
    args = [
        "--model",
        "clip",
        "--bundle",
        str(tmp_path),
        "--deployment",
        str(tmp_path),
        "--deployment-sha256",
        "frozen",
        "--output",
        str(tmp_path / "out"),
        "--authorization-note",
        "synthetic offline test",
        "--execute",
    ]
    calls = []
    monkeypatch.setattr(native_run, "execute", lambda *a, **kw: calls.append(True))
    # main() modifies the environment only within this test's monkeypatch scope.
    for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_HUB_DISABLE_XET"):
        monkeypatch.setenv(name, "test")
    with lock_path.open("a") as first:
        locking.lock_exclusive(first)
        with pytest.raises(OSError):
            native_run.main(args)
        assert not calls
    native_run.main(args)
    assert calls == [True]

    # main's context closes its descriptor even if the execution body raises.
    def fail(*a, **kw):
        raise RuntimeError("synthetic body failure")

    monkeypatch.setattr(native_run, "execute", fail)
    with pytest.raises(RuntimeError, match="synthetic body failure"):
        native_run.main(args)
    with lock_path.open("a") as next_owner:
        locking.lock_exclusive(next_owner)


def test_process_exit_releases_ledger_lock(tmp_path):
    path = tmp_path / "attempts.jsonl"
    result = child(
        """
import os, sys
from tools.unified_benchmark.retry import Ledger
ledger = Ledger(sys.argv[1], 'lock-test')
ledger.append({'unit':'one','request_sha256':'frozen','attempt':1,'type':'intent'})
os._exit(0)
""",
        path,
    )
    assert result.returncode == 0, result.stderr
    assert contender(path) == "1"


def test_owned_raw_hash_is_exact_bounded_and_restores_position(tmp_path, monkeypatch):
    import hashlib
    import json
    from contextlib import contextmanager

    from tools.unified_benchmark.retry import Ledger

    path = tmp_path / "attempts.jsonl"
    row = INTENT | {"context": "lock-test", "note": "é" * (1024 * 1024)}
    raw = json.dumps(row, ensure_ascii=False).encode("utf-8") + b"\r\n"
    path.write_bytes(raw)
    ledger = Ledger(path, "lock-test")
    try:
        ledger.file.seek(0)
        ledger.file.read(3)
        position = ledger.file.tell()
        original_reader = ledger.raw_reader
        sizes = []

        @contextmanager
        def counted_reader():
            with original_reader() as reader:

                def read(size):
                    sizes.append(size)
                    return reader.read(size)

                yield SimpleNamespace(read=read)

        monkeypatch.setattr(ledger, "raw_reader", counted_reader)
        assert ledger.file_hash() == hashlib.sha256(raw).hexdigest()
        assert len(sizes) >= 3 and set(sizes) == {1024 * 1024}
        assert ledger.file.tell() == position
        with pytest.raises(RuntimeError, match="reader failure"):
            with original_reader() as reader:
                assert reader.read(7) == raw[:7]
                assert contender(path) == "blocked"
                raise RuntimeError("reader failure")
        assert ledger.file.tell() == position
        ledger.append(INTENT | {"attempt": 2})
        with original_reader() as reader:
            final_bytes = reader.read()
        assert final_bytes.startswith(raw)
        assert ledger.file_hash() == hashlib.sha256(final_bytes).hexdigest()
    finally:
        ledger.close()
    assert path.read_bytes() == final_bytes
