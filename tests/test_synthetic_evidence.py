"""Offline failure evidence tests, without spawned load workers or wall sleeps."""

import hashlib
import os
import subprocess
import sys
from types import SimpleNamespace

import pytest

from tools import synthetic_evidence as evidence
from tools.benchmark_runtime import person_detector
from tools.vlm_benchmark import source_feeder


def item(tmp_path):
    root = tmp_path / "source"
    root.mkdir()
    node = SimpleNamespace(
        nodeid=sorted(evidence.TESTS)[0] + "[events]",
        config=SimpleNamespace(
            option=SimpleNamespace(ci_evidence=str(tmp_path / "evidence"))
        ),
    )
    evidence.register_synthetic_evidence(node, root, {"source": {}})
    return node, root


@pytest.mark.skipif(os.open not in os.supports_dir_fd, reason="Linux CI evidence")
def test_only_allowlisted_files_are_copied_with_hashes(tmp_path):
    node, root = item(tmp_path)
    for name in evidence.FILES:
        path = root / name
        path.parent.mkdir(exist_ok=True)
        path.write_text("{}\n")
    (root / "api-audit").mkdir()
    (root / "api-audit/secret.json").write_text("private")
    result = evidence.retain_synthetic_evidence(node, "call")
    assert result["complete"]
    assert len(result["files"]) == 5
    assert result["files"][0]["sha256"] == hashlib.sha256(b"{}\n").hexdigest()
    assert not list((tmp_path / "evidence").rglob("secret.json"))


@pytest.mark.parametrize(
    "damage",
    ["missing", "truncated", "oversize", "symlink", "parent_symlink", "directory"],
)
@pytest.mark.skipif(os.open not in os.supports_dir_fd, reason="Linux CI evidence")
def test_unsafe_or_incomplete_files_are_explicit(tmp_path, damage):
    node, root = item(tmp_path)
    path = root / evidence.FILES[0]
    path.parent.mkdir()
    if damage == "truncated":
        path.write_text("{")
    elif damage == "oversize":
        with path.open("wb") as f:
            f.truncate(evidence.MAX_FILE_BYTES + 1)
    elif damage in {"symlink", "parent_symlink"}:
        target = tmp_path / "private"
        target.mkdir()
        (target / "summary.json").write_text("{}")
        try:
            if damage == "parent_symlink":
                path.parent.rmdir()
                path.parent.symlink_to(target, target_is_directory=True)
            else:
                path.symlink_to(target / "summary.json")
        except OSError:
            pytest.skip("symlinks unavailable")
    elif damage == "directory":
        path.mkdir()
    result = evidence.retain_synthetic_evidence(node, "call")
    assert not result["complete"]
    row = result["files"][0]
    if damage == "truncated":
        assert row["content_status"] == "invalid_or_truncated_json"
    else:
        assert row["status"] == "omitted" and row["error_type"]


@pytest.mark.skipif(os.open not in os.supports_dir_fd, reason="Linux CI evidence")
def test_traversal_and_limits_and_opt_in(tmp_path, monkeypatch):
    node, root = item(tmp_path)
    with pytest.raises(ValueError, match="unsafe"):
        evidence._read(root, "../private.json")
    monkeypatch.setattr(evidence, "MAX_TOTAL_BYTES", 1)
    result = evidence.retain_synthetic_evidence(node, "call")
    assert result["files"][-1]["status"] == "omitted"
    monkeypatch.setattr(evidence, "MAX_FILES", 0)
    result = evidence.retain_synthetic_evidence(node, "teardown")
    assert all(row["reason"] == "file count limit" for row in result["files"])
    with pytest.raises(ValueError, match="unsafe"):
        evidence._read(root, str(root.absolute()))
    node.nodeid = "tests/test_other.py::test_other"
    assert evidence.retain_synthetic_evidence(node, "call") is None


def test_source_drop_and_broker_expiry_remain_distinct(tmp_path, monkeypatch):
    clock = SimpleNamespace(monotonic=lambda: 1.0)
    monkeypatch.setattr(source_feeder, "time", clock)
    branch = SimpleNamespace(
        config=SimpleNamespace(stream_session_id="synthetic"), offer=lambda *args: True
    )
    feeder = source_feeder.TimedSourceFeeder(
        [branch], [[b"synthetic"]], [(0, [(0, 0)]), (1, [(0, 1)])], 0, 2.5
    )
    feeder._run()
    rows = feeder.diagnostics()
    assert rows["complete"]
    assert rows["drop_counters_reconciled"]
    assert rows["missed"] == rows["derived_drop_count"] == 1
    assert rows["missed_by_stream"] == rows["derived_drops_by_stream"] == [1]
    assert rows["offer_rejected"] == [0]
    feeder.missed = 2
    assert not feeder.diagnostics()["complete"]
    assert not feeder.diagnostics()["drop_counters_reconciled"]
    feeder.missed = 1
    assert [r["source_drop"] for r in rows["samples"]] == [True, False]
    assert rows["samples"][1]["derived_measured_release"] == 1
    broker = person_detector.DetectorBroker(
        "http://localhost:1", transport=lambda *_: pytest.fail("expired")
    )
    monkeypatch.setattr(person_detector, "time", clock)
    broker.submit(0, "synthetic", 1, 0.4, b"synthetic")
    clock.monotonic = lambda: 4.0
    original = broker._finish

    def finish(*args, **kwargs):
        original(*args, **kwargs)
        broker.stop.set()

    monkeypatch.setattr(broker, "_finish", finish)
    broker._consume()
    assert broker.drain() == {"submitted": 1, "queue_expired": 1}
    assert broker.cache["synthetic", 1]["state"] == "queue_expired"
    # Interruptions leave an unmeasured suffix; extra lags are not hidden.
    feeder.release_lags.pop()
    assert not feeder.diagnostics()["complete"]
    assert "source_drop" not in feeder.diagnostics()["samples"][1]
    feeder.release_lags.extend([0, 0])
    assert feeder.diagnostics()["unmatched_measurement_count"] == 1
    feeder.error = ValueError("synthetic error")
    assert feeder.diagnostics()["feeder_error_type"] == "ValueError"


@pytest.mark.parametrize("outcome", ["pass", "fail", "collector_error"])
def test_ci_plugin_preserves_primary_outcome_in_subprocess(tmp_path, outcome):
    tests = tmp_path / "tests"
    tests.mkdir()
    # Exact synthetic opt-in identity, but no real load workers.
    test = tests / "test_tracked_decisions_flow.py"
    test.write_text(
        """from tools.synthetic_evidence import register_synthetic_evidence

def test_twelve_sources_native_decisions_and_person_reports(
    tmp_path, request, monkeypatch
):
    register_synthetic_evidence(request.node, tmp_path, {"source": {}})
"""
        + (
            """    import tools.ci_pytest as plugin
    def broken(*args):
        raise OSError("collector failure")
    monkeypatch.setattr(plugin, "retain_synthetic_evidence", broken)
"""
            if outcome == "collector_error"
            else ""
        )
        + (
            '    assert False, "PRIMARY_SYNTHETIC_FAILURE"\n'
            if outcome != "pass"
            else "    assert True\n"
        )
    )
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-p",
            "tools.ci_pytest",
            "--ci-shard=0",
            "--ci-shards=1",
            "--ci-evidence=output",
            "tests",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == (0 if outcome == "pass" else 1), (
        result.stdout + result.stderr
    )
    if outcome != "pass":
        assert "PRIMARY_SYNTHETIC_FAILURE" in result.stdout
        assert "INTERNALERROR" not in result.stdout
    manifests = list((tmp_path / "output").rglob("manifest.json"))
    assert bool(manifests) == (outcome == "fail" and os.open in os.supports_dir_fd)
    if outcome == "collector_error":
        assert "synthetic evidence retention error" in result.stdout


@pytest.mark.skipif(os.open not in os.supports_dir_fd, reason="Linux CI evidence")
@pytest.mark.parametrize("component", ["root", "intermediate", "leaf"])
def test_destination_symlinks_do_not_redirect_writes(tmp_path, component):
    node, _root = item(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    directory = tmp_path / "evidence"
    if component != "root":
        directory.mkdir()
        directory = directory / "synthetic-failures"
    if component == "leaf":
        directory.mkdir()
        directory = directory / (
            hashlib.sha256(node.nodeid.encode()).hexdigest() + "-call"
        )
    directory.symlink_to(outside, target_is_directory=True)
    with pytest.raises(OSError):
        evidence.retain_synthetic_evidence(node, "call")
    assert not list(outside.iterdir())


@pytest.mark.skipif(os.open not in os.supports_dir_fd, reason="Linux CI evidence")
@pytest.mark.parametrize("component", ["directory", "file"])
def test_payload_output_symlinks_are_rejected(tmp_path, component):
    root = tmp_path / "output"
    outside = tmp_path / "outside"
    outside.mkdir()
    with evidence._directory_fd(root) as fd:
        if component == "directory":
            (root / "stage").symlink_to(outside, target_is_directory=True)
        else:
            (root / "stage").mkdir()
            (root / "stage/data.json").symlink_to(outside / "data.json")
        with pytest.raises(OSError):
            evidence._write(fd, "stage/data.json", b"{}")
    assert not list(outside.iterdir())


def test_diagnostic_json_encoding_is_bounded(monkeypatch):
    monkeypatch.setattr(evidence, "MAX_FILE_BYTES", 64)
    with pytest.raises(ValueError, match="diagnostic byte limit"):
        evidence._bounded_json({"rows": list(range(100))})
    with pytest.raises(ValueError, match="diagnostic byte limit"):
        evidence._bounded_json({"value": "x" * 100})
    assert evidence._bounded_json({"ready": 1}) == b'{"ready": 1}\n'


@pytest.mark.skipif(os.open not in os.supports_dir_fd, reason="Linux CI evidence")
@pytest.mark.parametrize("force_depth_error", [False, True])
def test_deep_json_retention_keeps_hash_and_byte_accounting(
    tmp_path, monkeypatch, force_depth_error
):
    node, root = item(tmp_path)
    path = root / evidence.FILES[0]
    path.parent.mkdir()
    data = b"[" * 2000 + b"0" + b"]" * 2000
    path.write_bytes(data)
    if force_depth_error:

        def depth_error(*args, **kwargs):
            raise RecursionError("synthetic parser depth")

        monkeypatch.setattr(evidence.json, "loads", depth_error)
    result = evidence.retain_synthetic_evidence(node, "call")
    row = result["files"][0]
    assert row["status"] == "copied"
    assert row["bytes"] == len(data)
    assert row["sha256"] == hashlib.sha256(data).hexdigest()
    # Python 3.11 rejects this depth; 3.12 can parse it. Neither may hide a copy.
    assert row["content_status"] in {"valid", "unvalidated_json_depth"}
    if force_depth_error:
        assert row["content_status"] == "unvalidated_json_depth"
    copied = next((tmp_path / "evidence").rglob("summary.json"))
    assert copied.read_bytes() == data
    assert result["copied_bytes"] >= len(data)
    assert result["retained_bytes_upper_bound"] == result["copied_bytes"]


@pytest.mark.skipif(os.open not in os.supports_dir_fd, reason="Linux CI evidence")
@pytest.mark.parametrize("failure", ["partial", "short", "close", "cleanup"])
def test_failed_writes_cannot_hide_payload_bytes_or_exceed_cap(
    tmp_path, monkeypatch, failure
):
    node, root = item(tmp_path)
    for name in evidence.FILES:
        path = root / name
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(b" " * 60)
    monkeypatch.setattr(evidence, "MAX_TOTAL_BYTES", 100)
    original_fdopen = evidence.os.fdopen
    injected = False

    class FaultyOutput:
        def __init__(self, stream):
            self.stream = stream

        def __enter__(self):
            return self

        def write(self, data):
            if failure == "close":
                return self.stream.write(data)
            self.stream.write(data[:50])
            self.stream.flush()
            if failure == "short":
                return 50
            raise OSError("synthetic partial write failure")

        def __exit__(self, *args):
            self.stream.close()
            if failure == "close":
                raise OSError("synthetic close failure")

    def fdopen(fd, mode, *args, **kwargs):
        nonlocal injected
        stream = original_fdopen(fd, mode, *args, **kwargs)
        if mode == "wb" and not injected:
            injected = True
            return FaultyOutput(stream)
        return stream

    monkeypatch.setattr(evidence.os, "fdopen", fdopen)
    if failure == "cleanup":
        original_unlink = evidence.os.unlink

        def unlink(path, *args, **kwargs):
            if path == "summary.json":
                raise OSError("synthetic cleanup failure")
            return original_unlink(path, *args, **kwargs)

        monkeypatch.setattr(evidence.os, "unlink", unlink)
    result = evidence.retain_synthetic_evidence(node, "call")
    manifest = next((tmp_path / "evidence").rglob("manifest.json"))
    directory = manifest.parent
    actual = sum(
        path.stat().st_size
        for path in directory.rglob("*")
        if path.is_file() and path.name != "manifest.json"
    )
    assert actual <= result["retained_bytes_upper_bound"] <= 100
    first = result["files"][0]
    if failure == "cleanup":
        assert first["status"] == "write_failed_cleanup_failed"
        assert first["retained_bytes_upper_bound"] == 60
        assert (directory / evidence.FILES[0]).stat().st_size == 50
        assert actual == result["copied_bytes"] + 50
    else:
        assert first["status"] == "omitted"
        assert not (directory / evidence.FILES[0]).exists()
        assert actual == result["copied_bytes"]
    for row in result["files"]:
        if row["status"] == "copied":
            data = (directory / row["path"]).read_bytes()
            assert row["bytes"] == len(data)
            assert row["sha256"] == hashlib.sha256(data).hexdigest()
