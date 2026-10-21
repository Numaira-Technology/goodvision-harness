"""Memory admission preserves cross-process limits and has stage-owned lifetime."""

import json
import multiprocessing as mp
import os
import shutil
import time
from pathlib import Path
from threading import Event

import pytest
from benchmark_runtime_fixtures import jpeg

from tools import vlm_multistream_benchmark as benchmark
from tools.benchmark_runtime.vlm_admission import (
    AdmissionError,
    AdmissionSettings,
    SharedAdmission,
)
from tools.vlm_benchmark import admission_storage as storage
from tools.vlm_benchmark import modes


@pytest.mark.skipif(os.name == "nt", reason="Linux mount path semantics")
def test_mount_resolution_rejects_nested_disk_and_decodes_escapes():
    mounts = (
        "1 0 0:1 / / rw - overlay overlay rw\n"
        "2 1 0:2 / /dev/shm rw - tmpfs shm rw\n"
        "3 2 0:3 / /dev/shm/disk rw - ext4 /dev/sda rw\n"
        "4 1 0:4 / /memory\\040disk rw - tmpfs tmpfs rw\n"
    )
    assert storage.filesystem_type("/dev/shm/test", mounts) == "tmpfs"
    assert storage.filesystem_type("/dev/shm/disk/test", mounts) == "ext4"
    assert storage.filesystem_type("/dev/shmore/test", mounts) == "overlay"
    assert storage.filesystem_type("/memory disk/test", mounts) == "tmpfs"


def test_memory_requires_verified_tmpfs_without_silent_fallback(monkeypatch):
    monkeypatch.setattr(storage.sys, "platform", "linux")
    monkeypatch.setattr(
        Path, "read_text", lambda _: "1 0 0:1 / / rw - overlay overlay rw"
    )
    with pytest.raises(ValueError, match="verified tmpfs"):
        storage.memory_root()
    monkeypatch.setattr(storage.sys, "platform", "win32")
    with pytest.raises(ValueError, match="Linux tmpfs"):
        storage.memory_root()


def test_unique_memory_databases_cleanup_only_owned_directories(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "memory_root", lambda: tmp_path)
    stages = [tmp_path / name for name in ("a", "b")]
    for stage in stages:
        stage.mkdir()
    survivor = tmp_path / "unrelated.sqlite"
    survivor.write_text("keep")
    with storage.admission_database(stages[0], "memory") as a:
        with storage.admission_database(stages[1], "memory") as b:
            assert a.path != b.path
            if os.name != "nt":
                assert a.path.parent.stat().st_mode & 0o777 == 0o700
            ga, gb = (
                SharedAdmission(AdmissionSettings(str(a.path))),
                SharedAdmission(AdmissionSettings(str(b.path))),
            )
            assert ga.snapshot()["grants"] == gb.snapshot()["grants"] == 0
        assert a.path.exists() and not b.path.parent.exists()
    assert not a.path.parent.exists() and survivor.read_text() == "keep"
    assert (
        json.loads((stages[0] / "admission-storage.json").read_text())["cleanup"]
        == "removed"
    )


def test_disk_default_and_disabled_admission_do_not_need_tmpfs(tmp_path, monkeypatch):
    monkeypatch.setattr(
        storage, "memory_root", lambda: pytest.fail("unexpected tmpfs access")
    )
    with storage.admission_database(tmp_path) as d:
        SharedAdmission(AdmissionSettings(str(d.path)))
    assert d.path == tmp_path / "admission.sqlite" and d.path.exists()
    with storage.admission_database(tmp_path, "memory", enabled=False) as d:
        assert d.path is None and d.metadata["effective"] == "disabled"


def _take_slot(settings, acquired, release, results):
    gate = SharedAdmission(settings)
    try:
        with gate.slot("camera", stop=Event(), source_deadline=time.monotonic() + 3):
            acquired.set()
            if release is not None:
                assert release.wait(20)
        results.put("granted")
    except AdmissionError as exc:
        results.put(exc.code)


def test_distinct_processes_share_one_memory_path_and_limit(tmp_path, monkeypatch):
    # Location is injectable so macOS/Windows exercise the same lifecycle too.
    monkeypatch.setattr(storage, "memory_root", lambda: tmp_path)
    ctx = mp.get_context("spawn")
    acquired, release, contender_acquired = ctx.Event(), ctx.Event(), ctx.Event()
    results = ctx.Queue()
    with storage.admission_database(tmp_path, "memory") as d:
        settings = AdmissionSettings(str(d.path), 1, 1, 100, 0.2)
        gate = SharedAdmission(settings)
        holder = ctx.Process(
            target=_take_slot, args=(settings, acquired, release, results)
        )
        contender = ctx.Process(
            target=_take_slot, args=(settings, contender_acquired, None, results)
        )
        try:
            holder.start()
            assert acquired.wait(15)
            contender.start()
            assert results.get(timeout=15) == "admission_timeout"
            assert not contender_acquired.is_set()
            contender.join(5)
            assert contender.exitcode == 0
            assert gate.snapshot()["active"] == 1
            release.set()
            assert results.get(timeout=10) == "granted"
            holder.join(5)
            assert holder.exitcode == 0
            assert gate.snapshot()["grants"] == 1
            assert gate.snapshot()["active"] == gate.snapshot()["waiting"] == 0
        finally:
            release.set()
            for process in (holder, contender):
                if process.pid:
                    if process.is_alive():
                        process.terminate()
                    process.join(5)
            results.close()
            results.join_thread()
    assert not d.path.parent.exists()


def stage_options():
    args = modes.parser().parse_args(
        [
            "--modes",
            "parallel",
            "--parallel-scheduling",
            "paired",
            "--admission-storage",
            "memory",
            "--global-limit",
            "48",
            "--per-stream-limit",
            "4",
            "--duration",
            "3",
        ]
    )
    args.output_mode, args.deadline = "parallel", 3
    return args


def test_parallel_stage_preserves_results_and_cleans_after_workers(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(storage, "memory_root", lambda: tmp_path)
    args = stage_options()
    result = benchmark.run_stage(
        args, modes.configure(args), 2, [[jpeg()]], tmp_path, 1
    )
    assert result["all_drained"] and result["result_records_missing"] == 0
    assert result["shared_admission"]["grants"] == 12
    assert all(lane["complete_valid_results"] == 6 for lane in result["lanes"].values())
    meta = result["admission_storage"]
    assert meta["effective"] == "memory" and meta["cleanup"] == "removed"
    assert not Path(meta["database_path"]).parent.exists()
    stage = tmp_path / "streams-2-repeat-1"
    assert json.loads((stage / "summary.json").read_text())["admission_storage"] == meta
    assert len((stage / "reports.jsonl").read_text().splitlines()) == 12


@pytest.mark.parametrize("stop_confirmed", [True, False])
def test_startup_error_cleans_only_after_confirmed_shutdown(
    tmp_path, monkeypatch, stop_confirmed
):
    monkeypatch.setattr(storage, "memory_root", lambda: tmp_path)
    paths = []

    class Branch:
        def __init__(self, cfg, **kwargs):
            paths.append(Path(cfg.admission.database))
            self.state = "running"

        def start(self):
            raise RuntimeError("startup failed")

        def close(self, **kwargs):
            assert paths[0].exists(), "database removed while worker was still stopping"
            self.state = "stopped" if stop_confirmed else "stopping"

        def health(self):
            return {"state": self.state}

    monkeypatch.setattr(benchmark, "IndependentVLMBranch", Branch)
    args = stage_options()
    with pytest.raises(RuntimeError, match="startup failed"):
        benchmark.run_stage(args, modes.configure(args), 1, [[jpeg()]], tmp_path, 1)
    meta = json.loads(
        (tmp_path / "streams-1-repeat-1/admission-storage.json").read_text()
    )
    if stop_confirmed:
        assert meta["cleanup"] == "removed" and not paths[0].parent.exists()
    else:
        assert meta["cleanup"] == "retained_unconfirmed_worker_shutdown"
        assert paths[0].exists()
        shutil.rmtree(paths[0].parent)


def test_unavailable_memory_fails_before_starting_any_worker(tmp_path, monkeypatch):
    def unavailable():
        raise ValueError("tmpfs unavailable")

    monkeypatch.setattr(storage, "memory_root", unavailable)
    monkeypatch.setattr(
        benchmark, "IndependentVLMBranch", lambda *a, **k: pytest.fail("worker started")
    )
    args = stage_options()
    with pytest.raises(ValueError, match="tmpfs unavailable"):
        benchmark.run_stage(args, modes.configure(args), 1, [[jpeg()]], tmp_path, 1)
