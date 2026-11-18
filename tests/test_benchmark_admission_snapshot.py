"""Exercise the benchmark's own SQLite snapshot path under real contention."""

import sqlite3
from contextlib import closing
from threading import Event, Thread

import pytest

from tools.benchmark_runtime import vlm_admission as admission


def snapshot_reader(gate, monkeypatch):
    started, done = Event(), Event()
    snapshots, errors = [], []
    connect = sqlite3.connect

    class SnapshotConnection(sqlite3.Connection):
        def execute(self, sql, *args, **kwargs):
            if sql == "BEGIN IMMEDIATE":
                started.set()
            return super().execute(sql, *args, **kwargs)

    def snapshot_connect(*args, **kwargs):
        return connect(*args, factory=SnapshotConnection, **kwargs)

    def read():
        try:
            snapshots.append(gate.snapshot())
        except Exception as error:
            errors.append(error)
        finally:
            done.set()

    monkeypatch.setattr(admission.sqlite3, "connect", snapshot_connect)
    return Thread(target=read), started, done, snapshots, errors


@pytest.mark.parametrize("commit", [True, False], ids=["commit", "rollback"])
def test_benchmark_snapshot_waits_for_writer_and_reads_only_committed_state(
    tmp_path, monkeypatch, commit
):
    gate = admission.SharedAdmission(
        admission.AdmissionSettings(str(tmp_path / "admission.sqlite"))
    )
    with closing(sqlite3.connect(gate.settings.database, isolation_level=None)) as db:
        db.execute("BEGIN IMMEDIATE")
        db.execute("UPDATE policy SET grants=17 WHERE id=1")
        reader, started, done, snapshots, errors = snapshot_reader(gate, monkeypatch)
        reader.start()
        try:
            assert started.wait(3)
            # Exceed the normal 100 ms writer timeout without replacing SQLite
            # or retrying the snapshot after a failure.
            assert not done.wait(0.25), errors
            db.commit() if commit else db.rollback()
            assert done.wait(3)
            assert not errors
            assert len(snapshots) == 1
            assert snapshots[0]["grants"] == (17 if commit else 0)
            assert snapshots[0]["active"] == snapshots[0]["waiting"] == 0
        finally:
            db.rollback()
            reader.join(6)
    assert not reader.is_alive()


def test_benchmark_snapshot_persistent_writer_is_bounded_and_recovers(
    tmp_path, monkeypatch
):
    gate = admission.SharedAdmission(
        admission.AdmissionSettings(str(tmp_path / "admission.sqlite"))
    )
    with closing(sqlite3.connect(gate.settings.database, isolation_level=None)) as db:
        db.execute("BEGIN IMMEDIATE")
        db.execute("UPDATE policy SET grants=23 WHERE id=1")
        reader, started, done, snapshots, errors = snapshot_reader(gate, monkeypatch)
        reader.start()
        try:
            assert started.wait(3)
            # Generous scheduling margin over 0.5 s, still shorter than both
            # the 3 s admission deadline and the 5 s release retry window.
            assert done.wait(2), "snapshot waited through the admission deadline"
            assert not snapshots
            assert len(errors) == 1
            assert isinstance(errors[0], sqlite3.OperationalError)
            assert errors[0].sqlite_errorcode == sqlite3.SQLITE_BUSY
            assert db.in_transaction
            row = db.execute("SELECT grants FROM policy WHERE id=1").fetchone()
            assert row == (23,)
        finally:
            db.rollback()
            reader.join(6)
    assert not reader.is_alive()
    # Failure must close/roll back its own connection; a later snapshot works
    # and cannot observe the abandoned writer's uncommitted update.
    assert gate.snapshot()["grants"] == 0
