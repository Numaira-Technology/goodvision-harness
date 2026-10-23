"""Offline coordination keeps shared transaction and expiry guarantees."""

import sqlite3
import time
from contextlib import closing
from threading import Event
from types import SimpleNamespace

import pytest
from benchmark_runtime_fixtures import configure_ephemeral_admission

from tools.benchmark_runtime import vlm_admission
from tools.benchmark_runtime.vlm_admission import (
    AdmissionError,
    AdmissionSettings,
    SharedAdmission,
)


def test_ephemeral_fixture_keeps_shared_transactions_isolated(tmp_path, monkeypatch):
    original_connect = sqlite3.connect
    monkeypatch.setattr(vlm_admission, "sqlite3", sqlite3)
    settings = AdmissionSettings(str(tmp_path / "admission.sqlite"))
    config = SimpleNamespace(admission=settings)
    configure_ephemeral_admission(config)
    configure_ephemeral_admission(config)  # All warmup threads use the same gate.
    gate = SharedAdmission(settings)
    assert sqlite3.connect is original_connect

    with pytest.raises(ValueError, match="rollback probe"):
        with gate._transaction() as db:
            db.execute("INSERT INTO streams VALUES('rollback-probe',0,0)")
            with closing(sqlite3.connect(settings.database)) as observer:
                assert not observer.execute(
                    "SELECT 1 FROM streams WHERE id='rollback-probe'"
                ).fetchall()
            raise ValueError("rollback probe")
    with closing(sqlite3.connect(settings.database)) as observer:
        assert not observer.execute(
            "SELECT 1 FROM streams WHERE id='rollback-probe'"
        ).fetchall()

    with gate.slot("camera", stop=Event(), source_deadline=time.monotonic() + 3):
        with closing(sqlite3.connect(settings.database)) as observer:
            assert (
                observer.execute(
                    "SELECT COUNT(*) FROM tickets WHERE active=1"
                ).fetchone()[0]
                == 1
            )
    with pytest.raises(AdmissionError, match="admission_timeout"):
        with gate.slot("expired", stop=Event(), source_deadline=time.monotonic() - 1):
            pytest.fail("expired evidence was admitted")
    assert gate.snapshot()["grants"] == 1
    assert gate.snapshot()["active"] == gate.snapshot()["waiting"] == 0

    # Other databases and ordinary connections retain their original durability.
    with closing(vlm_admission.sqlite3.connect(tmp_path / "other.sqlite")) as other:
        with closing(sqlite3.connect(tmp_path / "normal.sqlite")) as normal:
            assert (
                other.execute("PRAGMA synchronous").fetchone()
                == normal.execute("PRAGMA synchronous").fetchone()
            )
    with closing(sqlite3.connect(settings.database)) as ordinary:
        assert ordinary.execute("PRAGMA synchronous").fetchone()[0] != 0
