"""Offline coordination keeps shared transaction and expiry guarantees."""

import sqlite3
import time
from contextlib import closing
from dataclasses import fields, replace
from threading import Event
from types import SimpleNamespace

import benchmark_runtime_fixtures as fixtures
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


def test_benchmark_config_preserves_shared_values_without_product_sampling(
    monkeypatch,
):
    source = replace(
        fixtures.main_config("source-camera", 7),
        model="shared-model", observation_fps=0.5, max_in_flight=3,
        parameters={"temperature": 0}, options={"max_image_side": 640},
        candidate_fps=4.0, observation_window_frames=2,
        observation_frame_delta_seconds=0.5,
    )
    monkeypatch.setattr(fixtures, "main_config", lambda session, generation: source)
    benchmark = fixtures.config("source-camera", 7)
    shared = {field.name for field in fields(source)} & {
        field.name for field in fields(benchmark)
    }
    assert all(getattr(benchmark, name) == getattr(source, name) for name in shared)
    assert benchmark.stream_session_id == "source-camera"
    assert benchmark.generation == 7
    assert benchmark.model == "shared-model"
    assert benchmark.parameters == {"temperature": 0}
    assert benchmark.options == {"max_image_side": 640}
    # Product-only candidate/window controls must not alter the frozen
    # benchmark's cadence, evidence selection or grounding route defaults.
    assert benchmark.observation_fps == 0.5
    assert benchmark.source_sample_fps is None
    assert benchmark.sample_fps == 2.0
    assert benchmark.grounding_source == "vlm"
    assert benchmark.grounding_format == "legacy"
    assert benchmark.parallel_scheduling == "independent"
