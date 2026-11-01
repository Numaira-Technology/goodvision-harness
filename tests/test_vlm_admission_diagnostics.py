import sqlite3
import time
from contextlib import contextmanager
from threading import Event

import pytest

from tools.benchmark_runtime.vlm_admission import (
    AdmissionError,
    AdmissionSettings,
    AdmissionTrace,
    SharedAdmission,
)


@pytest.mark.parametrize(
    "condition,possible,metric",
    [
        ("global", False, "global_full_polls"),
        ("stream", False, "stream_full_polls"),
        ("rate", False, "rate_spacing_polls"),
        ("fairness", False, "other_candidate_with_capacity_polls"),
        ("expired", False, "no_candidate_polls"),
        ("ready", True, None),
    ],
)
def test_diagnostics_observe_constraints_without_changing_admission(
    tmp_path,
    condition,
    possible,
    metric,
):
    settings = AdmissionSettings(str(tmp_path / "gate.sqlite"), 2, 1, 40, 1)
    gate = SharedAdmission(settings)
    now = time.monotonic()
    with gate._transaction() as db:
        db.execute("INSERT INTO streams VALUES('PRIVATE_TARGET',1,?)", (now,))
        db.execute("INSERT INTO streams VALUES('PRIVATE_OTHER',0,?)", (now,))
        db.execute(
            "INSERT INTO tickets(id,stream,active,created,expires) VALUES(?,?,0,?,?)",
            (
                "PRIVATE_TOKEN",
                "PRIVATE_TARGET",
                now,
                now - 1 if condition == "expired" else now + 5,
            ),
        )
        if condition == "global":
            for i in range(2):
                db.execute(
                    "INSERT INTO tickets(id,stream,active,created,expires) "
                    "VALUES(?,?,1,?,?)",
                    (str(i), str(i), now, now + 60),
                )
        if condition == "stream":
            db.execute(
                "INSERT INTO tickets(id,stream,active,created,expires) "
                "VALUES('holder','PRIVATE_TARGET',1,?,?)",
                (now, now + 60),
            )
        if condition == "rate":
            db.execute("UPDATE policy SET next_grant=?", (now + 5,))
        if condition == "fairness":
            db.execute(
                "INSERT INTO tickets(id,stream,active,created,expires) "
                "VALUES('other','PRIVATE_OTHER',0,?,?)",
                (now, now + 5),
            )
    trace = AdmissionTrace()
    assert gate._may_acquire("PRIVATE_TOKEN") is possible
    assert gate._may_acquire("PRIVATE_TOKEN", diagnostics=trace) is possible
    assert trace.values["polls"] == 1
    if metric:
        assert trace.values[metric] == 1
    if condition == "expired":
        assert not trace.values.get("other_candidate_polls")
        assert not trace.values.get("other_candidate_with_capacity_polls")
    assert "PRIVATE" not in str(trace.values)


def test_timeout_and_success_keep_separate_queue_and_occupancy_metrics(tmp_path):
    settings = AdmissionSettings(str(tmp_path / "gate.sqlite"), 1, 1, 40, 0.08)
    gate = SharedAdmission(settings)
    holder_trace, waiting_trace = AdmissionTrace(), AdmissionTrace()
    with gate.slot(
        "holder",
        stop=Event(),
        source_deadline=time.monotonic() + 3,
        diagnostics=holder_trace,
    ):
        with pytest.raises(AdmissionError, match="admission_timeout"):
            with gate.slot(
                "waiter",
                stop=Event(),
                source_deadline=time.monotonic() + 3,
                diagnostics=waiting_trace,
            ):
                pytest.fail("no capacity was released")
    assert waiting_trace.values["global_full_polls"] > 0
    assert waiting_trace.values["acquire_wait_seconds"] >= 0.08
    assert waiting_trace.values.get("acquired", 0) == 0
    assert "occupied_before_release_seconds" not in waiting_trace.values
    assert holder_trace.values["acquired"] == 1
    assert holder_trace.values["occupied_before_release_seconds"] >= 0.08
    assert holder_trace.values["release_failed"] == 0
    assert gate.snapshot()["active"] == gate.snapshot()["waiting"] == 0


def test_database_failure_is_observable_without_bypassing_limit(tmp_path, monkeypatch):
    gate = SharedAdmission(
        AdmissionSettings(
            str(tmp_path / "gate.sqlite"),
            queue_timeout_seconds=0.04,
        )
    )

    @contextmanager
    def busy():
        time.sleep(0.005)
        raise sqlite3.OperationalError("database is locked")
        yield  # pragma: no cover

    monkeypatch.setattr(gate, "_transaction", busy)
    trace = AdmissionTrace()
    with pytest.raises(AdmissionError, match="admission_timeout"):
        with gate.slot(
            "x", stop=Event(), source_deadline=time.monotonic() + 3, diagnostics=trace
        ):
            pytest.fail("database error must not grant a slot")
    assert trace.values["sqlite_errors"] > 0
    assert (
        trace.values["transaction_seconds"]
        >= trace.values["transaction_begin_seconds"]
        > 0
    )
    assert trace.values.get("acquired", 0) == 0
