import sqlite3
import time
from dataclasses import replace
from queue import Queue
from threading import Event, Thread
from types import SimpleNamespace

import pytest
from test_vlm_admission import settings, wait_until
from test_vlm_independent import config, jpeg, utc_now, verdict
from test_vlm_parallel import OutputQueue

from goodvision_harness import vlm_admission
from goodvision_harness.scene_evidence import EvidenceFrame
from goodvision_harness.vlm_admission import AdmissionError, SharedAdmission
from goodvision_harness.vlm_independent import SourceSample, run_vlm_process


def test_waiting_mate_survives_new_windows_and_goes_first_after_fast_peer_finishes(
    tmp_path,
):
    gate = SharedAdmission(
        settings(
            tmp_path,
            global_limit=1,
            per_stream_limit=1,
            requests_per_second=100,
            queue_timeout_seconds=3,
        )
    )
    order, errors = [], []
    # The marker must survive release of the first lane, not just active leases.
    with gate.slot(
        "camera",
        lane="events",
        pair_id="old",
        stop=Event(),
        source_deadline=time.monotonic() + 3,
    ):
        pass

    def wait(lane, pair):
        try:
            with gate.slot(
                "camera",
                lane=lane,
                pair_id=pair,
                stop=Event(),
                source_deadline=time.monotonic() + 3,
            ):
                order.append((lane, pair))
        except Exception as exc:
            errors.append(exc)

    threads = []
    with gate.slot("blocker", stop=Event(), source_deadline=time.monotonic() + 3):
        # Newer events registers before the older window's grounding mate.
        for index, (lane, pair) in enumerate(
            [("events", "new"), ("compact", "old"), ("compact", "new")], 1
        ):
            t = Thread(target=wait, args=(lane, pair))
            threads.append(t)
            t.start()
            wait_until(lambda index=index: gate.snapshot()["waiting"] == index)
    for t in threads:
        t.join(4)
        assert not t.is_alive()
    assert not errors
    assert order == [("compact", "old"), ("events", "new"), ("compact", "new")]
    assert gate.snapshot()["peak_active"] == 1
    assert gate.snapshot()["active"] == gate.snapshot()["waiting"] == 0


def test_pair_priority_is_scoped_and_keeps_cross_camera_fairness(tmp_path):
    gate = SharedAdmission(settings(tmp_path))
    now = time.monotonic()
    with gate._transaction() as db:
        db.executemany(
            "INSERT INTO streams(id,last_grant,touched) VALUES(?,?,?)",
            [("a", 2, now), ("b", 1, now)],
        )
        db.executemany(
            "INSERT INTO tickets(id,stream,active,created,expires,lane,pair_id) "
            "VALUES(?,?,0,?,?,?,?)",
            [
                ("a-mate", "a", now + 1, now + 3, "compact", "shared-id"),
                ("b-old", "b", now, now + 3, "compact", "other"),
                ("b-new", "b", now + 1, now + 3, "compact", "shared-id"),
            ],
        )
        db.execute(
            "INSERT INTO pair_grants VALUES(?,?,?,?)",
            ("a", "shared-id", "events", now + 3),
        )
        assert gate._candidate(db, now)[0] == "b-old"
        db.execute("UPDATE streams SET last_grant=3 WHERE id='b'")
        assert gate._candidate(db, now)[0] == "a-mate"
        gate._reap(db, now + 4)
        assert db.execute("SELECT COUNT(*) FROM pair_grants").fetchone()[0] == 0


@pytest.mark.parametrize("cancelled", [False, True])
def test_paired_waiter_expires_or_cancels_without_leaking_capacity(
    tmp_path, cancelled, monkeypatch
):
    # Observe the registered waiter before expiring it. Real SQLite I/O on a
    # busy host can otherwise consume its entire 100 ms lifetime before polling.
    now = [1000.0]
    monkeypatch.setattr(
        vlm_admission,
        "time",
        SimpleNamespace(monotonic=lambda: now[0], sleep=time.sleep),
    )
    gate = SharedAdmission(
        settings(
            tmp_path, global_limit=1, per_stream_limit=1, queue_timeout_seconds=0.1
        )
    )
    with gate.slot("holder", stop=Event(), source_deadline=now[0] + 3):
        stop, errors = Event(), []

        def wait():
            try:
                with gate.slot(
                    "camera",
                    lane="compact",
                    pair_id="w",
                    stop=stop,
                    source_deadline=now[0] + 3,
                ):
                    pytest.fail("capacity was bypassed")
            except AdmissionError as exc:
                errors.append(exc.code)

        t = Thread(target=wait)
        t.start()
        try:
            wait_until(lambda: gate.snapshot()["waiting"] == 1)
            if cancelled:
                stop.set()
            else:
                now[0] += 0.2
            t.join(2)
            assert not t.is_alive()
            assert errors == [
                "admission_cancelled" if cancelled else "admission_timeout"
            ]
            assert gate.snapshot()["waiting"] == 0
            assert gate.snapshot()["active"] == 1
        finally:
            stop.set()
            t.join(3)
    assert gate.snapshot()["active"] == 0


def test_paired_worker_skips_both_when_peer_busy_then_resumes_same_evidence(tmp_path):
    cfg = replace(
        config(),
        output_mode="parallel",
        parallel_scheduling="paired",
        max_in_flight=1,
        admission=settings(tmp_path, global_limit=2, per_stream_limit=2),
    )
    incoming, outgoing = Queue(), OutputQueue()
    stop, release = Event(), Event()
    calls, reports, snapshots = [], [], []

    class Inspector:
        def __init__(self, mode):
            self.mode = mode
            self.last_http_timing = {"http_request_attempted": False}

        def inspect(self, j):
            calls.append(
                (
                    self.mode,
                    j.request.request_id,
                    tuple(f.frame_sequence for f in j.frames),
                )
            )
            if self.mode == "compact":
                assert release.wait(6)
            return (verdict(),)

        def close(self):
            pass

    def observe():
        while not outgoing.empty():
            msg = outgoing.get_nowait()
            if msg["kind"] == "report":
                reports.append(msg["report"])
            else:
                snapshots.append(msg)
        return len(reports)

    def feed(seq, at):
        incoming.put(
            SourceSample(
                EvidenceFrame(cfg.stream_session_id, seq, at, jpeg()),
                utc_now(),
                time.monotonic(),
            )
        )

    t = Thread(
        target=run_vlm_process,
        args=(cfg, incoming, outgoing, stop, lambda c: Inspector(c.output_mode)),
    )
    for seq, at in enumerate([0, 0.4, 0.8, 1.2]):
        feed(seq, at)
    t.start()
    try:
        wait_until(lambda: observe() >= 1)
        feed(4, 2.0)

        def skipped():
            observe()
            return any(
                s.get("stats", {}).get("paired_busy_windows_skipped", 0)
                for s in snapshots
            )

        wait_until(skipped, timeout=3)
        assert len(calls) == 2  # No orphaned events job for frame 4.
        release.set()
        wait_until(lambda: observe() >= 2)
        feed(5, 2.8)
        wait_until(lambda: observe() >= 4)
        assert len(calls) == 4
        pairs = {}
        for lane, window, frames in calls:
            pairs.setdefault(window, {})[lane] = frames
        assert len(pairs) == 2
        assert all(p["events"] == p["compact"] for p in pairs.values())
        assert {p["events"][-1] for p in pairs.values()} == {3, 5}
    finally:
        release.set()
        stop.set()
        t.join(5)
    assert not t.is_alive()
    with sqlite3.connect(cfg.admission.database) as db:
        assert db.execute("SELECT COUNT(*) FROM tickets").fetchone()[0] == 0
