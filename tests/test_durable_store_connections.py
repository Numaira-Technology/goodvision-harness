"""Real SQLite transactions retain durability across reuse, failure and close."""

import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from goodvision_harness.face_observation_store import FaceObservationStore
from goodvision_harness.sam_tracking_spool import TrackingSpool
from goodvision_harness.sliding_window import FrameWindow, SampledFrame
from goodvision_harness.video_tracking import VideoFrame


@pytest.fixture(params=[TrackingSpool, FaceObservationStore])
def store(request, tmp_path):
    value = request.param(tmp_path / "durable.sqlite3", keep_open=True)
    try:
        yield value
    finally:
        value.close()


def test_reused_full_wal_transactions_are_visible_to_another_connection(store):
    connection = store._connection
    assert connection is not None
    with store._connect() as db:
        assert db.execute("PRAGMA synchronous").fetchone()[0] == 2
        assert db.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
        assert db.execute("PRAGMA wal_autocheckpoint").fetchone()[0] == 64
        db.execute("CREATE TABLE visibility (value INTEGER)")
    observer = sqlite3.connect(store.path)
    try:
        for value in range(3):
            with store._connect() as db:
                assert db is connection
                db.execute("INSERT INTO visibility VALUES (?)", (value,))
                # An independent reader cannot see a partially applied write.
                assert observer.execute(
                    "SELECT COUNT(*) FROM visibility"
                ).fetchone()[0] == value
            assert observer.execute(
                "SELECT COUNT(*) FROM visibility"
            ).fetchone()[0] == value + 1
        store.records()  # Its private Row conversion must not affect later reads.
        with store._connect() as db:
            assert db.execute("SELECT COUNT(*) FROM visibility").fetchone() == (3,)
    finally:
        observer.close()


def test_failed_transaction_rolls_back_before_reusing_the_connection(store):
    with store._connect() as db:
        db.execute("CREATE TABLE atomicity (value INTEGER UNIQUE)")
        db.execute("INSERT INTO atomicity VALUES (1)")
    with pytest.raises(sqlite3.IntegrityError), store._connect() as db:
        db.execute("INSERT INTO atomicity VALUES (2)")
        db.execute("INSERT INTO atomicity VALUES (1)")
    with store._connect() as db:
        assert db.execute("SELECT value FROM atomicity").fetchall() == [(1,)]
        db.execute("INSERT INTO atomicity VALUES (3)")
    observer = sqlite3.connect(store.path)
    try:
        assert observer.execute("SELECT value FROM atomicity").fetchall() == [
            (1,), (3,),
        ]
    finally:
        observer.close()


def test_parallel_transactions_serialize_and_preserve_every_operation(store):
    with store._connect() as db:
        db.execute("CREATE TABLE ordered (value INTEGER)")

    def append(_):
        with store._connect() as db:
            previous = db.execute(
                "SELECT COALESCE(MAX(value),0) FROM ordered"
            ).fetchone()[0]
            db.execute("INSERT INTO ordered VALUES (?)", (previous + 1,))

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(append, range(24)))
    with store._connect() as db:
        assert db.execute("SELECT value FROM ordered ORDER BY value").fetchall() == [
            (value,) for value in range(1, 25)
        ]


def test_close_waits_for_atomic_commit_and_allows_residual_writes(store):
    entered, release, closing, closed = (threading.Event() for _ in range(4))
    with store._connect() as db:
        db.execute("CREATE TABLE tail (value INTEGER)")

    def write():
        with store._connect() as db:
            db.execute("INSERT INTO tail VALUES (1)")
            entered.set()
            assert release.wait(3)

    def close():
        closing.set()
        store.close()
        closed.set()

    with ThreadPoolExecutor(max_workers=2) as pool:
        writing = pool.submit(write)
        try:
            assert entered.wait(2)
            shutdown = pool.submit(close)
            assert closing.wait(2)
            assert not closed.wait(0.03)
        finally:
            release.set()
        writing.result(2)
        shutdown.result(2)
    assert store._connection is None
    store.close()  # Repeated lifecycle cleanup is harmless.
    with store._connect() as db:
        assert db.execute("PRAGMA synchronous").fetchone()[0] == 2
        db.execute("INSERT INTO tail VALUES (2)")
    assert store._connection is None
    observer = sqlite3.connect(store.path)
    try:
        assert observer.execute("SELECT value FROM tail").fetchall() == [(1,), (2,)]
    finally:
        observer.close()


def test_face_admission_failure_does_not_commit_its_gallery(tmp_path):
    store = FaceObservationStore(tmp_path / "face.sqlite3", keep_open=True)
    frame = SampledFrame(0, 0, 0.0, b"source", 1, 1)
    window = FrameWindow(0, (frame,))
    try:
        store.admit(1, "run", window, {"version": "first"})
        with pytest.raises(sqlite3.IntegrityError):
            store.admit(1, "run", window, {"version": "must-rollback"})
        with store._connect() as db:
            assert db.execute("SELECT version FROM galleries").fetchall() == [
                ("first",),
            ]
        assert store.load(1)[0] == window
    finally:
        store.close()


def test_sam_batch_failure_rolls_back_source_assignments(tmp_path):
    store = TrackingSpool(tmp_path / "sam.sqlite3", keep_open=True)
    frame = VideoFrame(1, 0.1, b"source")
    try:
        store.begin(0, "run")
        store.admit(0, "run", frame, 1.0)
        with pytest.raises(sqlite3.ProgrammingError):
            store.batch("invalid", 0, 0, [1, object()])
        assert store.records() == []
        with store._connect() as db:
            assert db.execute("SELECT batch_id FROM sources").fetchall() == [(None,)]
        store.batch("valid", 0, 0, [1])
        assert store.sources_for_request("valid", "run") == (frame,)
    finally:
        store.close()


@pytest.mark.parametrize("kind", [TrackingSpool, FaceObservationStore])
def test_default_store_does_not_keep_a_connection_open(tmp_path, kind):
    store = kind(tmp_path / "fixture.sqlite3")
    assert store._connection is None
    assert store.records() == []
    assert store._connection is None
    store.close()
