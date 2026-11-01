"""Shared fixtures for the preserved experimental runtime, separate from main."""

import sqlite3
import time
from dataclasses import fields
from pathlib import Path

from test_vlm_independent import BOX, ROOT, await_condition, job, jpeg
from test_vlm_independent import config as main_config

from tools.benchmark_runtime.vlm_independent import (
    SourceSample,
    VLMProcessConfig,
    utc_now,
)
from tools.benchmark_runtime.vlm_independent_model import IndependentVerdict
from tools.vlm_benchmark.source_feeder import TimedSourceFeeder

__all__ = [
    "BOX",
    "ROOT",
    "assert_admission_drained",
    "await_condition",
    "config",
    "configure_ephemeral_admission",
    "ContractSourceFeeder",
    "job",
    "jpeg",
    "report",
    "utc_now",
    "verdict",
]


class ContractSourceFeeder(TimedSourceFeeder):
    """Replay every planned frame for adapter tests, allowing host scheduling gaps.

    Source timestamps, frame identity and real worker/admission paths are kept.
    Wall-clock stalls delay later releases instead of dropping test inputs.
    This fixture provides no throughput/freshness evidence; those behaviors use
    the production feeder in test_vlm_source_feeder and test_vlm_sender_lag.
    """

    def _run(self):
        previous_release = 0.0
        try:
            for release_at, actions in self.releases:
                delay = max(0.0, release_at - previous_release)
                before = time.monotonic()
                if self.stop.wait(delay):
                    return
                self.release_lags.append(max(0.0, time.monotonic() - before - delay))
                previous_release = release_at
                for camera, tick in actions:
                    if self.stop.is_set():
                        return
                    frames = self.framesets[camera % len(self.framesets)]
                    frame = frames[tick % len(frames)]
                    accepted = self.branches[camera].offer(frame, tick / self.sample_fps, tick)
                    if self.source_hook is not None:
                        self.source_hook(camera, self.branches[camera].config.stream_session_id,
                                         tick, tick / self.sample_fps, frame)
                    self.offer_rejected[camera] += int(not accepted)
                    self.samples.put_nowait((camera, tick, accepted))
        except BaseException as error:
            self.error = error


class _EphemeralAdmissionSQLite:
    """Keep real SQLite isolation without disk-flush timing in offline flows."""

    def __init__(self, database):
        self.database = Path(database).resolve()

    def __getattr__(self, name):
        return getattr(sqlite3, name)

    def connect(self, database, *args, **kwargs):
        connection = sqlite3.connect(database, *args, **kwargs)
        if Path(database).resolve() == self.database:
            try:
                connection.execute("PRAGMA synchronous=OFF")
            except BaseException:
                connection.close()
                raise
        return connection


def configure_ephemeral_admission(config):
    """Scope disposable coordination storage to this warmed worker's database.

    Formal runs use Linux tmpfs. These portable mock flows use a temporary disk
    file, whose durable flush latency must not decide adapter correctness.
    Keep shared transactions, source expiry, scheduling and counters unchanged;
    the global sqlite3 module and other databases retain their normal settings.
    """
    from tools.benchmark_runtime import vlm_admission

    database = Path(config.admission.database).resolve()
    current = vlm_admission.sqlite3
    if isinstance(current, _EphemeralAdmissionSQLite):
        if current.database != database:
            raise ValueError("offline worker admission database changed")
        return
    vlm_admission.sqlite3 = _EphemeralAdmissionSQLite(database)


def assert_admission_drained(result, args, expected_grants):
    gate = result["shared_admission"]
    assert gate["grants"] == expected_grants, gate
    assert gate["active"] == gate["waiting"] == 0, gate
    assert gate["global_limit"] == args.global_limit, gate
    assert gate["per_stream_limit"] == args.per_stream_limit, gate
    assert gate["requests_per_second"] == args.request_rate, gate
    assert 1 <= gate["peak_active"] <= args.global_limit, gate


def config(session="vlm-run", generation=1):
    base = main_config(session, generation)
    # Product-only sampling controls evolve separately from this preserved
    # runtime. Copy shared defaults without changing benchmark scheduling.
    supported = {field.name for field in fields(VLMProcessConfig)}
    return VLMProcessConfig(
        **{
            field.name: getattr(base, field.name)
            for field in fields(base) if field.name in supported
        }
    )


def verdict(detected=True, sid="p1", box=BOX):
    return IndependentVerdict("smoking", (sid,), detected, 0.9, ((sid, box),))


def report(state, at, sequence, verdicts):
    inspection = job(state.config, at, sequence)
    sample = SourceSample(inspection.frames[-1], utc_now(), time.monotonic())
    return state.report(inspection, sample, verdicts, {"model_seconds": 0.1})
