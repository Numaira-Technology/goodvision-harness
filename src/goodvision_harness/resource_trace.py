"""Per-process resource sampling for the streaming harness.

Each harness process serves at most one stream (multi-stream = a fleet of
sibling processes, see fleet.py), so a per-process CPU/RSS sample is already a
per-stream sample. This module runs one daemon thread that periodically emits a
`resource_timing` record through the shared `log_timing` channel, tagged with the
fleet camera index, so `diag/parse_pipeline_log.py` can roll several runs
(1/2/4 streams) into a "where did the marginal cost of stream N+1 land" table.

It answers the CPU half of that question; bandwidth is already counted per stream
(video_tracking.transport_stats) and GPU is sampled out-of-band on the remote box
(diag/gpu_sampler.sh). Optional per-thread CPU attribution maps named worker
threads to pipeline stages (prepare/commit/display/capture), so host CPU can be
split across stage 3 (transport encode), stage 6 (scene commit) and stage 7
(display render).

psutil is an optional dependency (the `diag` extra). When it is not installed the
sampler is a no-op, so importing/starting it never breaks a normal run.
"""

from __future__ import annotations

import os
import threading
from collections.abc import Iterable

from .latency_trace import log_timing

try:  # psutil is optional (pip install -e ".[diag]")
    import psutil
except Exception:
    psutil = None  # type: ignore[assignment]


def _env_float(name: str, default: float) -> float:
    try:
        value = float(os.environ.get(name, "") or default)
    except (TypeError, ValueError):
        return default
    return value if value > 0 else default


def _env_int(name: str) -> int | None:
    raw = os.environ.get(name)
    if raw is None:
        return None
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


# Thread-name substring -> pipeline stage bucket. First match wins. The substrings
# match the names the threads are actually created with:
#   sam_tracking_pipeline.py -> sam-prepare / sam-ordered-commit / sam-ordered-request
#   video_connection.py      -> video-connection-<id> / video-display-encoder
#   scene_delivery.py        -> scene-display
# The event loop / main thread cover the receive+decode (stage 1) and the remote
# SAM HTTP round-trip (stage 4), which run on the asyncio loop.
_STAGE_BY_THREAD: tuple[tuple[str, str], ...] = (
    ("sam-prepare", "prepare"),           # stage 3: transport encode/downscale
    ("sam-ordered-commit", "commit"),     # stage 6: scene state commit
    ("sam-ordered-request", "dispatch"),  # stage 4: SAM request dispatch
    ("scene-display", "display"),         # stage 7: overlay render + encode
    ("video-display", "preview_encode"),  # preview/display encoder
    ("video-connection", "capture"),      # stage 1: receive/decode drain
    ("resource-sampler", "sampler"),      # this module's own thread
    ("mainthread", "event_loop"),         # stages 1 & 4 on the loop
    ("asyncio", "event_loop"),
)


def _stage_for(thread_name: str) -> str:
    lowered = thread_name.lower()
    for needle, bucket in _STAGE_BY_THREAD:
        if needle in lowered:
            return bucket
    return "other"


def _thread_cpu_by_stage(proc: psutil.Process) -> dict[str, float] | None:
    """Cumulative CPU seconds per stage bucket, keyed by OS thread id -> name.

    Returns None when per-thread timing is unavailable on this platform.

    Platform note: this joins psutil's per-thread ids to Python thread names via
    `Thread.native_id`. On Linux both are the kernel TID, so the join is exact and
    per-stage CPU is accurate — which is where the production/fleet harness runs.
    On macOS psutil reports Mach thread indices, not pthread TIDs, so the join
    fails and all CPU collapses into the "other" bucket; process-level cpu_percent
    stays correct there. Read per-stage CPU only from Linux runs.
    """
    try:
        per_thread = proc.threads()
    except Exception:
        return None
    names = {
        t.native_id: t.name
        for t in threading.enumerate()
        if getattr(t, "native_id", None) is not None
    }
    totals: dict[str, float] = {}
    for entry in per_thread:
        stage = _stage_for(names.get(entry.id, ""))
        totals[stage] = totals.get(stage, 0.0) + float(entry.user_time) + float(
            entry.system_time
        )
    return totals


class ResourceSampler:
    """Background daemon that emits `resource_timing` every N seconds."""

    def __init__(self, *, interval_seconds: float | None = None) -> None:
        self._interval = (
            interval_seconds
            if interval_seconds is not None
            else _env_float("RESOURCE_SAMPLE_SECONDS", 5.0)
        )
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._proc = psutil.Process() if psutil is not None else None
        self._prev_thread_cpu: dict[str, float] | None = None
        self._camera_index = _env_int("HARNESS_FLEET_CAMERA_INDEX")
        self._fleet_capacity = _env_int("HARNESS_FLEET_CAPACITY")

    @property
    def enabled(self) -> bool:
        return self._proc is not None

    def start(self) -> None:
        if self._proc is None or self._thread is not None:
            return
        # Prime the counters so the first real sample has a valid delta base.
        try:
            self._proc.cpu_percent(None)
            psutil.cpu_percent(None, percpu=True)  # type: ignore[union-attr]
        except Exception:
            pass
        self._prev_thread_cpu = _thread_cpu_by_stage(self._proc)
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, name="resource-sampler", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=2.0)
        self._thread = None

    def _run(self) -> None:
        # Wait the first interval before the first emit so cpu_percent spans a
        # real window rather than process startup.
        while not self._stop.wait(self._interval):
            self._emit_once()

    def _emit_once(self) -> None:
        proc = self._proc
        if proc is None:
            return
        fields: dict[str, object] = {
            "interval_seconds": round(self._interval, 3),
        }
        if self._camera_index is not None:
            fields["camera_index"] = self._camera_index
        if self._fleet_capacity is not None:
            fields["fleet_capacity"] = self._fleet_capacity

        try:
            fields["cpu_percent"] = round(proc.cpu_percent(None), 2)
        except Exception:
            pass
        try:
            fields["num_threads"] = proc.num_threads()
        except Exception:
            pass
        try:
            fields["rss_bytes"] = int(proc.memory_info().rss)
        except Exception:
            pass
        try:
            fields["open_fds"] = proc.num_fds()
        except Exception:
            pass

        self._sample_system(fields)
        self._sample_threads(proc, fields)

        log_timing("resource_timing", **fields)

    def _sample_system(self, fields: dict[str, object]) -> None:
        if psutil is None:
            return
        try:
            per_cpu = psutil.cpu_percent(None, percpu=True)
            if per_cpu:
                fields["system_cpu_percent_mean"] = round(
                    sum(per_cpu) / len(per_cpu), 2
                )
                fields["system_cpu_percent_max"] = round(max(per_cpu), 2)
                fields["cpu_count"] = len(per_cpu)
        except Exception:
            pass
        try:
            fields["system_available_bytes"] = int(
                psutil.virtual_memory().available
            )
        except Exception:
            pass
        try:
            load1, load5, _load15 = os.getloadavg()
            fields["load_avg_1m"] = round(load1, 2)
            fields["load_avg_5m"] = round(load5, 2)
        except (OSError, AttributeError):
            pass

    def _sample_threads(
        self, proc: psutil.Process, fields: dict[str, object]
    ) -> None:
        current = _thread_cpu_by_stage(proc)
        if current is None:
            return
        prev = self._prev_thread_cpu or {}
        delta: dict[str, float] = {}
        for stage, total in current.items():
            consumed = total - prev.get(stage, 0.0)
            if consumed > 0.0005:
                delta[stage] = round(consumed, 4)
        self._prev_thread_cpu = current
        if delta:
            # CPU seconds each stage burned during this interval; divide by
            # interval for a utilization fraction downstream.
            fields["thread_cpu_seconds"] = delta


_SAMPLER: ResourceSampler | None = None


def start_resource_sampling(*, interval_seconds: float | None = None) -> None:
    """Start the global sampler (idempotent; no-op without psutil)."""
    global _SAMPLER
    if _SAMPLER is not None:
        return
    sampler = ResourceSampler(interval_seconds=interval_seconds)
    sampler.start()
    _SAMPLER = sampler


def stop_resource_sampling() -> None:
    global _SAMPLER
    sampler = _SAMPLER
    _SAMPLER = None
    if sampler is not None:
        sampler.stop()


__all__: Iterable[str] = (
    "ResourceSampler",
    "start_resource_sampling",
    "stop_resource_sampling",
)
