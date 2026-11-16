"""Synthetic source windows, native payloads and sparse-source recovery."""

import json
import time
from dataclasses import replace
from queue import Queue
from threading import Event, Thread

import pytest
from test_vlm_independent import config, jpeg, verdict
from test_vlm_parallel import OutputQueue

from goodvision_harness.model_api import ModelAPISettings
from goodvision_harness.neural_model import NeuralInspector
from goodvision_harness.runtime.scheduling import EvidenceFrame
from goodvision_harness.vlm_independent import (
    SourceSample,
    _observation_window,
    run_vlm_process,
    utc_now,
)


@pytest.mark.parametrize(
    "times,expected",
    [
        ([0, 0.25], []),  # Startup must wait for a complete span.
        ([0, 0.25, 0.5], [0, 0.5]),
        ([0, 0.25, 0.5, 0.75, 1], [0.5, 1]),
        ([0, 0.3, 0.6, 0.9], [0.3, 0.9]),
        ([0, 0.5, 1], [0.5, 1]),
        ([0, 0.8, 1.6], []),
        ([0, 1, 2, 3], []),
    ],
)
def test_native_windows_retain_distinct_source_evidence(times, expected):
    history = [
        SourceSample(EvidenceFrame("run", i, at, bytes([i])), "", 0)
        for i, at in enumerate(times)
    ]
    frames = _observation_window(history, 2, 0.5, 4.0)
    assert [frame.timestamp_seconds for frame in frames] == expected
    assert all(frame is history[frame.frame_sequence].frame for frame in frames)
    assert len({frame.frame_sequence for frame in frames}) == len(frames)


@pytest.mark.parametrize("source_interval", [0.8, 1.0])
def test_sparse_source_reports_unavailable_then_recovers_with_two_image_payload(
    source_interval,
):
    cfg = replace(
        config(), model=NeuralInspector.model_name, output_mode="events",
        model_settings=ModelAPISettings(provider="local", endpoint="http://local.test"),
        candidate_fps=4.0, observation_window_frames=2,
        observation_frame_delta_seconds=0.5,
    )
    incoming, outgoing, payloads = Queue(), OutputQueue(), Queue()
    stop = Event()

    class Inspector(NeuralInspector):
        def warmup(self):
            return True

        def inspect(self, job):
            payloads.put(json.loads(self._payload(job)[0]))
            return (verdict(),)

    worker = Thread(
        target=run_vlm_process,
        args=(cfg, incoming, outgoing, stop, Inspector),
    )
    worker.start()

    def offer(seq, at):
        incoming.put(SourceSample(
            EvidenceFrame(cfg.stream_session_id, seq, at, jpeg()),
            utc_now(), time.monotonic(),
        ))

    def next_message(predicate):
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            message = outgoing.get(timeout=5)
            if predicate(message):
                return message
        pytest.fail("worker did not report expected source state")

    try:
        offer(0, 0)
        offer(1, source_interval)
        unavailable = next_message(lambda m: m.get("stats", {}).get(
            "sampling_window_state"
        ) == "unavailable")
        assert unavailable["state"] == "running"
        assert unavailable["stats"].get("claimed_requests", 0) == 0
        assert payloads.empty()
        offer(2, source_interval + 0.5)
        report = next_message(lambda m: m["kind"] == "report")["report"]
        assert report["process_metrics"]["sampling_window_state"] == "ready"
        payload = payloads.get_nowait()
        assert len(payload["images"]) == 2
        assert payload["frame_times"] == [source_interval, source_interval + 0.5]
        assert payload["sampling"] == {
            "policy": "two-frame-target-spacing-v1", "candidate_fps": 4.0,
            "observation_window_frames": 2,
            "observation_frame_delta_seconds": 0.5, "observation_fps": 1.25,
        }
    finally:
        stop.set()
        worker.join(5)
        assert not worker.is_alive()
