"""Regression coverage for ordered VLM publication without inference-slot blocking."""

from __future__ import annotations

import io
import multiprocessing as mp
import time
from contextlib import contextmanager
from dataclasses import replace
from functools import partial
from pathlib import Path
from queue import Full

import pytest
from PIL import Image

from goodvision_harness.config import load_config
from goodvision_harness.contracts import NormalizedBBox
from goodvision_harness.scene_evidence import EvidenceFrame
from goodvision_harness.vlm_independent import (
    INPUT_CAPACITY,
    OUTPUT_CAPACITY,
    REORDER_CAPACITY,
    IndependentVLMBranch,
    SourceSample,
    VLMProcessConfig,
    run_vlm_process,
    utc_now,
)
from goodvision_harness.vlm_independent_model import IndependentVerdict

ROOT = Path(__file__).resolve().parents[1]
BOX = NormalizedBBox(0.1, 0.1, 0.5, 0.9)


def jpeg():
    with Image.new("RGB", (64, 48), "gray") as picture, io.BytesIO() as data:
        picture.save(data, "JPEG")
        return data.getvalue()


def config():
    base = load_config(ROOT / "config.yaml")
    return VLMProcessConfig(
        "vlm-reorder-capacity",
        "camera01",
        1,
        ("smoking", "fall"),
        base.model_api,
        "test",
        observation_fps=1.25,
        max_in_flight=2,
    )


def verdicts():
    return (
        IndependentVerdict("smoking", ("p1",), True, 0.9, (("p1", BOX),)),
        IndependentVerdict("fall", ("p1",), False, 0.9, (("p1", BOX),)),
    )


class HeadOfLineInspector:
    """Keep the earliest source request slow and later requests fast."""

    def inspect(self, inspection):
        endpoint = inspection.frames[-1].timestamp_seconds
        time.sleep(2.5 if endpoint < 1.5 else 0.02)
        return verdicts()

    def close(self):
        pass


def inspector_factory(_config):
    return HeadOfLineInspector()


def await_condition(predicate, timeout=10.0):
    deadline = time.monotonic() + timeout
    while not predicate():
        assert time.monotonic() < deadline, "independent VLM branch did not progress"
        time.sleep(0.02)


@pytest.mark.parametrize("output_mode", ["full", "compact"])
def test_completed_results_release_capacity_but_publication_stays_ordered(output_mode):
    cfg = replace(config(), output_mode=output_mode)
    branch = IndependentVLMBranch(cfg, inspector_factory=inspector_factory)
    branch.start()
    try:
        await_condition(lambda: branch.stats()["state"] == "running")
        for sequence in range(0, 97, 6):
            branch.offer(jpeg(), sequence / 15.0, sequence)
            time.sleep(0.08)

        # The slow first request still prevents public delivery, but completed
        # later requests must free model-call capacity for additional work. The
        # assertion below distinguishes this from the old implementation: there
        # the third claim only happened after the first report was published.
        await_condition(
            lambda: branch.stats().get("claimed_requests", 0) > cfg.max_in_flight,
            timeout=6.0,
        )
        blocked = branch.stats()
        assert branch.results.latest() is None
        assert blocked["pending_requests"] >= 1
        assert blocked["active_requests"] > cfg.max_in_flight
        assert blocked["reorder_capacity"] == REORDER_CAPACITY
        assert blocked["active_requests"] <= REORDER_CAPACITY
        assert blocked["inflight_requests"] <= cfg.max_in_flight

        await_condition(
            lambda: branch.stats().get("published_reports", 0) >= 4,
            timeout=8.0,
        )
        records = branch.results.records_after(0)
        sequences = [item.payload["frame"]["source_frame_index"] for item in records]
        assert sequences == sorted(set(sequences))
        assert any(
            item.payload["timing"]["reorder_wait_seconds"] > 1.0
            for item in records[1:]
        )
        assert all(
            item.payload["process_metrics"]["inflight_requests"]
            <= cfg.max_in_flight
            for item in records
        )
        assert all(
            item.payload["process_metrics"]["active_requests"] <= REORDER_CAPACITY
            for item in records
        )
    finally:
        branch.close(timeout_seconds=6.0)


class HeadTimeout(RuntimeError):
    code = "timeout"


class GatedInspector:
    def __init__(self, started, release, fail_head):
        self.started, self.release, self.fail_head = started, release, fail_head

    def inspect(self, inspection):
        endpoint = inspection.frames[-1].timestamp_seconds
        self.started.put(endpoint)
        if endpoint < 1.5:
            assert self.release.wait(20.0), "head request was never released"
            if self.fail_head:
                raise HeadTimeout()
        return verdicts()

    def close(self):
        pass


def gated_factory(_config, *, started, release, fail_head):
    return GatedInspector(started, release, fail_head)


class ObservedOutput:
    """Use the real bounded IPC queue and signal a rejected report admission."""

    def __init__(self, queue, full):
        self.queue, self.full = queue, full

    def put_nowait(self, message):
        try:
            self.queue.put_nowait(message)
        except Full:
            if message["kind"] == "report":
                self.full.set()
            raise

    def close(self):
        self.queue.close()

    def cancel_join_thread(self):
        self.queue.cancel_join_thread()


@contextmanager
def process_harness(*, fail_head=False, output_mode="full"):
    cfg = replace(config(), output_mode=output_mode)
    context = mp.get_context("spawn")
    incoming, outgoing = context.Queue(INPUT_CAPACITY), context.Queue(OUTPUT_CAPACITY)
    started = context.Queue()
    release, full, stop = context.Event(), context.Event(), context.Event()
    process = context.Process(
        target=run_vlm_process,
        args=(cfg, incoming, ObservedOutput(outgoing, full), stop),
        kwargs={"inspector_factory": partial(
            gated_factory, started=started, release=release, fail_head=fail_head,
        )},
    )
    process.start()
    try:
        assert outgoing.get(timeout=10)["state"] == "warming"
        assert outgoing.get(timeout=10)["state"] == "running"
        yield cfg, process, incoming, outgoing, started, release, full, stop
    finally:
        release.set()
        stop.set()
        process.join(5)
        if process.is_alive():
            process.terminate()
            process.join(5)
        for queue in (incoming, outgoing, started):
            queue.close()
            queue.cancel_join_thread()


def submit_request(cfg, incoming, started, number):
    # Advance source time without wall-clock sleeps. Acknowledged model starts
    # prevent input batching from skipping any of the expected request windows.
    endpoint = 2 * (number + 1)
    for sequence in range(0 if number == 0 else endpoint - 1, endpoint + 1):
        incoming.put(SourceSample(
            EvidenceFrame(cfg.stream_session_id, sequence * 6, sequence * 0.4, jpeg()),
            utc_now(), time.monotonic(),
        ), timeout=5)
    for _ in range(2 if cfg.output_mode == "parallel" else 1):
        assert started.get(timeout=5) == pytest.approx(endpoint * 0.4)


@pytest.mark.parametrize("fail_head", [False, True])
@pytest.mark.parametrize("output_mode", ["full", "compact"])
def test_full_reorder_burst_delivers_every_report_once_in_source_order(
    fail_head, output_mode,
):
    with process_harness(fail_head=fail_head, output_mode=output_mode) as harness:
        cfg, _process, incoming, outgoing, started, release, full, _stop = harness
        for number in range(REORDER_CAPACITY):
            submit_request(cfg, incoming, started, number)
        incoming.put(None, timeout=5)

        while True:
            message = outgoing.get(timeout=5)
            assert message["kind"] == "status"
            assert message["state"] != "drained"
            if message["stats"].get("pending_requests") == REORDER_CAPACITY - 1:
                assert message["stats"]["active_requests"] == REORDER_CAPACITY
                assert message["stats"]["inflight_requests"] == 1
                break

        # Pause consumption until a real report admission hits the full queue.
        release.set()
        assert full.wait(5), "test did not exercise output backpressure"
        reports = []
        while True:
            message = outgoing.get(timeout=5)
            if message["kind"] == "report":
                reports.append(message["report"])
            elif message["state"] == "drained":
                final = message["stats"]
                break

        assert len(reports) == REORDER_CAPACITY
        assert [r["sequence"] for r in reports] == list(range(1, REORDER_CAPACITY + 1))
        assert [r["frame"]["source_frame_index"] for r in reports] == [
            12 * n for n in range(1, REORDER_CAPACITY + 1)
        ]
        assert len({r["message_id"] for r in reports}) == REORDER_CAPACITY
        assert reports[0]["state"] == ("failed" if fail_head else "completed")
        assert reports[0]["error_code"] == ("timeout" if fail_head else None)
        assert all(r["state"] == "completed" for r in reports[1:])
        assert final["claimed_requests"] == REORDER_CAPACITY
        assert final.get("failed_requests", 0) == int(fail_head)
        assert final["succeeded_requests"] == REORDER_CAPACITY - int(fail_head)
        assert final["active_requests"] == final["pending_requests"] == 0
        assert all(
            r["process_metrics"]["active_requests"] <= REORDER_CAPACITY
            and r["process_metrics"]["inflight_requests"] <= cfg.max_in_flight
            for r in reports
        )


@pytest.mark.parametrize("output_mode", ["full", "compact", "events", "parallel"])
def test_full_output_keeps_inference_progressing_and_process_stoppable(output_mode):
    with process_harness(output_mode=output_mode) as harness:
        cfg, process, incoming, _outgoing, started, release, full, stop = harness
        release.set()
        for number in range(OUTPUT_CAPACITY + 1):
            submit_request(cfg, incoming, started, number)
        assert full.wait(5)
        # A blocked report must not block admission or completion of later work.
        for number in range(OUTPUT_CAPACITY + 1, OUTPUT_CAPACITY + 5):
            submit_request(cfg, incoming, started, number)
        stop.set()
        process.join(3)
        assert not process.is_alive(), "shutdown waited on a full output queue"
        assert process.exitcode == 0
