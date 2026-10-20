"""Ready VLM replies bypass slow calls while history and live state stay separate."""

from __future__ import annotations

import io
import multiprocessing as mp
import time
from contextlib import contextmanager
from dataclasses import replace
from functools import partial
from pathlib import Path
from queue import Full
from types import SimpleNamespace

import pytest
from PIL import Image

from goodvision_harness.config import load_config
from goodvision_harness.contracts import NormalizedBBox
from goodvision_harness.scene_evidence import EvidenceFrame
from tools.benchmark_runtime.result_bus import ResultBus
from tools.benchmark_runtime.vlm_independent import (
    INPUT_CAPACITY,
    OUTPUT_CAPACITY,
    REORDER_CAPACITY,
    IndependentVLMBranch,
    SourceSample,
    VLMProcessConfig,
    run_vlm_process,
    utc_now,
)
from tools.benchmark_runtime.vlm_independent_model import IndependentVerdict

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


def await_condition(predicate, timeout=10.0):
    deadline = time.monotonic() + timeout
    while not predicate():
        assert time.monotonic() < deadline, "independent VLM branch did not progress"
        time.sleep(0.02)


class HeadTimeout(RuntimeError):
    code = "timeout"


class GatedInspector:
    def __init__(self, started, release, fail_head, fail_newer=False):
        self.started, self.release, self.fail_head = started, release, fail_head
        self.fail_newer = fail_newer

    def inspect(self, inspection):
        endpoint = inspection.frames[-1].timestamp_seconds
        self.started.put(endpoint)
        if endpoint < 1.5:
            assert self.release.wait(20.0), "head request was never released"
            if self.fail_head:
                raise HeadTimeout()
        elif self.fail_newer:
            raise HeadTimeout()
        return tuple(replace(v, detected=endpoint < 1.5) for v in verdicts())

    def close(self):
        pass


def gated_factory(_config, *, started, release, fail_head, fail_newer=False):
    return GatedInspector(started, release, fail_head, fail_newer)


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
        kwargs={
            "inspector_factory": partial(
                gated_factory,
                started=started,
                release=release,
                fail_head=fail_head,
            )
        },
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
        incoming.put(
            SourceSample(
                EvidenceFrame(
                    cfg.stream_session_id, sequence * 6, sequence * 0.4, jpeg()
                ),
                utc_now(),
                time.monotonic(),
            ),
            timeout=5,
        )
    for _ in range(2 if cfg.output_mode == "parallel" else 1):
        assert started.get(timeout=5) == pytest.approx(endpoint * 0.4)


@pytest.mark.parametrize("fail_head", [False, True])
@pytest.mark.parametrize("output_mode", ["full", "compact"])
def test_output_backpressure_preserves_every_reply_and_late_head(
    fail_head,
    output_mode,
):
    with process_harness(fail_head=fail_head, output_mode=output_mode) as harness:
        cfg, _process, incoming, outgoing, started, release, full, _stop = harness
        for number in range(REORDER_CAPACITY):
            submit_request(cfg, incoming, started, number)
        incoming.put(None, timeout=5)

        # Later replies fill the real IPC queue before the slow head finishes.
        # The remaining completed work must stay retained under backpressure.
        assert full.wait(5), "test did not exercise output backpressure"
        release.set()
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
        assert sorted(r["frame"]["source_frame_index"] for r in reports) == [
            12 * n for n in range(1, REORDER_CAPACITY + 1)
        ]
        assert len({r["message_id"] for r in reports}) == REORDER_CAPACITY
        head = next(r for r in reports if r["frame"]["source_frame_index"] == 12)
        assert reports[0] is not head
        assert head["historical_only"] is True
        assert head["state"] == ("failed" if fail_head else "completed")
        assert head["error_code"] == ("timeout" if fail_head else None)
        assert all(r["state"] == "completed" for r in reports if r is not head)
        assert final["late_grounding_reports_preserved"] >= 1
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


@pytest.mark.parametrize("output_mode", ["full", "compact", "parallel"])
@pytest.mark.parametrize(
    "fail_head,fail_newer", [(False, False), (True, False), (False, True)]
)
def test_grounding_ready_reply_bypasses_head_without_live_state_rollback(
    output_mode,
    fail_head,
    fail_newer,
):
    cfg = replace(config(), output_mode=output_mode)
    context = mp.get_context("spawn")
    started, release = context.Queue(), context.Event()
    fused, displayed, delivered = [], [], []
    fusion = SimpleNamespace(
        start=lambda: None,
        close=lambda: None,
        finish=lambda: None,
        stats=lambda: {},
        offer=lambda r, frames: fused.append(r),
    )
    display = SimpleNamespace(grounding_required=False, offer_vlm=displayed.append)
    shared = ResultBus()
    branch = IndependentVLMBranch(
        cfg,
        results=shared,
        fusion=fusion,
        display_fusion=display,
        delivery=SimpleNamespace(submit=delivered.append),
        inspector_factory=partial(
            gated_factory,
            started=started,
            release=release,
            fail_head=fail_head,
            fail_newer=fail_newer,
        ),
    )
    bus = branch.grounding_results if output_mode == "parallel" else branch.results
    lane = "grounding" if output_mode == "parallel" else "combined"
    branch.start()
    try:
        for number in range(3):
            endpoint = 2 * (number + 1)
            for seq in range(0 if number == 0 else endpoint - 1, endpoint + 1):
                assert branch.offer(jpeg(), seq * 0.4, seq * 6)
            for _ in range(2 if output_mode == "parallel" else 1):
                assert started.get(timeout=10) == pytest.approx(endpoint * 0.4)
            if number:
                # The head cannot finish until the parent releases it. This
                # checks causality rather than relying on a short sleep/P95.
                await_condition(lambda n=number: len(bus.records_after(0)) == n)
        assert not release.is_set()
        latest = bus.latest()
        assert latest.payload["frame"]["source_frame_index"] == 36
        await_condition(lambda: len(displayed) == 2)
        await_condition(lambda: len(fused) == (0 if fail_newer else 2))
        await_condition(
            lambda: branch.health()[lane + "_last_result"]["frame_sequence"] == 36
        )
        health = branch.health()
        assert health[lane + "_last_result"]["frame_sequence"] == 36
        release.set()
        assert branch.wait_until_idle(10)
        rows = [r.payload for r in bus.records_after(0)]
        assert [r["frame"]["source_frame_index"] for r in rows] == [24, 36, 12]
        assert rows[-1]["historical_only"] is True
        assert rows[-1]["state"] == ("failed" if fail_head else "completed")
        assert bus.latest() == latest
        if fail_newer:
            assert bus.latest(feature="smoking") is None
        else:
            assert bus.latest(feature="smoking") == latest
        assert len(displayed) == 2  # Old boxes never reach the live display.
        assert len(fused) == (0 if fail_newer else 2)  # No old alarm votes.
        assert branch.health()[lane + "_last_result"] == health[lane + "_last_result"]
        assert branch.stats()["late_grounding_reports_preserved"] == 1
        count = branch.stats()["request_accounting"]["grounding"]
        assert count["submitted"] == count["finished"] == count["published"] == 3
        if output_mode != "parallel":
            assert shared.latest().payload["message_id"] == latest.payload["message_id"]
            assert len(shared.records_after(0)) == 3
            assert len(delivered) == 3 - int(fail_head) - 2 * int(fail_newer)
        assert all(r["timing"]["completion_to_publish_seconds"] >= 0 for r in rows)
    finally:
        release.set()
        branch.close(timeout_seconds=6)
        started.close()
        started.cancel_join_thread()
