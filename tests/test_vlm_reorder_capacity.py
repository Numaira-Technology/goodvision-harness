"""Regression coverage for ordered VLM publication without inference-slot blocking."""

from __future__ import annotations

import io
import time
from pathlib import Path
from threading import Lock

from PIL import Image

from goodvision_harness.config import load_config
from goodvision_harness.contracts import NormalizedBBox
from goodvision_harness.vlm_independent import (
    REORDER_CAPACITY,
    IndependentVLMBranch,
    VLMProcessConfig,
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
    """Keep the first request slow while every later request completes quickly."""

    _lock = Lock()
    _calls = 0

    def inspect(self, _inspection):
        with type(self)._lock:
            type(self)._calls += 1
            call = type(self)._calls
        time.sleep(2.5 if call == 1 else 0.02)
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


def test_completed_results_release_capacity_but_publication_stays_ordered():
    cfg = config()
    branch = IndependentVLMBranch(cfg, inspector_factory=inspector_factory)
    branch.start()
    try:
        await_condition(lambda: branch.stats()["state"] == "running")
        for sequence in range(0, 97, 6):
            branch.offer(jpeg(), sequence / 15.0, sequence)
            time.sleep(0.04)

        # The slow first request still prevents public delivery, but completed
        # later requests must free model-call capacity for additional work.
        await_condition(
            lambda: branch.stats().get("claimed_requests", 0) > cfg.max_in_flight,
            timeout=1.4,
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
