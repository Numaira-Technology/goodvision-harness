"""Default prompts and publication behavior stay aligned with current main."""

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from queue import Queue
from threading import Event, Thread

import pytest
from test_vlm_independent import config, job, verdict

from goodvision_harness.event_sensitivity import EVENT_SENSITIVITY_DEFAULTS
from goodvision_harness.vlm_compact_model import (
    CompactGroundingInspector,
    EventJSONInspector,
)
from goodvision_harness.vlm_event_policy import event_policy_snapshot
from goodvision_harness.vlm_independent import _inspector, run_vlm_process
from goodvision_harness.vlm_independent_model import IndependentVLMInspector
from goodvision_harness.vlm_offload import decode_inspection, encode_inspection
from tools.vlm_benchmark import modes


@pytest.mark.parametrize(
    "cls", [IndependentVLMInspector, CompactGroundingInspector, EventJSONInspector]
)
@pytest.mark.parametrize("level", ["standard", "high"])
@pytest.mark.parametrize("followup", [False, True])
def test_default_system_prompts_match_main_snapshot(cls, level, followup):
    labels = tuple(EVENT_SENSITIVITY_DEFAULTS)
    cfg = replace(config(), labels=labels)
    request = replace(
        job(cfg, 1.2, 18),
        event_sensitivities=tuple((label, level) for label in labels),
        grounding_required=followup,
    )
    inspector = cls(cfg.model_settings, model="qwen3-vl-test", api_key="test-only")
    try:
        prompt = inspector.build_messages(request)[0]["content"]
    finally:
        inspector.close()
    frozen = json.loads(
        (Path(__file__).parent / "fixtures/vlm/main_baseline_prompts.json").read_text()
    )
    key = f"{cls.__name__}:{level}:{int(followup)}"
    assert (
        hashlib.sha256(prompt.encode()).hexdigest()
        == frozen["system_prompt_sha256"][key]
    )


def test_research_policy_is_explicit_and_has_a_different_fingerprint():
    cfg = config()
    assert (
        cfg.event_policy
        == modes.parser().parse_args([]).event_policy
        == "main-baseline"
    )
    assert cfg.grounding_format == "legacy" and cfg.parallel_scheduling == "independent"
    assert cfg.retain_historical_results is False
    baseline = _inspector(cfg)
    research = _inspector(replace(cfg, event_policy="research-v1"))
    try:
        assert baseline.event_policy == "main-baseline"
        assert research.event_policy == "research-v1"
        assert baseline.build_messages(job(cfg, 1.2, 18)) != research.build_messages(
            job(cfg, 1.2, 18)
        )
    finally:
        baseline.close()
        research.close()
    assert (
        event_policy_snapshot(cfg.labels, profile="main-baseline")["sha256"]
        != event_policy_snapshot(cfg.labels)["sha256"]
    )
    with pytest.raises(ValueError, match="policy"):
        replace(cfg, event_policy="unknown")


@pytest.mark.parametrize("profile", ["main-baseline", "research-v1"])
def test_offload_preserves_selected_policy_without_changing_baseline_envelope(profile):
    cfg = config()
    payload = encode_inspection(
        job(cfg, 1.2, 18), "compact", "test", {}, {}, 5, event_policy=profile
    )
    assert ("event_policy" in payload) == (profile != "main-baseline")
    assert decode_inspection(payload).event_policy == profile
    payload["event_policy"] = "unknown"
    with pytest.raises(ValueError, match="policy"):
        decode_inspection(payload)


def test_default_parallel_runtime_discards_superseded_information():
    """Use gates rather than wall-clock response ordering to prove the default."""
    cfg = replace(config(), output_mode="parallel")

    class OutputQueue(Queue):
        def close(self):
            pass

        def cancel_join_thread(self):
            pass

    incoming, outgoing = Queue(), OutputQueue()
    stop, old_started, newer_done, release_old = (Event() for _ in range(4))

    class Inspector:
        def __init__(self, lane):
            self.lane = lane

        def inspect(self, request):
            sequence = request.frames[-1].frame_sequence
            if self.lane == "events":
                if sequence == 18:
                    old_started.set()
                    assert release_old.wait(5)
                else:
                    newer_done.set()
            return (verdict(detected=sequence == 18),)

        def close(self):
            pass

    runner = Thread(
        target=run_vlm_process,
        args=(cfg, incoming, outgoing, stop, lambda c: Inspector(c.output_mode)),
    )
    runner.start()
    try:
        from time import monotonic

        from goodvision_harness.vlm_independent import SourceSample, utc_now

        for sequence, at in ((16, 0.4), (17, 0.8), (18, 1.2)):
            incoming.put(
                SourceSample(job(cfg, at, sequence).frames[-1], utc_now(), monotonic())
            )
        assert old_started.wait(5)
        for sequence, at in ((19, 1.6), (20, 2.0)):
            incoming.put(
                SourceSample(job(cfg, at, sequence).frames[-1], utc_now(), monotonic())
            )
        assert newer_done.wait(5)
        rows = []
        deadline = monotonic() + 5
        while not any(
            row.get("type") == "vlm_report"
            and row.get("vlm_lane") == "information"
            and row["frame"]["source_frame_index"] == 20
            for row in rows
        ):
            assert monotonic() < deadline
            item = outgoing.get(timeout=5)
            rows.append(item.get("report", item))
        release_old.set()
        incoming.put(None)
        deadline = monotonic() + 5
        while not any(
            row.get("kind") == "status" and row.get("state") == "drained"
            for row in rows
        ):
            assert monotonic() < deadline
            item = outgoing.get(timeout=5)
            rows.append(item.get("report", item))
        stop.set()
        runner.join(5)
        assert not runner.is_alive()
        while not outgoing.empty():
            item = outgoing.get_nowait()
            rows.append(item.get("report", item))
        reports = [
            row
            for row in rows
            if row.get("type") == "vlm_report" and row.get("vlm_lane") == "information"
        ]
        assert [row["frame"]["source_frame_index"] for row in reports] == [20]
        metrics = [row for row in rows if row.get("kind") == "status"]
        assert metrics[-1]["stats"]["late_information_reports_discarded"] == 1
    finally:
        release_old.set()
        stop.set()
        runner.join(6)


def test_baseline_lean_keeps_legacy_temporal_event_rules():
    cfg = replace(config(), output_mode="compact", grounding_format="lean-v1")
    inspector = _inspector(cfg)
    try:
        prompt = inspector.build_messages(job(cfg, 1.2, 18))[0]["content"]
    finally:
        inspector.close()
    assert "original frames in source-time order" in prompt
    assert "even with one visible participant" in prompt
    assert "deliberate bending or picking objects up remains negative" in prompt
    assert "A single-frame absence does not prove an event ended" in prompt
