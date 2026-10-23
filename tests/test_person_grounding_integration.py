"""Isolated benchmark ownership and main grounding regression checks."""

import json
import time
from dataclasses import replace
from pathlib import Path
from threading import Event, Thread

import pytest
from benchmark_runtime_fixtures import await_condition, config, jpeg
from test_tracked_grounding import LABELS
from test_tracked_grounding import fake_models as fake_models

from goodvision_harness.model_api import ModelAPISettings
from tools.benchmark_runtime.person_detector import DetectorBroker
from tools.benchmark_runtime.person_grounding import TrackedPersonInspector
from tools.benchmark_runtime.vlm_compact_model import CompactGroundingInspector
from tools.benchmark_runtime.vlm_independent import IndependentVLMBranch, _inspector
from tools.tracked_grounding.inspector import inspector
from tools.tracked_grounding.monitor import PersonMonitor
from tools.tracked_grounding.suite import arguments

ROOT = Path(__file__).resolve().parents[1]


def test_isolated_adapters_select_the_same_runtime_inspector():
    cfg = replace(config(), output_mode="compact")
    old = _inspector(cfg)
    tracked = _inspector(
        replace(
            cfg,
            grounding_source="detector_ids",
            detector_broker_url="http://localhost:1/token",
        )
    )
    bench = inspector(cfg, broker_url="http://localhost:1/token")
    try:
        assert type(old) is CompactGroundingInspector
        assert type(tracked) is type(bench) is TrackedPersonInspector
    finally:
        for client in (old, tracked, bench):
            client.close()


@pytest.mark.parametrize("mode", ["compact", "parallel"])
def test_isolated_branch_detects_each_source_once_and_joins_owned_threads(
    fake_models, monkeypatch, mode
):
    url, calls = fake_models
    monkeypatch.setattr(DetectorBroker, "check_ready", lambda _: {"ready": True})
    cfg = replace(
        config(),
        labels=LABELS,
        model="fixture",
        output_mode=mode,
        model_settings=ModelAPISettings(
            provider="local", endpoint=url + "/chat/completions"
        ),
        grounding_source="detector_ids",
        detector_url=url,
    )
    branch = IndependentVLMBranch(cfg)
    try:
        branch.start()
        for seq in range(4):
            assert branch.offer(jpeg(), seq * 0.4, seq)
            time.sleep(0.42)
        expected = 2 if mode == "parallel" else 1
        await_condition(lambda: branch.stats().get("published_reports", 0) >= expected)
        bus = branch.grounding_results if mode == "parallel" else branch.results
        records = [r.payload for r in bus.records_after(0)]
        grounding = [r for r in records if r.get("output_mode") == "compact"]
        assert grounding and all(r["analysis_complete"] for r in grounding)
        assert grounding[-1]["subject_id_scope"] == "stream_session"
        assert len(calls["detector"]) == len(set(calls["detector"])) == 4
        assert any(
            "Allowed last-frame IDs:" in p["messages"][0]["content"]
            for p in calls["model"]
        )
        broker = branch._detector_broker
    finally:
        branch.close(timeout_seconds=10)
    assert branch.health()["state"] == "stopped"
    assert not broker.thread.is_alive() and not broker.server_thread.is_alive()


def test_failed_detector_preflight_starts_no_model_process(monkeypatch):
    def fail(_):
        raise ValueError("detector is not ready")

    monkeypatch.setattr(DetectorBroker, "check_ready", fail)
    branch = IndependentVLMBranch(
        replace(
            config(), grounding_source="detector_ids", detector_url="http://localhost:1"
        )
    )
    with pytest.raises(ValueError, match="not ready"):
        branch.start()
    branch.close(timeout_seconds=5)
    assert branch._process.pid is None
    assert branch.health()["state"] == "stopped"


def test_detector_start_failure_releases_listener_and_audit_file(tmp_path, monkeypatch):
    broker = DetectorBroker("http://localhost:1", tmp_path)

    def fail():
        raise RuntimeError("thread unavailable")

    monkeypatch.setattr(broker.thread, "start", fail)
    with pytest.raises(RuntimeError, match="thread unavailable"):
        broker.__enter__()
    assert broker.log.closed
    assert broker.server.socket.fileno() == -1
    assert not broker.server_thread.is_alive()


def test_stop_during_detector_preflight_does_not_leak_or_start_vlm(monkeypatch):
    checking, release = Event(), Event()

    def readiness(_):
        checking.set()
        assert release.wait(5)

    monkeypatch.setattr(DetectorBroker, "check_ready", readiness)
    branch = IndependentVLMBranch(
        replace(
            config(), grounding_source="detector_ids", detector_url="http://localhost:1"
        )
    )
    starter = Thread(target=branch.start)
    starter.start()
    try:
        assert checking.wait(5)
        branch.close()
    finally:
        release.set()
        starter.join(10)
        branch.close(timeout_seconds=10)
    assert not starter.is_alive()
    assert branch._process.pid is None
    assert branch.health()["state"] == "stopped"
    assert not branch._detector_broker.thread.is_alive()
    assert not branch._detector_broker.server_thread.is_alive()


@pytest.mark.parametrize("display", ["ids", "boxes", "off"])
def test_display_does_not_change_detection_owner_or_add_preview_calls(
    tmp_path, monkeypatch, display
):
    monkeypatch.setenv("PREVIEW_DETECTOR_URL", "http://localhost:1")
    monkeypatch.setenv("PREVIEW_SELECT_EVENTS", "1")
    args = arguments()
    args.person_display = display
    # The legacy display flag is deliberately false: it cannot own inference.
    args.person_ids_only = False
    view = PersonMonitor(tmp_path, [], args)
    assert view.backend_person_tracks
    assert view.person_display == display
    assert not view.detector_url and not view._select_events


def test_pr48_preview_remains_available_independently(tmp_path, monkeypatch):
    monkeypatch.setenv("PREVIEW_DETECTOR_URL", "http://localhost:1")
    monkeypatch.setenv("PREVIEW_SELECT_EVENTS", "1")
    args = arguments(baseline=True)
    view = PersonMonitor(tmp_path, [], args)
    assert not view.backend_person_tracks
    assert view.detector_url and view._select_events


def test_portable_provenance_survives_no_git_checkout(tmp_path):
    from tools.vlm_benchmark.provenance import source_identity

    saved = {
        "integration": "v2",
        "checkout_commit": "test",
        "uncommitted_changes": True,
    }
    (tmp_path / "bundle-manifest.json").write_text(
        json.dumps({"source_identity": saved})
    )
    assert source_identity(tmp_path) == saved


def test_fp8_display_plan_is_optional_and_does_not_change_formal_load(capsys):
    from tools.vast_vlm.fp8 import main

    plans = []
    for display in ("ids", "boxes", "off"):
        assert main(["plan", "--person-display", display]) == 0
        plan = json.loads(capsys.readouterr().out)
        assert plan.pop("person_display") == display
        plans.append(plan)
    assert plans[0] == plans[1] == plans[2]
    assert plans[0]["formal_vlm_requests"] == 6624


def test_production_grounding_keeps_main_bbox_inspector():
    from test_vlm_independent import config as main_config

    from goodvision_harness.vlm_compact_model import (
        CompactGroundingInspector as MainCompact,
    )
    from goodvision_harness.vlm_independent import _inspector as main_inspector

    cfg = replace(main_config(), output_mode="compact")
    assert not hasattr(cfg, "grounding_source")
    client = main_inspector(cfg)
    try:
        assert type(client) is MainCompact
        assert type(client) is not CompactGroundingInspector
    finally:
        client.close()


def test_production_has_no_dependency_on_benchmark_runtime():
    for path in (ROOT / "src/goodvision_harness").rglob("*.py"):
        source = path.read_text(encoding="utf-8")
        assert "tools.benchmark_runtime" not in source, str(path)
        assert "grounding_source" not in source, str(path)
