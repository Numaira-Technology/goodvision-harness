"""Scripted preview events never claim model recognition and remain reproducible."""

import hashlib
import json
from dataclasses import replace
from types import SimpleNamespace

import pytest
from test_vlm_independent import ROOT, config, job
from test_vlm_multistream_benchmark import report

from goodvision_harness.config import load_config
from goodvision_harness.model_api import ModelAPISettings
from tools import vlm_multistream_benchmark as benchmark
from tools.vlm_benchmark import modes


def simulator_config(**parameters):
    return replace(config(), parameters={"simulation_seconds": 0, **parameters})


@pytest.mark.parametrize("value", [None, "", "0", "false", "False", "no", "OFF"])
def test_default_and_disabled_demo_preserve_negative_simulation(monkeypatch, value):
    if value is None:
        monkeypatch.delenv("PREVIEW_DEMO_EVENTS", raising=False)
    else:
        monkeypatch.setenv("PREVIEW_DEMO_EVENTS", value)
    cfg = simulator_config()
    inspector = benchmark.SimulatedInspector(cfg)
    assert not inspector.demo_events
    assert all(v.detected is False for v in inspector.inspect(job(cfg, 20, 50)))
    assert not benchmark.preview_demo_settings(False)["enabled"]


def test_demo_cycle_uses_stable_source_hash_and_source_time(monkeypatch):
    monkeypatch.setenv("PREVIEW_DEMO_EVENTS", "1")

    # No host clock may affect which synthetic verdict is positive.
    def forbidden():
        raise AssertionError("demo cycle must not read the host clock")

    monkeypatch.setattr(
        benchmark,
        "time",
        SimpleNamespace(
            sleep=lambda _seconds: None,
            monotonic=forbidden,
        ),
    )
    cfg = simulator_config()
    expected = int.from_bytes(
        hashlib.sha256(cfg.source_id.encode()).digest()[:8], "big"
    )
    expected = expected % 1000 / 1000 * 14
    first, second = (benchmark.SimulatedInspector(cfg) for _ in range(2))
    assert first.demo_offset == second.demo_offset == expected
    phases = [(28.1, cfg.labels[0]), (34.0, None), (42.1, cfg.labels[1])]
    for clock, positive in phases:
        request = job(cfg, clock - expected, 50)
        results = first.inspect(request)
        assert results == second.inspect(request)
        assert [v.event_label for v in results if v.detected] == (
            [positive] if positive else []
        )


@pytest.mark.parametrize("frozen", [False, True])
def test_serialized_demo_selection_wins_over_inherited_environment(monkeypatch, frozen):
    monkeypatch.setenv("PREVIEW_DEMO_EVENTS", "0" if frozen else "1")
    cfg = simulator_config(simulation_demo_events=frozen)
    assert benchmark.SimulatedInspector(cfg).demo_events is frozen
    with pytest.raises(ValueError, match="boolean"):
        benchmark.SimulatedInspector(simulator_config(simulation_demo_events="false"))


@pytest.mark.parametrize("real", [False, True])
def test_real_factory_bypasses_demo_and_simulated_reports_are_labelled(
    monkeypatch,
    tmp_path,
    real,
):
    monkeypatch.setenv("PREVIEW_DEMO_EVENTS", "1")
    captured, factories, offers = [], [], []

    class Clock:
        now = 0.0

        def monotonic(self):
            return self.now

        def sleep(self, seconds):
            self.now += seconds

    clock = Clock()

    def real_factory(cfg):
        factories.append(("real", cfg.parameters))
        return None

    def simulator_factory(cfg):
        factories.append(("simulated", cfg.parameters))
        return None

    class Branch:
        def __init__(self, cfg, *, inspector_factory):
            self.results, self.finished = self, False
            captured.append(cfg)
            inspector_factory(cfg)

        def start(self):
            pass

        def health(self):
            return {"state": "running"}

        def offer(self, jpeg, source_pts, sequence):
            offers.append((sequence, source_pts))
            return True

        def finish_source(self):
            self.finished = True

        def stats(self):
            return {"state": "drained" if self.finished else "running"}

        def records_after(self, cursor):
            if cursor or not self.finished:
                return []
            sequence, pts = offers[-1]
            payload = {
                **report(),
                "type": "vlm_report",
                "vlm_lane": "information",
                "model": {"name": "REAL-STUB"},
                "frame": {"source_frame_index": sequence, "timestamp_seconds": pts},
            }
            return [SimpleNamespace(cursor=1, payload=payload)]

        def close(self, **_kwargs):
            pass

    monkeypatch.setattr(benchmark, "time", clock)
    monkeypatch.setattr(benchmark, "IndependentVLMBranch", Branch)
    monkeypatch.setattr(benchmark, "simulated_factory", simulator_factory)
    args = SimpleNamespace(
        real=real,
        duration=3,
        observation_fps=1.25,
        output_mode="events",
        simulated_latency=0,
        local_concurrency=4,
        global_limit=0,
        per_stream_limit=4,
        request_rate=40,
        queue_timeout=1,
        deadline=3,
    )
    base = replace(
        load_config(ROOT / "config.yaml"), aiclub=ModelAPISettings(provider="local")
    )
    result = benchmark.run_stage(
        args,
        base,
        1,
        [[b"fixture"]],
        tmp_path,
        1,
        inspector_factory=real_factory,
        source_wait=clock.sleep,
    )
    assert factories[0][0] == ("real" if real else "simulated")
    assert ("simulation_demo_events" in captured[0].parameters) is not real
    demo = result["preview_demo_events"]
    assert demo["requested"] and demo["enabled"] is not real
    row = json.loads((tmp_path / "streams-1-repeat-1/reports.jsonl").read_text())[
        "report"
    ]
    if real:
        assert row["model"]["name"] == "REAL-STUB"
        assert "synthetic" not in row
    else:
        assert row["synthetic"] and row["model"]["is_simulated"]
        assert row["model"]["name"] == "SIMULATED"
        assert row["preview_demo_events"] == demo
        assert row["measurement_kind"] == "simulated_scheduler_only"


def test_mode_manifest_and_saved_reports_record_effective_synthetic_demo(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setenv("PREVIEW_DEMO_EVENTS", "true")
    args = modes.parser().parse_args(
        [
            "--streams",
            "1",
            "--modes",
            "events",
            "--duration",
            "3",
            "--repeats",
            "1",
            "--warmup-seconds",
            "3",
            "--simulated-latency",
            "0",
            "--output",
            str(tmp_path / "demo"),
        ]
    )
    output = modes.run(args)
    manifest = json.loads((output / "manifest.json").read_text())
    demo = manifest["preview_demo_events"]
    assert demo["enabled"] and demo["requested"]
    assert demo["algorithm"] == "source-time-sha256-v1"
    assert demo["source_phase_offsets_seconds"] == {
        "camera01": benchmark.preview_demo_offset("camera01"),
    }
    assert "no model recognition or inference" in demo["meaning"]
    assert manifest["measurement_kind"] == "simulated_scheduler_only"
    assert manifest["model"] == "SIMULATED"
    summary = json.loads((output / "summary.json").read_text())
    assert summary["settings"]["preview_demo_events"] is True
    stage = summary["jobs"][0]
    assert (
        stage["preview_demo_events"] == stage["result"]["preview_demo_events"] == demo
    )
    rows = [
        json.loads(line)["report"]
        for line in next(output.rglob("reports.jsonl")).read_text().splitlines()
    ]
    assert rows and all(
        row["synthetic"] and row["preview_demo_events"] == demo for row in rows
    )
