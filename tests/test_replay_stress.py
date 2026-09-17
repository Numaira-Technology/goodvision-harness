"""The replay tool drives the real pipeline with a deterministic simulator."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import pytest

from goodvision_harness.config import load_config
from goodvision_harness.grounding import AIClubMultiGroundingClient
from goodvision_harness.model_api import ModelAPIError
from goodvision_harness.sliding_window import FrameWindow, SampledFrame

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "replay_stress", ROOT / "tools/replay_stress.py",
)
assert SPEC and SPEC.loader
replay = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = replay
SPEC.loader.exec_module(replay)


def _no_sleep(_seconds: float) -> None:
    return None


def test_event_arguments_are_validated():
    event = replay.parse_event("smoking:1.5-2")
    assert (event.label, event.start, event.end) == ("smoking", 1.5, 2.0)
    assert event.scaled(2).end == 1.0
    for bad in ("smoking", "dance:1-2", "fight:3-2"):
        with pytest.raises(argparse.ArgumentTypeError):
            replay.parse_event(bad)


def _window(*timestamps: float) -> FrameWindow:
    ok, jpeg = cv2.imencode(".jpg", np.zeros((36, 64, 3), np.uint8))
    assert ok
    return FrameWindow(0, tuple(
        SampledFrame(i, i, ts, jpeg.tobytes(), 64, 36)
        for i, ts in enumerate(timestamps)
    ))


def _client(transport):
    settings = load_config(ROOT / "config.yaml").inference
    return AIClubMultiGroundingClient(
        settings, transport, event_labels=("fight", "smoking"),
    )


def test_simulated_replies_pass_the_real_parser():
    events = [replay.LabelledEvent("smoking", 1.0, 1.4)]
    profile = replay.SimulationProfile(latency_ms=0)
    transport = replay.SimulatedAnalysisTransport(
        profile, events, ("fight", "smoking"), sleep=_no_sleep,
    )
    client = _client(transport)
    positive = client.classify_window(_window(0.6, 1.2))
    judgments = {j.event_label: j.detected for j in positive.classification.judgments}
    assert judgments == {"fight": False, "smoking": True}
    negative = client.classify_window(_window(2.0))
    assert not any(j.detected for j in negative.classification.judgments)
    assert [c.timestamps for c in transport.calls] == [(0.6, 1.2), (2.0,)]


@pytest.mark.parametrize(
    ("rates", "outcome"),
    [({"auth_rate": 1.0}, "auth"), ({"timeout_rate": 1.0}, "timeout"),
     ({"parse_error_rate": 1.0}, "parse")],
)
def test_simulated_failures_use_the_real_error_paths(rates, outcome):
    profile = replay.SimulationProfile(latency_ms=0, timeout_ms=0, **rates)
    transport = replay.SimulatedAnalysisTransport(
        profile, [], ("smoking",), sleep=_no_sleep,
    )
    if outcome == "parse":
        response = _client(transport).classify_window(_window(0.0))
        assert response.classification is None
    else:
        with pytest.raises(ModelAPIError):
            transport.chat_completion(messages=[{"content": []}])
    assert transport.calls[-1].outcome == outcome


def test_profile_rejects_impossible_rates():
    with pytest.raises(ValueError):
        replay.SimulationProfile(auth_rate=0.6, timeout_rate=0.6).validate()


@pytest.fixture(scope="module")
def synthetic_video(tmp_path_factory) -> Path:
    path = tmp_path_factory.mktemp("replay") / "clip.avi"
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (64, 36),
    )
    assert writer.isOpened()
    for index in range(30):
        writer.write(np.full((36, 64, 3), index * 8, np.uint8))
    writer.release()
    return path


def _run(video, features, speed, **overrides):
    config = replay.session_config(
        load_config(ROOT / "config.yaml"), features, **overrides,
    )
    clip = replay.Clip("synthetic", video, (
        replay.LabelledEvent("smoking", 1.55, 1.75),
    ))
    profile = replay.SimulationProfile(latency_ms=0, face_latency_ms=0)
    return replay.replay_clip(
        clip, config, features, profile, speed=speed, realtime=False,
    )


def test_short_event_is_sampled_at_normal_speed_but_not_when_compressed(
    synthetic_video,
):
    normal = _run(synthetic_video, ("smoking",), 1.0, window_size=1)
    assert normal["summary"]["detected"] == 1
    assert normal["summary"]["false_alarms"] == 0
    fast = _run(synthetic_video, ("smoking",), 3.0, window_size=1)
    assert fast["events"][0]["verdict"] == "not_sampled"
    assert fast["events"][0]["duration_seconds"] == pytest.approx(0.0667, abs=1e-3)


def test_mixed_replay_keeps_independent_branch_cadences(synthetic_video):
    report = _run(
        synthetic_video, ("smoking", "stranger"), 1.0,
        event_fps=5.0, window_size=3, face_fps=2.5,
    )
    sampling = report["diagnostics"]["sampling"]
    assert sampling["safety"]["fps"] == 5.0
    assert sampling["safety"]["size_frames"] == 3
    safety_frames = sampling["safety"]["selected_frames"]
    assert sampling["stranger"]["selected_frames"] < safety_frames
    assert report["summary"]["detected"] == 1
    assert set(report["diagnostics"]["branches"]) == {"safety", "stranger"}


def test_cli_writes_a_report(synthetic_video, tmp_path, capsys):
    output = tmp_path / "report.json"
    code = replay.main([
        str(synthetic_video), "--event", "smoking:1.55-1.75", "--no-realtime",
        "--latency-ms", "0", "--features", "smoking", "--output", str(output),
    ])
    assert code == 0
    assert output.is_file()
    assert "1/1 detected" in capsys.readouterr().out


class _RealService:
    """Stands in for ModelAPIClient: always an all-negative verdict."""

    def __init__(self, error=None):
        self.error = error

    def chat_completion(self, **_kwargs):
        if self.error is not None:
            raise self.error
        return {"text": '{"no_events": 0.9}'}

    def assistant_text(self, response):
        return response["text"]


def test_recording_transport_classifies_real_service_failures():
    events = [replay.LabelledEvent("smoking", 0.0, 1.0)]
    failing = replay.RecordingTransport(
        _RealService(ModelAPIError("x", code="http_error", status_code=401)),
        events, ("smoking",),
    )
    assert _client(failing).classify_window(_window(0.5)).classification is None
    assert failing.calls[0].outcome == "auth"
    working = replay.RecordingTransport(_RealService(), events, ("smoking",))
    working.chat_completion(messages=[{"content": [
        {"type": "text", "text": "Sample timestamps: f0=0.500s"},
    ]}])
    assert working.calls[0].outcome == "ok"
    assert working.calls[0].oracle_positives == ("smoking",)


def test_real_mode_reports_evidence_the_service_did_not_recognize(synthetic_video):
    config = replay.session_config(
        load_config(ROOT / "config.yaml"), ("smoking",), window_size=1,
    )
    clip = replay.Clip("synthetic", synthetic_video, (
        replay.LabelledEvent("smoking", 1.55, 1.75),
    ))
    report = replay.replay_clip(
        clip, config, ("smoking",), replay.SimulationProfile(),
        realtime=False, real_transport=_RealService(),
    )
    assert report["analysis_service"] == "real"
    assert report["events"][0]["verdict"] == "not_recognized"
    assert report["events"][0]["request_outcomes"] == {"ok": 1}


def test_manifest_ignore_intervals_suppress_boundary_false_alarms(
    synthetic_video, tmp_path,
):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"clips": [{
        "id": "synthetic", "path": str(synthetic_video),
        "events": [{"label": "smoking", "start": 1.55, "end": 1.75}],
        "ignore": [{"start": 0.0, "end": 1.0}],
    }]}), encoding="utf-8")
    clip = replay.load_manifest(manifest)[0]
    assert clip.ignore == ((0.0, 1.0),)
    config = replay.session_config(
        load_config(ROOT / "config.yaml"), ("smoking",), window_size=1,
    )
    # Every successful simulated reply is positive.
    profile = replay.SimulationProfile(latency_ms=0, false_positive_rate=1.0)
    with_ignore = replay.replay_clip(
        clip, config, ("smoking",), profile, realtime=False,
    )
    without = replay.replay_clip(
        replace(clip, ignore=()), config, ("smoking",), profile, realtime=False,
    )
    # Samples at 0.0, 0.4, 0.8 fall in the ignored span; 2.0-2.8 do not.
    assert without["summary"]["false_alarms"] - with_ignore["summary"][
        "false_alarms"
    ] == 3
    assert with_ignore["summary"]["detected"] == 1


class _CapturingService(_RealService):
    def __init__(self):
        super().__init__()
        self.messages = []

    def chat_completion(self, **kwargs):
        self.messages.append(json.dumps(kwargs["messages"], ensure_ascii=False))
        return super().chat_completion(**kwargs)


@pytest.mark.parametrize(
    ("override", "expected", "absent"),
    [(None, "SMOKING_HIGH_RULE", "SMOKING_EVIDENCE_RULE"),
     ({"smoking": "standard"}, "SMOKING_EVIDENCE_RULE", "SMOKING_HIGH_RULE")],
)
def test_real_mode_uses_the_service_smoking_policy(
    synthetic_video, override, expected, absent,
):
    from goodvision_harness import event_sensitivity, grounding

    rules = {
        "SMOKING_HIGH_RULE": event_sensitivity.SMOKING_HIGH_RULE,
        "SMOKING_EVIDENCE_RULE": grounding.SMOKING_EVIDENCE_RULE,
    }
    config = replay.session_config(
        load_config(ROOT / "config.yaml"), ("smoking",), window_size=1,
    )
    clip = replay.Clip("synthetic", synthetic_video, ())
    service = _CapturingService()
    report = replay.replay_clip(
        clip, config, ("smoking",), replay.SimulationProfile(),
        realtime=False, real_transport=service, event_sensitivities=override,
    )
    assert service.messages
    assert all(rules[expected] in text for text in service.messages)
    assert not any(rules[absent] in text for text in service.messages)
    level = "high" if override is None else "standard"
    assert report["event_sensitivities"]["smoking"] == level


def test_request_budget_stops_sending_once_spent():
    service = _CapturingService()
    budget = replay.RequestBudget(1)
    transport = replay.RecordingTransport(
        service, [], ("smoking",), budget=budget,
    )
    message = [{"content": [{"type": "text", "text": "f0=0.500s"}]}]
    transport.chat_completion(messages=message)
    with pytest.raises(ModelAPIError):
        transport.chat_completion(messages=message)
    assert len(service.messages) == 1
    assert [c.outcome for c in transport.calls] == ["ok", "budget_exhausted"]
    assert (budget.used, budget.limit) == (1, 1)
    with pytest.raises(ValueError):
        replay.RequestBudget(0)


def test_real_budget_is_shared_across_a_replay(synthetic_video):
    config = replay.session_config(
        load_config(ROOT / "config.yaml"), ("smoking",), window_size=1,
    )
    clip = replay.Clip("synthetic", synthetic_video, ())
    service = _CapturingService()
    budget = replay.RequestBudget(2)
    report = replay.replay_clip(
        clip, config, ("smoking",), replay.SimulationProfile(),
        realtime=False, real_transport=service, request_budget=budget,
    )
    assert len(service.messages) == 2
    outcomes = report["summary"]["request_outcomes"]
    assert outcomes["ok"] == 2
    assert outcomes["budget_exhausted"] >= 1


def test_request_estimate_bounds_the_replayed_windows(synthetic_video):
    config = replay.session_config(
        load_config(ROOT / "config.yaml"), ("smoking",), window_size=1,
    )
    clip = replay.Clip("synthetic", synthetic_video, ())
    estimate = replay.estimate_safety_requests([clip], config, [1.0])
    report = replay.replay_clip(
        clip, config, ("smoking",), replay.SimulationProfile(latency_ms=0),
        realtime=False,
    )
    sent = sum(report["summary"]["request_outcomes"].values())
    assert 0 < sent <= estimate
    assert replay.estimate_safety_requests([clip], config, [1.0, 2.0]) > estimate
    assert replay.estimate_safety_requests([clip, clip], config, [1.0]) == (
        2 * estimate
    )


def test_real_cli_refuses_before_creating_a_client(
    synthetic_video, monkeypatch, capsys,
):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("the analysis client must not be created")

    monkeypatch.setattr(replay, "ModelAPIClient", forbidden)
    with pytest.raises(SystemExit) as exc:
        replay.main([
            str(synthetic_video), "--real", "--no-realtime",
            "--features", "smoking", "--max-requests", "1",
        ])
    assert "exceed --max-requests 1" in str(exc.value)
    with pytest.raises(SystemExit):
        replay.main([str(synthetic_video), "--real", "--max-requests", "0"])
    assert "real service" not in capsys.readouterr().out


def test_real_cli_reports_paid_requests(synthetic_video, monkeypatch, tmp_path):
    service = _CapturingService()
    monkeypatch.setattr(replay, "ModelAPIClient", lambda _settings: service)
    output = tmp_path / "real.json"
    code = replay.main([
        str(synthetic_video), "--real", "--no-realtime", "--features", "smoking",
        "--event-sensitivity", "smoking=standard", "--output", str(output),
    ])
    assert code == 0
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["analysis_service"] == "real"
    assert report["event_sensitivities"] == {"smoking": "standard"}
    assert report["paid_requests"] == len(service.messages) > 0


def test_event_sensitivity_arguments_are_validated():
    assert replay._event_sensitivity(" smoking = high ") == ("smoking", "high")
    for bad in ("smoking", "smoking=max", "stranger=high", "dance=standard"):
        with pytest.raises(argparse.ArgumentTypeError):
            replay._event_sensitivity(bad)


def test_model_canvas_follows_the_config_unless_overridden():
    base = load_config(ROOT / "config.yaml")
    assert base.inference.image_size == (360, 202)
    kept = replay.session_config(base, ("smoking",), safety_quality=90)
    assert kept.inference.image_size == (360, 202)
    assert kept.inference.image_jpeg_quality == 90
    plain = replay.session_config(base, ("smoking",), model_canvas=None)
    assert plain.inference.image_size is None
    other = replay.session_config(base, ("smoking",), model_canvas=(320, 180))
    assert other.inference.image_size == (320, 180)
    assert replay._model_canvas("none") is None
    assert replay._model_canvas("config") == "config"
    assert replay._model_canvas("320x180") == (320, 180)
    for bad in ("320", "0x180", "big"):
        with pytest.raises(argparse.ArgumentTypeError):
            replay._model_canvas(bad)


class _SizeRecordingService(_RealService):
    def __init__(self):
        super().__init__()
        self.sizes = []

    def chat_completion(self, **kwargs):
        import base64

        for message in kwargs["messages"]:
            content = message["content"]
            for part in content if isinstance(content, list) else ():
                if part.get("type") == "image_url":
                    data = base64.b64decode(part["image_url"]["url"].split(",", 1)[1])
                    image = cv2.imdecode(np.frombuffer(data, np.uint8), 1)
                    self.sizes.append((image.shape[1], image.shape[0]))
        return super().chat_completion(**kwargs)


@pytest.mark.parametrize(
    ("canvas", "expected"), [(None, (48, 27)), ("config", (360, 202))],
)
def test_the_model_receives_safety_side_only_without_a_canvas(
    synthetic_video, canvas, expected,
):
    config = replay.session_config(
        load_config(ROOT / "config.yaml"), ("smoking",), window_size=1,
        safety_side=48, model_canvas=canvas,
    )
    service = _SizeRecordingService()
    replay.replay_clip(
        replay.Clip("synthetic", synthetic_video, ()), config, ("smoking",),
        replay.SimulationProfile(), realtime=False, safety_side=48,
        real_transport=service,
    )
    assert service.sizes and set(service.sizes) == {expected}
