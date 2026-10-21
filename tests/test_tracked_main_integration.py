"""Main integration guards; no external services, weights or credentials."""

import hashlib
import json
from contextlib import nullcontext
from types import SimpleNamespace

import pytest

from tools.decisions_benchmark import profile, suite
from tools.tracked_grounding.monitor import PersonMonitor
from tools.vlm_benchmark import modes, monitor


def test_backend_ids_disable_preview_api_work_even_with_inherited_flags(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("PREVIEW_DETECTOR_URL", "http://127.0.0.1:1")
    monkeypatch.setenv("PREVIEW_SELECT_EVENTS", "1")
    monkeypatch.setenv("OPENROUTER_API_KEY", "unused-preview-key")
    from tools.tracked_decisions.suite import arguments

    view = PersonMonitor(tmp_path, [], arguments())
    assert not view.detector_url and not view._select_events and not view._select_key
    with pytest.raises(ValueError, match="tracking disabled"):
        view.track_preview("stage", 0, {})


def test_native_manifest_records_actual_protocol_and_evidence(tmp_path, monkeypatch):
    class Frames(list):
        media_seconds = (0, 0.4)

    source = tmp_path / "video.mp4"
    source.write_bytes(b"offline-placeholder")
    output = tmp_path / "run"
    args = suite.arguments(source, output)
    monkeypatch.setattr(modes, "decode", lambda *a: Frames([b"frame0", b"frame1"]))
    view = SimpleNamespace(begin=lambda *a: None, finish=lambda *a, **kw: None)
    monkeypatch.setattr(monitor, "LiveMonitor", lambda *a: nullcontext(view))

    def stop(*a, **kw):
        raise RuntimeError("offline stop after manifest")

    monkeypatch.setattr(modes, "run_stage", stop)
    with pytest.raises(RuntimeError, match="offline stop after manifest"):
        modes.run(args)
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["decisions_profile"] == profile.VERSION
    assert manifest["transport"] == "decisions-json-v1"
    assert manifest["http_timeouts"] == {"events": 20}
    assert manifest["output_limits"] is None
    assert manifest["event_policy"]["sha256"] == profile.POLICY_SHA256
    assert manifest["decoded_evidence"][0][0] == {
        "sha256": hashlib.sha256(b"frame0").hexdigest(),
        "media_seconds": 0,
    }
    assert json.loads((output / "summary.json").read_text())["state"] == "failed"


def test_local_route_cannot_accept_cloud_provider_preferences():
    args = modes.parser().parse_args(
        ["--backend", "local", "--provider-prefer", "alibaba"]
    )
    with pytest.raises(ValueError, match="provider routing requires OpenRouter"):
        modes.validate(args)
