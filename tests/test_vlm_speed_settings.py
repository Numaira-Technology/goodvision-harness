from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest
import yaml

from goodvision_harness import service
from goodvision_harness import vlm_rich_labeling as vlm
from goodvision_harness.config import load_config
from goodvision_harness.grounding import DryRunGroundingClient
from goodvision_harness.model_api import ModelAPISettings
from goodvision_harness.rich_labeling import RichLabelProtocolError

ROOT = Path(__file__).resolve().parents[1]
PERSON_FIELDS = (
    "a", "g", "d", "s", "uc", "lc", "b", "us", "ls", "gl", "ht", "bt", "ho",
)


@pytest.fixture(autouse=True)
def isolate_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("goodvision_harness.config.find_dotenv", lambda **kwargs: "")
    for name in (
        "INFERENCE_PROVIDER", "INFERENCE_MODEL", "INFERENCE_ENDPOINT",
        "RICH_LABEL_ENABLED", "RICH_LABEL_ENDPOINT", "RICH_LABEL_BACKEND",
    ):
        monkeypatch.delenv(name, raising=False)


def _config_path(tmp_path: Path, **settings: Any) -> Path:
    payload = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    payload["rich_labeling"].update(settings)
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return path


def test_transport_and_completeness_settings_keep_compatible_defaults() -> None:
    settings = load_config(ROOT / "config.yaml").rich_labeling
    assert settings.vlm_http2 is False
    assert settings.vlm_trust_env is True
    assert settings.vlm_keepalive_expiry_seconds == 60.0
    assert settings.vlm_require_complete is False


def test_yaml_transport_and_completeness_settings_reach_runtime_client(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_config(_config_path(
        tmp_path, enabled=True, backend="vlm", vlm_wire_format="compact",
        vlm_http2=True, vlm_trust_env=False, vlm_keepalive_expiry_seconds=42.5,
        vlm_require_complete=True,
    ))
    captured: dict[str, Any] = {}

    def client(settings: ModelAPISettings, **kwargs: Any) -> object:
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(service, "VLMRichLabelClient", client)
    monkeypatch.setattr(
        service, "build_grounding_client",
        lambda settings, *args, **kwargs: DryRunGroundingClient(settings),
    )
    runtime = service.HarnessRuntime(config)
    try:
        assert captured["http2"] is True
        assert captured["trust_env"] is False
        assert captured["keepalive_expiry_seconds"] == 42.5
        assert captured["require_complete"] is True
    finally:
        runtime.stop()


@pytest.mark.parametrize(("field", "value"), [
    ("vlm_http2", "true"), ("vlm_http2", 1),
    ("vlm_trust_env", "false"), ("vlm_trust_env", 0),
    ("vlm_require_complete", "true"), ("vlm_require_complete", 1),
    ("vlm_keepalive_expiry_seconds", 0),
    ("vlm_keepalive_expiry_seconds", -1),
    ("vlm_keepalive_expiry_seconds", True),
    ("vlm_keepalive_expiry_seconds", float("inf")),
    ("vlm_keepalive_expiry_seconds", float("nan")),
])
def test_yaml_rejects_invalid_speed_settings(
    tmp_path: Path, field: str, value: Any,
) -> None:
    with pytest.raises((ValueError, TypeError), match=field):
        load_config(_config_path(tmp_path, **{field: value}))


def test_yaml_complete_validation_requires_compact_format(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="requires vlm_wire_format: compact"):
        load_config(_config_path(
            tmp_path, vlm_wire_format="objects", vlm_require_complete=True,
        ))


@pytest.mark.parametrize("settings", [
    {"http2": 1}, {"trust_env": "false"}, {"require_complete": 1},
    {"keepalive_expiry_seconds": 0}, {"keepalive_expiry_seconds": True},
    {"keepalive_expiry_seconds": float("nan")},
    {"keepalive_expiry_seconds": float("inf")},
    {"require_complete": True, "wire_format": "objects"},
])
def test_native_client_rejects_invalid_speed_settings(settings: dict[str, Any]) -> None:
    with pytest.raises((ValueError, TypeError)):
        vlm.VLMRichLabelClient(ModelAPISettings(), model="test-model", **settings)


class _Pool:
    closed = False

    async def aclose(self) -> None:
        self.closed = True


def test_native_client_reuses_pool_with_configured_transport_flags(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pool = _Pool()
    calls: list[dict[str, Any]] = []

    def create(**kwargs: Any) -> _Pool:
        calls.append(kwargs)
        return pool

    monkeypatch.setattr(vlm, "create_vlm_http_client", create)
    client = vlm.VLMRichLabelClient(
        ModelAPISettings(), model="test-model", http2=True,
        trust_env=False, keepalive_expiry_seconds=33.5,
    )
    try:
        assert client._connection_pool() is pool
        assert client._connection_pool() is pool
        assert calls == [{
            "http2": True, "trust_env": False, "keepalive_expiry": 33.5,
        }]
    finally:
        client.close()


@pytest.mark.parametrize("mode", ["single", "parallel"])
@pytest.mark.parametrize("missing_field", PERSON_FIELDS)
def test_incomplete_reply_never_replaces_previous_snapshot_or_track_state(
    monkeypatch: pytest.MonkeyPatch, mode: str, missing_field: str,
) -> None:
    malformed = [False]
    pool = _Pool()
    captured_attributes: list[dict[str, None]] = []

    async def complete(_pool: _Pool, **request: Any) -> dict[str, Any]:
        schema_name = request["request_parameters"]["response_format"]["json_schema"][
            "name"
        ]
        if schema_name == "rich_items":
            rows = [["chair", 100, 200, 600, 800]]
        else:
            attributes = dict.fromkeys(PERSON_FIELDS)
            if malformed[0]:
                attributes.pop(missing_field)
            captured_attributes.append(attributes)
            rows = [["p", 100, 100, 400, 900, attributes]]
        return {"choices": [{
            "finish_reason": "stop", "message": {"content": json.dumps({"o": rows})},
        }]}

    monkeypatch.setattr(vlm, "create_vlm_http_client", lambda **kwargs: pool)
    monkeypatch.setattr(vlm, "complete_stream", complete)
    client = vlm.VLMRichLabelClient(
        ModelAPISettings(), model="test-model", api_key="fake-key",
        wire_format="compact", execution_mode=mode,
        require_complete=True, structured_output=True,
    )
    ok, encoded = cv2.imencode(".jpg", np.zeros((40, 60, 3), dtype=np.uint8))
    assert ok

    def infer(sequence: int) -> Any:
        return client.infer(
            encoded.tobytes(), stream_session_id="test-session",
            frame_sequence=sequence, timestamp_seconds=float(sequence),
        )

    try:
        first = infer(1)
        people = [obj for obj in first.objects if obj.label == "person"]
        assert people
        assert all(dict(person.attributes) == {"annotation_source": "vlm"}
                   for person in people)
        assert all(len(attributes) == 13 for attributes in captured_attributes)
        first_track_ids = [obj.track_id for obj in first.objects]
        next_id = client._next_id

        malformed[0] = True
        with pytest.raises(RichLabelProtocolError, match="complete compact contract"):
            infer(2)
        assert client._previous is first
        assert client._next_id == next_id

        malformed[0] = False
        third = infer(3)
        assert [obj.track_id for obj in third.objects] == first_track_ids
        assert third.frame_sequence == 3
    finally:
        client.close()
    assert pool.closed


def test_opt_out_still_accepts_legacy_partial_person_attributes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def complete(_pool: _Pool, **request: Any) -> dict[str, Any]:
        content = json.dumps({
            "o": [["p", 100, 100, 500, 800, {"s": "s"}]],
        })
        return {"choices": [{
            "finish_reason": "stop", "message": {"content": content},
        }]}

    monkeypatch.setattr(vlm, "create_vlm_http_client", lambda **kwargs: _Pool())
    monkeypatch.setattr(vlm, "complete_stream", complete)
    client = vlm.VLMRichLabelClient(
        ModelAPISettings(), model="test-model", api_key="fake-key",
        wire_format="compact", require_complete=False,
    )
    ok, encoded = cv2.imencode(".jpg", np.zeros((40, 60, 3), dtype=np.uint8))
    assert ok
    try:
        snapshot = client.infer(
            encoded.tobytes(), stream_session_id="test-session",
            frame_sequence=1, timestamp_seconds=0,
        )
        assert snapshot.objects[0].attributes["sleeve_type"] == "ShortSleeve"
    finally:
        client.close()
