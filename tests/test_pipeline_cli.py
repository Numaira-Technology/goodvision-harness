"""Offline checks for the production CLI's streamed transport lifecycle."""

from pathlib import Path
from unittest.mock import Mock

import pytest

from goodvision_harness import pipeline
from goodvision_harness.config import load_config

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("source", ["--video", "--stream-url"])
@pytest.mark.parametrize("fail_start", [False, True])
def test_cli_uses_real_streaming_route_and_closes_resources(
    monkeypatch, source, fail_start,
):
    config = load_config(ROOT / "config.yaml")
    transport, archive, session = Mock(), Mock(), Mock()
    build_client = Mock(return_value=Mock())
    monkeypatch.setattr(pipeline, "load_config", lambda _: config)
    transport_factory = Mock(return_value=transport)
    monkeypatch.setattr(pipeline, "PooledModelAPIClient", transport_factory)
    monkeypatch.setattr(pipeline, "build_grounding_client", build_client)
    monkeypatch.setattr(pipeline, "EventArchive", lambda _: archive)

    def create_session(_config, _client, callbacks, **kwargs):
        assert kwargs["archive"] is archive
        session.organizer.snapshot.return_value = {}
        session.start.side_effect = (
            RuntimeError("fixture start failure") if fail_start
            else lambda: callbacks.on_end("completed", None)
        )
        return session

    monkeypatch.setattr(pipeline.StreamingSession, "for_file", create_session)
    monkeypatch.setattr(pipeline.StreamingSession, "for_stream", create_session)
    args = [source, "fixture", "--stream-decisions", "--transport-timing"]
    assert pipeline.main(args) == (1 if fail_start else 0)
    assert build_client.call_args.kwargs["dry_run"] is False
    assert build_client.call_args.kwargs["stream_decisions"] is True
    assert build_client.call_args.kwargs["transport"] is transport
    assert transport_factory.call_args.kwargs["trace_transport"] is True
    archive.close.assert_called_once_with(timeout=15.0)
    transport.close.assert_called_once_with()


def test_cli_does_not_restore_retired_synthetic_mode():
    with pytest.raises(SystemExit) as error:
        pipeline._parse_args(["--dry-run"])
    assert error.value.code == 2
