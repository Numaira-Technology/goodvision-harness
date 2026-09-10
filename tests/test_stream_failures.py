from __future__ import annotations

from pathlib import Path

import pytest

from goodvision_harness.config import load_config
from goodvision_harness.inference import DryRunWindowClient
from goodvision_harness.pipeline import SessionCallbacks, StreamingSession
from goodvision_harness.privacy import PUBLIC_STREAM_ERROR, PUBLIC_STREAM_SUPPORT_ERROR
from goodvision_harness.stream_adapter import StreamAdapterError

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    ("code", "public_message"),
    [("missing_support", PUBLIC_STREAM_SUPPORT_ERROR),
     ("connection_failed", PUBLIC_STREAM_ERROR)],
)
def test_video_source_failure_is_not_reported_as_model_failure(
    caplog, code, public_message,
) -> None:
    class FailedStream:
        realtime = True

        def frames(self):
            raise StreamAdapterError("private-source?token=secret", code=code)

    config = load_config(ROOT / "config.yaml")
    ended = []
    session = StreamingSession(
        config,
        DryRunWindowClient(config.inference),
        SessionCallbacks(on_end=lambda reason, error: ended.append((reason, error))),
        mode="stream",
        video_stream=FailedStream(),
    )

    session._run()

    assert ended == [("error", public_message)]
    assert f"stream_failure code={code}" in caplog.text
    assert "private-source" not in caplog.text
    assert "secret" not in caplog.text
