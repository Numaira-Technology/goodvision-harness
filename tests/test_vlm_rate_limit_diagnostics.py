import asyncio
import json

import httpx
import pytest

from tools.benchmark_runtime.vlm_rate_limit import (
    read_rate_limit_diagnostics,
    sanitize_rate_limit_diagnostics,
)


@pytest.mark.parametrize(
    "body",
    [
        b"PRIVATE_SECRET",
        b"[]",
        b'{"error": []}',
        b'{"error": {"metadata": "PRIVATE_SECRET"}}',
        b"[" * 2000,
        json.dumps(
            {
                "error": {
                    "metadata": {
                        "limit_source": "upstream_provider_shared_pool",
                        "raw": "PRIVATE_SECRET" * 2000,
                    }
                }
            }
        ).encode(),
    ],
)
def test_malformed_or_oversized_metadata_stays_unknown(body):
    async def check():
        response = httpx.Response(429, content=body)
        assert await read_rate_limit_diagnostics(response) == {}

    asyncio.run(check())


def test_metadata_filter_rejects_unknown_values_and_non_boolean_byok():
    assert (
        sanitize_rate_limit_diagnostics(
            {
                "limit_source": ["PRIVATE_SECRET"],
                "provider_name": "PRIVATE_SECRET",
                "error_type": "PRIVATE_SECRET",
                "provider_code": "PRIVATE_SECRET",
                "is_byok": 1,
                "raw": "PRIVATE_SECRET",
                "reason": "PRIVATE_SECRET",
            }
        )
        == {}
    )


def test_slow_error_body_is_bounded_but_parent_cancellation_survives():
    class SlowBody(httpx.AsyncByteStream):
        async def __aiter__(self):
            await asyncio.sleep(60)
            yield b"PRIVATE_SECRET"

    async def check():
        response = httpx.Response(429, stream=SlowBody())
        try:
            assert (
                await asyncio.wait_for(
                    read_rate_limit_diagnostics(response),
                    timeout=1,
                )
                == {}
            )
        finally:
            await response.aclose()
        response = httpx.Response(429, stream=SlowBody())
        try:
            task = asyncio.create_task(read_rate_limit_diagnostics(response))
            await asyncio.sleep(0)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        finally:
            await response.aclose()

    asyncio.run(check())


class CreditLimitedInspector:
    """Spawn-safe stub for the actual worker-to-result transport."""

    def __init__(self, config):
        self.last_rate_limit_diagnostics = {
            "limit_source": "openrouter_in_flight_budget",
            "reason": "in_flight_budget_exhausted",
            "raw": "PRIVATE_SECRET",
        }
        self.last_http_timing = {
            "http_request_attempted": True,
            "retry_after_seconds": 12,
        }

    def inspect(self, job):
        from goodvision_harness.model_api import ModelAPIError

        raise ModelAPIError("redacted", code="http_error", status_code=402)

    def close(self):
        pass


def test_402_evidence_survives_actual_worker_delivery():
    import io
    import time
    from pathlib import Path

    from PIL import Image

    from goodvision_harness.config import load_config
    from tools.benchmark_runtime.vlm_independent import (
        IndependentVLMBranch,
        VLMProcessConfig,
    )

    base = load_config(Path(__file__).resolve().parents[1] / "config.yaml")
    cfg = VLMProcessConfig(
        "limit-test",
        "camera01",
        1,
        ("smoking",),
        base.model_api,
        "test",
        max_in_flight=1,
        output_mode="events",
    )
    branch = IndependentVLMBranch(cfg, inspector_factory=CreditLimitedInspector)
    with Image.new("RGB", (64, 48), "gray") as frame, io.BytesIO() as buffer:
        frame.save(buffer, "JPEG")
        jpeg = buffer.getvalue()
    branch.start()
    try:
        deadline = time.monotonic() + 10
        while branch.health()["state"] != "running":
            assert time.monotonic() < deadline
            time.sleep(0.02)
        for sequence in range(4):
            branch.offer(jpeg, sequence * 0.4, sequence)
            time.sleep(0.03)
        while not (records := branch.results.records_after(0)):
            assert time.monotonic() < deadline
            time.sleep(0.02)
        timing = records[0].payload["timing"]
        assert timing["http_status"] == 402
        assert timing["http_request_attempted"] is True
        assert timing["rate_limit_diagnostics"] == {
            "limit_source": "analysis engine",
            "limit_category": "in_flight_spending_budget",
            "reason": "in_flight_budget_exhausted",
        }
        assert timing["http_timing"]["retry_after_seconds"] == 12
        assert "PRIVATE_SECRET" not in json.dumps(records[0].payload)
    finally:
        branch.close(timeout_seconds=5)
