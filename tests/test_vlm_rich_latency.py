from __future__ import annotations

import asyncio
import base64
import json
import time
from collections.abc import AsyncIterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import cv2
import httpx
import numpy as np
import pytest

from goodvision_harness.model_api import ModelAPISettings
from tools import vlm_rich_latency as latency


class _Chunks(httpx.AsyncByteStream):
    def __init__(self, chunks: list[tuple[float, bytes]]) -> None:
        self.chunks = chunks
        self.closed = False
        self.exhausted = False

    async def __aiter__(self) -> AsyncIterator[bytes]:
        for delay, chunk in self.chunks:
            await asyncio.sleep(delay)
            yield chunk
        self.exhausted = True

    async def aclose(self) -> None:
        self.closed = True


@pytest.fixture(autouse=True)
def _no_real_config_or_network(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*args: Any, **kwargs: Any) -> Any:
        pytest.fail("latency tests must not load dotenv or open a real connection")

    monkeypatch.setattr(latency, "load_config", forbidden)
    monkeypatch.setattr(latency, "environment_api_key", forbidden)
    monkeypatch.setattr(latency, "_http_client", forbidden)


def _kwargs(**overrides: Any) -> dict[str, Any]:
    return {
        "settings": ModelAPISettings(provider="openrouter"),
        "model": "test-model", "messages": [{"role": "user", "content": "test"}],
        "api_key": "fake-key-never-save", "execute": True, **overrides,
    }


def _event(content: str | None = None, finish: str | None = None) -> bytes:
    value = {
        "id": "gen-test", "provider": "DeepInfra",
        "choices": [{"index": 0, "delta": {"content": content},
                     "finish_reason": finish}],
    }
    return ("data: " + json.dumps(value) + "\n\n").encode()


def _mock_stream(
    monkeypatch: pytest.MonkeyPatch, chunks: list[tuple[float, bytes]], *,
    status: int = 200, content_type: str = "text/event-stream",
) -> tuple[_Chunks, list[httpx.Request]]:
    stream = _Chunks(chunks)
    sent: list[httpx.Request] = []

    async def handle(request: httpx.Request) -> httpx.Response:
        sent.append(request)
        trace = request.extensions["trace"]
        for phase in ("connect_tcp", "start_tls"):
            await trace(f"connection.{phase}.started", {})
            await asyncio.sleep(0.001)
            await trace(f"connection.{phase}.complete", {})
        return httpx.Response(
            status, headers={"content-type": content_type}, stream=stream,
        )

    monkeypatch.setattr(
        latency, "_http_client",
        lambda deadline: httpx.AsyncClient(transport=httpx.MockTransport(handle)),
    )
    return stream, sent


def test_default_never_calls_network_or_resolves_key() -> None:
    options = _kwargs()
    options.pop("execute")
    options.pop("api_key")
    result = latency.probe_rich_latency(**options)
    assert result["metrics"]["status"] == "not_run"
    assert result["metrics"]["api_requests"] == 0


def test_endpoint_must_match_exactly_before_key_or_network() -> None:
    with pytest.raises(ValueError, match="exactly match"):
        latency.probe_rich_latency(**_kwargs(expected_endpoint="https://other.example"))


def test_aggregates_sse_and_separates_content_timing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    usage = {
        "choices": [], "usage": {"prompt_tokens": 7, "completion_tokens": 4,
                                  "total_tokens": 11, "api_key": "do-not-save"},
    }
    stream, sent = _mock_stream(monkeypatch, [
        (0, b": heartbeat\n\n"), (0, _event()),
        (0.005, _event('{"objects":')),
        (0.005, _event("[]}", "stop")),
        (0, ("data: " + json.dumps(usage) + "\n\n").encode()),
        (0, b"data: [DONE]\n\n"),
    ])
    result = latency.probe_rich_latency(**_kwargs(
        request_parameters={"provider": {"only": ["deepinfra"]}, "max_tokens": 100},
    ))
    metrics = result["metrics"]
    assert metrics["status"] == "success"
    assert metrics["client_reused"] is False
    assert metrics["client_setup_included"] is True
    assert metrics["client_setup_seconds"] >= 0
    assert metrics["connection_reused"] is False
    assert metrics["tcp_connect_seconds"] >= 0
    assert metrics["tls_handshake_seconds"] >= 0
    assert metrics["connection_ready_seconds"] <= metrics["response_headers_seconds"]
    assert metrics["response_headers_seconds"] < metrics["first_content_token_seconds"]
    assert metrics["first_content_token_seconds"] < metrics["complete_seconds"]
    assert result["assistant_text"] == '{"objects":[]}'
    assert result["response"]["provider"] == "DeepInfra"
    assert result["response"]["id"] == "gen-test"
    assert result["response"]["usage"]["total_tokens"] == 11
    payload = json.loads(sent[0].content)
    assert payload["stream"] is True
    assert payload["provider"]["only"] == ["deepinfra"]
    assert len(sent) == 1 and stream.closed and stream.exhausted
    saved = json.dumps(result)
    assert "fake-key-never-save" not in saved
    assert "do-not-save" not in saved
    assert "Authorization" not in saved


@pytest.mark.parametrize(("chunks", "category"), [
    ([_event("{}", "stop")], "missing_done"),
    ([_event("{", "length"), b"data: [DONE]\n\n"], "truncated"),
    ([b'data: {"error":{"message":"remote-secret"}}\n\n'], "remote_stream_error"),
    ([b"data: invalid remote-secret\n\n"], "invalid_sse_json"),
    ([_event("{}", "stop"), b"data: [DONE]\n\n", b"data: [DONE]\n\n"],
     "data_after_done"),
    ([_event("{}", "stop"), b"data: [DONE]\n\n", _event("extra")],
     "data_after_done"),
    ([_event("{}", "stop"), b"data: [DONE]\n\n",
      b'data: {"error":{"message":"remote-secret"}}\n\n'], "data_after_done"),
])
def test_incomplete_or_error_stream_is_not_success(
    monkeypatch: pytest.MonkeyPatch, chunks: list[bytes], category: str,
) -> None:
    stream, sent = _mock_stream(monkeypatch, [(0, chunk) for chunk in chunks])
    result = latency.probe_rich_latency(**_kwargs())
    assert result["metrics"]["status"] == "failure"
    assert result["metrics"]["error_category"] == category
    assert len(sent) == 1 and stream.closed
    assert "remote-secret" not in json.dumps(result)


def test_http_failure_does_not_read_or_save_remote_body(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stream, sent = _mock_stream(
        monkeypatch, [(0, b"secret remote error")], status=504,
        content_type="application/json",
    )
    result = latency.probe_rich_latency(**_kwargs())
    assert result["metrics"]["http_status"] == 504
    assert result["metrics"]["error_category"] == "http_error"
    assert "secret remote error" not in json.dumps(result)
    assert len(sent) == 1 and stream.closed


@pytest.mark.parametrize("after_done", [False, True])
def test_wall_deadline_cancels_slow_stream_even_without_socket_timeout(
    monkeypatch: pytest.MonkeyPatch, after_done: bool,
) -> None:
    chunks = [(0.5, _event("{}", "stop"))]
    if after_done:
        chunks = [
            (0, _event("{}", "stop")), (0, b"data: [DONE]\n\n"),
            (0.5, b": trailing keepalive\n\n"),
        ]
    stream = _Chunks(chunks)
    sent: list[httpx.Request] = []

    async def handle(request: httpx.Request) -> httpx.Response:
        sent.append(request)
        return httpx.Response(
            200, headers={"content-type": "text/event-stream"}, stream=stream,
        )

    async def run() -> tuple[dict[str, Any], float]:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
            started = time.perf_counter()
            result = await latency.probe_rich_latency_async(
                **_kwargs(deadline_seconds=0.04, http_client=client),
            )
            assert not client.is_closed
            return result, time.perf_counter() - started

    result, elapsed = asyncio.run(run())
    assert result["metrics"]["error_category"] == "wall_deadline"
    assert result["metrics"]["status"] == "failure"
    assert result["metrics"]["done_received"] is after_done
    assert elapsed < 0.4
    assert len(sent) == 1 and stream.closed
    assert not stream.exhausted


def test_external_client_is_reused_in_one_loop_and_not_closed_by_probe() -> None:
    loops: list[int] = []
    streams: list[_Chunks] = []

    async def handle(request: httpx.Request) -> httpx.Response:
        if streams:
            assert streams[-1].exhausted
        loops.append(id(asyncio.get_running_loop()))
        trace = request.extensions["trace"]
        if len(loops) == 1:
            await trace("connection.connect_tcp.started", {})
            await trace("connection.connect_tcp.complete", {})
        await trace("http11.send_request_headers.started", {})
        stream = _Chunks([
            (0, _event("{}", "stop")), (0, b"data: [DONE]\n\n"),
        ])
        streams.append(stream)
        return httpx.Response(
            200, headers={"content-type": "text/event-stream"}, stream=stream,
        )

    async def run() -> list[dict[str, Any]]:
        results = []
        client = httpx.AsyncClient(transport=httpx.MockTransport(handle))
        async with client:
            for _ in range(2):
                results.append(await latency.probe_rich_latency_async(
                    **_kwargs(http_client=client),
                ))
                assert not client.is_closed
                assert streams[-1].exhausted
                assert not [
                    task for task in asyncio.all_tasks()
                    if task is not asyncio.current_task() and not task.done()
                ]
        assert client.is_closed
        return results

    results = asyncio.run(run())
    assert len(loops) == 2 and loops[0] == loops[1]
    assert all(stream.closed for stream in streams)
    assert results[0]["metrics"]["connection_reused"] is False
    assert results[1]["metrics"]["connection_reused"] is True
    for result in results:
        assert result["metrics"]["status"] == "success"
        assert result["metrics"]["client_reused"] is True
        assert result["metrics"]["client_setup_included"] is False
        assert result["metrics"]["client_setup_seconds"] == 0


def test_drains_keepalive_tail_without_moving_done_timestamp(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stream, sent = _mock_stream(monkeypatch, [
        (0, _event("{}", "stop")), (0, b"data: [DONE]\n\n"),
        (0.02, b": keepalive after done\n\ndata:\n\n"),
    ])
    result = latency.probe_rich_latency(**_kwargs())
    metrics = result["metrics"]
    assert metrics["status"] == "success"
    assert metrics["response_eof_seconds"] - metrics["complete_seconds"] >= 0.015
    assert stream.exhausted and stream.closed and len(sent) == 1


@pytest.mark.parametrize("wire_format", ["objects", "compact"])
def test_cli_resizes_and_reencodes_image_for_selected_wire_format(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, wire_format: str,
) -> None:
    config = SimpleNamespace(
        aiclub=ModelAPISettings(provider="openrouter"),
        rich_labeling=SimpleNamespace(
            vlm_execution_mode="single", vlm_wire_format=wire_format,
            vlm_max_image_side=80,
        ),
    )
    monkeypatch.setattr(latency, "load_config", lambda _: config)
    image_path = tmp_path / "office.png"
    ok, encoded = cv2.imencode(".png", np.zeros((100, 200, 3), dtype=np.uint8))
    assert ok
    image_path.write_bytes(encoded.tobytes())
    captured: dict[str, Any] = {}

    async def capture(args: Any, selected_config: Any, messages: Any) -> int:
        captured.update(config=selected_config, messages=messages, execute=args.execute)
        return 0

    monkeypatch.setattr(latency, "_run_cli_samples", capture)
    assert latency.main([
        "--image", str(image_path), "--output-dir", str(tmp_path / "result"),
    ]) == 0
    assert captured["execute"] is False
    assert captured["config"] is config
    messages = captured["messages"]
    image_url = next(
        block["image_url"]["url"] for message in messages
        if isinstance(message["content"], list) for block in message["content"]
        if block.get("type") == "image_url"
    )
    assert image_url.startswith("data:image/jpeg;base64,")
    jpeg = base64.b64decode(image_url.split(",", 1)[1])
    assert jpeg.startswith(b"\xff\xd8")
    resized = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
    assert resized.shape[:2] == (40, 80)
    builder = (
        latency.build_compact_rich_annotation_messages if wire_format == "compact"
        else latency.build_rich_annotation_messages
    )
    assert messages == builder(jpeg)


def test_cli_rejects_parallel_mode_before_reading_image(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config = SimpleNamespace(
        aiclub=ModelAPISettings(provider="openrouter"),
        rich_labeling=SimpleNamespace(vlm_execution_mode="parallel"),
    )
    monkeypatch.setattr(latency, "load_config", lambda _: config)

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        pytest.fail("parallel CLI rejection must precede media and request preparation")

    monkeypatch.setattr(latency, "_build_cli_messages", forbidden)
    output = tmp_path / "result"
    assert latency.main([
        "--image", "unused.jpg", "--output-dir", str(output), "--execute",
    ]) == 1
    error = capsys.readouterr().err
    assert "single-request latency diagnostic" in error
    assert "tools/vlm_rich_smoke.py" in error
    assert "complete parallel rich annotation result" in error
    assert not output.exists()
