"""Real summary scheduling, cancellation and transport contracts without a model."""

import asyncio
import io
import json
import threading
import time
from types import SimpleNamespace

import httpx
import pytest
from PIL import Image

from goodvision_harness import workflow_summary
from goodvision_harness.model_api import ModelAPISettings
from goodvision_harness.workflow_summary import (
    SummaryManager,
    SummarySnapshot,
    snapshot_from_connection,
)


class Clock:
    def __init__(self):
        self.now = 100.0

    def __call__(self):
        return self.now

    def advance(self, manager, seconds):
        self.now += seconds
        manager.wake()


def wait_for(predicate, timeout=2):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.005)
    assert predicate(), "summary worker did not reach the expected state"


def snapshot(sequence=1, connection_id="connection-a", jpeg=b"test JPEG"):
    return SummarySnapshot(jpeg, sequence, sequence / 10, connection_id)


def make_manager(specs=None, **options):
    async def infer(spec, frame):
        return f"Observed frame {frame.source_frame_sequence} for {spec['id']}."

    defaults = {
        "source_id": "camera02",
        "stream_session_id": "session-two",
        "generation": 7,
        "frame_provider": snapshot,
        "inference": infer,
        "timeout_seconds": 1,
    }
    return SummaryManager(
        specs or [{"id": "summary-a", "interval_seconds": 10}], **(defaults | options)
    )


def test_independent_intervals_use_the_latest_frame_and_source_qualified_results():
    clock, current, records = Clock(), [snapshot()], []
    manager = make_manager(
        [
            {"id": "fast", "interval_seconds": 10},
            {"id": "slow", "interval_seconds": 30},
        ],
        clock=clock,
        frame_provider=lambda: current[0],
        on_result=records.append,
    )
    try:
        manager.start()
        assert not manager.history()
        clock.advance(manager, 10)
        wait_for(lambda: len(records) == 1)
        assert records[0]["node_id"] == "fast"
        current[0] = snapshot(9)
        clock.advance(manager, 20)
        wait_for(lambda: len(records) == 3)
        assert {r["node_id"] for r in records[1:]} == {"fast", "slow"}
        for record in records:
            assert record["source_id"] == "camera02"
            assert record["stream_session_id"] == "session-two"
            assert record["generation"] == 7
            assert record["connection_id"] == "connection-a"
            assert record["synthetic"] is False
            assert record["status"] == "completed"
            assert record["summary_translations"] == {}
            assert record["summary_language"] is None
        assert manager.latest("slow")["source_frame_sequence"] == 9
        assert manager.latest("slow")["timestamp_seconds"] == 0.9
        json.dumps(manager.status(), allow_nan=False)
        json.dumps(manager.history(), allow_nan=False)
    finally:
        manager.stop(1)


def test_busy_component_does_not_block_other_summary_or_accumulate_requests():
    clock, release, calls = Clock(), threading.Event(), []

    async def infer(spec, frame):
        calls.append(spec["id"])
        if spec["id"] == "blocked":
            while not release.is_set():
                await asyncio.sleep(0.005)
        return "Actual injected inference result."

    manager = make_manager(
        [
            {"id": "blocked", "interval_seconds": 10},
            {"id": "free", "interval_seconds": 10},
        ],
        clock=clock,
        inference=infer,
        frame_provider=lambda: snapshot(int(clock.now)),
    )
    try:
        manager.start()
        clock.advance(manager, 10)
        wait_for(lambda: manager.latest("free") is not None)
        for _ in range(3):
            before = len(manager.history("free"))
            clock.advance(manager, 10)
            wait_for(lambda before=before: len(manager.history("free")) > before)
        assert calls.count("blocked") == 1
        assert calls.count("free") == 4
        assert manager.status()["components"][0]["skipped_busy"] == 3
        release.set()
        wait_for(lambda: manager.latest("blocked") is not None)
        assert calls.count("blocked") == 1
    finally:
        release.set()
        manager.stop(1)


def test_stop_rejects_late_publication_and_stays_busy_until_resistant_work_drains():
    clock, release, entered, records = Clock(), threading.Event(), threading.Event(), []

    async def infer(spec, frame):
        entered.set()
        while not release.is_set():
            try:
                await asyncio.sleep(0.005)
            except asyncio.CancelledError:
                pass
        return "A late result that must never be published."

    manager = make_manager(clock=clock, inference=infer, on_result=records.append)
    manager.start()
    clock.advance(manager, 10)
    assert entered.wait(1)
    manager.stop(0.01)
    try:
        assert manager.state == "stopping"
        assert manager.busy
        with pytest.raises(RuntimeError, match="busy"):
            manager.start()
        assert not records and manager.latest() is None
    finally:
        release.set()
        wait_for(lambda: not manager.busy)
    assert manager.state == "idle"
    assert not records and manager.latest() is None


def test_deadline_cancels_request_reports_error_and_allows_a_later_period():
    clock, cancelled, count = Clock(), threading.Event(), []

    async def infer(spec, frame):
        count.append(frame.source_frame_sequence)
        if len(count) == 1:
            try:
                await asyncio.sleep(10)
            finally:
                cancelled.set()
        return "The next request succeeded."

    manager = make_manager(
        clock=clock,
        inference=infer,
        timeout_seconds=0.03,
        frame_provider=lambda: snapshot(int(clock.now)),
    )
    try:
        manager.start()
        clock.advance(manager, 10)
        wait_for(lambda: manager.latest() is not None)
        assert cancelled.is_set()
        assert manager.latest()["error"]["code"] == "timeout"
        assert manager.latest()["summary"] is None
        clock.advance(manager, 10)
        wait_for(lambda: len(manager.history()) == 2)
        assert manager.latest()["summary"] == "The next request succeeded."
        assert manager.status()["components"][0]["failed"] == 1
    finally:
        manager.stop(1)


def test_no_new_frame_and_changed_connection_are_errors_without_model_calls():
    clock, current, calls = Clock(), [snapshot()], []

    async def infer(spec, frame):
        calls.append(frame)
        return "Visible scene."

    manager = make_manager(
        clock=clock, frame_provider=lambda: current[0], inference=infer
    )
    try:
        manager.start()
        clock.advance(manager, 10)
        wait_for(lambda: len(manager.history()) == 1)
        clock.advance(manager, 10)
        wait_for(lambda: len(manager.history()) == 2)
        assert manager.latest()["error"]["code"] == "no_new_frame"
        current[0] = snapshot(2, "different-camera")
        clock.advance(manager, 10)
        wait_for(lambda: len(manager.history()) == 3)
        assert manager.latest()["error"]["code"] == "source_changed"
        assert len(calls) == 1
    finally:
        manager.stop(1)


def test_run_once_is_real_isolated_and_does_not_consume_a_scheduled_frame():
    clock = Clock()
    manager = make_manager(clock=clock)
    result = manager.run_once("summary-a", snapshot(20))
    wait_for(lambda: not manager.busy)
    assert result["test"] is True
    assert result["summary"] == "Observed frame 20 for summary-a."
    assert result["source_frame_sequence"] == 20
    assert not manager.history() and manager.latest() is None
    assert manager.status()["components"][0]["requests"] == 0
    try:
        manager.start()
        clock.advance(manager, 10)
        wait_for(lambda: manager.latest() is not None)
        assert manager.latest()["source_frame_sequence"] == 1
    finally:
        manager.stop(1)


def test_run_once_refuses_overlap_with_a_running_instance():
    clock, entered = Clock(), threading.Event()

    async def infer(spec, frame):
        entered.set()
        await asyncio.sleep(10)
        return "Unreachable."

    manager = make_manager(clock=clock, inference=infer)
    try:
        manager.start()
        clock.advance(manager, 10)
        assert entered.wait(1)
        with pytest.raises(RuntimeError, match="busy"):
            manager.run_once("summary-a", snapshot(2))
        assert not manager.history()
    finally:
        manager.stop(1)


def test_dry_run_is_explicitly_synthetic_and_does_not_require_credentials():
    async def forbidden(spec, frame):
        pytest.fail("dry run must not invoke a model")

    manager = make_manager(dry_run=True, inference=forbidden)
    result = manager.run_once("summary-a", snapshot())
    wait_for(lambda: not manager.busy)
    assert result["synthetic"] is True
    assert "no model was called" in result["summary"]
    assert result["summary_translations"] == {
        "zh": "模拟运行的示例描述；未调用模型。",  # noqa: RUF001
        "en": result["summary"],
    }
    assert result["summary_language"] == "en"
    assert result["status"] == "completed"


def test_history_is_bounded_and_return_values_do_not_mutate_manager_state():
    clock = Clock()
    manager = make_manager(
        clock=clock, history_size=2, frame_provider=lambda: snapshot(int(clock.now))
    )
    try:
        manager.start()
        for sequence in range(4):
            clock.advance(manager, 10)
            wait_for(
                lambda sequence=sequence: (
                    manager.status()["components"][0]["succeeded"] == sequence + 1
                )
            )
        assert len(manager.history()) == 2
        latest = manager.latest()
        latest["summary"] = "tampered"
        assert manager.latest()["summary"] != "tampered"
    finally:
        manager.stop(1)


def jpeg_bytes():
    buffer = io.BytesIO()
    Image.new("RGB", (30, 20), "green").save(buffer, format="JPEG")
    return buffer.getvalue()


@pytest.mark.parametrize("provider", ["local", "openrouter"])
def test_http_summary_uses_configured_compatible_route_including_goodvision_local(
    monkeypatch,
    provider,
):
    requests = []

    async def handler(request):
        requests.append(request)
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": json.dumps({
                "zh": "绿色的画面。", "en": "A green scene.",
            })}}]},
        )

    def client():
        return httpx.AsyncClient(
            transport=httpx.MockTransport(handler),
            headers={"Authorization": "Bearer inherited-cloud-secret"},
            cookies={"cloud": "private"},
        )

    monkeypatch.setattr(workflow_summary, "create_vlm_http_client", client)
    settings = ModelAPISettings(
        provider=provider,
        endpoint="http://configured.test/v1/chat/completions",
        inspection_endpoint="http://configured.test/v1/inspect"
        if provider == "local"
        else "",
        extra_headers={"Authorization": "Bearer wrong-key", "X-Route": "selected"},
    )
    config = SimpleNamespace(
        model_api=settings,
        inference=SimpleNamespace(
            model="chosen-model", request_parameters={"top_p": 0.8}
        ),
        runtime=SimpleNamespace(
            scene_timeout_seconds=1, scene_max_image_side=720, scene_jpeg_quality=85
        ),
    )
    manager = SummaryManager.from_config(
        [{"id": "summary-1", "prompt": "Describe the picture."}],
        config,
        source_id="camera01",
        stream_session_id="session",
        generation=3,
        frame_provider=lambda: None,
        api_key="selected-key",
    )
    result = manager.run_once("summary-1", snapshot(jpeg=jpeg_bytes()))
    wait_for(lambda: not manager.busy)
    assert result["summary"] == "A green scene."
    assert result["summary_translations"] == {
        "zh": "绿色的画面。", "en": "A green scene.",
    }
    assert result["summary_language"] == "en"
    assert len(requests) == 1, "Both UI languages share one scheduled inference"
    assert requests[0].url.path == "/v1/chat/completions"
    assert requests[0].headers["authorization"] == "Bearer selected-key"
    assert "cookie" not in requests[0].headers
    assert requests[0].headers["x-route"] == "selected"
    payload = json.loads(requests[0].content)
    assert payload["model"] == "chosen-model"
    assert payload["top_p"] == 0.8
    assert payload["stream"] is False
    assert payload["max_tokens"] == 512
    assert "Simplified Chinese" in payload["messages"][0]["content"]
    assert '"en" in English' in payload["messages"][0]["content"]
    assert "same observations" in payload["messages"][0]["content"]
    assert payload["messages"][1]["content"][0]["text"] == "Describe the picture."
    assert payload["messages"][1]["content"][1]["image_url"]["url"].startswith(
        "data:image/jpeg;base64,"
    )


@pytest.mark.parametrize("fence", [False, True])
def test_http_bilingual_summary_normalizes_both_languages_without_changing_focus(
    monkeypatch, fence,
):
    requests = []
    content = json.dumps(
        {"zh": "  一个人站在桌旁。\n", "en": " A person stands by a table. "}
    )
    if fence:
        content = "```json\n" + content + "\n```"

    def handler(request):
        requests.append(json.loads(request.content))
        return httpx.Response(
            200, json={"choices": [{"message": {"content": content}}]}
        )

    monkeypatch.setattr(
        workflow_summary, "create_vlm_http_client",
        lambda: httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    manager = make_manager(
        [{"id": "summary-a", "prompt": "只关注桌子周围的情况。"}],
        inference=None, model_settings=ModelAPISettings(provider="local"),
        model="local-model",
    )
    result = manager.run_once("summary-a", snapshot(jpeg=jpeg_bytes()))
    wait_for(lambda: not manager.busy)
    assert result["status"] == "completed"
    assert result["summary_translations"] == {
        "zh": "一个人站在桌旁。", "en": "A person stands by a table.",
    }
    assert result["summary"] == result["summary_translations"]["en"]
    assert len(requests) == 1
    assert requests[0]["messages"][1]["content"][0]["text"] == "只关注桌子周围的情况。"


@pytest.mark.parametrize("content,finish_reason", [
    ("A plain English description.", "stop"),
    ('{"zh":"一个人。","en":"A person.', "length"),
    ('{"zh":"一个人。","en":"A person."}', "length"),
    ('{"zh":"一个人。","en":"A person."}', "content_filter"),
    ('{"en":"A person."}', "stop"),
    ('{"zh":"一个人。","en":" "}', "stop"),
    ('{"zh":null,"en":"A person."}', "stop"),
    ('{"zh":[],"en":"A person."}', "stop"),
    ('{"zh":"一个人。","en":"A person.","extra":"unexpected"}', "stop"),
    (json.dumps({"zh": "人" * 8001, "en": "A person."}), "stop"),
    (json.dumps({"zh": "一个人。", "en": "x" * 8001}), "stop"),
], ids=[
    "plain-text",
    "truncated-json",
    "length-finish-reason",
    "content-filtered",
    "missing-chinese",
    "blank-english",
    "null-chinese",
    "non-string-chinese",
    "unexpected-field",
    "chinese-too-long",
    "english-too-long",
])
def test_invalid_or_incomplete_bilingual_output_is_not_published_as_summary(
    monkeypatch, content, finish_reason,
):
    monkeypatch.setattr(
        workflow_summary, "create_vlm_http_client",
        lambda: httpx.AsyncClient(transport=httpx.MockTransport(
            lambda request: httpx.Response(200, json={"choices": [{
                "finish_reason": finish_reason, "message": {"content": content},
            }]}),
        )),
    )
    manager = make_manager(
        inference=None, model_settings=ModelAPISettings(provider="local"),
        model="local-model",
    )
    result = manager.run_once("summary-a", snapshot(jpeg=jpeg_bytes()))
    wait_for(lambda: not manager.busy)
    assert result["status"] == "error"
    assert result["error"]["code"] == "invalid_response"
    assert result["summary"] is None
    assert result["summary_translations"] == {}
    assert result["summary_language"] is None


@pytest.mark.parametrize(
    "response,code",
    [
        (httpx.Response(401, text="PRIVATE_KEY provider detail"), "http_error"),
        (httpx.Response(200, json={"choices": []}), "invalid_response"),
        (
            httpx.Response(200, json={"choices": [{"message": {"content": ""}}]}),
            "empty_response",
        ),
        (httpx.Response(200, text="x" * (256 * 1024 + 1)), "invalid_response"),
    ],
)
def test_transport_errors_never_fabricate_summary_or_expose_provider_body(
    monkeypatch, response, code
):
    monkeypatch.setattr(
        workflow_summary,
        "create_vlm_http_client",
        lambda: httpx.AsyncClient(
            transport=httpx.MockTransport(lambda request: response),
        ),
    )
    manager = make_manager(
        inference=None,
        model_settings=ModelAPISettings(provider="local"),
        model="local-model",
    )
    result = manager.run_once("summary-a", snapshot(jpeg=jpeg_bytes()))
    wait_for(lambda: not manager.busy)
    assert result["status"] == "error"
    assert result["error"]["code"] == code
    assert result["summary"] is None
    assert "PRIVATE_KEY" not in json.dumps(result)


def test_snapshot_helper_uses_public_source_metadata_and_rejects_disconnect():
    statuses = [
        {"connected": True, "connection_id": "connection-a"},
        {"connected": True, "connection_id": "connection-a"},
    ]
    connection = SimpleNamespace(
        status=lambda: statuses.pop(0),
        display_frames=lambda **kw: [(9, 1.5, 30, 20, b"jpeg")],
    )
    frame = snapshot_from_connection(connection)
    assert frame == SummarySnapshot(b"jpeg", 9, 1.5, "connection-a")
    statuses.extend(
        [
            {"connected": True, "connection_id": "connection-a"},
            {"connected": False, "connection_id": "connection-a"},
        ]
    )
    assert snapshot_from_connection(connection) is None


@pytest.mark.parametrize(
    "specs",
    [
        [{"id": "same"}, {"id": "same"}],
        [{"id": ""}],
        [{"id": "x", "interval_seconds": 0}],
        [{"id": "x", "interval_seconds": float("nan")}],
        [{"id": "x", "prompt": ""}],
    ],
)
def test_invalid_specifications_are_rejected_before_any_worker_starts(specs):
    with pytest.raises(ValueError):
        make_manager(specs)


def test_restart_then_run_once_does_not_reuse_tasks_from_a_closed_loop():
    clock = Clock()
    manager = make_manager(clock=clock)
    manager.start()
    manager.stop(1)
    assert manager.state == "idle"
    first = manager.run_once("summary-a", snapshot(2))
    assert first["status"] == "completed"
    wait_for(lambda: not manager.busy)
    second = manager.run_once("summary-a", snapshot(3))
    assert second["source_frame_sequence"] == 3
    wait_for(lambda: not manager.busy)
    assert not manager.history()


def test_http_body_deadline_closes_the_stream(monkeypatch):
    closed = threading.Event()

    class SlowBody(httpx.AsyncByteStream):
        async def __aiter__(self):
            await asyncio.sleep(10)
            yield b"{}"

        async def aclose(self):
            closed.set()

    monkeypatch.setattr(
        workflow_summary,
        "create_vlm_http_client",
        lambda: httpx.AsyncClient(
            transport=httpx.MockTransport(
                lambda request: httpx.Response(200, stream=SlowBody()),
            )
        ),
    )
    manager = make_manager(
        inference=None,
        model_settings=ModelAPISettings(provider="local"),
        model="local-model",
        timeout_seconds=0.03,
    )
    result = manager.run_once("summary-a", snapshot(jpeg=jpeg_bytes()))
    wait_for(lambda: not manager.busy)
    assert result["error"]["code"] == "timeout"
    assert result["summary"] is None
    assert closed.is_set()


def test_invalid_snapshot_returns_image_error_before_any_network_request(monkeypatch):
    def forbidden_client():
        pytest.fail("invalid JPEG must not create a network client")

    monkeypatch.setattr(workflow_summary, "create_vlm_http_client", forbidden_client)
    manager = make_manager(
        inference=None,
        model_settings=ModelAPISettings(provider="local"),
        model="local-model",
    )
    result = manager.run_once("summary-a", snapshot(jpeg=b"invalid JPEG"))
    wait_for(lambda: not manager.busy)
    assert result["error"]["code"] == "invalid_image"
    assert result["summary"] is None


def test_stop_while_loop_is_starting_cannot_leave_an_idle_thread_running(monkeypatch):
    original = asyncio.new_event_loop
    entered, release = threading.Event(), threading.Event()

    def delayed_loop():
        entered.set()
        assert release.wait(1)
        return original()

    monkeypatch.setattr(workflow_summary.asyncio, "new_event_loop", delayed_loop)
    manager = make_manager()
    starter = threading.Thread(target=manager.start)
    starter.start()
    assert entered.wait(1)
    manager.stop(0.01)
    assert manager.busy
    release.set()
    starter.join(1)
    assert not starter.is_alive()
    wait_for(lambda: not manager.busy)


def test_largest_compilable_summary_graph_initializes_without_http():
    from goodvision_harness.workflows import MAX_NODES, compile_graph

    nodes = [{"id": "source", "type": "source"}, {"id": "output", "type": "output"}]
    edges = []
    for index in range(MAX_NODES - 2):
        node_id = f"summary-{index}"
        nodes.append({"id": node_id, "type": "summary"})
        edges.extend(
            [
                {
                    "id": f"input-{index}",
                    "from_node": "source",
                    "from_port": "frames",
                    "to_node": node_id,
                    "to_port": "frames",
                },
                {
                    "id": f"output-{index}",
                    "from_node": node_id,
                    "from_port": "summary",
                    "to_node": "output",
                    "to_port": "summary",
                },
            ]
        )
    plan = compile_graph({"nodes": nodes, "edges": edges})
    assert len(plan["summaries"]) > 32
    manager = make_manager(plan["summaries"], dry_run=True)
    try:
        manager.start()
        assert len(manager.status()["components"]) == MAX_NODES - 2
        assert not manager.history()
    finally:
        manager.stop(1)
