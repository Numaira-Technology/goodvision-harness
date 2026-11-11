"""Result subscriptions replay safely and remain independent of analysis control."""

import asyncio
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect
from test_branch_result_delivery import observation, runtime_without_workers
from test_fleet import make_app

from goodvision_harness import result_websocket as transport
from goodvision_harness.fleet_outputs import output_urls
from goodvision_harness.result_bus import ResultBus
from goodvision_harness.service import create_app


@pytest.fixture
def runtime():
    runtime, _ = runtime_without_workers()
    return runtime


def publish(runtime, label="smoking", timestamp=1):
    return runtime.results.publish(observation(label, timestamp).as_dict())


def test_live_fanout_replay_and_disconnect_do_not_control_analysis(runtime):
    first = publish(runtime)
    with TestClient(create_app(lambda: runtime)) as client:
        with client.websocket_connect("/v1/results/ws") as one:
            assert one.receive_json()["type"] == "hello"
            envelope = one.receive_json()
            assert envelope == {
                "type": "analysis_result",
                "id": f"{runtime.results.epoch}:1",
                "data": first.payload,
            }
            with client.websocket_connect("/v1/results/ws?after=1") as two:
                assert two.receive_json()["type"] == "hello"
                second = publish(runtime, "stranger", 2)
                assert one.receive_json()["data"] == second.payload
                assert two.receive_json()["data"] == second.payload
            third = publish(runtime, "smoking", 3)
            assert one.receive_json()["data"] == third.payload
        assert runtime.results.latest().payload == third.payload


@pytest.mark.parametrize("use_header", [False, True])
def test_resume_token_replays_only_later_arrivals(runtime, use_header):
    publish(runtime)
    token = f"{runtime.results.epoch}:1"
    second = publish(runtime, "stranger", 2)
    path = "/v1/results/ws"
    headers = {"Last-Event-ID": token} if use_header else {}
    if not use_header:
        path += "?last_event_id=" + token
    with TestClient(create_app(lambda: runtime)) as client:
        with client.websocket_connect(path, headers=headers) as ws:
            assert ws.receive_json()["type"] == "hello"
            assert ws.receive_json()["data"] == second.payload


def test_nonzero_after_takes_precedence_over_resume_token(runtime):
    publish(runtime)
    second = publish(runtime, timestamp=2)
    with TestClient(create_app(lambda: runtime)) as client:
        with client.websocket_connect(
            "/v1/results/ws?after=1&last_event_id=old:999",
        ) as ws:
            assert ws.receive_json()["type"] == "hello"
            assert ws.receive_json()["data"] == second.payload


@pytest.mark.parametrize("channel", ["overall", *transport.DETECTION_EVENT_LABELS])
def test_projections_equal_existing_http_payloads(runtime, channel):
    label = "smoking" if channel == "overall" else channel
    publish(runtime, label)
    with TestClient(create_app(lambda: runtime)) as client:
        expected = client.get(f"/v1/results/{channel}/latest").json()
        with client.websocket_connect(f"/v1/results/{channel}/ws") as ws:
            assert ws.receive_json()["type"] == "hello"
            result = ws.receive_json()
            assert result["data"] == expected
            assert result["type"] == (
                "overall_description" if channel == "overall" else "analysis_result"
            )


def test_filtered_stream_skips_unrelated_results_and_heartbeats_advance(
    runtime,
    monkeypatch,
):
    monkeypatch.setattr(transport, "HEARTBEAT_SECONDS", 0.02)
    monkeypatch.setattr(transport, "POLL_SECONDS", 0.01)
    publish(runtime, "stranger")
    with TestClient(create_app(lambda: runtime)) as client:
        with client.websocket_connect("/v1/results/smoking/ws") as ws:
            assert ws.receive_json()["type"] == "hello"
            assert ws.receive_json() == {
                "type": "heartbeat",
                "id": f"{runtime.results.epoch}:1",
            }
            expected = publish(runtime, timestamp=2)
            while (message := ws.receive_json())["type"] == "heartbeat":
                pass
            assert message["data"] == expected.payload


@pytest.mark.parametrize(
    "resume,reason",
    [
        ("last_event_id=previous:999", "stream_changed"),
        ("after=999", "cursor_ahead"),
        ("after=1", "history_lost"),
    ],
)
def test_unavailable_history_is_reported_before_retained_replay(
    runtime,
    resume,
    reason,
):
    for timestamp in range(1, 5):
        publish(runtime, timestamp=timestamp)
    with TestClient(create_app(lambda: runtime)) as client:
        with client.websocket_connect("/v1/results/ws?" + resume) as ws:
            hello = ws.receive_json()
            assert hello["oldest_cursor"] == 3
            reset = ws.receive_json()
            assert reset["reason"] == reason
            assert reset["id"] == f"{runtime.results.epoch}:2"
            assert [
                ws.receive_json()["data"]["frame"]["timestamp_seconds"]
                for _ in range(2)
            ] == [3, 4]


@pytest.mark.parametrize("use_header", [False, True])
def test_explicit_zero_resume_reports_evicted_history(runtime, use_header):
    token = f"{runtime.results.epoch}:0"
    for timestamp in range(1, 5):
        publish(runtime, timestamp=timestamp)
    path = "/v1/results/ws"
    headers = {"Last-Event-ID": token} if use_header else {}
    if not use_header:
        path += "?last_event_id=" + token
    with TestClient(create_app(lambda: runtime)) as client:
        with client.websocket_connect(path, headers=headers) as ws:
            assert ws.receive_json()["type"] == "hello"
            assert ws.receive_json() == {
                "type": "reset",
                "reason": "history_lost",
                "id": f"{runtime.results.epoch}:2",
                "oldest_cursor": 3,
                "latest_cursor": 4,
            }
            assert [
                ws.receive_json()["data"]["frame"]["timestamp_seconds"]
                for _ in range(2)
            ] == [3, 4]


@pytest.mark.parametrize("query", ["", "?after=0"])
def test_fresh_subscription_replays_retained_history_without_reset(runtime, query):
    for timestamp in range(1, 5):
        publish(runtime, timestamp=timestamp)
    with TestClient(create_app(lambda: runtime)) as client:
        with client.websocket_connect("/v1/results/ws" + query) as ws:
            assert ws.receive_json()["type"] == "hello"
            assert [
                ws.receive_json()["data"]["frame"]["timestamp_seconds"]
                for _ in range(2)
            ] == [3, 4]


def test_live_subscription_reports_eviction_before_first_result(runtime, monkeypatch):
    snapshot_after = runtime.results.snapshot_after

    def snapshot_then_publish(cursor):
        snapshot = snapshot_after(cursor)
        if snapshot.cursor == 0:
            # Evict the first arrivals after the initial empty snapshot, without
            # relying on a race between publications and the next polling tick.
            for timestamp in range(1, 5):
                publish(runtime, timestamp=timestamp)
        return snapshot

    monkeypatch.setattr(runtime.results, "snapshot_after", snapshot_then_publish)
    with TestClient(create_app(lambda: runtime)) as client:
        with client.websocket_connect("/v1/results/ws") as ws:
            assert ws.receive_json()["type"] == "hello"
            reset = ws.receive_json()
            assert reset["type"] == "reset"
            assert reset["reason"] == "history_lost"
            assert reset["id"] == f"{runtime.results.epoch}:2"
            assert [
                ws.receive_json()["data"]["frame"]["timestamp_seconds"]
                for _ in range(2)
            ] == [3, 4]


def test_open_subscription_observes_session_reset(runtime):
    publish(runtime)
    with TestClient(create_app(lambda: runtime)) as client:
        with client.websocket_connect("/v1/results/ws") as ws:
            old_epoch = ws.receive_json()["epoch"]
            assert ws.receive_json()["id"] == f"{old_epoch}:1"
            runtime.results.reset()
            second = publish(runtime, timestamp=2)
            reset = ws.receive_json()
            assert reset["type"] == "reset" and reset["reason"] == "stream_changed"
            assert reset["id"] == f"{runtime.results.epoch}:1"
            assert ws.receive_json()["data"] == second.payload


@pytest.mark.parametrize(
    "query", ["after=-1", "after=abc", "last_event_id=nope", "last_event_id=epoch:-1"]
)
def test_bad_resume_is_actionable_and_closed(runtime, query):
    with TestClient(create_app(lambda: runtime)) as client:
        with client.websocket_connect("/v1/results/ws?" + query) as ws:
            assert ws.receive_json()["type"] == "error"
            with pytest.raises(WebSocketDisconnect) as error:
                ws.receive_json()
            assert error.value.code == 1008


def test_commands_cannot_start_or_stop_analysis(runtime):
    with TestClient(create_app(lambda: runtime)) as client:
        with client.websocket_connect("/v1/results/ws") as ws:
            assert ws.receive_json()["type"] == "hello"
            ws.send_json({"type": "start", "mode": "camera"})
            assert ws.receive_json()["type"] == "error"
            with pytest.raises(WebSocketDisconnect) as error:
                ws.receive_json()
            assert error.value.code == 1008
        assert runtime.results.latest() is None


def test_fleet_local_identity_and_discovery(monkeypatch):
    app = make_app(monkeypatch)
    app.state.runtime.results = ResultBus()
    first = publish(app.state.runtime)
    with TestClient(app) as client:
        path = output_urls("camera01")["results/ws"]
        with client.websocket_connect(path) as ws:
            assert ws.receive_json()["camera_id"] == "camera01"
            result = ws.receive_json()
            assert result["camera_id"] == "camera01"
            assert result["data"] == {**first.payload, "camera_id": "camera01"}


@pytest.mark.parametrize("index,camera", [(1, "camera99"), (2, "camera02")])
def test_fleet_invalid_camera_and_secondary_host_rejected(monkeypatch, index, camera):
    with TestClient(make_app(monkeypatch, index=index)) as client:
        with client.websocket_connect(f"/v1/fleet/cameras/{camera}/results/ws") as ws:
            assert ws.receive_json()["type"] == "error"
            with pytest.raises(WebSocketDisconnect) as error:
                ws.receive_json()
            assert error.value.code == 1008


def test_slow_writer_times_out_and_cancels_disconnect_reader(monkeypatch):
    monkeypatch.setattr(transport, "SEND_TIMEOUT", 0.01)

    async def exercise():
        cancelled = asyncio.Event()
        closed = []

        async def send(_text):
            await asyncio.Event().wait()

        async def receive():
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()

        async def close(code):
            closed.append(code)

        socket = SimpleNamespace(
            query_params={}, headers={}, send_text=send, receive=receive, close=close
        )
        await asyncio.wait_for(
            transport.stream_results(
                socket,
                SimpleNamespace(results=ResultBus()),
            ),
            1,
        )
        assert cancelled.is_set() and closed[0] == 1013

    asyncio.run(exercise())


def test_result_size_is_bounded(runtime):
    runtime.results.publish({"description": "x" * transport.MESSAGE_LIMIT})
    with TestClient(create_app(lambda: runtime)) as client:
        with client.websocket_connect("/v1/results/ws") as ws:
            assert ws.receive_json()["type"] == "hello"
            assert ws.receive_json()["type"] == "error"
            with pytest.raises(WebSocketDisconnect) as error:
                ws.receive_json()
            assert error.value.code == 1009


def test_snapshot_epoch_bounds_and_payload_are_isolated():
    bus = ResultBus(2)
    for sequence in range(3):
        bus.publish({"sequence": sequence})
    snapshot = bus.snapshot_after(1)
    assert (snapshot.oldest_cursor, snapshot.cursor) == (2, 3)
    snapshot.records[0].payload["sequence"] = -1
    assert bus.records_after(1)[0].payload["sequence"] == 1
    bus.reset()
    reset = bus.snapshot_after(0)
    assert reset.epoch != snapshot.epoch
    assert (reset.oldest_cursor, reset.cursor, reset.records) == (4, 3, ())
