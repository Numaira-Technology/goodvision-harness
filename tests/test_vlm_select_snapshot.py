"""Selection chips follow one live viewer and its detector source session."""

import threading
from http.server import ThreadingHTTPServer
from types import SimpleNamespace

import httpx
import pytest
import test_vlm_detector_preview as detector

from tools.vlm_benchmark import monitor as monitor_module

detector_monitor = detector.monitor
CLIENT_A = "a" * 36
CLIENT_B = "b" * 36


@pytest.fixture
def monitor(detector_monitor, monkeypatch):
    instance = detector_monitor
    instance._select_events = True
    instance._select_key = "offline-test-only"
    instance.previews = [None]
    instance.state["state"] = "playing"
    monkeypatch.setattr(
        monitor_module, "time", SimpleNamespace(monotonic=lambda: 100.0)
    )
    for client, label, epoch, generation in (
        (CLIENT_A, "fight", 0, 0),
        (CLIENT_B, "smoking", 1, 3),
    ):
        key = (client, 0)
        session = f"session-{client}"
        instance.preview_viewers[client] = 100.0
        instance.preview_trackers[key] = object()
        instance._detector_sessions[key] = {
            "id": session,
            "stage": instance.state["stage"],
            "epoch": epoch,
            "generation": generation,
        }
        instance._select_states[key] = {
            "session": session,
            "stage": instance.state["stage"],
            "tags": {"P0001": (label,)},
            "expires": 105.0,
        }
    return instance


def test_snapshot_selection_is_scoped_to_viewer_and_source_identity(monitor):
    first = monitor.snapshot(CLIENT_A)["cameras"][0]
    second = monitor.snapshot(CLIENT_B)["cameras"][0]
    assert first["select_events"] == ["fight"]
    assert (first["select_epoch"], first["select_generation"]) == (0, 0)
    assert second["select_events"] == ["smoking"]
    assert (second["select_epoch"], second["select_generation"]) == (1, 3)
    assert monitor.state["cameras"][0].get("select_events") is None


@pytest.mark.parametrize("client", [None, "invalid", "c" * 36])
def test_anonymous_or_unmatched_viewer_cannot_inherit_selection(monitor, client):
    camera = monitor.snapshot(client)["cameras"][0]
    assert camera["select_events"] == []
    assert camera["select_epoch"] is None
    assert camera["select_generation"] is None


@pytest.mark.parametrize(
    "change",
    ["expiry", "viewer", "tracker", "stage", "session", "detector-stage", "terminal"],
)
def test_expired_or_invalid_source_state_cannot_surface_selection(monitor, change):
    key = (CLIENT_A, 0)
    if change == "expiry":
        monitor._select_states[key]["expires"] = 100.0
    elif change == "viewer":
        monitor.preview_viewers[CLIENT_A] = 69.0
    elif change == "tracker":
        monitor.preview_trackers.pop(key)
    elif change == "stage":
        monitor._select_states[key]["stage"] = "previous-stage"
    elif change == "session":
        monitor._detector_sessions[key]["id"] = "new-session"
    elif change == "detector-stage":
        monitor._detector_sessions[key]["stage"] = "previous-stage"
    else:
        monitor.state["state"] = "finished"
    camera = monitor.snapshot(CLIENT_A)["cameras"][0]
    assert camera["select_events"] == []
    assert camera["select_epoch"] is None
    assert camera["select_generation"] is None


def test_http_state_polls_forward_viewer_identity_without_global_union(monitor):
    server = ThreadingHTTPServer(("127.0.0.1", 0), monitor.handler())
    thread = threading.Thread(
        target=lambda: server.serve_forever(poll_interval=0.01), daemon=True
    )
    thread.start()
    try:
        url = f"http://127.0.0.1:{server.server_port}/{monitor.token}/state"
        with httpx.Client(trust_env=False, timeout=2) as client:
            for viewer, expected in (
                (CLIENT_A, ["fight"]), (CLIENT_B, ["smoking"]), (None, [])
            ):
                headers = {"X-GoodVision-Preview-Client": viewer} if viewer else {}
                response = client.get(url, headers=headers)
                assert response.status_code == 200
                assert response.json()["cameras"][0]["select_events"] == expected
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
    assert not thread.is_alive()
