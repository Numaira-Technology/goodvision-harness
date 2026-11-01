"""The backend-person monitor preserves the base state-poll viewer contract."""

import threading
from http.server import ThreadingHTTPServer
from types import SimpleNamespace

import httpx
import pytest

from tools.tracked_grounding.monitor import PersonMonitor
from tools.vlm_benchmark.monitor import LiveMonitor


@pytest.mark.parametrize("client", [None, "a" * 36])
def test_person_monitor_http_state_forwards_client_and_preserves_backend_people(
    tmp_path, monkeypatch, client
):
    view = PersonMonitor(
        tmp_path,
        [[b"source-frame"] * 4],
        SimpleNamespace(
            model="offline",
            real=False,
            backend="local",
            observation_fps=1.25,
            backend_person_tracks=True,
        ),
    )
    view.previews = [None]
    view.begin(
        {
            "id": "stage",
            "mode": "compact",
            "phase": "measurement",
            "duration": 10,
            "repeat": 1,
            "streams": 1,
        },
        tmp_path,
    )
    people = [
        {
            "track_id": "P0001",
            "bbox": {"x1": 0.1, "y1": 0.1, "x2": 0.5, "y2": 0.9},
        }
    ]
    view.broker = SimpleNamespace(
        condition=threading.Condition(),
        latest={
            0: {
                "state": "ready",
                "frame_sequence": 12,
                "timestamp_seconds": 4.8,
                "people": people,
            }
        },
    )
    expected = {"sequence": 12, "timestamp": 4.8, "objects": people}
    forwarded = []
    original = LiveMonitor.snapshot

    def snapshot(self, client=None):
        forwarded.append(client)
        return original(self, client)

    monkeypatch.setattr(LiveMonitor, "snapshot", snapshot)
    assert view.snapshot()["cameras"][0]["people"] == expected
    server = ThreadingHTTPServer(("127.0.0.1", 0), view.handler())
    thread = threading.Thread(
        target=lambda: server.serve_forever(poll_interval=0.01), daemon=True
    )
    thread.start()
    try:
        url = f"http://127.0.0.1:{server.server_port}/{view.token}/state"
        headers = {"X-GoodVision-Preview-Client": client} if client else {}
        with httpx.Client(trust_env=False, timeout=2) as connection:
            response = connection.get(url, headers=headers)
        assert response.status_code == 200
        data = response.json()
        assert data["stage"] == "stage"
        assert data["cameras"][0]["people"] == expected
        assert forwarded == [None, client]
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
        view.log.close()
    assert not thread.is_alive()
