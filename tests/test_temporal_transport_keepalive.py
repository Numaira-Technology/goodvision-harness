"""Exercise cross-run keepalive with real HTTP/1.1 on loopback only."""

from __future__ import annotations

import io
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Event, Thread

from PIL import Image

from goodvision_harness import scene_model
from goodvision_harness.contracts import NormalizedBBox
from goodvision_harness.model_api import ModelAPISettings
from goodvision_harness.scene_evidence import EvidenceFrame, InspectionRequest
from goodvision_harness.scene_model import TemporalVLMInspector
from goodvision_harness.scene_runtime import TemporalInspection, TemporalVerdict
from goodvision_harness.temporal_client_pool import TemporalClientPool
from goodvision_harness.vlm_transport import create_vlm_http_client


def _job(session_id: str) -> TemporalInspection:
    with Image.new("RGB", (48, 32), "red") as image, io.BytesIO() as buffer:
        image.save(buffer, "JPEG")
        jpeg = buffer.getvalue()
    box = NormalizedBBox(0.1, 0.1, 0.6, 0.9)
    return TemporalInspection(
        session_id,
        "loopback-test-source",
        InspectionRequest(
            f"{session_id}-request", ("person-1",), "new_person", ("smoking",), 0, 1,
        ),
        (
            EvidenceFrame(session_id, 0, 0, jpeg),
            EvidenceFrame(session_id, 1, 1, jpeg),
        ),
        ((0, "person-1", box), (1, "person-1", box)),
    )


def test_head_and_two_runs_reuse_one_real_tcp_connection(monkeypatch):
    requests = []
    connections = []
    handler_closed = Event()

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def setup(self):
            super().setup()
            # A failed assertion must not leave a handler waiting indefinitely.
            self.connection.settimeout(3.0)
            connections.append(self.client_address)

        def log_message(self, *args):
            pass

        def finish(self):
            try:
                super().finish()
            finally:
                handler_closed.set()

        def do_HEAD(self):
            requests.append((self.command, self.client_address, dict(self.headers)))
            self.send_response(405)
            self.send_header("Content-Length", "0")
            self.end_headers()

        def do_POST(self):
            requests.append((self.command, self.client_address, dict(self.headers)))
            self.rfile.read(int(self.headers["Content-Length"]))
            payload = json.dumps({
                "choices": [{
                    "message": {"content": json.dumps([{
                        "event_label": "smoking", "subject_ids": ["person-1"],
                        "detected": True, "confidence": 0.9,
                    }])},
                    "finish_reason": "stop",
                }],
            }).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            self.wfile.flush()

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = False
    server_thread = Thread(
        target=lambda: server.serve_forever(poll_interval=0.05),
        name="test-temporal-loopback", daemon=True,
    )
    server_thread.start()
    clients, inspectors = [], []

    def create_transport():
        # Keep real httpcore/TCP handling but ignore proxy and CA environment
        # settings, so this test can never send its requests outside loopback.
        client = create_vlm_http_client(trust_env=False)
        clients.append(client)
        return client

    monkeypatch.setattr(scene_model, "create_vlm_http_client", create_transport)
    settings = ModelAPISettings(
        provider="local",
        endpoint=f"http://127.0.0.1:{server.server_port}/v1/chat/completions",
        timeout_seconds=1.5,
    )

    def create_inspector():
        inspector = TemporalVLMInspector(
            settings, model="local-test-model", api_key="local-test-credential",
        )
        inspectors.append(inspector)
        return inspector

    pool = TemporalClientPool()
    try:
        expected = (TemporalVerdict("smoking", ("person-1",), True, 0.9),)
        first, = pool.prepare("same-config", create_inspector, 1)
        assert first.inspect(_job("run-1")) == expected
        first.close()
        assert not clients[0].is_closed

        second, = pool.prepare("same-config", create_inspector, 1)
        assert second.inspect(_job("run-2")) == expected
        second.close()

        assert len(inspectors) == len(clients) == 1
        assert [method for method, _, _ in requests] == ["HEAD", "POST", "POST"]
        assert len(connections) == 1
        assert {address for _, address, _ in requests} == {connections[0]}
        assert "authorization" not in {
            name.lower() for name in requests[0][2]
        }
        assert inspectors[0].stats()["requests"] == 2
    finally:
        pool.close(timeout_seconds=3.0)
        server.shutdown()
        server.server_close()
        server_thread.join(timeout=1.0)

    assert not server_thread.is_alive()
    assert all(not actor.thread.is_alive() for actor in pool._actors)
    assert handler_closed.is_set()
    assert all(client.is_closed for client in clients)
    assert all(inspector._loop.is_closed() for inspector in inspectors)
