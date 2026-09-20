"""Idle reclamation, lane safety and authenticated telemetry contract."""

import time
from contextlib import nullcontext

import httpx
import pytest
from fastapi.testclient import TestClient
from test_contour_node import AUTH, KEY
from test_sam_multistream import _path, _payload, _setup

from workers.contour.app import create_app
from workers.contour.settings import Settings
from workers.sam_scene.gpu_memory import GPUMemory, maintain_idle_gpu


class CUDA:
    allocated = 100
    reserved = 900
    calls = 0
    fail = False

    def is_initialized(self):
        return True

    def current_device(self):
        return 0

    def memory_allocated(self, _):
        return self.allocated

    def memory_reserved(self, _):
        return self.reserved

    def max_memory_allocated(self, _):
        return 500

    def mem_get_info(self, _):
        return 1000 - self.reserved, 1000

    def device(self, _):
        return nullcontext()

    def synchronize(self, _):
        pass

    def empty_cache(self):
        self.calls += 1
        if self.fail:
            raise RuntimeError("private driver details")
        self.reserved = self.allocated


def test_memory_accounting_and_failed_reclaim_retry():
    cuda = CUDA()
    memory = GPUMemory(cuda)
    assert memory.snapshot()["unused_reserved_bytes"] == 800
    assert not memory.reclaim_if_idle(False)
    memory.mark_pending()
    assert not memory.reclaim_if_idle(True)
    cuda.fail = True
    assert not memory.reclaim_if_idle(False)
    assert memory.snapshot()["reclaim_pending"]
    assert memory.snapshot()["error_code"] == "cache_reclaim_failed"
    cuda.fail = False
    assert memory.reclaim_if_idle(False)
    status = memory.snapshot()
    assert status["allocated_bytes"] == 100  # Models stay loaded.
    assert status["unused_reserved_bytes"] == 0
    assert status["last_reclaim"]["released_bytes"] == 800
    assert status["reclaim_count"] == 1
    assert status["error_code"] is None
    assert not memory.reclaim_if_idle(False)  # No repeated idle flushing.


@pytest.mark.parametrize("ending", ["delete", "eof", "expire", "cancel"])
def test_reclaim_waits_for_last_lane_and_gpu_worker(ending):
    app, fleet, gate, registry = _setup(2)
    cuda = CUDA()
    memory = GPUMemory(cuda)
    registry.on_state_discarded = memory.mark_pending
    try:
        with TestClient(app) as client:
            tokens = []
            for index in (1, 2):
                opened = client.post(
                    _path(index), json={"stream_session_id": f"source-{index}"}
                ).json()
                tokens.append(opened["video_session_id"])
                assert (
                    client.post(
                        f"{_path(index)}/{tokens[-1]}/append", json=_payload(index)
                    ).status_code
                    == 200
                )
            assert client.delete(f"{_path(1)}/{tokens[0]}").status_code == 200
            assert not maintain_idle_gpu(registry, gate, memory)
            assert cuda.calls == 0
            lane = fleet.engines["camera-02"]
            if ending == "delete":
                assert client.delete(f"{_path(2)}/{tokens[1]}").status_code == 200
            elif ending == "eof":
                payload = _payload(2, start=1)
                payload["is_last_batch"] = True
                assert (
                    client.post(
                        f"{_path(2)}/{tokens[1]}/append", json=payload
                    ).status_code
                    == 200
                )
            elif ending == "expire":
                lane.session.touched = time.monotonic() - lane.settings.idle_seconds - 1
            else:
                lane.invalidate(tokens[1])
            assert gate.acquire()
            try:
                assert not maintain_idle_gpu(registry, gate, memory)
                assert cuda.calls == 0  # An abandoned worker still owns admission.
            finally:
                gate.release()
            assert maintain_idle_gpu(registry, gate, memory)
            assert cuda.calls == 1
            assert not registry.has_active_session
            assert not maintain_idle_gpu(registry, gate, memory)
            # Same model can immediately serve a fresh session after reclamation.
            opened = client.post(_path(2), json={"stream_session_id": "source-2"})
            assert opened.status_code == 200
            assert (
                client.post(
                    f"{_path(2)}/{opened.json()['video_session_id']}/append",
                    json=_payload(2),
                ).status_code
                == 200
            )
    finally:
        registry.close()


def test_uninitialized_cuda_does_not_initialize_or_claim_memory():
    class Uninitialized:
        def is_initialized(self):
            return False

    memory = GPUMemory(Uninitialized())
    memory.mark_pending()
    assert not memory.reclaim_if_idle(False)
    assert memory.snapshot()["available"] is False
    assert memory.snapshot()["reason"] == "cuda_not_initialized"


def test_gateway_memory_authentication_allowlist_and_invalid_metrics():
    value = GPUMemory(CUDA()).snapshot()
    value["private"] = "secret paths and credentials"
    app = create_app(
        Settings(api_key=KEY, engine_url="http://engine"),
        transport=httpx.MockTransport(lambda _: httpx.Response(200, json=value)),
    )
    with TestClient(app) as client:
        assert client.get("/api/gpu-memory").status_code == 401
        response = client.get("/api/gpu-memory", headers=AUTH)
        assert response.status_code == 200
        assert "private" not in response.text
        assert response.json()["allocated_bytes"] == 100
        value["allocated_bytes"] = -1
        assert client.get("/api/gpu-memory", headers=AUTH).status_code == 502
        value["allocated_bytes"] = None
        assert client.get("/api/gpu-memory", headers=AUTH).status_code == 502
