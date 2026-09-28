"""Spawned VLM -> bounded diagnostic IPC -> real HTTP routes, without model calls."""

import base64
import io
import json
import time
from dataclasses import replace
from queue import Queue
from threading import Event, Thread

import pytest
from fastapi.testclient import TestClient
from PIL import Image
from test_vlm_independent import (
    ROOT,
    SourceSample,
    await_condition,
    config,
    job,
    jpeg,
    utc_now,
    verdict,
)
from test_vlm_parallel import OutputQueue

from goodvision_harness import service
from goodvision_harness.config import load_config
from goodvision_harness.independent_diagnostics import (
    IndependentDiagnostics,
    IndependentRequestDiagnostics,
)
from goodvision_harness.model_api import ModelAPIError
from goodvision_harness.vlm_compact_model import (
    CompactGroundingInspector,
    EventJSONInspector,
)
from goodvision_harness.vlm_independent import IndependentVLMBranch, run_vlm_process
from goodvision_harness.vlm_independent_model import IndependentVLMInspector


class MockHTTP:
    def warmup(self):
        return None

    def inspect(self, j):
        self.current_job = j
        try:
            return super().inspect(j)
        finally:
            self.current_job = None

    async def _request(self, payload, headers, *, timing):
        # Assert equality at the actual serialized multimodal payload boundary,
        # inside the spawned child; a mismatch fails the ordinary report too.
        sent = [
            base64.b64decode(p["image_url"]["url"].split(",")[1])
            for p in payload["messages"][1]["content"]
            if p["type"] == "image_url"
        ]
        j = self.current_job
        assert sent == [
            j.diagnostics.jpeg(j.request.request_id, f.frame_sequence) for f in j.frames
        ]
        timing["http_status"] = 401 if self.model == "failure" else 200
        timing["request_body_completed"] = True
        if self.model == "failure":
            raise ModelAPIError(
                "SECRET provider error", code="http_error", status_code=401
            )
        raw = (
            {"smoking": True, "fall": None}
            if isinstance(self, EventJSONInspector)
            else {
                "subjects": [],
                "verdicts": [
                    {
                        "event_label": "smoking",
                        "subject_ids": [],
                        "detected": True,
                        "confidence": 0.9,
                    },
                    {
                        "event_label": "fall",
                        "subject_ids": [],
                        "detected": None,
                        "confidence": None,
                    },
                ],
            }
        )
        return {
            "choices": [
                {"finish_reason": "stop", "message": {"content": json.dumps(raw)}}
            ]
        }


class FullHTTP(MockHTTP, IndependentVLMInspector):
    pass


class EventsHTTP(MockHTTP, EventJSONInspector):
    pass


class CompactHTTP(MockHTTP, CompactGroundingInspector):
    pass


def factory(cfg):
    cls = {"events": EventsHTTP, "compact": CompactHTTP}.get(cfg.output_mode, FullHTTP)
    return cls(cfg.model_settings, model=cfg.model, api_key="SECRET", max_image_side=24)


class DiagnosticBranch(IndependentVLMBranch):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, inspector_factory=factory, **kwargs)


def send_window(branch):
    assert branch.offer(jpeg(), 0, 0)
    assert branch.offer(jpeg(), 0.8, 12)


@pytest.mark.parametrize("mode", ["full", "compact", "events", "parallel"])
def test_spawned_requests_preserve_images_lanes_and_unknowns(mode):
    cfg = replace(config(), output_mode=mode)
    branch = DiagnosticBranch(cfg)
    branch.start()
    try:
        send_window(branch)
        count = 2 if mode == "parallel" else 1
        await_condition(
            lambda: len(branch.diagnostics.snapshot(())["records"]) == count
        )
        await_condition(lambda: branch.health().get("published_reports", 0) == count)
        data = branch.diagnostics.snapshot(())
        assert data["scope"] == "independent_model_response"
        assert data["unanalyzed_gaps"] == {}
        assert data["counts"]["response_validated"] == count
        assert len({row["request_id"] for row in data["records"]}) == count
        assert len({row["window_id"] for row in data["records"]}) == 1
        assert {row["output_mode"] for row in data["records"]} == (
            {"events", "compact"} if mode == "parallel" else {mode}
        )
        for row in data["records"]:
            assert row["result"] == "partial"
            assert row["unknown_event_labels"] == ["fall"]
            assert row["detected_labels"] == ["smoking"]
            assert row["request_attempted"] and row["request_body_completed"]
            assert row["evidence_outcome"] == "not_observed"
            assert row["accepted_verdicts"] is None
            assert row["confirmed_event_labels"] is None
            assert len(row["frames"]) == 2
            for f in row["frames"]:
                raw = branch.diagnostics.jpeg(row["request_id"], f["frame_sequence"])
                with Image.open(io.BytesIO(raw)) as image:
                    assert image.size == (24, 18)
                assert f["bytes"] == len(raw)
        assert "SECRET" not in json.dumps(data)
        assert branch.results.latest().payload["state"] == "completed"
    finally:
        branch.close(timeout_seconds=6)


def test_spawned_failure_remains_failure_with_images():
    branch = DiagnosticBranch(replace(config(), model="failure"))
    branch.start()
    try:
        send_window(branch)
        await_condition(lambda: bool(branch.diagnostics.snapshot(())["records"]))
        data = branch.diagnostics.snapshot(())
        assert data["failures"] == {"auth": 1}
        row = data["records"][0]
        assert row["response_outcome"] == "failed" and row["result"] == "unavailable"
        assert row["images_available"] and row["confirmed_event_labels"] is None
        assert "SECRET" not in json.dumps(data)
    finally:
        branch.close(timeout_seconds=6)


def packet(cfg, sequence, lane="full"):
    j = job(cfg, sequence + 0.8, sequence)
    sample = SourceSample(j.frames[-1], utc_now(), time.monotonic())
    diag = IndependentRequestDiagnostics(j, lane, sample, cfg.generation)
    diag.prepared(j, [(f, f.jpeg_bytes, 64, 48) for f in j.frames])
    output = Queue()
    assert diag.publish(output, verdicts=(verdict(),))
    return output.get_nowait()


def test_cache_bounds_eviction_duplicates_and_run_isolation():
    cfg = config()
    cache = IndependentDiagnostics(
        cfg.stream_session_id,
        cfg.generation,
        history=2,
        image_history=1,
        max_image_bytes=2000,
    )
    first = packet(cfg, 1)
    cache.receive(first)
    cache.receive(first)
    cache.receive(packet(replace(cfg, generation=2), 8))
    cache.receive(packet(replace(cfg, stream_session_id="old-run"), 9))
    assert cache.snapshot(())["counts"]["received_requests"] == 1
    for seq in (2, 3):
        cache.receive(packet(cfg, seq))
    data = cache.snapshot(())
    assert len(data["records"]) == 2 and data["image_bytes"] <= 2000
    assert sum(row["images_available"] for row in data["records"]) == 1
    assert cache.jpeg(first["record"]["request_id"], 1) is None


def test_full_diagnostic_queue_does_not_block_reports():
    incoming, outgoing, diagnostics = Queue(), OutputQueue(), OutputQueue(maxsize=1)
    diagnostics.put("occupied")
    stop = Event()
    cfg = config()
    thread = Thread(
        target=run_vlm_process,
        args=(cfg, incoming, outgoing, stop, factory),
        kwargs={"diagnostic_outgoing": diagnostics},
    )
    thread.start()
    try:
        for frame in job(cfg, 0.8, 12).frames:
            incoming.put(SourceSample(frame, utc_now(), time.monotonic()))
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            message = outgoing.get(timeout=10)
            if message["kind"] == "report":
                assert message["report"]["state"] == "completed"
                break
        else:
            pytest.fail("diagnostic backpressure blocked normal output")
        assert diagnostics.get_nowait() == "occupied"
    finally:
        stop.set()
        thread.join(10)
        assert not thread.is_alive()


def test_service_routes_use_live_independent_cache_without_scene_lock(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("HARNESS_AUTOSTART", "false")
    monkeypatch.setattr("goodvision_harness.config.find_dotenv", lambda **kwargs: "")
    monkeypatch.setattr(service, "IndependentVLMBranch", DiagnosticBranch)
    base = load_config(ROOT / "config.yaml")
    cfg = replace(
        base,
        runtime=replace(base.runtime, scene_mode="active", scene_independent_vlm=True),
        rich_labeling=replace(
            base.rich_labeling, enabled=False, tracking_enabled=False, backend="paddle"
        ),
        events=replace(base.events, directory=tmp_path / "events"),
        service=replace(base.service, webhook=replace(base.service.webhook, url=None)),
    )
    runtime = service.HarnessRuntime(cfg)
    runtime._aiclub_api_key = "SECRET"
    release, held = Event(), Event()
    blocker = None
    try:
        with TestClient(service.create_app(runtime_factory=lambda: runtime)) as client:
            runtime.start_camera(dry_run=False, features=["smoking", "fall"])
            branch = runtime._vlm_branch
            send_window(branch)
            await_condition(lambda: bool(branch.diagnostics.snapshot(())["records"]))

            def hold_scene():
                with runtime._state_lock:
                    held.set()
                    release.wait(10)

            blocker = Thread(target=hold_scene)
            blocker.start()
            assert held.wait(2)
            started = time.monotonic()
            response = client.get("/v1/scene/diagnostics")
            assert time.monotonic() - started < 2
            assert response.status_code == 200
            assert response.headers["cache-control"] == "no-store"
            row = response.json()["records"][0]
            image_path = "/v1/scene/diagnostics/" + row["request_id"] + "/0.jpg"
            image = client.get(image_path)
            assert image.content == branch.diagnostics.jpeg(row["request_id"], 0)
            assert image.status_code == 200
            release.set()
            blocker.join(2)
            scoped = client.get("/v1/fleet/cameras/camera01/scene/diagnostics")
            assert scoped.status_code == 200
            assert scoped.json()["records"] == response.json()["records"]
            scoped_image = client.get("/v1/fleet/cameras/camera01" + image_path[3:])
            assert scoped_image.content == image.content
            assert (
                client.get(image_path.replace(row["request_id"], "old-run")).status_code
                == 404
            )
    finally:
        release.set()
        if blocker:
            blocker.join(2)
        runtime.stop()


def test_oversized_request_keeps_metadata_without_jpegs(monkeypatch):
    monkeypatch.setattr(
        "goodvision_harness.independent_diagnostics.REQUEST_IMAGE_BYTES", 1
    )
    cfg = config()
    value = packet(cfg, 1)
    assert value["images"] == {}
    assert value["record"]["frames"]
    assert value["record"]["response_outcome"] == "validated"
    assert not value["record"]["images_available"]
    cache = IndependentDiagnostics(cfg.stream_session_id, cfg.generation)
    cache.receive(value)
    assert cache.snapshot(())["image_bytes"] == 0
