"""Plugin lifetime/source isolation; doubles never attest GPU performance."""

from __future__ import annotations

import threading
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from test_sam31_continuous_worker import Backend, append, create, engine, payload

from workers.sam_scene.app import decode_jpeg
from workers.sam_scene.continuous import (
    SessionEngine,
    SessionProfile,
    SessionRegistry,
    SessionSettings,
    register_session_routes,
)
from workers.sam_scene.perception_routes import register_perception_routes
from workers.sam_scene.sam31 import VideoSettings, native_snapshot
from workers.sam_scene.sam_live import CausalPostprocessing, SamLiveContinuation
from workers.sam_scene.streaming_state import FRAME_FIELDS, BoundedRetention, RingFrames


class CausalBackend(Backend):
    def __init__(self, backend_name="sam3.1_video"):
        super().__init__(hotstart_delay=1)
        self.backend_name = backend_name

    def serialize_snapshot(self, output, **kwargs):
        snapshot = native_snapshot(output, **kwargs)
        for item in snapshot["objects"]:
            item["attributes"]["native_track"]["backend"] = self.backend_name
        return snapshot

    def route_metadata(self, session, snapshots):
        assert session.native_state is not None
        return {
            "confirmation_policy": "test_causal",
            "observations": [
                {
                    "frame_sequence": frame["frame_sequence"],
                    "provisional_object_ids": ["7"],
                }
                for frame in snapshots
            ],
        }


def causal_engine(route="sam_live", *, registry=None):
    backend_name = "sam3.1_video"
    selected = SessionEngine(
        SimpleNamespace(
            settings=replace(VideoSettings(), enabled=True), gpu_ready=True
        ),
        replace(SessionSettings(), enabled=True),
        CausalBackend(backend_name),
        profile=SessionProfile(route, backend_name),
        registry=registry,
    )
    selected.warmup()
    return selected


def test_causal_first_source_append_and_eof_preserve_exact_epoch_without_pending():
    selected = causal_engine()
    token = create(selected)
    responses = [append(selected, token, payload(1))]
    for start in (1, 5, 9, 13, 17, 21, 25, 29):
        responses.append(
            append(selected, token, payload(4, start=start, request_id=str(start)))
        )
    responses.append(
        append(selected, token, payload(0, start=33, request_id="eof", final=True))
    )
    assert [f["frame_sequence"] for r in responses for f in r["frames"]] == list(
        range(0, 66, 2)
    )
    assert all(r["pending_frame_count"] == 0 for r in responses)
    assert len({r["predictor_epoch"] for r in responses}) == 1
    assert responses[0]["route_metadata"]["observations"] == [
        {"frame_sequence": 0, "provisional_object_ids": ["7"]}
    ]
    assert responses[-1]["route_metadata"]["observations"] == []
    assert responses[-1]["state"] == "closed"
    assert not selected.has_active_session


def test_route_capacity_is_enforced_before_decode_and_does_not_advance():
    selected = causal_engine()
    token = create(selected)
    with pytest.raises(HTTPException) as error:
        append(selected, token, payload(5))
    assert error.value.status_code == 413
    assert selected.backend.calls == 0
    result = append(selected, token, payload(1))
    assert result["accepted_frame_sequence"] == 0


def test_shared_ownership_prevents_sam_and_legacy_sessions_overlapping():
    registry = SessionRegistry()
    legacy = engine()
    registry.add(legacy)
    live = causal_engine(registry=registry)
    token = create(live)
    append(live, token, payload(1))
    with pytest.raises(HTTPException, match="active_session_exists"):
        create(legacy)
    live.invalidate(token)
    assert registry.has_active_session
    registry.expire()
    assert not registry.has_active_session
    legacy_token = create(legacy)
    assert legacy_token != token
    with pytest.raises(HTTPException) as error:
        append(live, legacy_token, payload(1))
    assert error.value.status_code == 404
    assert legacy.session.status == "active"
    legacy.delete(legacy_token)
    assert create(live)


def test_production_routes_mount_only_sam_live_and_preserve_legacy_contract():
    registry = SessionRegistry()
    legacy = engine()
    registry.add(legacy)
    routes = {"sam_live": causal_engine(registry=registry)}
    app = FastAPI()
    admission = threading.Lock()
    register_session_routes(app, legacy, admission, decode_jpeg)
    register_perception_routes(app, routes, admission, decode_jpeg)
    with TestClient(app) as client:
        catalog = client.get("/v1/perception/routes").json()
        assert [entry["id"] for entry in catalog["routes"]] == ["sam_live"]
        assert all(not entry["ready"] for entry in catalog["routes"])
        first = client.post(
            "/v1/perception/sam_live/sessions", json={"stream_session_id": "camera"}
        ).json()
        assert (
            first["limits"]["startup_frames"] == first["limits"]["hotstart_delay"] == 1
        )
        assert first["limits"]["max_pending_frames"] == 0
        assert first["backend"] == "sam3.1_video"
        assert (
            client.post(
                "/v1/video/sessions", json={"stream_session_id": "camera"}
            ).status_code
            == 409
        )
        assert (
            client.post(
                "/v1/perception/hybrid/sessions", json={"stream_session_id": "camera"}
            ).status_code
            == 404
        )
        assert client.get("/health/perception/hybrid").status_code == 404
        token = first["video_session_id"]
        assert (
            client.post(
                f"/v1/perception/hybrid/sessions/{token}/append", content=payload(1)
            ).status_code
            == 404
        )
        result = client.post(
            f"/v1/perception/sam_live/sessions/{token}/append",
            content=payload(1, final=True),
        ).json()
        assert result["backend"] == "sam3.1_video"
        assert (
            result["frames"][0]["objects"][0]["attributes"]["native_track"]["backend"]
            == "sam3.1_video"
        )
        legacy_create = client.post(
            "/v1/video/sessions", json={"stream_session_id": "camera"}
        ).json()
        assert "route_metadata" not in legacy_create
        assert legacy_create["limits"]["startup_frames"] == 16


@pytest.mark.parametrize("route", ["legacy", "sam_live"])
@pytest.mark.parametrize("failure", ["engine", "backend"])
def test_failed_readiness_still_allows_owned_cleanup_under_shared_admission(
    route, failure
):
    selected = engine() if route == "legacy" else causal_engine(route)
    token = create(selected)
    append(selected, token, payload(16 if route == "legacy" else 1))
    selected.ready = failure != "engine"
    selected._production = True
    selected.backend.gpu_ready = False
    app, admission = FastAPI(), threading.Lock()
    prefix = "/v1/video" if route == "legacy" else f"/v1/perception/{route}"
    register_session_routes(app, selected, admission, decode_jpeg, api_prefix=prefix)
    with TestClient(app) as client:
        assert (
            client.post(
                prefix + "/sessions", json={"stream_session_id": "camera"}
            ).status_code
            == 503
        )
        assert (
            client.post(
                f"{prefix}/sessions/{token}/append", content=payload(1, start=16)
            ).status_code
            == 503
        )
        assert (
            client.delete(f"{prefix}/sessions/not-the-owned-token").status_code == 404
        )
        assert selected.has_active_session
        with admission:
            assert client.delete(f"{prefix}/sessions/{token}").status_code == 429
        assert selected.has_active_session
        invalid_body = client.request(
            "DELETE", f"{prefix}/sessions/{token}", content=b"unexpected"
        )
        assert invalid_body.status_code == 400
        assert selected.has_active_session
        removed = client.delete(f"{prefix}/sessions/{token}")
        assert removed.status_code == 200
        assert removed.json()["state"] == "closed"
        assert not selected.has_active_session
        assert selected.session.native_state is None


def test_sam_health_reports_global_ownership_until_legacy_teardown_finishes():
    registry = SessionRegistry()
    legacy = engine()
    registry.add(legacy)
    routes = {"sam_live": causal_engine(registry=registry)}
    app, admission = FastAPI(), threading.Lock()
    register_session_routes(app, legacy, admission, decode_jpeg)
    register_perception_routes(app, routes, admission, decode_jpeg)
    with TestClient(app) as client:
        token = create(legacy)
        append(legacy, token, payload(16))
        legacy.invalidate(token)
        assert (
            client.get("/health/perception/sam_live").json()["active_session"] is True
        )
        legacy.delete(token)
        assert (
            client.get("/health/perception/sam_live").json()["active_session"] is False
        )
        token = create(routes["sam_live"])
        assert (
            client.get("/health/perception/sam_live").json()["active_session"] is True
        )
        assert client.get("/health/video/session").json()["active_session"] is False
        routes["sam_live"].delete(token)
        assert (
            client.get("/health/perception/sam_live").json()["active_session"] is False
        )


class Frame:
    shape = (3, 32, 32)
    device = "cpu"

    def __init__(self, value):
        self.value = value

    def clone(self):
        return Frame(self.value)


def test_partial_ring_has_no_synthetic_images_and_rejects_unarrived_access():
    ring = RingFrames.from_tensor([Frame(0)], allow_partial=True)
    assert len(ring) == 16
    assert ring[0].value == 0
    assert sum(frame is not None for frame in ring.frames) == 1
    for index in (1, 8, 15, 16):
        with pytest.raises(IndexError):
            ring[index]
    for index in range(1, 40):
        ring.put(index, Frame(index))
        assert ring[index].value == index
        assert ring[index % 16].value == index
    with pytest.raises(IndexError):
        ring[20]


def test_partial_retention_expands_only_address_metadata_and_preserves_real_count():
    stage = SimpleNamespace(
        img_ids=np.array([0]), img_ids_np=np.array([0]), text_ids=np.array([2])
    )
    batch = SimpleNamespace(
        img_batch=SimpleNamespace(tensors=[Frame(0)]),
        find_inputs=[stage],
        find_targets=[None],
        find_metadatas=[None],
    )
    state = {
        "input_batch": batch,
        "num_frames": 1,
        **{name: [None] for name in FRAME_FIELDS},
    }
    BoundedRetention.__new__(BoundedRetention).install(state, allow_partial=True)
    assert len(batch.find_inputs) == 16
    assert state["num_frames"] == 1
    assert all(
        values == {0: None} for name, values in state.items() if name in FRAME_FIELDS
    )
    assert [int(item.img_ids.item()) for item in batch.find_inputs] == list(range(16))
    batch.find_inputs[1].text_ids[...] = 9
    assert batch.find_inputs[0].text_ids.item() == 2
    assert batch.find_inputs[2].text_ids.item() == 2


def test_causal_postprocess_never_reads_future_confirmation_and_emits_provisional():
    calls = []

    class Model(CausalPostprocessing):
        def _compile_model(self):
            pass

        def _get_processing_order(self, state, start, count, reverse):
            return range(start, start + count), start + count - 1

        def _run_single_frame_inference(self, state, index, reverse):
            calls.append(index)
            return {
                "removed_obj_ids": set(),
                "unconfirmed_obj_ids": {7} if index < 2 else set(),
                "suppressed_obj_ids": set(),
                "obj_id_to_mask": {7: "real"},
            }

        def _cache_frame_outputs(self, state, index, masks, **kwargs):
            assert kwargs["unconfirmed_obj_ids"] is None

        def _postprocess_output(self, state, raw, removed, suppressed, unconfirmed):
            assert unconfirmed is None
            return {"out_obj_ids": [7]}

    state = {
        "num_frames": 3,
        "feature_cache": {},
        "generator_state": {
            "hotstart_buffer": [],
            "postprocess_yield_list": [],
            "unconfirmed_obj_ids_per_frame": {},
        },
    }
    model = Model()
    iterator = model.propagate_in_video(state, 0, 3)
    assert next(iterator)[0] == 0
    assert calls == [0]  # first source delivered before even processing the next
    assert state["causal_observations"] == {0: ["7"]}
    assert [index for index, _ in iterator] == [1, 2]
    assert state["causal_observations"] == {0: ["7"], 1: ["7"], 2: []}
    assert state["generator_state"]["hotstart_buffer"] == []


def test_causal_prompt_initialization_does_not_count_first_source_twice():
    calls = []

    class Model(CausalPostprocessing):
        TEXT_ID_FOR_TEXT = 0

        def reset_state(self, state):
            calls.append("reset")

        def _init_backbone_out(self, state):
            calls.append("text_features")
            return {"features": "text_only"}

        def _run_single_frame_inference(self, *args, **kwargs):
            pytest.fail("initialization must not infer or count a source frame")

    state = {
        "num_frames": 1,
        "input_batch": SimpleNamespace(
            find_text_batch=["placeholder", "visual", "geometric"],
            find_inputs=[SimpleNamespace(text_ids=np.array([9]))],
        ),
    }
    Model().initialize_causal_prompt(state)
    assert calls == ["reset", "text_features"]
    assert state["num_frames"] == 1
    assert state["text_prompt"] == "person"
    assert state["input_batch"].find_inputs[0].text_ids.item() == 0


def test_confirmation_metadata_omits_native_masks_removed_by_geometry_validation():
    backend = SamLiveContinuation.__new__(SamLiveContinuation)
    session = SimpleNamespace(
        emitted_index=0, native_state={"causal_observations": {0: ["7", "8"]}}
    )
    snapshots = [
        {
            "frame_sequence": 12,
            "objects": [{"attributes": {"native_track": {"object_id": "7"}}}],
        }
    ]
    assert backend.route_metadata(session, snapshots)["observations"] == [
        {"frame_sequence": 12, "provisional_object_ids": ["7"]}
    ]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"startup_frames": 0},
        {"max_append_frames": 17},
        {"output_delay_frames": -1},
        {"startup_frames": 4, "max_append_frames": 1},
    ],
)
def test_invalid_plugin_profiles_fail_before_serving(kwargs):
    with pytest.raises(ValueError):
        SessionProfile("sam_live", "sam3.1_video", **kwargs)
