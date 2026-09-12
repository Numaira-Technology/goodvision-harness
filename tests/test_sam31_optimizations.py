"""Latency-profile and compile contracts; doubles never attest CUDA speed/quality."""

from contextlib import nullcontext
from types import SimpleNamespace

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from test_sam31_continuous_worker import append, create, engine, payload

from workers.sam_scene.app import decode_jpeg
from workers.sam_scene.continuous import (
    NativeContinuation,
    Session,
    SessionSettings,
    register_session_routes,
)
from workers.sam_scene.sam31 import VideoSettings, compile_vision_encoder
from workers.sam_scene.streaming_state import hotstart_parameters


@pytest.mark.parametrize("delay", [3, 5, 15])
def test_profile_preserves_source_prefix_across_small_appends_and_empty_eof(delay):
    selected = engine(hotstart_delay=delay)
    token = create(selected)
    limits = selected.limits()
    assert limits["startup_frames"] == 16
    assert limits["hotstart_delay"] == delay
    assert limits["max_pending_frames"] == delay - 1
    responses = [append(selected, token, payload())]
    for index in range(16, 23):
        responses.append(
            append(selected, token, payload(1, start=index, request_id=str(index)))
        )
        assert responses[-1]["pending_frame_count"] == delay - 1
    responses.append(
        append(selected, token, payload(0, start=23, request_id="eof", final=True))
    )
    assert responses[-1]["pending_frame_count"] == 0
    assert responses[-1]["state"] == "closed"
    snapshots = [frame for response in responses for frame in response["frames"]]
    assert [frame["frame_sequence"] for frame in snapshots] == list(range(0, 46, 2))
    assert {response["predictor_epoch"] for response in responses} == {
        responses[0]["predictor_epoch"]
    }


@pytest.mark.parametrize("delay", [3, 5, 15])
def test_profiles_have_explicit_early_rejection_tradeoff(delay, monkeypatch):
    assert hotstart_parameters(delay) == {
        "hotstart_delay": delay,
        "hotstart_unmatch_thresh": min(8, delay),
        "hotstart_dup_thresh": min(8, delay),
    }
    monkeypatch.setenv("SAM31_SESSION_HOTSTART_DELAY", str(delay))
    assert SessionSettings.from_env().hotstart_delay == delay


@pytest.mark.parametrize("delay", [True, 0, 1, 2, 4, 6, 16, 3.0, "5"])
def test_unaudited_hotstart_profiles_rejected(delay):
    with pytest.raises(ValueError):
        SessionSettings(hotstart_delay=delay)
    with pytest.raises(ValueError):
        hotstart_parameters(delay)


def test_native_exceeding_advertised_pending_bound_invalidates_epoch():
    selected = engine(hotstart_delay=5)
    selected.backend.hotstart_delay = 15  # Misconfigured or incompatible backend.
    token = create(selected)
    with pytest.raises(RuntimeError, match="pending count"):
        append(selected, token, payload())
    assert selected.session.status == "invalidated"
    assert selected.session.native_state is None
    assert selected.ready is False


def test_low_profile_does_not_accept_unaudited_short_first_live_append():
    selected = engine(hotstart_delay=3)
    with pytest.raises(HTTPException, match="startup_requires_16_frames"):
        append(selected, create(selected), payload(3))
    assert selected.backend.calls == 0


@pytest.mark.parametrize("delay", [3, 5, 15])
@pytest.mark.parametrize("count", [1, 3, 16])
def test_first_final_uses_selected_prod_profile_without_fabricated_input(delay, count):
    calls = []

    class Model:
        hotstart_delay = delay

        def init_state(self, *, resource_path, **kwargs):
            calls.append(("init", len(resource_path)))
            return {"tracker_metadata": {"max_obj_id": 0}}

        def add_prompt(self, **kwargs):
            calls.append(("prompt", kwargs["frame_idx"]))

        def propagate_in_video(self, **kwargs):
            assert kwargs["is_last_batch"] is True
            assert kwargs["start_frame_idx"] == 0
            assert kwargs["max_frame_num_to_track"] == count
            calls.append(("propagate", self.hotstart_delay))
            yield from ((i, {}) for i in range(count))

    def forbidden(*args, **kwargs):
        raise AssertionError("finite EOF must not use the batch model or input ring")

    torch = SimpleNamespace(
        inference_mode=nullcontext,
        autocast=lambda **kwargs: nullcontext(),
        bfloat16="bf16",
        cuda=SimpleNamespace(synchronize=lambda: calls.append(("sync", True))),
    )
    video = SimpleNamespace(backend=SimpleNamespace(_torch=torch, infer=forbidden))
    native = NativeContinuation(video, SessionSettings(hotstart_delay=delay))
    native.model = Model()
    native.retention = SimpleNamespace(install=forbidden, prune=forbidden)
    session = Session("token", "source", "epoch", 0.0)
    outputs = list(native.run(session, [object() for _ in range(count)], final=True))
    assert [index for index, _ in outputs] == list(range(count))
    assert calls == [
        ("init", count), ("prompt", 0), ("propagate", delay), ("sync", True)
    ]
    assert session.native_retention_installed is False
    native.prune(session)
    native.discard(session)
    assert session.native_state is None


@pytest.mark.parametrize("mode", [True, None, "full", "auto", "fa3", "Encoder"])
def test_unvalidated_compile_modes_rejected(mode):
    with pytest.raises(ValueError, match="compile_mode"):
        VideoSettings(compile_mode=mode)


def test_compile_configuration_is_opt_in_and_keeps_existing_limits(monkeypatch):
    assert VideoSettings().compile_mode == "off"
    baseline = VideoSettings().limits()
    monkeypatch.setenv("SAM31_COMPILE_MODE", "encoder")
    optimized = VideoSettings.from_env()
    assert optimized.compile_mode == "encoder"
    assert optimized.limits() == baseline


def test_encoder_compilation_failure_is_propagated_without_eager_fallback():
    def original(value):
        return value

    encoder = SimpleNamespace(forward=original)
    model = SimpleNamespace(
        detector=SimpleNamespace(
            backbone=SimpleNamespace(vision_backbone=SimpleNamespace(trunk=encoder))
        ),
        state_dict=lambda: {},
    )

    def fail(*args, **kwargs):
        raise RuntimeError("compiler unavailable")

    with pytest.raises(RuntimeError, match="compiler unavailable"):
        compile_vision_encoder(SimpleNamespace(compile=fail), model)
    assert encoder.forward is original
    assert not getattr(encoder, "_goodvision_compiled_forward", False)


def test_encoder_compile_cannot_change_verified_tensor_names_or_storage():
    def original(value):
        return value

    encoder = SimpleNamespace(forward=original)
    weight = SimpleNamespace(shape=(2,), data_ptr=lambda: 1)
    weights = {"detector.weight": weight}
    model = SimpleNamespace(
        detector=SimpleNamespace(
            backbone=SimpleNamespace(vision_backbone=SimpleNamespace(trunk=encoder))
        ),
        state_dict=lambda: weights,
    )

    def compile_with_mutation(forward, **kwargs):
        weights["unexpected.derived.weight"] = weight
        return lambda value: forward(value)

    with pytest.raises(RuntimeError, match="changed verified model tensors"):
        compile_vision_encoder(SimpleNamespace(compile=compile_with_mutation), model)
    assert encoder.forward is original
    assert not getattr(encoder, "_goodvision_compiled_forward", False)


def test_encoder_compiles_exactly_once_without_replacing_the_module():
    encoder = SimpleNamespace(forward=lambda value: value + 1)
    model = SimpleNamespace(
        detector=SimpleNamespace(
            backbone=SimpleNamespace(vision_backbone=SimpleNamespace(trunk=encoder))
        ),
        state_dict=lambda: {},
    )
    torch = SimpleNamespace(compile=lambda forward, **kwargs: lambda x: forward(x))
    compile_vision_encoder(torch, model)
    assert model.detector.backbone.vision_backbone.trunk is encoder
    assert encoder.forward(2) == 3
    with pytest.raises(RuntimeError, match="already compiled"):
        compile_vision_encoder(torch, model)


def test_request_stage_logging_contains_only_counts_and_timings(caplog):
    import threading

    selected = engine(hotstart_delay=5)
    app = FastAPI()
    register_session_routes(app, selected, threading.Lock(), decode_jpeg)
    with caplog.at_level("INFO", logger="goodvision.sam31.session"):
        with TestClient(app) as client:
            opened = client.post(
                "/v1/video/sessions", json={"stream_session_id": "camera"}
            )
            token = opened.json()["video_session_id"]
            response = client.post(
                f"/v1/video/sessions/{token}/append", content=payload()
            )
    assert response.status_code == 200
    messages = "\n".join(
        record.getMessage() for record in caplog.records
        if record.name == "goodvision.sam31.session"
    )
    for key in (
        "received_bytes", "body_read_seconds", "processing_seconds", "total_seconds",
        "validation_decode_seconds", "native_output_seconds",
        "retention_response_seconds",
    ):
        assert f"{key}=" in messages
    assert "input_frames=16 output_frames=12 pending_frames=4" in messages
    for private in (token, opened.json()["predictor_epoch"], "camera", "jpeg_base64"):
        assert private not in messages
