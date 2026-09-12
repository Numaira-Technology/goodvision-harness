"""Protocol/lifecycle tests use injected detections, never claim SAM accuracy."""

from __future__ import annotations

import io
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from dataclasses import replace
from types import ModuleType, SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from goodvision_harness.rich_labeling import RichLabelSnapshot
from workers.sam_scene.app import (
    SEQUENCE_HEADER,
    SESSION_HEADER,
    TIMESTAMP_HEADER,
    BusyError,
    Detection,
    FrameOrderError,
    SamImageDetector,
    SceneEngine,
    Settings,
    _validated_image_detector_state,
    create_app,
)

Image = pytest.importorskip("PIL.Image")


class DetectorFixture:
    # This deliberately misleading flag must not cause GPU readiness.
    gpu_ready = True

    def __init__(self, batches):
        self.batches = list(batches)
        self.calls = 0
        self.closed = False

    def warmup(self):
        pass

    def detect(self, image):
        result = self.batches[min(self.calls, len(self.batches) - 1)]
        self.calls += 1
        return result

    def close(self):
        self.closed = True


def detection(x=0.1, *, label="person", score=0.95, width=0.2):
    return Detection(label, (x, 0.1, x + width, 0.7), score)


def engine_for(batches=None, **kwargs):
    detector = DetectorFixture(batches or [[detection()]])
    settings = replace(Settings(), **kwargs)
    engine = SceneEngine(settings, detector=detector)
    engine.warmup()
    return engine, detector


def infer(
    engine,
    sequence=0,
    timestamp=0.0,
    *,
    session="camera-a",
    size=(100, 100),
    deadline=None,
):
    with Image.new("RGB", size) as image:
        return engine.infer(
            image,
            stream_session_id=session,
            frame_sequence=sequence,
            timestamp_seconds=timestamp,
            deadline=deadline,
        )


def jpeg(size=(100, 100), *, format="JPEG"):
    stream = io.BytesIO()
    Image.new("RGB", size).save(stream, format=format)
    return stream.getvalue()


def headers(sequence=0, timestamp=0.0, session="camera-a"):
    return {
        "Content-Type": "image/jpeg",
        SESSION_HEADER: session,
        SEQUENCE_HEADER: str(sequence),
        TIMESTAMP_HEADER: str(timestamp),
    }


def test_observations_and_predictions_follow_strict_harness_contract():
    engine, detector = engine_for([[detection(0.1)], [detection(0.14)]])
    first = RichLabelSnapshot.from_dict(infer(engine))
    observed = RichLabelSnapshot.from_dict(infer(engine, 8, 0.4))
    predicted = RichLabelSnapshot.from_dict(infer(engine, 9, 0.5))
    assert first.objects[0].track_id == observed.objects[0].track_id
    assert predicted.objects[0].track_id == observed.objects[0].track_id
    assert predicted.objects[0].observation_state == "predicted"
    assert predicted.objects[0].last_observed_timestamp_seconds == 0.4
    assert predicted.objects[0].last_observed_frame_sequence == 8
    assert predicted.objects[0].bbox.x1 == pytest.approx(0.15)
    assert detector.calls == 2


def test_keyframes_count_accepted_requests_instead_of_source_frame_numbers():
    engine, detector = engine_for(keyframe_requests=3, keyframe_seconds=10)
    infer(engine, 1, 0.0)
    infer(engine, 200, 0.1)
    infer(engine, 400, 0.2)
    assert detector.calls == 1
    infer(engine, 600, 0.3)
    assert detector.calls == 2


def test_unmatched_track_is_predicted_without_forged_fresh_observation():
    engine, _ = engine_for(
        [[detection()], [], []], keyframe_requests=1, track_ttl_seconds=0.6
    )
    original = infer(engine)["objects"][0]
    missing = infer(engine, 1, 0.4)["objects"][0]
    assert missing["track_id"] == original["track_id"]
    assert missing["observation_state"] == "predicted"
    assert missing["last_observed_frame_sequence"] == 0
    assert infer(engine, 2, 0.7)["objects"] == []


def test_clear_multiobject_associations_preserve_ids_with_reordered_detections():
    engine, _ = engine_for(
        [[detection(0.05), detection(0.7)], [detection(0.68), detection(0.08)]],
        keyframe_requests=1,
    )
    original = sorted(infer(engine)["objects"], key=lambda x: x["bbox"]["x1"])
    updated = sorted(infer(engine, 1, 0.1)["objects"], key=lambda x: x["bbox"]["x1"])
    assert [x["track_id"] for x in original] == [x["track_id"] for x in updated]


def test_ambiguous_merge_retires_both_old_identities():
    engine, _ = engine_for(
        [[detection(0.1), detection(0.3)], [detection(0.2)]], keyframe_requests=1
    )
    original_ids = {x["track_id"] for x in infer(engine)["objects"]}
    merged = infer(engine, 1, 0.1)["objects"]
    assert len(merged) == 1
    assert merged[0]["track_id"] not in original_ids


def test_one_track_with_two_equally_plausible_detections_gets_fresh_ids():
    engine, _ = engine_for(
        [[detection(0.2)], [detection(0.15), detection(0.25)]], keyframe_requests=1
    )
    original = infer(engine)["objects"][0]["track_id"]
    split = infer(engine, 1, 0.1)["objects"]
    assert len(split) == 2
    assert all(x["track_id"] != original for x in split)


def test_labels_cannot_transfer_identity():
    engine, _ = engine_for(
        [[detection()], [detection(label="backpack")]], keyframe_requests=1
    )
    original = infer(engine)["objects"][0]["track_id"]
    updated = infer(engine, 1, 0.1)["objects"]
    bag = next(x for x in updated if x["label"] == "backpack")
    assert bag["track_id"] != original


@pytest.mark.parametrize("sequence,timestamp", [(0, 0.1), (4, 0.1), (6, 0.01)])
def test_stale_source_is_rejected_without_resetting_identity(sequence, timestamp):
    engine, _ = engine_for()
    original = infer(engine, 5, 0.05)
    with pytest.raises(FrameOrderError):
        infer(engine, sequence, timestamp)
    assert engine.sessions["camera-a"].sequence == 5
    assert (
        infer(engine, 7, 0.2)["objects"][0]["track_id"]
        == original["objects"][0]["track_id"]
    )


@pytest.mark.parametrize("change", ["session", "dimensions", "gap"])
def test_source_restart_uses_new_ids(change):
    engine, _ = engine_for()
    original = infer(engine)["objects"][0]["track_id"]
    kwargs = (
        {"session": "camera-b"}
        if change == "session"
        else {"size": (200, 100)}
        if change == "dimensions"
        else {}
    )
    updated = infer(engine, 1, 3 if change == "gap" else 0.1, **kwargs)
    assert updated["objects"][0]["track_id"] != original


def test_sessions_are_bounded_and_expiry_does_not_reuse_ids():
    clock = [0.0]
    detector = DetectorFixture([[detection()]])
    engine = SceneEngine(
        replace(Settings(), max_sessions=1, session_idle_seconds=2),
        detector=detector,
        clock=lambda: clock[0],
    )
    engine.warmup()
    old = infer(engine)["objects"][0]["track_id"]
    with pytest.raises(BusyError):
        infer(engine, session="camera-b")
    clock[0] = 3
    new = infer(engine)["objects"][0]["track_id"]
    assert old != new
    assert len(engine.sessions) == 1


def test_track_count_and_duplicate_detections_are_bounded():
    engine, _ = engine_for(
        [[detection(0.1), detection(0.101), detection(0.4), detection(0.7)]],
        max_tracks=2,
    )
    objects = infer(engine)["objects"]
    assert len(objects) == 2
    assert objects[0]["bbox"]["x1"] == 0.1
    assert objects[1]["bbox"]["x1"] == 0.4


def test_expired_inference_never_commits_results():
    engine, _ = engine_for()
    with pytest.raises(TimeoutError):
        infer(engine, deadline=0)
    assert engine.sessions == {}


def test_injected_detector_cannot_assert_gpu_readiness():
    engine, _ = engine_for()
    with TestClient(create_app(engine)) as client:
        for path in ("/health/ready", "/health/tracking"):
            response = client.get(path)
            assert response.status_code == 200
            assert response.json()["ready"] is True
            assert response.json()["gpu_ready"] is False
            assert response.json()["native_sam_video_tracking"] is False


def test_http_contract_for_tracks_and_labels():
    engine, detector = engine_for()
    with TestClient(create_app(engine)) as client:
        for sequence, path in enumerate(
            ("/v1/tracks?include_attributes=false", "/v1/labels")
        ):
            response = client.post(
                path, content=jpeg(), headers=headers(sequence, sequence / 10)
            )
            assert response.status_code == 200
            snapshot = RichLabelSnapshot.from_dict(response.json())
            assert snapshot.frame_sequence == sequence
            assert snapshot.stream_session_id == "camera-a"
    assert detector.closed


@pytest.mark.parametrize(
    "mutations,expected",
    [
        ({SESSION_HEADER: ""}, 400),
        ({SESSION_HEADER: "x" * 161}, 400),
        ({SEQUENCE_HEADER: "-1"}, 400),
        ({SEQUENCE_HEADER: "one"}, 400),
        ({TIMESTAMP_HEADER: "NaN"}, 400),
        ({TIMESTAMP_HEADER: "inf"}, 400),
        ({"Content-Type": "application/octet-stream"}, 415),
    ],
)
def test_http_rejects_invalid_source_metadata(mutations, expected):
    engine, detector = engine_for()
    with TestClient(create_app(engine)) as client:
        response = client.post(
            "/v1/tracks", content=jpeg(), headers=headers() | mutations
        )
    assert response.status_code == expected
    assert detector.calls == 0


@pytest.mark.parametrize(
    "payload,settings,expected",
    [
        (b"broken jpeg", {}, 400),
        (jpeg(format="PNG"), {}, 400),
        (jpeg(), {"max_jpeg_bytes": 20}, 413),
        (jpeg((101, 100)), {"max_image_pixels": 10000}, 413),
    ],
)
def test_http_rejects_bad_or_excessive_image_before_inference(
    payload, settings, expected
):
    engine, detector = engine_for(**settings)
    with TestClient(create_app(engine)) as client:
        response = client.post("/v1/tracks", content=payload, headers=headers())
    assert response.status_code == expected
    assert detector.calls == 0


def test_http_stale_frame_is_conflict():
    engine, _ = engine_for()
    with TestClient(create_app(engine)) as client:
        assert (
            client.post("/v1/tracks", content=jpeg(), headers=headers(5, 1)).status_code
            == 200
        )
        assert (
            client.post(
                "/v1/tracks", content=jpeg(), headers=headers(4, 0.9)
            ).status_code
            == 409
        )


def test_http_timeout_keeps_worker_busy_and_discards_late_result():
    started = threading.Event()
    release = threading.Event()

    class BlockingDetector(DetectorFixture):
        def detect(self, image):
            started.set()
            assert release.wait(2)
            return [detection()]

    engine = SceneEngine(
        replace(Settings(), request_timeout_seconds=0.05), detector=BlockingDetector([])
    )
    with TestClient(create_app(engine)) as client:
        try:
            with ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(
                    client.post, "/v1/tracks", content=jpeg(), headers=headers()
                )
                assert started.wait(1)
                assert future.result(timeout=1).status_code == 504
                assert (
                    client.post(
                        "/v1/tracks", content=jpeg(), headers=headers(1, 0.1)
                    ).status_code
                    == 429
                )
                release.set()
        finally:
            release.set()
    assert engine.sessions == {}


def test_checkpoint_must_exist_before_importing_heavy_model_runtime(tmp_path):
    detector = SamImageDetector(
        replace(Settings(), checkpoint_path=str(tmp_path / "missing.pt"))
    )
    with pytest.raises(RuntimeError, match="checkpoint missing"):
        detector.warmup()
    assert detector.gpu_ready is False


def test_production_startup_failure_remains_live_but_unready(monkeypatch, tmp_path):
    monkeypatch.setenv("SAM_SCENE_CHECKPOINT_PATH", str(tmp_path / "missing.pt"))
    with TestClient(create_app()) as client:
        assert client.get("/health/live").json() == {"live": True}
        response = client.get("/health/tracking")
        assert response.status_code == 503
        assert response.json()["gpu_ready"] is False
        assert (
            client.post("/v1/tracks", content=jpeg(), headers=headers()).status_code
            == 503
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"labels": ()},
        {"max_sessions": 0},
        {"request_timeout_seconds": float("nan")},
        {"confidence": 0},
    ],
)
def test_invalid_settings_fail_early(kwargs):
    with pytest.raises(ValueError):
        replace(Settings(), **kwargs)


def test_real_adapter_uses_official_image_api_and_strict_local_checkpoint(
    monkeypatch, tmp_path
):
    """API-contract doubles only: this does not execute SAM or CUDA."""
    checkpoint = tmp_path / "sam3.pt"
    checkpoint.write_bytes(b"fixture")
    calls = {}
    image_weights, checkpoint_weights = official_neck_checkpoint_fixture()

    class TensorFixture:
        def __init__(self, data):
            self.data = data

        def detach(self):
            return self

        def float(self):
            return self

        def cpu(self):
            return self

        def tolist(self):
            return self.data

    class ModelFixture:
        def state_dict(self):
            return image_weights

        def load_state_dict(self, state, *, strict):
            calls["state"] = state
            calls["strict"] = strict

        def to(self, *, device):
            calls["device"] = device
            return self

        def eval(self):
            return self

    class ProcessorFixture:
        def __init__(self, model, *, device, confidence_threshold):
            calls["processor"] = (device, confidence_threshold)

        def set_image(self, image):
            return {"size": image.size}

        def reset_all_prompts(self, state):
            calls["resets"] = calls.get("resets", 0) + 1

        def set_text_prompt(self, *, state, prompt):
            width, height = state["size"]
            calls.setdefault("prompts", []).append(prompt)
            return {
                "boxes": TensorFixture(
                    [[width * 0.1, height * 0.2, width * 0.4, height * 0.7]]
                ),
                "scores": TensorFixture([0.91]),
            }

    def build_model(**kwargs):
        calls["builder"] = kwargs
        return ModelFixture()

    def load_model(path, *, map_location, weights_only):
        calls["checkpoint"] = (path, map_location, weights_only)
        return {"model": checkpoint_weights}

    torch = ModuleType("torch")
    torch.cuda = SimpleNamespace(
        is_available=lambda: True,
        synchronize=lambda: calls.update(synchronized=True),
        empty_cache=lambda: None,
    )
    torch.load = load_model
    torch.inference_mode = nullcontext
    torch.autocast = lambda **kwargs: nullcontext()
    torch.bfloat16 = "bf16"
    sam3 = ModuleType("sam3")
    sam3_model = ModuleType("sam3.model")
    processor_module = ModuleType("sam3.model.sam3_image_processor")
    processor_module.Sam3Processor = ProcessorFixture
    builder_module = ModuleType("sam3.model_builder")
    builder_module.build_sam3_image_model = build_model
    for name, module in (
        ("torch", torch),
        ("sam3", sam3),
        ("sam3.model", sam3_model),
        ("sam3.model.sam3_image_processor", processor_module),
        ("sam3.model_builder", builder_module),
    ):
        monkeypatch.setitem(sys.modules, name, module)
    detector = SamImageDetector(
        replace(
            Settings(), checkpoint_path=str(checkpoint), labels=("person", "backpack")
        )
    )
    detector.warmup()
    assert calls["builder"] == {
        "device": "cpu",
        "eval_mode": True,
        "checkpoint_path": None,
        "load_from_HF": False,
        "enable_segmentation": True,
        "enable_inst_interactivity": False,
        "compile": False,
    }
    assert calls["state"] == image_weights
    assert calls["strict"] is True
    assert calls["checkpoint"] == (str(checkpoint), "cpu", True)
    assert calls["device"] == "cuda"
    assert calls["prompts"] == ["person", "backpack"]
    assert calls["synchronized"] is True
    assert detector.detect(Image.new("RGB", (1600, 900)))[0].bbox == pytest.approx(
        (0.1, 0.2, 0.4, 0.7)
    )


def official_neck_checkpoint_fixture():
    """Tensor-shape doubles from the pinned neck (1024 channels, d_model 256).

    No tensors or checkpoint weights are downloaded or allocated. The real
    video builder creates 22 extra sam2_convs tensors by deepcopy of this neck.
    """
    shapes = {
        "0.dconv_2x2_0.weight": (1024, 512, 2, 2),
        "0.dconv_2x2_0.bias": (512,),
        "0.dconv_2x2_1.weight": (512, 256, 2, 2),
        "0.dconv_2x2_1.bias": (256,),
        "1.dconv_2x2.weight": (1024, 512, 2, 2),
        "1.dconv_2x2.bias": (512,),
    }
    for index, channels in enumerate((256, 512, 1024, 1024)):
        shapes[f"{index}.conv_1x1.weight"] = (256, channels, 1, 1)
        shapes[f"{index}.conv_1x1.bias"] = (256,)
        shapes[f"{index}.conv_3x3.weight"] = (256, 256, 3, 3)
        shapes[f"{index}.conv_3x3.bias"] = (256,)
    expected = {
        f"backbone.visual.convs.{key}": SimpleNamespace(shape=shape)
        for key, shape in shapes.items()
    }
    # Representative required non-neck parameter; never an optional branch.
    expected["backbone.visual.trunk.pos_embed"] = SimpleNamespace(shape=(1, 577, 1024))
    checkpoint = {f"detector.{key}": value for key, value in expected.items()}
    checkpoint.update(
        {
            f"detector.backbone.visual.sam2_convs.{key}": SimpleNamespace(shape=shape)
            for key, shape in shapes.items()
        }
    )
    checkpoint["tracker.weight"] = SimpleNamespace(shape=(1,))
    return expected, checkpoint


def test_official_optional_neck_is_accepted_without_weakening_image_keys():
    expected, checkpoint = official_neck_checkpoint_fixture()
    assert len([key for key in checkpoint if ".sam2_convs." in key]) == 22
    assert _validated_image_detector_state(checkpoint, expected) == expected
    image_only = {f"detector.{key}": value for key, value in expected.items()}
    assert _validated_image_detector_state(image_only, expected) == expected


@pytest.mark.parametrize(
    ("mutation", "error"),
    [
        ("missing_image_neck", "missing required"),
        ("missing_image_trunk", "missing required"),
        ("wrong_image_shape", "incompatible"),
        ("wrong_optional_shape", "incompatible"),
        ("unknown_optional_key", "unexpected"),
        ("unknown_detector_key", "unexpected"),
    ],
)
def test_checkpoint_rejects_damage_and_unknown_extras(mutation, error):
    expected, checkpoint = official_neck_checkpoint_fixture()
    image_key = "detector.backbone.visual.convs.0.conv_1x1.weight"
    optional_key = "detector.backbone.visual.sam2_convs.0.conv_1x1.weight"
    if mutation == "missing_image_neck":
        del checkpoint[image_key]
    elif mutation == "missing_image_trunk":
        del checkpoint["detector.backbone.visual.trunk.pos_embed"]
    elif mutation == "wrong_image_shape":
        checkpoint[image_key] = SimpleNamespace(shape=(1,))
    elif mutation == "wrong_optional_shape":
        checkpoint[optional_key] = SimpleNamespace(shape=(1,))
    elif mutation == "unknown_optional_key":
        checkpoint["detector.backbone.visual.sam2_convs.9.conv_1x1.weight"] = (
            checkpoint[optional_key]
        )
    else:
        checkpoint["detector.unexpected.weight"] = SimpleNamespace(shape=(1,))
    with pytest.raises(RuntimeError, match=error):
        _validated_image_detector_state(checkpoint, expected)


def test_colliding_motion_predictions_stop_id_propagation():
    engine, _ = engine_for(
        [[detection(0.1), detection(0.7)], [detection(0.2), detection(0.6)]],
        keyframe_requests=1,
    )
    infer(engine)
    infer(engine, 1, 0.4)
    engine.settings = replace(
        engine.settings, keyframe_seconds=10, keyframe_requests=100
    )
    result = infer(engine, 2, 1.1)
    assert result["objects"] == []
    assert engine.sessions["camera-a"].last_keyframe_timestamp == -float("inf")
