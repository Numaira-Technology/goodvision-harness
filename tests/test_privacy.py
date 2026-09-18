from __future__ import annotations

import io
import json
import threading
from types import SimpleNamespace
from urllib.error import HTTPError

import httpx
import numpy as np
import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

import goodvision_harness.config as config_module
import goodvision_harness.model_api as transport_module
import goodvision_harness.overlay as overlay_module
from goodvision_harness.aiclub import (
    AIClubClient,
    AIClubError,
    AIClubSettings,
)
from goodvision_harness.camera_display import CameraDisplayMetadata
from goodvision_harness.contracts import (
    PUBLIC_MODEL_NAME,
    AnalysisResult,
    EventJudgment,
    FrameReference,
    GroundedObject,
    NormalizedBBox,
)
from goodvision_harness.delivery import WebhookDelivery
from goodvision_harness.privacy import (
    PUBLIC_ANALYSIS_ERROR,
    PUBLIC_INSPECTION_ERROR,
    PUBLIC_TRACKING_ERROR,
    contains_private_implementation_detail,
    public_error_message,
    public_safe_value,
    redact_public_text,
)
from goodvision_harness.result_bus import ResultBus
from goodvision_harness.service import HarnessRuntime, create_app
from goodvision_harness.vision_result import VisionResult

PRIVATE_MARKERS = (
    "aiclub",
    "zimaocloud",
    "openrouter",
    "qwen",
    "coreweave",
    "deepinfra",
    "insightface",
    "buffalo_l",
    "onnxruntime",
    "cpuexecutionprovider",
    "dmlexecutionprovider",
    "paddle",
    "pp-yoloe",
    "pp-lcnet",
    "picodet",
    "nvidia",
    "opencv",
    "huggingface",
    "sam3", "sam_live", "segment anything", "reka", "ultralytics", "vllm",
)
PRIVATE_TEXT = (
    "AIClub https://aiclub.zimaocloud.com qwen3.5-plus "
    "OpenRouter https://openrouter.ai qwen/qwen3.5-35b-a3b CoreWeave DeepInfra "
    "InsightFace buffalo_l onnxruntime CPUExecutionProvider DmlExecutionProvider"
    " PaddleX PaddleRuntimeError PP-YOLOE_plus-L PP-LCNet_x1_0_pedestrian_attribute"
    " PicoDet-S NVIDIA OpenCV https://huggingface.co/example/model"
    " SAM3.1 sam_live Segment Anything 3.1 Reka Ultralytics vLLM"
)


def test_branch_failure_messages_are_fixed_allowlisted_text_not_exception_echoes():
    for message in (PUBLIC_TRACKING_ERROR, PUBLIC_INSPECTION_ERROR):
        assert public_error_message(message) == message
        assert public_safe_value({"last_error": message}) == {"last_error": message}
    private = "https://private.example/path?token=secret raw RemoteProtocolError"
    assert public_error_message(RuntimeError(private)) == PUBLIC_ANALYSIS_ERROR
    assert public_error_message(private) == PUBLIC_ANALYSIS_ERROR
    assert public_error_message(
        PUBLIC_TRACKING_ERROR + private,
    ) == PUBLIC_ANALYSIS_ERROR


def _assert_public(value: object) -> None:
    rendered = json.dumps(value, ensure_ascii=False, default=str).lower()
    assert all(marker not in rendered for marker in PRIVATE_MARKERS)


def test_recursive_public_projection_removes_implementation_markers() -> None:
    projected = public_safe_value(
        {"description": PRIVATE_TEXT, "nested": [PRIVATE_TEXT]}
    )

    _assert_public(projected)
    assert "analysis engine" in str(projected)


def test_serialized_result_records_never_emit_private_model_details() -> None:
    result = AnalysisResult(
        message_id="message-1",
        source_id="camera-1",
        stream_session_id="session-1",
        sequence=1,
        emitted_at="2026-08-28T00:00:00.000Z",
        frame=FrameReference(
            source_frame_index=1,
            timestamp_seconds=1.0,
            observed_at="2026-08-28T00:00:00.000Z",
            width=320,
            height=180,
        ),
        judgments=(
            EventJudgment(
                event_label="fight",
                detected=True,
                confidence=0.9,
                description=PRIVATE_TEXT,
                object_ids=("object-1",),
            ),
        ),
        objects=(
            GroundedObject(
                object_id="object-1",
                label=f"person {PRIVATE_TEXT}",
                bbox=NormalizedBBox(0.1, 0.1, 0.8, 0.8),
                attributes={"engine_note": PRIVATE_TEXT},
            ),
        ),
        model_name="qwen3.5-plus",
        model_latency_seconds=0.2,
    )
    vision = VisionResult(
        event_label="fight",
        detected=True,
        confidence=0.9,
        description=PRIVATE_TEXT,
        window_index=1,
        window_start_seconds=0.0,
        window_end_seconds=1.0,
    )

    payload = result.as_dict()
    assert payload["model"]["name"] == PUBLIC_MODEL_NAME
    _assert_public(payload)
    _assert_public(vision.as_dict())


def test_overlay_redacts_description_label_and_enrolled_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[str] = []

    def capture(_image: object, text: str, *_args: object, **_kwargs: object) -> None:
        captured.append(text)

    monkeypatch.setattr(overlay_module, "_text_with_background", capture)
    image = np.zeros((100, 160, 3), dtype=np.uint8)
    bbox = NormalizedBBox(0.1, 0.1, 0.8, 0.8)

    overlay_module.draw_overlay(
        image,
        event_label="fight",
        detected=True,
        confidence=0.9,
        description=PRIVATE_TEXT,
        objects=(SimpleNamespace(label=PRIVATE_TEXT, bbox=bbox, confidence=0.8),),
    )
    overlay_module.draw_overlay(
        image,
        event_label="unknown_person",
        detected=False,
        confidence=0.9,
        description=PRIVATE_TEXT,
        objects=(
            SimpleNamespace(
                label="known_person",
                bbox=bbox,
                confidence=0.8,
                attributes={"status": "known", "name": PRIVATE_TEXT},
            ),
        ),
        analysis_kind="face_recognition",
    )

    _assert_public(captured)


def test_provider_error_body_and_network_exception_are_not_echoed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    error = HTTPError(
        "https://aiclub.zimaocloud.com/model/openApi/v1/chat/completions",
        500,
        "failed",
        {},
        io.BytesIO(PRIVATE_TEXT.encode("utf-8")),
    )

    def fail(*_args: object, **_kwargs: object) -> object:
        raise error

    monkeypatch.setattr(transport_module, "urlopen", fail)
    client = AIClubClient(
        AIClubSettings(max_retries=0),
        api_key="not-a-real-key",
    )

    with pytest.raises(AIClubError) as raised:
        client.chat_completion(model="private-model", messages=[{"role": "user"}])

    assert str(raised.value) == "model API returned HTTP 500"
    _assert_public(str(raised.value))
    _assert_public(redact_public_text(PRIVATE_TEXT))


@pytest.mark.parametrize("marker", [
    "Paddle", "PaddleX", "paddlepaddle-gpu", "PaddleDetection",
    "PaddleRuntimeError", "PP-YOLOE_plus-L", "PP-LCNet_x1_0_pedestrian_attribute",
    "PicoDet-S", "InsightFaceError", "OpenRouterError", "AIClubError",
    "qwen3.5-plus", "使用Qwen模型", "NVIDIA", "CUDAExecutionProvider", "cuDNN",
    "TensorRT", "OpenCV", "FFmpeg", "FastAPI", "Uvicorn", "Swagger", "ReDoc",
    "OpenAI", "Anthropic", "DeepSeek", "HuggingFace", "ModelScope", "ONNXRuntimeError",
    "https://paddle-model-ecology.bj.bcebos.com/model.tar",
    "https://huggingface.co/team/model/file", "飞桨", "通义千问", "英伟达",
    "SAM", "SAM 3.1", "sam3.1", "sam31_stream", "sam_live", "sam_scene",
    "SamRuntimeError", "Segment Anything 3.1", "segment_anything",
    "facebook/sam3.1", "facebookresearch/sam3", "Parasail", "Novita", "Chutes",
    "https://github.com/facebookresearch/sam3", "Reka", "vLLM", "Ultralytics",
])
def test_all_known_runtime_markers_are_removed(marker: str) -> None:
    assert contains_private_implementation_detail(marker)
    result = redact_public_text(marker)
    assert not contains_private_implementation_detail(result)
    assert marker not in result


def test_nested_metadata_and_keys_are_safe() -> None:
    source = {"PaddleRuntimeError": [{"engine": "PaddleX", "model": "PP-YOLOE_plus-L"}]}
    result = public_safe_value(source)
    assert not contains_private_implementation_detail(json.dumps(result))
    assert source["PaddleRuntimeError"][0]["engine"] == "PaddleX"


def test_product_contract_and_common_video_words_are_unchanged() -> None:
    text = (
        "numaira-beta-fast person bottle track_id frame_sequence "
        "JPEG GPU CPU HTTP JSON OpenAPI"
    )
    assert redact_public_text(text) == text


def test_validation_keeps_error_contract_without_echoing_input() -> None:
    client = TestClient(create_app())
    response = client.get("/v1/results/events", params={"after": PRIVATE_TEXT})
    assert response.status_code == 422
    error = response.json()["detail"][0]
    assert error["loc"] == ["query", "after"]
    assert error["type"] == "int_parsing"
    assert error["input"] == "[redacted]"
    assert isinstance(error["msg"], str)
    _assert_public(response.json())


def test_http_exception_detail_is_projected_and_status_and_headers_remain() -> None:
    app = create_app()

    @app.get("/audit-error")
    async def audit_error() -> None:
        raise HTTPException(
            409, detail={"reason": PRIVATE_TEXT}, headers={"Retry-After": "5"}
        )

    response = TestClient(app).get("/audit-error")
    assert response.status_code == 409
    assert response.headers["retry-after"] == "5"
    assert "reason" in response.json()["detail"]
    _assert_public(response.json())


def test_result_bus_projects_public_records_without_mutating_caller() -> None:
    payload = {
        "message_id": "observation-1",
        "description": PRIVATE_TEXT,
        "nested": [PRIVATE_TEXT],
    }
    bus = ResultBus()
    published = bus.publish(payload)
    _assert_public(published.payload)
    _assert_public(bus.latest().payload)
    _assert_public(bus.records_after(0)[0].payload)
    assert payload["description"] == PRIVATE_TEXT


def test_webhook_projects_payload_and_preserves_matching_idempotency() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(204)

    delivery = WebhookDelivery(
        "http://receiver.test/events", transport=httpx.MockTransport(handler)
    )
    message_ids = ["Qwen-event-1", "Qwen-event-2"]
    delivery.start()
    try:
        for message_id in message_ids:
            assert delivery.submit({
                "message_id": message_id, "description": PRIVATE_TEXT,
            })
        assert delivery.flush(timeout_seconds=2)
    finally:
        delivery.close()
    assert len(requests) == 2
    for request, message_id in zip(requests, message_ids, strict=True):
        payload = json.loads(request.content)
        assert request.headers["idempotency-key"] == payload["message_id"] == message_id
        _assert_public(payload["description"])
    assert delivery.stats()["delivered"] == 2


def test_public_projection_preserves_business_identifiers_and_reference_lists() -> None:
    identifiers = {
        "message_id": "Paddle-event-1",
        "observation_id": "Paddle-event-1",
        "source_id": "azure-camera-1",
        "stream_session_id": "Qwen-session-1",
        "object_id": "NVIDIA-object-1",
        "track_id": "OpenRouter-track-1",
        "person_id": "claude",
        "stranger_id": "Paddle-stranger-1",
        "memory_id": "Qwen-memory-1",
        "analysis_message_id": "Paddle-event-1",
        "next_analysis_message_id": "Paddle-event-2",
        "association_message_id": "Paddle-event-1",
    }
    source = {
        **identifiers,
        "attributes": {"person_id": "claude", "engine": PRIVATE_TEXT},
        "model": {"id": "Qwen", "name": PRIVATE_TEXT},
        "description": PRIVATE_TEXT,
    }
    projected = public_safe_value(source)
    assert all(projected[key] == value for key, value in identifiers.items())
    assert projected["attributes"]["person_id"] == "claude"
    _assert_public(projected["attributes"]["engine"])
    _assert_public(projected["model"])
    _assert_public(projected["description"])
    for field in ("object_ids", "memory_ids", "matched_memory_ids"):
        for values in (["Paddle-1", "Paddle-2"], ("Paddle-1", "Paddle-2")):
            result = public_safe_value({field: values})[field]
            assert result == values
            assert type(result) is type(values)
            if isinstance(values, list):
                assert result is not values


def test_identifier_fields_do_not_bypass_nested_metadata_redaction() -> None:
    invalid_identifiers = {
        "person_id": {"engine": PRIVATE_TEXT},
        "object_ids": [{"provider": PRIVATE_TEXT}, PRIVATE_TEXT],
    }
    _assert_public(public_safe_value(invalid_identifiers))


def test_face_gallery_business_identifiers_round_trip_through_http() -> None:
    people: dict[str, SimpleNamespace] = {}

    def enroll(person_id: str, name: str, _image: bytes) -> SimpleNamespace:
        person = SimpleNamespace(person_id=person_id, name=name, sample_count=1)
        people[person_id] = person
        return person

    face = SimpleNamespace(
        enroll=enroll,
        list_people=lambda: tuple(people.values()),
        delete_person=lambda person_id: people.pop(person_id, None) is not None,
    )
    # Exercise the real public endpoints with a memory-only gallery adapter;
    # construction deliberately avoids model setup, environment, and storage.
    runtime = object.__new__(HarnessRuntime)
    runtime._state_lock = threading.RLock()
    runtime._identity_control_lock = threading.RLock()
    runtime._run_postprocess = {}
    runtime._camera_display_metadata = CameraDisplayMetadata()
    runtime._scene = None
    runtime._session = None
    runtime._ensure_face_runtime = lambda: face
    runtime._face_runtime = face
    app = create_app()
    app.state.runtime = runtime
    client = TestClient(app)
    identifiers = ("claude", "claude-2")
    for person_id in identifiers:
        response = client.post(
            f"/v1/faces/gallery/{person_id}",
            params={"name": "Staff member"},
            content=b"memory-only adapter fixture",
        )
        assert response.status_code == 200
        assert response.json()["person"]["person_id"] == person_id
    gallery = client.get("/v1/faces/gallery").json()
    assert {person["person_id"] for person in gallery["people"]} == set(identifiers)
    for person in gallery["people"]:
        response = client.delete(f"/v1/faces/gallery/{person['person_id']}")
        assert response.status_code == 200
        assert response.json() == {"deleted": True, "person_id": person["person_id"]}
    assert client.get("/v1/faces/gallery").json()["people"] == []


def test_public_result_bus_keeps_distinct_source_identifiers() -> None:
    bus = ResultBus()
    source_ids = ("azure-camera-1", "azure-camera-2")
    for index, source_id in enumerate(source_ids):
        bus.publish({
            "message_id": f"Paddle-event-{index}",
            "source_id": source_id,
            "description": PRIVATE_TEXT,
        })
    records = bus.records_after(0)
    assert tuple(record.payload["source_id"] for record in records) == source_ids
    assert len({record.payload["message_id"] for record in records}) == 2
    for record in records:
        _assert_public(record.payload["description"])


DOMAIN_NAMES = ("Claude", "Gemini", "Google", "Amazon", "Timm", "Azure", "Sam")


@pytest.fixture(autouse=True)
def isolated_privacy_configuration(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config_module, "find_dotenv", lambda **_: "")
    monkeypatch.setenv("HARNESS_AUTOSTART", "false")
    monkeypatch.setenv("RICH_LABEL_ENABLED", "false")
    monkeypatch.delenv("RESULT_WEBHOOK_URL", raising=False)


@pytest.mark.parametrize("value", [
    *DOMAIN_NAMES, "torch", "torches", "torchbearer", "pandas", "buffalo", "antelope",
    "Claude-user", "Gemini-person", "GoogleSmith", "Amazonia",
    "Claude carries a torch near Amazon's entrance.",
])
def test_ordinary_names_objects_and_ids_are_preserved(value: str) -> None:
    assert redact_public_text(value) == value
    assert public_safe_value({value: [value]}) == {value: [value]}
    # Logs and release audits remain strict; data projection has more context.
    assert contains_private_implementation_detail(value)


@pytest.mark.parametrize("value", [
    "torch.nn.Module", "torchvision", "TorchRuntimeError", "torch-2.5.1",
    "pandas.DataFrame", "timm.create_model", "import torch", "torch backend",
    "claude-3.5-sonnet", "claude-sonnet-4", "gemini-2.5-flash", "Gemini model",
    "GOOGLE_API_KEY", "AWS_SDK", "AmazonClientError", "Google Cloud",
    "https://generativelanguage.googleapis.com/v1/models",
    "https://api.anthropic.com/v1/messages", "https://bedrock.amazonaws.com",
    "buffalo_l", "antelopev2",
])
def test_actual_implementation_forms_stay_redacted(value: str) -> None:
    projected = redact_public_text(value)
    assert projected != value
    assert not contains_private_implementation_detail(projected)


def _domain_result() -> AnalysisResult:
    return AnalysisResult(
        message_id="Google",
        source_id="Amazon",
        stream_session_id="Gemini-person",
        sequence=1,
        emitted_at="2026-09-03T00:00:00Z",
        frame=FrameReference(1, 1.0, "2026-09-03T00:00:00Z", 320, 180),
        judgments=(EventJudgment(
            "weapon", True, 0.9, "Claude carries a torch.",
            object_ids=("torch-object",),
        ),),
        objects=(GroundedObject(
            "torch-object", "torch", NormalizedBBox(0.1, 0.1, 0.8, 0.8),
            confidence=0.8, attributes={"person_id": "Claude-user", "name": "Claude"},
        ),),
        model_name="qwen3.5-plus",
        model_latency_seconds=0.1,
    )


def test_serialization_sse_history_and_webhook_preserve_domain_values() -> None:
    payload = _domain_result().as_dict()
    assert payload["message_id"] == "Google"
    assert payload["source_id"] == "Amazon"
    assert payload["objects"][0]["label"] == "torch"
    assert payload["objects"][0]["attributes"]["name"] == "Claude"
    assert payload["judgments"][0]["object_ids"] == ["torch-object"]
    assert payload["judgments"][0]["description"] == "Claude carries a torch."
    assert payload["description"] == "weapon: Claude carries a torch."
    assert payload["model"]["name"] == PUBLIC_MODEL_NAME
    bus = ResultBus()
    assert bus.publish(payload).payload == payload
    assert bus.latest().payload == payload
    assert bus.records_after(0)[0].payload == payload

    calls: list[httpx.Request] = []

    def capture(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(204)

    delivery = WebhookDelivery(
        "http://receiver.test/events", transport=httpx.MockTransport(capture),
    )
    delivery.start()
    try:
        assert delivery.submit(payload)
        assert delivery.flush(timeout_seconds=2)
    finally:
        delivery.close()
    assert json.loads(calls[0].content) == payload
    assert calls[0].headers["idempotency-key"] == "Google"


@pytest.mark.parametrize("name", DOMAIN_NAMES)
def test_object_and_face_overlay_keep_domain_labels(
    monkeypatch: pytest.MonkeyPatch, name: str,
) -> None:
    labels: list[str] = []

    def capture(_image: object, text: str, *_args: object, **_kwargs: object) -> None:
        labels.append(text)

    monkeypatch.setattr(overlay_module, "_text_with_background", capture)
    image = np.zeros((100, 160, 3), dtype=np.uint8)
    bbox = NormalizedBBox(0.1, 0.1, 0.8, 0.8)
    overlay_module.draw_overlay(
        image, event_label="weapon", detected=True, confidence=0.9,
        description=f"{name} carries a torch.",
        objects=(SimpleNamespace(label="torch", bbox=bbox, confidence=0.8),),
    )
    overlay_module.draw_overlay(
        image, event_label="unknown_person", detected=False, confidence=0.9,
        description="Known person.", analysis_kind="face_recognition",
        objects=(SimpleNamespace(
            label="known_person", bbox=bbox, confidence=0.9,
            attributes={"status": "known", "name": name},
        ),),
    )
    assert "torch 0.80" in labels
    assert f"{name} carries a torch." in labels
    assert name in labels


@pytest.mark.parametrize("field", [
    "engine", "provider", "backend", "model", "model_name",
    "execution_provider", "execution_providers",
])
def test_explicit_implementation_metadata_is_strict_but_business_values_are_kept(
    field: str,
) -> None:
    business = {
        "person_id": "Claude-user",
        "name": "Claude",
        "label": "torch",
        "description": "Claude carries a torch near Google's entrance.",
    }
    metadata = {
        "name": "Claude",
        "provider": "Google",
        "modules": ["torch", {"Amazon": "pandas"}],
    }
    source = {**business, field: metadata}
    projected = public_safe_value(source)

    assert all(projected[key] == value for key, value in business.items())
    assert not contains_private_implementation_detail(projected[field])
    assert source[field] == metadata
    assert public_safe_value({field: "Claude"}) == {field: "analysis engine"}


@pytest.mark.parametrize("text", [
    "Samantha", "sample_fps", "same frame", "samples", "Sam carries a bag.",
])
def test_tracking_redaction_preserves_unrelated_words(text):
    assert redact_public_text(text) == text


def test_tracking_metadata_has_neutral_aliases_and_keeps_correlation_ids():
    source = {
        "processing_method": "sam_live",
        "backend": "sam3.1_video",
        "native_sam_video_tracking": True,
        "model": "SAM 3.1",
        "person_id": "sam-person-1",
        "epoch": "sam-run-1",
        "name": "Sam",
    }
    result = public_safe_value(source)
    assert result == {
        "processing_method": "person_tracking",
        "backend": "person_tracking_video",
        "native_person_tracking": True,
        "model": "人员追踪模块",
        "person_id": "sam-person-1",
        "epoch": "sam-run-1",
        "name": "Sam",
    }
    assert public_safe_value(result) == result
    assert source["processing_method"] == "sam_live"


def test_camera_display_http_and_socket_metadata_are_projected_without_mutation():
    metadata = CameraDisplayMetadata()
    metadata.reset("sam-session-1")
    metadata.generation = 1
    metadata.identities = [{
        "person_id": "sam-person-1", "epoch": "sam-epoch-1",
        "name": PRIVATE_TEXT,
    }, {"person_id": "sam-person-2", "name": "Sam"}]
    payloads = (
        metadata.snapshot(), metadata.delta(None, stream_session_id="sam-session-1"),
    )
    for payload in payloads:
        assert payload["stream_session_id"] == "sam-session-1"
        assert payload["identities"][0]["person_id"] == "sam-person-1"
        assert payload["identities"][0]["epoch"] == "sam-epoch-1"
        _assert_public(payload["identities"][0]["name"])
        assert payload["identities"][1]["name"] == "Sam"
    assert metadata.identities[0]["name"] == PRIVATE_TEXT
