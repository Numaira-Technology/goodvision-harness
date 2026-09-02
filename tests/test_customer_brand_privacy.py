from __future__ import annotations

import json

import pytest

from goodvision_harness.privacy import (
    contains_private_implementation_detail,
    public_safe_value,
    redact_public_text,
)


@pytest.mark.parametrize("marker", [
    "Paddle", "PaddleX", "paddlepaddle-gpu", "PaddleDetection",
    "PaddleRuntimeError", "PP-YOLOE_plus-L", "PP-LCNet_x1_0_pedestrian_attribute",
    "PicoDet-S", "InsightFaceError", "OpenRouterError", "AIClubError",
    "qwen3.5-plus", "使用Qwen模型", "NVIDIA", "CUDAExecutionProvider", "cuDNN",
    "TensorRT", "OpenCV", "FFmpeg", "FastAPI", "Uvicorn", "Swagger", "ReDoc",
    "OpenAI", "Anthropic", "DeepSeek", "HuggingFace", "ModelScope", "ONNXRuntimeError",
    "https://paddle-model-ecology.bj.bcebos.com/model.tar",
    "https://huggingface.co/team/model/file", "飞桨", "通义千问", "英伟达",
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
    text = "numaira-beta-fast person bottle track_id frame_sequence JPEG GPU CPU HTTP JSON OpenAPI"
    assert redact_public_text(text) == text
