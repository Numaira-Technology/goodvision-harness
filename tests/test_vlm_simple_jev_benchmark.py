from __future__ import annotations

import asyncio
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

SPEC = importlib.util.spec_from_file_location(
    "simple_benchmark",
    Path(__file__).parents[1] / "tools" / "vlm_simple_jev_benchmark.py",
)
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


class Inspector:
    def build_messages(self, job):
        return [
            {"role": "system", "content": "full JSON"},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/jpeg;base64,AA=="},
                    }
                    for _ in range(4)
                ],
            },
        ]

    _parse = staticmethod(module.IndependentVLMInspector._parse)


def job():
    return SimpleNamespace(
        request=SimpleNamespace(end_seconds=12.8, event_labels=module.LABELS)
    )


def response():
    return {
        "model": module.CLASSIFIER,
        "answers": {
            label: {
                "type": "choice",
                "choice": "uncertain",
                "confidence": 0.8,
                "probabilities": {"present": 0.1, "absent": 0.1, "uncertain": 0.8},
            }
            for label in module.LABELS
        },
        "usage": {"input_tokens": 100, "output_tokens": 4},
    }


def test_public_demo_receives_original_images_and_no_credentials_or_prose():
    def handler(request):
        assert request.url.host == "simple-jev-demo-api.featherless.ai"
        assert "authorization" not in request.headers
        body = json.loads(request.content)
        assert set(body) == {"model", "messages", "questions"}
        assert len(body["messages"][1]["content"]) == 4
        assert all(p["type"] == "image_url" for p in body["messages"][1]["content"])
        return httpx.Response(200, json=response())

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await module.trial(
                client, "simple_vision", Inspector(), job(), "private-key"
            )

    row = asyncio.run(run())
    assert row["ok"] and row["complete_labels"]
    assert not row["localization_complete"]
    assert row["scope"] == "scene_events_only"
    assert row["assembled_json"]["subjects"] == []
    assert all(v is None for v in row["decisions"].values())
    assert "private-key" not in json.dumps(row)
    assert "data:image" not in json.dumps(row)


def test_case_inputs_keep_the_same_four_images_and_disable_flash_reasoning():
    bodies = {case: module.payload(case, Inspector(), job()) for case in module.CASES}
    pictures = [body["messages"][1]["content"] for body in bodies.values()]
    assert all(value == pictures[0] for value in pictures)
    assert bodies["flash_full"]["reasoning"] == {"enabled": False}
    assert bodies["8b_full"]["provider"]["only"] == ["alibaba"]


def test_missing_event_answer_is_failure_not_negative():
    data = response()
    del data["answers"]["fall"]
    with pytest.raises(ValueError, match="coverage"):
        module.decode("simple_vision", data, Inspector(), job())


@pytest.mark.parametrize("bad", [1, "false", [], {}])
def test_scene_json_does_not_coerce_invalid_boolean(bad):
    text = {label: False for label in module.LABELS}
    text["fall"] = bad
    data = {
        "choices": [{"finish_reason": "stop", "message": {"content": json.dumps(text)}}]
    }
    with pytest.raises(ValueError, match="invalid_scene_json"):
        module.decode("8b_events", data, Inspector(), job())


def test_control_and_warmup_excluded_but_failed_trials_count():
    common = {
        "case": "simple_vision",
        "endpoint": 4.0,
        "scope": "scene_events_only",
        "ok": False,
    }
    rows = [
        {**common, "total_seconds": 100, "warmup": True},
        {**common, "total_seconds": 200, "control": True},
        {**common, "total_seconds": 2, "warmup": False, "control": False},
    ]
    summary = module.summarize(rows)["simple_vision"]
    assert summary["attempts"] == 1
    assert summary["successes"] == 0
    assert summary["median_seconds"] is None
    assert summary["all_attempt_median_seconds"] == 2
