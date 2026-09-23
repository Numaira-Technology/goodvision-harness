from __future__ import annotations

import asyncio
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import httpx

SPEC = importlib.util.spec_from_file_location(
    "jev_benchmark",
    Path(__file__).parents[1] / "tools" / "vlm_jev_benchmark.py",
)
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


class Inspector:
    def build_messages(self, job):
        return [
            {"role": "system", "content": "json"},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/jpeg;base64,AA=="},
                    }
                ],
            },
        ]

    _parse = staticmethod(module.IndependentVLMInspector._parse)


def test_real_contract_path_sends_only_text_to_jev_and_keeps_keys_separate():
    calls = []

    def handler(request):
        body = json.loads(request.content)
        calls.append(request.url.host)
        if request.url.host == "openrouter.ai":
            assert request.headers["authorization"] == "Bearer vlm-secret"
            assert "response_format" not in body
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "finish_reason": "stop",
                            "message": {"content": "p1 at (10,20,300,900): standing."},
                        }
                    ],
                    "usage": {"completion_tokens": 20},
                    "provider": "fixture",
                },
            )
        assert request.url.host == "api.typesafe.ai"
        assert request.headers["authorization"] == "Bearer jev-secret"
        assert set(body) == {"model", "state", "questions"}
        assert "data:image" not in json.dumps(body)
        return httpx.Response(
            200,
            json={
                "model": "fixture",
                "answers": {
                    key: {
                        "type": "choice",
                        "choice": "uncertain",
                        "confidence": 0.8,
                        "probabilities": {
                            "present": 0.1,
                            "absent": 0.0,
                            "uncertain": 0.9,
                        },
                    }
                    for key in body["questions"]
                },
            },
        )

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            job = SimpleNamespace(
                request=SimpleNamespace(end_seconds=4, event_labels=module.LABELS),
                event_sensitivities=(),
            )
            args = SimpleNamespace(
                model="fixture", max_tokens=1600, description_style="brief"
            )
            return await module.trial(
                client, Inspector(), job, "jev", ("vlm-secret", "jev-secret"), args
            )

    result = asyncio.run(run())
    assert calls == ["openrouter.ai", "api.typesafe.ai"]
    assert result["ok"] and result["complete_labels"]
    assert all(value is None for value in result["decisions"].values())
    assert "secret" not in json.dumps(result)
    assert result["total_seconds"] >= result["vlm_seconds"] + result["jev_seconds"]


def test_failed_trials_stay_in_denominator_and_warmup_is_excluded():
    rows = [
        {"mode": "json", "warmup": True, "ok": False, "total_seconds": 100},
        {"mode": "json", "warmup": False, "ok": False, "total_seconds": 2},
    ]
    result = module.summarize(rows)
    assert result["json"]["attempts"] == 1
    assert result["json"]["successes"] == 0
    assert result["json"]["median_seconds"] is None
    assert result["json"]["all_attempt_median_seconds"] == 2
    assert result["pairs"] == []
