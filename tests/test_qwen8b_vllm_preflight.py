"""Offline artifact, bundle, smoke and host-selection regressions; no real services."""

from __future__ import annotations

import base64
import io
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from tools import build_env, registry
from tools import start_qwen8b_vllm as launcher
from tools.qwen8b import eval_local
from tools.unified_benchmark.adapters import request_body
from tools.unified_benchmark.contract import RULES, VERSION, digest, file_hash
from tools.unified_benchmark.preflight import freeze_sources, model_configs
from tools.unified_benchmark.retry import Ledger, Queue

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def no_external_calls(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("real network/R2/model operation forbidden in this offline test")

    monkeypatch.setattr(build_env.model_source, "r2_client", forbidden)
    monkeypatch.setattr(eval_local.urllib.request, "urlopen", forbidden)
    monkeypatch.setattr(launcher, "restore_env", forbidden)
    monkeypatch.setattr(launcher, "prefetch_models", forbidden)
    monkeypatch.setattr(launcher.S, "create_from_offers", forbidden)


def seal(bundle):
    manifest = bundle / "bundle-files-sha256.json"
    manifest.write_text(
        json.dumps(
            {
                p.relative_to(bundle).as_posix(): file_hash(p)
                for p in bundle.rglob("*")
                if p.is_file() and p != manifest
            }
        )
    )


@pytest.fixture
def frozen(tmp_path):
    """Synthetic request bundle, not real reviewed exam inputs or model evidence."""
    bundle = tmp_path / "bundle"
    for name in ("frames", "configs", "plans"):
        (bundle / name).mkdir(parents=True)
    config = model_configs(["qwen8b_r2_vllm"])["qwen8b_r2_vllm"]
    (bundle / "configs/qwen8b_r2_vllm.json").write_text(json.dumps(config))
    frame = bundle / "frames/synthetic.jpg"
    Image.new("RGB", (720, 404), (100, 20, 40)).save(frame)
    for split in ("validation", "test"):
        unit = {
            "id": f"synthetic-{split}/0",
            "source_id": f"synthetic-{split}",
            "targets": ["fight"],
            "question": "Is fighting visible in the current image?",
            "frames": [
                {
                    "path": "frames/synthetic.jpg",
                    "sha256": file_hash(frame),
                    "time": 0.0,
                }
            ],
        }
        unit["request_sha256"] = digest(request_body(config, unit, bundle)[1])
        plan = {
            "protocol": VERSION,
            "split": split,
            "config_sha256": digest(config),
            "rules_sha256": digest(RULES),
            "units": [unit],
            "questions": [
                {
                    "number": 1,
                    "event": "fight",
                    "reference": "no",
                    "unit_ids": [unit["id"]],
                }
            ],
        }
        (bundle / f"plans/qwen8b_r2_vllm-{split}.json").write_text(json.dumps(plan))
    (bundle / "rules.json").write_text(json.dumps(RULES))
    sources = freeze_sources(bundle)
    (bundle / "preflight.json").write_text(
        json.dumps({"offline_passed": True, "sources_sha256": digest(sources),
        "models": {alias: {split: {"plan_sha256": digest(json.loads(
            (bundle / f"plans/{alias}-{split}.json").read_text()))}
            for split in ("validation", "test")} for alias in ('qwen8b_r2_vllm',)}})
    )
    seal(bundle)
    return bundle


def test_valid_bundle_is_checked_without_external_services(frozen):
    assert launcher._verify_eval_bundle(
        frozen, "both", registry.env_of("qwen8b-vllm")
    ) == file_hash(frozen / "bundle-files-sha256.json")


@pytest.mark.parametrize(
    "change",
    ["legacy", "key", "sha", "size", "tokens", "temperature", "weights", "engine"],
)
def test_even_resealed_stale_or_changed_profiles_fail(frozen, change):
    path = frozen / "configs/qwen8b_r2_vllm.json"
    cfg = json.loads(path.read_text())
    if change == "legacy":
        del cfg["env_artifact"]
        cfg["env_source"] = "envs/qwen8b-vllm/synthetic-wrong.tar.gz"
    elif change == "key":
        cfg["env_artifact"]["r2_key"] = "envs/qwen8b-vllm/synthetic-wrong.tar.gz"
    elif change == "sha":
        cfg["env_artifact"]["sha256"] = "0" * 64
    elif change == "size":
        cfg["env_artifact"]["size"] += 1
    else:
        field, value = {
            "tokens": ("max_tokens", 300),
            "temperature": ("temperature", 1),
            "weights": ("weight_revision", "0" * 40),
            "engine": ("engine", "transformers"),
        }[change]
        cfg[field] = value
    path.write_text(json.dumps(cfg))
    # Rebind the synthetic preflight to isolate the independent active-profile
    # gate. Otherwise the new plan/config binding rejects this earlier.
    report_path = frozen / "preflight.json"
    report = json.loads(report_path.read_text())
    for split in ("validation", "test"):
        plan_path = frozen / f"plans/qwen8b_r2_vllm-{split}.json"
        plan = json.loads(plan_path.read_text())
        plan["config_sha256"] = digest(cfg)
        plan_path.write_text(json.dumps(plan))
        report["models"]["qwen8b_r2_vllm"][split]["plan_sha256"] = digest(plan)
    report_path.write_text(json.dumps(report))
    seal(frozen)
    before = {
        p.relative_to(frozen).as_posix(): file_hash(p)
        for p in frozen.rglob("*")
        if p.is_file()
    }
    with pytest.raises(ValueError, match="profile differs"):
        launcher._verify_eval_bundle(frozen, "both", registry.env_of("qwen8b-vllm"))
    assert before == {
        p.relative_to(frozen).as_posix(): file_hash(p)
        for p in frozen.rglob("*")
        if p.is_file()
    }


@pytest.mark.parametrize(
    "change",
    ["request", "config_hash", "rules_hash", "split", "unsealed", "empty", "duplicate"],
)
def test_changed_request_plan_fails_before_sending(frozen, change):
    path = frozen / "plans/qwen8b_r2_vllm-validation.json"
    plan = json.loads(path.read_text())
    if change == "request":
        plan["units"][0]["question"] = "A different question"
    elif change == "config_hash":
        plan["config_sha256"] = "0" * 64
    elif change == "rules_hash":
        plan["rules_sha256"] = "0" * 64
    elif change == "split":
        plan["split"] = "test"
    elif change == "empty":
        plan["units"] = []
    elif change == "duplicate":
        plan["units"] *= 2
    path.write_text(json.dumps(plan))
    seal(frozen)
    if change == "unsealed":
        manifest = frozen / "bundle-files-sha256.json"
        hashes = json.loads(manifest.read_text())
        del hashes[path.relative_to(frozen).as_posix()]
        manifest.write_text(json.dumps(hashes))
    with pytest.raises(ValueError):
        launcher._verify_eval_bundle(frozen, "both", registry.env_of("qwen8b-vllm"))


def test_test_phase_needs_no_calibration_but_preserves_preflight_inventory(frozen):
    launcher._verify_eval_bundle(frozen, "test", registry.env_of("qwen8b-vllm"))
    (frozen / "plans/qwen8b_r2_vllm-validation.json").unlink()
    seal(frozen)
    with pytest.raises(ValueError, match="unsealed bundle plan"):
        launcher._verify_eval_bundle(frozen, "test", registry.env_of("qwen8b-vllm"))


def test_invalid_bundle_stops_main_before_credentials_or_rental(
    frozen, tmp_path, monkeypatch
):
    (frozen / "configs/qwen8b_r2_vllm.json").write_text("{}")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "launcher",
            "--eval-bundle",
            str(frozen),
            "--eval-output",
            str(tmp_path / "out"),
        ],
    )
    monkeypatch.setattr(
        launcher.S,
        "required",
        lambda *_: pytest.fail("credentials read before verification"),
    )
    with pytest.raises(ValueError, match="bundle file differs"):
        launcher.main()


def payload(content='{"answer":false}'):
    return {
        "model": launcher.MODEL_ID,
        "choices": [{"finish_reason": "stop", "message": {"content": content}}],
    }


def test_synthetic_smoke_never_consumes_scored_attempts(frozen, tmp_path, monkeypatch):
    cfg = json.loads((frozen / "configs/qwen8b_r2_vllm.json").read_text())
    unit = json.loads((frozen / "plans/qwen8b_r2_vllm-validation.json").read_text())[
        "units"
    ][0]
    requests = []

    class FakeOpener:
        def open(self, req, **kwargs):
            requests.append(json.loads(req.data))
            return io.BytesIO(
                json.dumps(
                    payload() if len(requests) == 1 else payload("not JSON")
                ).encode()
            )

    opener = FakeOpener()
    receipt = tmp_path / "smoke.json"
    launcher._smoke(opener, receipt)
    record = json.loads(receipt.read_text())
    assert record["state"] == "completed" and record["included_in_evaluation"] is False
    assert record["request_sha256"] == digest(requests[0])
    assert record["response"] == payload()
    image_uri = requests[0]["messages"][0]["content"][1]["image_url"]["url"]
    with Image.open(io.BytesIO(base64.b64decode(image_uri.split(",", 1)[1]))) as image:
        image.load()  # Force decoding the exact transmitted JPEG, not only its header.
        assert image.format == "JPEG" and image.size == (720, 404)
    assert requests[0]["max_tokens"] == 30 and requests[0]["temperature"] == 0
    monkeypatch.setattr(eval_local.urllib.request, "urlopen", opener.open)
    ledger = Ledger(tmp_path / "attempts.jsonl", {"synthetic": True})
    try:
        transport = eval_local.LocalTransport(
            frozen, cfg, "http://127.0.0.1:1", tmp_path / "run", ledger
        )
        now = [0.0]
        queue = Queue(
            ledger,
            "choice",
            transport,
            clock=lambda: now[0],
            sleep=lambda n: now.__setitem__(0, now[0] + n),
        )
        queue.one(unit)
        queue.one(unit)  # An exhausted budget cannot be reset by continuation.
        assert len(requests) == 4
        assert digest(requests[0]) != unit["request_sha256"]
        assert all(digest(body) == unit["request_sha256"] for body in requests[1:])
        assert len([r for r in ledger.rows if r["type"] == "intent"]) == 3
    finally:
        ledger.close()


def test_existing_smoke_receipt_prevents_another_call(tmp_path):
    receipt = tmp_path / "smoke.json"
    receipt.write_text("retained evidence")
    opener = SimpleNamespace(
        open=lambda *a, **k: pytest.fail("duplicate diagnostic was sent")
    )
    with pytest.raises(FileExistsError):
        launcher._smoke(opener, receipt)
    assert receipt.read_text() == "retained evidence"


@pytest.mark.parametrize(
    "response", [payload("not JSON"), {"choices": []}, payload() | {"model": "wrong"}]
)
def test_invalid_smoke_response_is_preserved_and_rejected(tmp_path, response):
    opener = SimpleNamespace(
        open=lambda *a, **k: io.BytesIO(json.dumps(response).encode())
    )
    receipt = tmp_path / "smoke.json"
    with pytest.raises(ValueError):
        launcher._smoke(opener, receipt)
    record = json.loads(receipt.read_text())
    assert record["state"] == "failed" and record["response"] == response


@pytest.mark.parametrize("raw", [b"not JSON", b'{"score":NaN}', b'{"score":Infinity}'])
def test_malformed_smoke_json_keeps_raw_failure_receipt(tmp_path, raw):
    opener = SimpleNamespace(open=lambda *a, **k: io.BytesIO(raw))
    receipt = tmp_path / "smoke.json"
    with pytest.raises(ValueError):
        launcher._smoke(opener, receipt)
    record = json.loads(receipt.read_text())
    assert record["state"] == "failed" and record["response_body"] == raw.decode()
