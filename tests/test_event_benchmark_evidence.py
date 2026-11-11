"""Controlled-provider evidence and calibration regressions; synthetic artifacts only."""
import copy
import hashlib
import io
import json
from pathlib import Path

import pytest

from tools.event_benchmark import evidence, tune_threshold

MODELS = [
    ("language", "qwen/qwen3-vl-8b-instruct"),
    ("language", "google/gemini-2.5-flash"),
    ("decision", "openai/gpt-6-luna-decisions"),
]


@pytest.fixture(autouse=True)
def forbid_network(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("evidence regressions must not contact a real endpoint")
    monkeypatch.setattr("urllib.request.urlopen", forbidden)
    monkeypatch.setattr("socket.create_connection", forbidden)


def context(model, backend="decision"):
    return {"sha256": "a" * 64, "config": {
        "schema_version": 1, "model": model, "backend": backend,
        "expected_provider_name": "Synthetic Provider",
        "provider": {"only": ["synthetic-endpoint"], "allow_fallbacks": False},
        "selection": "synthetic fixture", "selected_at": "2026-11-10T00:00:00Z",
    }}


def receipt(model):
    return {"ok": True, "score": .8, "generation_id": "synthetic-generation",
            "actual_provider": "Synthetic Provider", "actual_model": model}


@pytest.mark.parametrize("backend,model", MODELS)
def test_exact_model_preserves_existing_summary_schema_and_values(backend, model):
    assert evidence.provider_summary([receipt(model)], context(model, backend)) == {
        "actual_providers": ["Synthetic Provider"], "actual_models": [model],
        "successful_attempts": 1, "successful_attempts_missing_metadata": 0,
        "provider_mismatch": False, "complete": True,
        "actual_endpoint_verified": False, "actual_precision": None,
        "server_weight_revision": None,
        "limits": "Actual provider/model come from receipts. Endpoint, precision and weight revision are not independently attested.",
    }


@pytest.mark.parametrize("backend,model", MODELS)
@pytest.mark.parametrize("alteration", ["unrelated", "case", "whitespace", "bare", "suffix"])
def test_model_names_are_not_implicitly_normalized(backend, model, alteration):
    actual = {
        "unrelated": "other/unrelated-model",
        "case": model.upper(),
        "whitespace": model + " ",
        "bare": model.split("/", 1)[1],
        "suffix": model + "-unverified-version",
    }[alteration]
    observed = evidence.provider_summary([receipt(actual)], context(model, backend))
    assert observed["actual_models"] == [actual]  # retain the returned identity
    assert observed["provider_mismatch"] is False
    assert observed["complete"] is False


def test_mixed_attempts_cannot_hide_a_model_change_after_retry():
    model = MODELS[-1][1]
    attempts = [dict(receipt("other/model"), ok=False), receipt(model)]
    original = copy.deepcopy(attempts)
    assert evidence.provider_summary(attempts, context(model))["complete"] is False
    assert attempts == original


@pytest.mark.parametrize("missing", ["model", "generation_id", "provider", "frozen_model", "context"])
def test_missing_identity_remains_incomplete(missing):
    model = MODELS[-1][1]
    attempt, frozen = receipt(model), context(model)
    if missing == "frozen_model":
        frozen["config"].pop("model")
    elif missing == "context":
        frozen = None
    else:
        attempt.pop({"model": "actual_model", "provider": "actual_provider"}.get(missing, missing))
    assert evidence.provider_summary([attempt], frozen)["complete"] is False


@pytest.mark.parametrize("matching", [True, False])
def test_generation_lookup_cannot_attest_a_different_model(monkeypatch, matching):
    model = MODELS[-1][1]
    returned = model if matching else "other/model"
    attempt = {"ok": True, "generation_id": "synthetic-generation", "score": .8}
    calls = []

    def respond(request, **kwargs):
        calls.append(request)
        assert request.get_method() == "GET"
        return io.BytesIO(json.dumps({"data": {
            "id": attempt["generation_id"], "provider_name": "Synthetic Provider", "model": returned,
        }}).encode())

    monkeypatch.setattr(evidence, "_metadata_open", respond)
    evidence.enrich_attempts([attempt], "synthetic-key")
    assert len(calls) == 1
    assert attempt["actual_model"] == returned
    assert evidence.provider_summary([attempt], context(model))["complete"] is matching


def controlled_run(tmp_path, prefix, actual_model):
    """Build complete validation24-shaped evidence without media or model access."""
    model = MODELS[-1][1]
    rows, members = [], []
    for event in tune_threshold.EVENTS:
        for index in range(6):
            number = len(rows) + 1
            filename = f"{prefix}-{event}-{index}.mp4"
            sha = hashlib.sha256(filename.encode()).hexdigest()
            positive = index >= 3
            gt, score = ("yes", .8) if positive else ("no", .2)
            attempt = dict(receipt(actual_model), score=score, generation_id=f"{prefix}-{number}")
            rows.append({
                "number": number, "filename": filename, "sha256": sha, "event": event,
                "gt": gt, "score": score, "pred": gt, "correct": True,
                "frames_planned": 1, "frames_ok": 1, "frames_failed": 0,
                "frames": [{"score": score, "attempts": [attempt]}],
            })
            members.append({
                "number": number, "filename": filename, "sha256": sha, "event": event,
                "proposed_answer": gt, "question": f"Is {event} visible?",
                "human_review": "user_reviewed", "source_video_group": filename,
            })
    manifest_path = tmp_path / (prefix + "-manifest.json")
    manifest_path.write_text(json.dumps({
        "status": "human_reviewed", "samples_json_sha256": "b" * 64, "files": members,
    }))
    attempts = [frame["attempts"][0] for row in rows for frame in row["frames"]]
    frozen = context(model)
    observed = evidence.provider_summary(attempts, frozen)
    # Model a persisted pre-fix run which claimed completion for wrong-model receipts.
    observed["complete"] = True
    metadata = {
        "spec_version": "v2", "score_precision": "lossless", "backend": "decision", "model": model,
        "samples_json_sha256": "b" * 64,
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "runner_sha256": "c" * 64, "spec_sha256": "d" * 64,
        "evidence_sha256": hashlib.sha256(Path(evidence.__file__).read_bytes()).hexdigest(),
        "config": {"gate": .5, "interval": 1.0, "frame_rule": "n=max(1,round(dur/interval))",
                   "frame_width": 720, "temperature": 0, "retries": 3, "timeout": 90,
                   "decision": "max-frame-score >= gate"},
        "overall": {"attempted": 24, "total": 24, "errors": 0},
        "protocol_variant": evidence.VARIANT, "provider_context": frozen,
        "provider_observed": observed, "per_video": rows,
    }
    path = tmp_path / (prefix + "-run.json")
    path.write_text(json.dumps(metadata))
    return path, manifest_path, metadata


@pytest.mark.parametrize("matching", [True, False])
def test_receipt_verifier_recomputes_frozen_model_match(tmp_path, matching):
    actual = MODELS[-1][1] if matching else "other/model"
    _path, _manifest, run = controlled_run(tmp_path, "verify", actual)
    if matching:
        evidence.verify_provider_evidence(run, run["per_video"])
    else:
        with pytest.raises(ValueError, match="actual provider/model evidence"):
            evidence.verify_provider_evidence(run, run["per_video"])


@pytest.mark.parametrize("matching", [True, False])
def test_threshold_selection_requires_requested_model(tmp_path, matching):
    actual = MODELS[-1][1] if matching else "other/model"
    path, manifest, _run = controlled_run(tmp_path, "validation", actual)
    if matching:
        selected = tune_threshold.select_thresholds(path, manifest)
        assert set(selected["thresholds"]) == set(tune_threshold.EVENTS)
        assert selected["val_counts"] == {"attempted": 24, "total": 24, "errors": 0}
    else:
        with pytest.raises(ValueError, match="actual provider/model evidence"):
            tune_threshold.select_thresholds(path, manifest)


@pytest.mark.parametrize("matching", [True, False])
def test_threshold_evaluation_requires_requested_model(tmp_path, matching):
    model = MODELS[-1][1]
    val_path, val_manifest, _val = controlled_run(tmp_path, "validation", model)
    artifact = tune_threshold.select_thresholds(val_path, val_manifest)
    if not matching:
        # A pre-fix selection could claim the requested model while recording
        # a different one. Matching that old observed model is not sufficient.
        artifact["validation_run"]["provider_observed"]["actual_models"] = ["other/model"]
    artifact_path = tmp_path / "thresholds.json"
    artifact_hash = tune_threshold.write_new_json(artifact_path, artifact)
    test_path, test_manifest, _test = controlled_run(tmp_path, "test", model if matching else "other/model")
    if matching:
        result = tune_threshold.evaluate_thresholds(
            artifact_path, artifact_hash, [test_path], [test_manifest])
        assert len(result["test"]) == 1
    else:
        with pytest.raises(ValueError, match="actual provider/model evidence"):
            tune_threshold.evaluate_thresholds(
                artifact_path, artifact_hash, [test_path], [test_manifest])


@pytest.mark.parametrize("code", [301, 302, 303, 307, 308])
@pytest.mark.parametrize(
    "destination",
    ["https://synthetic.invalid/redirect", "http://synthetic.invalid/redirect"],
)
def test_metadata_redirect_never_forwards_bearer(monkeypatch, code, destination):
    import urllib.response
    from email.message import Message

    requests = []
    build = evidence.urllib.request.build_opener

    class MemoryTransport(evidence.urllib.request.BaseHandler):
        handler_order = 100

        def https_open(self, request):
            requests.append(request)
            assert len(requests) == 1  # default redirect machinery must stop here
            headers = Message()
            headers["Location"] = destination
            response = urllib.response.addinfourl(
                io.BytesIO(b""), headers, request.full_url, code
            )
            response.msg = "synthetic redirect"
            return response

        http_open = https_open

    monkeypatch.setattr(
        evidence.urllib.request,
        "build_opener",
        lambda *handlers: build(
            evidence.urllib.request.ProxyHandler({}), MemoryTransport(), *handlers
        ),
    )
    attempt = {"generation_id": "synthetic/id?x=1", "ok": True}
    evidence.enrich_attempts([attempt], "SYNTHETIC-NOT-A-KEY")
    assert len(requests) == 1
    assert (
        requests[0].full_url
        == "https://openrouter.ai/api/v1/generation?id=synthetic%2Fid%3Fx%3D1"
    )
    assert attempt["metadata_lookup_error"] == "unavailable"
    assert "actual_provider" not in attempt and "actual_model" not in attempt


@pytest.mark.parametrize(
    "payload",
    [
        None,
        [],
        {},
        {"data": None},
        {"data": []},
        {"data": "bad"},
        {"data": {"id": "other"}},
    ],
)
def test_malformed_metadata_is_unavailable_and_cached(monkeypatch, payload):
    calls = []

    def response(request):
        calls.append(request)
        return io.BytesIO(json.dumps(payload).encode())

    monkeypatch.setattr(evidence, "_metadata_open", response)
    attempts = [{"generation_id": "synthetic"}, {"generation_id": "synthetic"}]
    evidence.enrich_attempts(attempts, "SYNTHETIC")
    assert len(calls) == 1
    assert all(a["metadata_lookup_error"] == "unavailable" for a in attempts)


@pytest.mark.parametrize(
    "bad", [10**1000, -(10**1000), float("inf"), float("nan"), True]
)
def test_unrepresentable_metadata_numbers_do_not_discard_receipt_identity(
    monkeypatch, bad
):
    data = {
        "id": "synthetic",
        "provider_name": "Synthetic Provider",
        "model": "model",
        "total_cost": bad,
        "native_tokens_prompt": bad,
        "native_tokens_completion": bad,
        "usage": {
            "cost": bad,
            "prompt_tokens": bad,
            "completion_tokens": 12,
            "total_tokens": 20,
            "input_tokens": 8,
            "output_tokens": 12,
        },
    }
    monkeypatch.setattr(
        evidence,
        "_metadata_open",
        lambda request: io.BytesIO(json.dumps({"data": data}).encode()),
    )
    attempt = {"generation_id": "synthetic"}
    evidence.enrich_attempts([attempt], "SYNTHETIC")
    expected = {
        "generation_id": "synthetic",
        "actual_provider": "Synthetic Provider",
        "actual_model": "model",
        "usage": {
            "completion_tokens": 12,
            "total_tokens": 20,
            "input_tokens": 8,
            "output_tokens": 12,
        },
    }
    assert attempt == expected
    assert evidence.response_metadata(data) == expected


def test_normal_metadata_numbers_remain_unchanged():
    values = {
        "usage": {"cost": 0.002, "prompt_tokens": 100},
        "total_cost": 0.002,
        "native_tokens_prompt": 100,
        "native_tokens_completion": 8,
    }
    assert evidence.response_metadata(values) == values
