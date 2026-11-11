"""Nonfinite decoded JSON stays diagnostic evidence, never a valid score."""

import io
import json
import math
import threading
from collections import Counter

import pytest
from test_native_workflow import FakeBackend, execute, frozen, immediate_queue
from test_stack_transport_regressions import unit as request_unit

from tools.qwen8b.eval_local import LocalTransport
from tools.unified_benchmark import api_run, native_run, workflow
from tools.unified_benchmark.adapters import D3_OPTIONS, parse_response
from tools.unified_benchmark.contract import canonical, digest, file_hash, write_new
from tools.unified_benchmark.native_deploy import profile
from tools.unified_benchmark.preflight import model_configs
from tools.unified_benchmark.receipts import JSON_TEXT, raw_fields, read_raw_field
from tools.unified_benchmark.retry import Ledger, PausedRun

__all__ = ["frozen"]
TOKENS = ["1e400", "-1e400", "NaN"]


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    FakeBackend.calls, FakeBackend.instances, FakeBackend.behavior = [], [], None

    def forbidden(*args, **kwargs):
        pytest.fail("nonfinite regression attempted real network/model IO")

    monkeypatch.setattr("socket.create_connection", forbidden)
    monkeypatch.setattr(native_run.NativeBackend, "start", forbidden)


def strict_read(path):
    def invalid(token):
        raise AssertionError(
            "nonfinite token escaped the diagnostic JSON string: " + token
        )

    return json.loads(path.read_text(), parse_constant=invalid)


@pytest.mark.parametrize("alias", ["decision3", "videojev", "clip"])
@pytest.mark.parametrize("token", TOKENS)
@pytest.mark.parametrize("field", ["score", "metadata", "timings"])
def test_native_nonfinite_receipt_retains_retry_and_resume(
    frozen, tmp_path, alias, token, field
):
    uid = "validation/fight/0/yes"
    bad = json.loads(token)

    def behavior(self, unit, body):
        if unit["id"] != uid or Counter(self.calls)[uid] != 1:
            return None
        raw = self.raw(unit, "yes")
        timings = {}
        if field == "timings":
            timings = {"optional_estimate": bad}
        elif field == "metadata":
            raw["optional_metadata"] = {"estimate": bad}
        elif alias == "decision3":
            raw["raw"]["answers"]["fight"]["probabilities"][D3_OPTIONS[0]] = bad
        elif alias == "videojev":
            raw["raw"][0]["probabilities"][0] = bad
        else:
            raw["fight"]["margin"] = bad
        return 200, raw, timings

    FakeBackend.behavior = behavior
    out = tmp_path / "run"
    execute(frozen, out, alias)
    expected_calls = 2 if field == "score" else 1
    assert Counter(FakeBackend.calls)[uid] == expected_calls
    receipt_path = out / "validation/receipts" / f"{digest(uid)}-1.json"
    receipt = strict_read(receipt_path)
    encoded = "timings" if field == "timings" else "response"
    assert receipt[encoded + "_encoding"] == JSON_TEXT
    decoded = read_raw_field(receipt, encoded)
    if field == "timings":
        actual = decoded["optional_estimate"]
    elif field == "metadata":
        actual = decoded["optional_metadata"]["estimate"]
    elif alias == "decision3":
        actual = decoded["raw"]["answers"]["fight"]["probabilities"][D3_OPTIONS[0]]
    elif alias == "videojev":
        actual = decoded["raw"][0]["probabilities"][0]
    else:
        actual = decoded["fight"]["margin"]
    assert math.isnan(actual) if math.isnan(bad) else actual == bad
    rows = [
        json.loads(line)
        for line in (out / "validation/attempts.jsonl").read_text().splitlines()
    ]
    first = next(row for row in rows if row["type"] == "result" and row["unit"] == uid)
    assert first["complete"] is (field != "score")
    assert first["metadata"]["receipt_sha256"] == file_hash(receipt_path)
    before = receipt_path.read_bytes()
    assert json.loads((out / "test/report.json").read_text())["correct"] == 120
    execute(frozen, out, alias, resume=True)
    assert Counter(FakeBackend.calls)[uid] == expected_calls
    assert receipt_path.read_bytes() == before
    assert all(backend.closed for backend in FakeBackend.instances)


@pytest.mark.parametrize("field", ["score", "metadata"])
def test_encoded_orphan_receipt_replays_without_losing_first_answer(
    frozen, tmp_path, field
):
    bundle = frozen[0]
    cfg = profile("clip")
    plan = json.loads((bundle / "plans/clip-validation.json").read_text())
    unit = plan["units"][0]
    phase = tmp_path / "run/validation"
    ledger = Ledger(
        phase / "attempts.jsonl",
        workflow.context_for(plan, cfg, workflow.verify_bundle(bundle)),
    )
    ledger.append(
        {
            "unit": unit["id"],
            "attempt": 1,
            "type": "intent",
            "request_sha256": unit["request_sha256"],
        }
    )
    raw = {"fight": {"margin": 0.3, "positive_cosine": 0.5, "contrast_cosine": 0.2}}
    if field == "score":
        raw["fight"]["margin"] = json.loads("1e400")
    else:
        raw["diagnostic"] = json.loads("1e400")
    parsed = parse_response(cfg, unit, raw)
    path = phase / "receipts" / f"{digest(unit['id'])}-1.json"
    write_new(
        path,
        {
            "unit": unit["id"],
            "attempt": 1,
            "request_sha256": unit["request_sha256"],
            "config_sha256": digest(cfg),
            "frame_sha256": [f["sha256"] for f in unit["frames"]],
            "http_status": 200,
            "error": None,
            "parsed": parsed,
            "transport_seconds": 0.1,
            **raw_fields(response=raw),
        },
    )
    before = path.read_bytes()
    try:
        native_run.reconcile_receipts(phase, ledger, plan, cfg, phase.parent)
        assert ledger.state(unit, cfg["kind"]) == (1, parsed)
        calls = []
        queue = immediate_queue(
            ledger,
            cfg["kind"],
            lambda u: calls.append(u) or ({"fight": {"score": 0.3}}, {}),
        )
        queue.one(unit)
        assert len(calls) == (1 if field == "score" else 0)
        assert path.read_bytes() == before
    finally:
        ledger.close()


@pytest.mark.parametrize(
    "kind",
    ["api_score", "api_usage", "api_generation", "local_usage", "local_bad_answer"],
)
@pytest.mark.parametrize("token", TOKENS)
def test_api_and_local_nonfinite_receipts_use_actual_decode_and_queue(
    tmp_path, monkeypatch, kind, token
):
    local = kind.startswith("local")
    alias = "qwen8b_r2_host" if local else "luna"
    config = model_configs([alias])[alias]
    unit = request_unit(tmp_path, config)
    ledger = Ledger(tmp_path / "ledger.jsonl", {"synthetic": True})
    output = tmp_path / "output"
    if local:
        transport = LocalTransport(
            tmp_path, config, "http://synthetic.invalid", output, ledger
        )
    else:
        transport = api_run.Transport(
            tmp_path, config, "SYNTHETIC", output, ledger, threading.Event()
        )
    calls = []

    def respond(request, **kwargs):
        calls.append(request.full_url)
        generation = "/generation?" in request.full_url
        if generation:
            return io.BytesIO(
                (
                    '{"data":{"id":"synthetic","model":'
                    + json.dumps(config["model"])
                    + ',"provider_name":"OpenAI","total_cost":'
                    + token
                    + "}}"
                ).encode()
            )
        payload = {"id": "synthetic", "model": config["model"]}
        if local:
            payload["choices"] = [
                {"finish_reason": "stop", "message": {"content": '{"answer":true}'}}
            ]
        else:
            payload.update(
                provider="OpenAI", answers={"answer": {"type": "noul", "noul": 0.8}}
            )
        first = len([url for url in calls if "/generation?" not in url]) == 1
        if first and kind == "api_score":
            payload["answers"]["answer"]["noul"] = "NONFINITE_TOKEN"
        elif first and kind == "local_bad_answer":
            payload["choices"][0]["message"]["content"] = '{"answer":' + token + "}"
        elif first and kind in ("api_usage", "local_usage"):
            payload["usage"] = {"cost": "NONFINITE_TOKEN"}
        elif kind == "api_generation":
            payload.pop("provider")
        return io.BytesIO(
            json.dumps(payload).replace('"NONFINITE_TOKEN"', token).encode()
        )

    if local:
        monkeypatch.setattr("urllib.request.urlopen", respond)
    else:
        monkeypatch.setattr(transport.opener, "open", respond)
    try:
        result = immediate_queue(ledger, config["kind"], transport).run([unit])
        retry = kind in ("api_score", "local_bad_answer")
        assert ledger.state(unit, config["kind"])[0] == (2 if retry else 1)
        assert result[unit["id"]]["fight"] == (
            {"answer": "yes"} if local else {"score": 0.8}
        )
        receipts = sorted((output / "receipts").glob("*.json"))
        assert len(receipts) == (2 if retry else 1)
        first = strict_read(receipts[0])
        field = "generation_metadata" if kind == "api_generation" else "response"
        if kind == "local_bad_answer":
            assert field + "_encoding" not in first  # nested answer already is text
        else:
            assert first[field + "_encoding"] == JSON_TEXT
        rows = [row for row in ledger.rows if row["type"] == "result"]
        assert "cost" not in rows[0]["metadata"].get("usage", {})
        assert "total_cost" not in rows[0]["metadata"]
    finally:
        ledger.close()


@pytest.mark.parametrize("data", [None, [], "bad"])
def test_malformed_generation_metadata_keeps_the_inference_receipt(tmp_path, data):
    config = model_configs(["luna"])["luna"]
    unit = request_unit(tmp_path, config)
    ledger = Ledger(tmp_path / "ledger.jsonl", {"synthetic": True})
    output = tmp_path / "output"
    transport = api_run.Transport(
        tmp_path, config, "SYNTHETIC", output, ledger, threading.Event()
    )
    payload = {
        "id": "synthetic",
        "model": config["model"],
        "answers": {"answer": {"type": "noul", "noul": 0.8}},
    }
    transport.call = lambda uri, body=None: (
        payload if body is not None else {"data": data}
    )
    try:
        with pytest.raises(PausedRun, match="receipt identity unavailable"):
            immediate_queue(ledger, config["kind"], transport).run([unit])
        receipt = strict_read(next((output / "receipts").glob("*.json")))
        assert receipt["response"] == payload
        assert receipt["generation_metadata"] == data
        assert ledger.state(unit, config["kind"]) == (1, {})
    finally:
        ledger.close()


def test_finite_receipt_fields_and_strict_contract_are_unchanged():
    values = {
        "response": {"ok": True, "usage": {"cost": 0.01}},
        "timings": {"seconds": 0.1},
    }
    assert raw_fields(**values) == values
    assert canonical(raw_fields(**values)) == canonical(values)
    with pytest.raises(ValueError):
        canonical({"score": json.loads("1e400")})
    with pytest.raises(ValueError, match="encoding"):
        read_raw_field({"response": "{}", "response_encoding": "unknown"}, "response")


def test_nonfinite_scores_exhaust_budget_once_and_resume_without_resending(
    frozen, tmp_path
):
    uid = "test/fight/0/yes"

    def behavior(self, unit, body):
        if unit["id"] != uid:
            return None
        raw = self.raw(unit, "yes")
        raw["raw"]["answers"]["fight"]["probabilities"][D3_OPTIONS[0]] = json.loads(
            "1e400"
        )
        return 200, raw, {}

    FakeBackend.behavior = behavior
    out = tmp_path / "run"
    execute(frozen, out, "decision3", "test")
    assert Counter(FakeBackend.calls)[uid] == 3
    result = json.loads((out / "test/report.json").read_text())
    assert result["correct"] == 119 and result["accuracy_denominator"] == 120
    assert result["final_missing_unique_units"] == result["incomplete_questions"] == 1
    receipts = [out / "test/receipts" / f"{digest(uid)}-{i}.json" for i in (1, 2, 3)]
    before = [path.read_bytes() for path in receipts]
    assert all(strict_read(path)["response_encoding"] == JSON_TEXT for path in receipts)
    execute(frozen, out, "decision3", "test", resume=True)
    assert Counter(FakeBackend.calls)[uid] == 3
    assert [path.read_bytes() for path in receipts] == before
