"""Real ledger/recovery/transport consumers; synthetic evidence, no model or IO."""

import json
import threading

import pytest
from test_native_workflow import FakeBackend, execute, frozen, immediate_queue, seal

from tools.unified_benchmark import adapters, api_run, native_run, workflow
from tools.unified_benchmark.contract import digest, file_hash, write_new
from tools.unified_benchmark.receipts import raw_fields, read_receipt
from tools.unified_benchmark.retry import Ledger, PausedRun, Queue

__all__ = ["frozen"]


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    FakeBackend.calls, FakeBackend.instances, FakeBackend.behavior = [], [], None

    def forbidden(*args, **kwargs):
        pytest.fail("real network/backend forbidden in synthetic recovery tests")

    monkeypatch.setattr("socket.create_connection", forbidden)
    monkeypatch.setattr(native_run.NativeBackend, "start", forbidden)


def prepare(frozen, tmp_path, *, outcome="success", codec="plain", recorded=False):
    bundle, deployment, deployment_sha = frozen
    cfg = json.loads((bundle / "configs/decision3.json").read_text())
    plan = json.loads((bundle / "plans/decision3-test.json").read_text())
    unit = plan["units"][0]
    output = tmp_path / "run"
    write_new(
        output / "run.json",
        {
            "model": "decision3",
            "phase": "test",
            "bundle_sha256": workflow.verify_bundle(bundle),
            "deployment_sha256": deployment_sha,
            "initial_threshold_sha256": None,
        },
    )
    ledger = Ledger(
        output / "test/attempts.jsonl",
        workflow.context_for(plan, cfg, workflow.verify_bundle(bundle)),
    )
    identity = {
        "unit": unit["id"],
        "attempt": 1,
        "request_sha256": unit["request_sha256"],
    }
    ledger.append({**identity, "type": "intent"})
    backend = FakeBackend("decision3", cfg, bundle, deployment, output, 1)
    raw = backend.raw(unit, "yes")
    FakeBackend.instances.clear()
    if codec != "plain":
        raw["diagnostic"] = float("inf")
    parsed = adapters.parse_response(cfg, unit, raw)
    if outcome != "success":
        raw, parsed = None, {}
    receipt = {
        **identity,
        "config_sha256": digest(cfg),
        "frame_sha256": [f["sha256"] for f in unit["frames"]],
        "http_status": {"success": 200, "fatal": 401, "retry": 429}[outcome],
        "error": None if outcome == "success" else "synthetic HTTP failure",
        "parsed": parsed,
        "transport_seconds": 0.01,
        "fatal": outcome == "fatal",
        "retry_after": 90 if outcome == "retry" else 0,
        "completed_at_unix": 1000.0,
    }
    if codec == "global":
        receipt.update(
            response=None,
            response_json=json.dumps(raw),
            raw_json_encoding="python-json-nonfinite-v1",
        )
    else:
        receipt.update(raw_fields(response=raw))
    path = output / "test/receipts" / f"{digest(unit['id'])}-1.json"
    write_new(path, receipt)
    if recorded:
        ledger.append(
            {
                **identity,
                "type": "result",
                "answers": parsed,
                "complete": outcome == "success",
                "fatal": outcome == "fatal",
                "http_status": receipt["http_status"],
                "retry_after": receipt["retry_after"],
                "elapsed_seconds": 0.01,
                "metadata": {
                    "receipt": str(path.relative_to(output)),
                    "receipt_sha256": file_hash(path),
                },
            }
        )
    ledger.close()
    return output, path, plan, cfg, unit


def rows(output):
    return [
        json.loads(line)
        for line in (output / "test/attempts.jsonl").read_text().splitlines()
    ]


def edit_json(path, update):
    value = json.loads(path.read_text())
    update(value)
    path.write_text(json.dumps(value))


def test_newly_recovered_fatal_pauses_before_backend_then_explicit_resume(
    frozen, tmp_path
):
    output, receipt, _, _, unit = prepare(frozen, tmp_path, outcome="fatal")
    original = receipt.read_bytes()
    with pytest.raises(PausedRun, match="newly recovered fatal"):
        execute(frozen, output, "decision3", "test", resume=True)
    assert not FakeBackend.instances and not FakeBackend.calls
    assert not (output / "invocations").exists()
    recovered = rows(output)
    assert len(recovered) == 2 and recovered[1]["fatal"] is True
    assert recovered[1]["http_status"] == 401
    assert recovered[1]["metadata"]["receipt_sha256"] == file_hash(receipt)
    # A second explicit resume acknowledges the now-recorded failure, as before.
    execute(frozen, output, "decision3", "test", resume=True)
    assert FakeBackend.calls.count(unit["id"]) == 1
    assert [
        r["attempt"]
        for r in rows(output)
        if r["unit"] == unit["id"] and r["type"] == "intent"
    ] == [1, 2]
    assert receipt.read_bytes() == original


@pytest.mark.parametrize("codec", ["plain", "field", "global"])
@pytest.mark.parametrize("recorded", [False, True])
def test_historical_formats_reparse_without_rewriting_or_resending(
    frozen, tmp_path, codec, recorded
):
    output, receipt, _, _, unit = prepare(
        frozen, tmp_path, codec=codec, recorded=recorded
    )
    original = receipt.read_bytes()
    execute(frozen, output, "decision3", "test", resume=True)
    assert unit["id"] not in FakeBackend.calls
    assert len(FakeBackend.calls) == 119
    assert receipt.read_bytes() == original
    assert json.loads((output / "test/report.json").read_text())["correct"] == 120


@pytest.mark.parametrize(
    "field", ["fatal", "retry_after", "completed_at_unix", "http_status", "error"]
)
def test_incomplete_legacy_missing_outcome_stops_without_inventing_metadata(
    frozen, tmp_path, field
):
    output, receipt, _, _, _ = prepare(frozen, tmp_path, outcome="retry")
    edit_json(receipt, lambda r: r.pop(field))
    before = receipt.read_bytes(), (output / "test/attempts.jsonl").read_bytes()
    with pytest.raises(ValueError, match="missing authoritative.*" + field):
        execute(frozen, output, "decision3", "test", resume=True)
    assert not FakeBackend.instances and not FakeBackend.calls
    assert before == (
        receipt.read_bytes(),
        (output / "test/attempts.jsonl").read_bytes(),
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("fatal", "false"),
        ("fatal", 0),
        ("retry_after", -1),
        ("retry_after", True),
        ("completed_at_unix", -1),
        ("completed_at_unix", True),
        ("http_status", "429"),
        ("transport_seconds", -1),
    ],
)
def test_invalid_recovery_outcome_fails_before_start(frozen, tmp_path, field, value):
    output, receipt, _, _, _ = prepare(frozen, tmp_path, outcome="retry")
    edit_json(receipt, lambda r: r.update({field: value}))
    with pytest.raises(ValueError, match="invalid receipt"):
        execute(frozen, output, "decision3", "test", resume=True)
    assert not FakeBackend.instances and len(rows(output)) == 1


@pytest.mark.parametrize("final", [False, True])
def test_recovery_retains_completion_based_shared_429_cooldown(frozen, tmp_path, final):
    output, path, plan, cfg, unit = prepare(frozen, tmp_path, outcome="retry")
    ledger = Ledger(
        output / "test/attempts.jsonl",
        workflow.context_for(plan, cfg, workflow.verify_bundle(frozen[0])),
    )
    if final:
        # Earlier interrupted sends consume their slots without synthetic results.
        path.unlink()
        for attempt in (2, 3):
            ledger.append(
                {
                    "type": "intent",
                    "unit": unit["id"],
                    "request_sha256": unit["request_sha256"],
                    "attempt": attempt,
                }
            )
        path = path.with_name(f"{digest(unit['id'])}-3.json")
        write_new(
            path,
            {
                "unit": unit["id"],
                "attempt": 3,
                "request_sha256": unit["request_sha256"],
                "config_sha256": digest(cfg),
                "frame_sha256": [f["sha256"] for f in unit["frames"]],
                "http_status": 429,
                "response": None,
                "parsed": {},
                "error": "HTTP 429",
                "fatal": False,
                "retry_after": 90,
                "completed_at_unix": 1000,
                "transport_seconds": 0.01,
            },
        )
    try:
        native_run.reconcile_receipts(output / "test", ledger, plan, cfg, output)
        assert ledger.rows[-1]["cooldown_until"] == 1090
        now, sent = [1050.0], []
        queue = Queue(
            ledger,
            cfg["kind"],
            lambda u: (
                sent.append((u["id"], now[0])) or {"fight": {"answer": "yes"}},
                {},
            ),
            slots=1,
            clock=lambda: now[0],
            sleep=lambda seconds: now.__setitem__(0, now[0] + seconds),
        )
        queue.one(unit)
        queue.one(plan["units"][1])
        assert all(at >= 1090 for _, at in sent)
        assert len(sent) == (1 if final else 2)
    finally:
        ledger.close()


@pytest.mark.parametrize(
    "fault",
    [
        "phantom",
        "duplicate",
        "before_intent",
        "boolean",
        "wrong_request",
        "unknown_type",
    ],
)
def test_malformed_ledger_cannot_supply_unverified_answers(frozen, tmp_path, fault):
    output, _, _, _, _ = prepare(frozen, tmp_path, recorded=True)
    data = rows(output)
    if fault == "phantom":
        data[-1]["attempt"] = 2
    elif fault == "duplicate":
        data.append(data[-1])
    elif fault == "before_intent":
        data.reverse()
    elif fault == "boolean":
        data[-1]["attempt"] = True
    elif fault == "wrong_request":
        data[-1]["request_sha256"] = "changed"
    else:
        data[-1]["type"] = "other"
    path = output / "test/attempts.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in data))
    before = path.read_bytes()
    with pytest.raises(ValueError):
        execute(frozen, output, "decision3", "test", resume=True)
    assert not FakeBackend.instances and not (output / "invocations").exists()
    assert path.read_bytes() == before


def test_recorded_answer_must_match_hash_verified_raw_receipt(frozen, tmp_path):
    output, receipt, _, _, unit = prepare(frozen, tmp_path, recorded=True)
    data = rows(output)
    data[-1]["answers"][unit["targets"][0]]["answer"] = "no"
    (output / "test/attempts.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in data)
    )
    before = receipt.read_bytes()
    with pytest.raises(ValueError, match="ledger answers differ"):
        execute(frozen, output, "decision3", "test", resume=True)
    assert not FakeBackend.instances and receipt.read_bytes() == before


@pytest.mark.parametrize(
    "payload",
    [
        {"choices": None},
        {"choices": [None]},
        {"choices": [{"message": []}]},
        {"choices": [{"finish_reason": "stop", "message": {"content": None}}]},
        {"answers": {"answer": []}},
        {"answers": {"answer": {"type": "noul", "noul": True}}},
        {"answers": {"answer": {"type": "noul", "noul": 1.2}}},
    ],
)
def test_actual_api_queue_retries_malformed_language_and_luna(
    tmp_path, monkeypatch, payload
):
    decision = "answers" in payload
    cfg = {
        "backend": "decision" if decision else "language",
        "model": "model",
        "kind": "probability" if decision else "choice",
        "provider_context": {"config": {"expected_provider_name": "provider"}},
    }
    unit = {"id": "unit", "targets": ["fight"], "request_sha256": digest({})}
    ledger = Ledger(tmp_path / "attempts.jsonl", {"synthetic": True})
    transport = api_run.Transport(
        tmp_path, cfg, "not-a-key", tmp_path, ledger, threading.Event()
    )
    monkeypatch.setattr(api_run, "request_body", lambda *a: ("unused", {}))
    good = (
        {"answers": {"answer": {"type": "noul", "noul": 0.7}}}
        if decision
        else {
            "choices": [
                {"finish_reason": "stop", "message": {"content": '{"answer":true}'}}
            ]
        }
    )
    payloads = [payload, good]
    transport.call = lambda *a: {
        **payloads.pop(0),
        "model": "model",
        "provider": "provider",
    }
    try:
        immediate_queue(ledger, cfg["kind"], transport).one(unit)
        assert not payloads and len(ledger.rows) == 4
        assert ledger.rows[1]["answers"] == {} and ledger.rows[-1]["complete"]
        for row in (ledger.rows[1], ledger.rows[-1]):
            assert row["metadata"]["receipt_sha256"] == file_hash(
                tmp_path / row["metadata"]["receipt"]
            )
    finally:
        ledger.close()


def test_identity_failure_keeps_receipt_hash_in_actual_queue(tmp_path, monkeypatch):
    cfg = {
        "backend": "language",
        "kind": "choice",
        "model": "wanted",
        "provider_context": {"config": {"expected_provider_name": "provider"}},
    }
    unit = {"id": "unit", "targets": ["fight"], "request_sha256": digest({})}
    ledger = Ledger(tmp_path / "attempts.jsonl", {"synthetic": True})
    transport = api_run.Transport(
        tmp_path, cfg, "not-a-key", tmp_path, ledger, threading.Event()
    )
    monkeypatch.setattr(api_run, "request_body", lambda *a: ("unused", {}))
    transport.call = lambda *a: {"model": "wrong", "provider": "provider"}
    try:
        with pytest.raises(PausedRun, match="actual model mismatch"):
            immediate_queue(ledger, "choice", transport).one(unit)
        metadata = ledger.rows[-1]["metadata"]
        assert metadata["receipt_sha256"] == file_hash(tmp_path / metadata["receipt"])
    finally:
        ledger.close()


@pytest.mark.parametrize(
    "fault",
    ["missing_model", "plan_drift", "extra_plan", "frame_binding", "extra_file"],
)
def test_bundle_requires_inventory_and_semantic_preflight_bindings(frozen, fault):
    bundle = frozen[0]
    if fault == "missing_model":
        edit_json(bundle / "preflight.json", lambda r: r["models"].pop("clip"))
    elif fault == "plan_drift":
        edit_json(
            bundle / "plans/clip-test.json",
            lambda r: r["questions"][0].update(reference="no"),
        )
    elif fault == "extra_plan":
        (bundle / "plans/rogue-test.json").write_text("{}")
    elif fault == "frame_binding":
        plan_path = bundle / "plans/clip-test.json"
        edit_json(
            plan_path, lambda r: r["units"][0]["frames"][0].update(sha256="0" * 64)
        )
        edit_json(
            bundle / "preflight.json",
            lambda r: r["models"]["clip"]["test"].update(
                plan_sha256=digest(json.loads(plan_path.read_text()))
            ),
        )
    else:
        (bundle / "unsealed.txt").write_text("not in seal")
    if fault != "extra_file":
        seal(bundle)
    with pytest.raises(ValueError):
        workflow.verify_bundle(bundle)


def test_failed_strict_serialization_leaves_no_empty_receipt(tmp_path):
    path = tmp_path / "receipts/receipt.json"
    with pytest.raises(ValueError):
        write_new(path, {"unencoded": float("inf")})
    assert not path.exists()


@pytest.mark.parametrize(
    "encoding", ["unknown", "mixed", "ambiguous", "outer_overflow", "duplicate"]
)
def test_dual_reader_rejects_ambiguous_or_nonfinite_envelopes(tmp_path, encoding):
    data = {
        "response": None,
        "response_json": '{"diagnostic":Infinity}',
        "raw_json_encoding": "python-json-nonfinite-v1",
    }
    if encoding == "unknown":
        data["raw_json_encoding"] = "other"
    elif encoding == "mixed":
        data["response_encoding"] = "json-text-nonfinite-v1"
    elif encoding == "ambiguous":
        data["response"] = {"different": 1}
    text = json.dumps(data)
    if encoding == "outer_overflow":
        text = '{"response":1e400}'
    elif encoding == "duplicate":
        text = '{"response":1,"response":2}'
    path = tmp_path / "receipt.json"
    path.write_text(text)
    with pytest.raises(ValueError):
        read_receipt(path)
    assert path.read_text() == text


def test_earlier_orphan_answer_precedes_later_recorded_answer(frozen, tmp_path):
    output, first, plan, cfg, unit = prepare(frozen, tmp_path)
    later = json.loads(first.read_text())
    later["attempt"] = 2
    raw = FakeBackend("decision3", cfg, frozen[0], frozen[1], output, 1).raw(unit, "no")
    FakeBackend.instances.clear()
    later.update(response=raw, parsed=adapters.parse_response(cfg, unit, raw))
    path = first.with_name(f"{digest(unit['id'])}-2.json")
    write_new(path, later)
    ledger = Ledger(
        output / "test/attempts.jsonl",
        workflow.context_for(plan, cfg, workflow.verify_bundle(frozen[0])),
    )
    identity = {
        "unit": unit["id"],
        "attempt": 2,
        "request_sha256": unit["request_sha256"],
    }
    ledger.append({**identity, "type": "intent"})
    ledger.append(
        {
            **identity,
            "type": "result",
            "answers": later["parsed"],
            "complete": True,
            "metadata": {
                "receipt": str(path.relative_to(output)),
                "receipt_sha256": file_hash(path),
            },
        }
    )
    ledger.close()
    before = first.read_bytes(), path.read_bytes()
    execute(frozen, output, "decision3", "test", resume=True)
    assert unit["id"] not in FakeBackend.calls
    assert json.loads((output / "test/report.json").read_text())["correct"] == 120
    assert before == (first.read_bytes(), path.read_bytes())


def test_partial_first_answers_follow_attempt_order_even_after_recovery(tmp_path):
    ledger = Ledger(tmp_path / "attempts.jsonl", {"synthetic": True})
    unit = {"id": "unit", "request_sha256": "request", "targets": ["fight", "smoking"]}
    base = {"unit": "unit", "request_sha256": "request"}
    try:
        for attempt in (1, 2):
            ledger.append({**base, "type": "intent", "attempt": attempt})
        ledger.append(
            {
                **base,
                "type": "result",
                "attempt": 2,
                "answers": {
                    "fight": {"answer": "yes"},
                    "smoking": {"answer": "unknown"},
                },
            }
        )
        ledger.append(
            {
                **base,
                "type": "result",
                "attempt": 1,
                "answers": {"fight": {"answer": "no"}},
            }
        )
        assert ledger.state(unit, "choice") == (
            2,
            {"fight": {"answer": "no"}, "smoking": {"answer": "unknown"}},
        )
    finally:
        ledger.close()


def test_anchored_recorded_legacy_failure_uses_ledger_outcome(frozen, tmp_path):
    output, receipt, _, _, unit = prepare(
        frozen, tmp_path, outcome="fatal", recorded=True
    )
    edit_json(
        receipt,
        lambda r: [r.pop(k) for k in ("fatal", "retry_after", "completed_at_unix")],
    )
    data = rows(output)
    data[-1]["metadata"]["receipt_sha256"] = file_hash(receipt)
    path = output / "test/attempts.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in data))
    before = receipt.read_bytes(), path.read_bytes()
    execute(frozen, output, "decision3", "test", resume=True)
    assert FakeBackend.calls.count(unit["id"]) == 1
    assert receipt.read_bytes() == before[0] and path.read_bytes().startswith(before[1])


@pytest.mark.parametrize("value", [None, "1000", True, -1])
def test_recorded_receipt_supplied_outcome_fields_are_typed(frozen, tmp_path, value):
    output, receipt, _, _, _ = prepare(frozen, tmp_path, recorded=True)
    edit_json(receipt, lambda r: r.update(completed_at_unix=value))
    data = rows(output)
    data[-1]["metadata"]["receipt_sha256"] = file_hash(receipt)
    path = output / "test/attempts.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in data))
    with pytest.raises(ValueError, match="invalid receipt completed_at_unix"):
        execute(frozen, output, "decision3", "test", resume=True)
    assert not FakeBackend.instances


def test_new_fatal_after_earlier_complete_orphan_pauses_once(frozen, tmp_path):
    output, first, plan, cfg, unit = prepare(frozen, tmp_path)
    later = json.loads(first.read_text())
    later.update(
        attempt=2,
        response=None,
        parsed={},
        http_status=401,
        error="HTTP 401",
        fatal=True,
    )
    second = first.with_name(f"{digest(unit['id'])}-2.json")
    write_new(second, later)
    ledger = Ledger(
        output / "test/attempts.jsonl",
        workflow.context_for(plan, cfg, workflow.verify_bundle(frozen[0])),
    )
    ledger.append(
        {
            "type": "intent",
            "unit": unit["id"],
            "request_sha256": unit["request_sha256"],
            "attempt": 2,
        }
    )
    ledger.close()
    before = first.read_bytes(), second.read_bytes()
    with pytest.raises(PausedRun, match="newly recovered fatal"):
        execute(frozen, output, "decision3", "test", resume=True)
    assert not FakeBackend.instances
    assert rows(output)[-1]["fatal"] is True
    execute(frozen, output, "decision3", "test", resume=True)
    assert unit["id"] not in FakeBackend.calls
    assert before == (first.read_bytes(), second.read_bytes())


def test_invalid_existing_phase_does_not_append_recovery(frozen, tmp_path):
    output, receipt, plan, cfg, _ = prepare(frozen, tmp_path)
    write_new(
        output / "test/run.json",
        {
            "context": workflow.context_for(
                plan, cfg, workflow.verify_bundle(frozen[0])
            ),
            "threshold_artifact_sha256": None,
            "native_deployment_sha256": "different-deployment",
        },
    )
    ledger = output / "test/attempts.jsonl"
    before = ledger.read_bytes(), receipt.read_bytes()
    with pytest.raises(ValueError, match="resume phase identity differs"):
        execute(frozen, output, "decision3", "test", resume=True)
    assert not FakeBackend.instances
    assert ledger.read_bytes() == before[0], "incompatible phase ledger was mutated"
    assert receipt.read_bytes() == before[1]
    assert not (output / "invocations").exists()


@pytest.mark.parametrize("diagnostic", [1.5, float("inf")], ids=["finite", "nonfinite"])
def test_finite_native_score_with_nested_nonfinite_diagnostic_is_durable(
    frozen, tmp_path, diagnostic
):
    def behavior(self, unit, body):
        if self.alias != "clip":
            return None
        answer = unit["id"].rsplit("/", 1)[1]
        raw = self.raw(unit, answer)
        for value in raw.values():
            value["optional_diagnostic"] = {"estimate": diagnostic}
        return 200, raw, {}

    FakeBackend.behavior = behavior
    output = tmp_path / "run"
    execute(frozen, output, "clip", "both")
    assert json.loads((output / "test/report.json").read_text())["correct"] == 120
    receipts = list((output / "test/receipts").glob("*.json"))
    assert len(receipts) == 120
    for path in receipts:
        record = json.loads(path.read_text())
        if diagnostic == float("inf"):
            assert record["response_encoding"] == "json-text-nonfinite-v1"
        else:
            assert "response_encoding" not in record
        decoded = read_receipt(path)
        for event, score in record["parsed"].items():
            assert set(score["native_scores"]) == {
                "margin",
                "positive_cosine",
                "contrast_cosine",
            }
            assert (
                decoded["response"][event]["optional_diagnostic"]["estimate"]
                == diagnostic
            )
        json.dumps(record["parsed"], allow_nan=False)
        assert file_hash(path)


def test_completed_phase_anchors_checked_before_invocation_or_recovery(
    frozen, tmp_path
):
    output = tmp_path / "run"
    execute(frozen, output, "decision3", "both")
    edit_json(output / "test/report.json", lambda r: r.update(correct=0))
    before = {
        p.relative_to(output): p.read_bytes() for p in output.rglob("*") if p.is_file()
    }
    FakeBackend.instances.clear()
    with pytest.raises(ValueError, match="completed phase evidence differs"):
        execute(frozen, output, "decision3", "both", resume=True)
    assert not FakeBackend.instances
    assert before == {
        p.relative_to(output): p.read_bytes() for p in output.rglob("*") if p.is_file()
    }


def test_all_phase_identities_checked_before_first_orphan_recovery(frozen, tmp_path):
    output, first, _, cfg, _ = prepare(frozen, tmp_path)
    edit_json(output / "run.json", lambda r: r.update(phase="both"))
    bundle = frozen[0]
    plan = json.loads((bundle / "plans/decision3-validation.json").read_text())
    unit = plan["units"][0]
    ledger = Ledger(
        output / "validation/attempts.jsonl",
        workflow.context_for(plan, cfg, workflow.verify_bundle(bundle)),
    )
    identity = {
        "unit": unit["id"],
        "attempt": 1,
        "request_sha256": unit["request_sha256"],
    }
    ledger.append({**identity, "type": "intent"})
    ledger.close()
    raw = FakeBackend("decision3", cfg, bundle, frozen[1], output, 1).raw(unit, "yes")
    FakeBackend.instances.clear()
    receipt = {
        **json.loads(first.read_text()),
        **identity,
        "response": raw,
        "parsed": adapters.parse_response(cfg, unit, raw),
    }
    write_new(output / "validation/receipts" / f"{digest(unit['id'])}-1.json", receipt)
    write_new(output / "test/run.json", {"context": "wrong"})
    before = {
        p.relative_to(output): p.read_bytes() for p in output.rglob("*") if p.is_file()
    }
    with pytest.raises(ValueError, match="resume phase identity differs"):
        execute(frozen, output, "decision3", "both", resume=True)
    assert not FakeBackend.instances
    assert before == {
        p.relative_to(output): p.read_bytes() for p in output.rglob("*") if p.is_file()
    }
