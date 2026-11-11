"""Synthetic contract tests; these are not benchmark/model accuracy results."""
import json
from pathlib import Path
import pytest
pytest.importorskip("fcntl", reason="The durable benchmark queue currently requires Linux/macOS")

from tools.unified_benchmark import adapters as a
from tools.unified_benchmark.contract import EVENTS, aggregate, digest, sample_times, score, select_thresholds
from tools.unified_benchmark.retry import Ledger, PausedRun, Queue, RequestFailure


def unit(uid="u", targets=None):
    return {"id": uid, "targets": targets or ["fight"], "request_sha256": "frozen"}


def plan():
    return {"split": "test", "units": [unit("u"), unit("v")], "questions": [
        {"number": 1, "event": "fight", "reference": "yes", "unit_ids": ["u", "v"]}]}


class Clock:
    def __init__(self): self.t = 0
    def now(self): return self.t
    def sleep(self, d): self.t += d


def queue(tmp_path, send, kind="choice"):
    ledger, clock = Ledger(tmp_path / "attempts.jsonl", {"frozen": 1}), Clock()
    return ledger, clock, Queue(ledger, kind, send, slots=1, clock=clock.now, sleep=clock.sleep)


def test_incomplete_yes_is_not_scored():
    r = score(plan(), {"u": {"fight": {"answer": "yes"}}}, "choice")
    assert r["accuracy"] == 0 and r["accuracy_denominator"] == 1 and r["incomplete_questions"] == 1 and r["final_missing_unique_units"] == 1


@pytest.mark.parametrize("answers,expected", [(["no", "unknown"], "no"), (["yes", "unknown"], "yes"), (["unknown", "unknown"], "unknown")])
def test_native_aggregation(answers, expected):
    assert aggregate(answers) == expected


def test_unknown_is_valid_incorrect_not_missing():
    r = score(plan(), {i: {"fight": {"answer": "unknown"}} for i in ("u", "v")}, "choice")
    assert (r["accuracy"], r["complete_valid_questions"], r["final_unknown_questions"], r["final_missing_unique_units"]) == (0, 1, 1, 0)


def test_shared_window_count_not_double_counted():
    p = plan()
    p["units"][0]["targets"] = ["fight", "smoking"]
    p["questions"] = [dict(p["questions"][0], unit_ids=["u"]), {"number": 2, "event": "smoking", "reference": "no", "unit_ids": ["u"]}]
    p["units"] = p["units"][:1]
    r = score(p, {}, "choice")
    assert r["final_missing_unique_units"] == 1 and r["incomplete_questions"] == 2


def test_exact_second_grid():
    assert sample_times(2.1) == [0., 1., 2.]
    assert sample_times(2) == [0., 1.]
    with pytest.raises(ValueError): sample_times(float("nan"))


@pytest.mark.parametrize("answer", ["yes", "no", "unknown"])
def test_first_valid_response_never_retried(tmp_path, answer):
    calls = []
    def send(u): calls.append(u); return {"fight": {"answer": answer}}, {}
    ledger, clock, q = queue(tmp_path, send)
    try:
        q.run([unit()]); q.run([unit()])
        assert len(calls) == 1
    finally: ledger.close()


def test_three_attempts_cumulative_across_resume(tmp_path):
    calls = []
    def send(u): calls.append(u); raise RequestFailure("rate_limit", status=429)
    ledger, clock, q = queue(tmp_path, send)
    try: q.run([unit()])
    finally: ledger.close()
    assert len(calls) == 3 and clock.t == 90
    ledger, clock, q = queue(tmp_path, send)
    try:
        q.run([unit()]); assert len(calls) == 3
    finally: ledger.close()


def test_retry_after_and_success_clears_final_missing(tmp_path):
    n = 0
    def send(u):
        nonlocal n
        n += 1
        if n == 1: raise RequestFailure("rate_limit", status=429, retry_after=45)
        return {"fight": {"answer": "yes"}}, {}
    ledger, clock, q = queue(tmp_path, send)
    try:
        r = q.run([unit()]); assert r["u"]["fight"]["answer"] == "yes" and n == 2 and clock.t == 45
    finally: ledger.close()


def test_partial_native_answer_is_preserved(tmp_path):
    n = 0
    def send(u):
        nonlocal n
        n += 1
        return ({"fight": {"answer": "no"}} if n == 1 else {"fight": {"answer": "yes"}, "smoking": {"answer": "unknown"}}), {}
    ledger, clock, q = queue(tmp_path, send)
    try:
        r = q.run([unit(targets=["fight", "smoking"])])
        assert r["u"] == {"fight": {"answer": "no"}, "smoking": {"answer": "unknown"}}
    finally: ledger.close()


def test_crash_intent_consumes_attempt_and_changed_request_rejected(tmp_path):
    ledger, clock, q = queue(tmp_path, lambda u: ({"fight": {"answer": "yes"}}, {}))
    try:
        ledger.append({"unit": "u", "type": "intent", "request_sha256": "frozen", "attempt": 1})
        q.run([unit()])
        assert ledger.state(unit(), "choice")[0] == 2
        with pytest.raises(ValueError): q.run([unit() | {"request_sha256": "changed"}])
    finally: ledger.close()


def test_fatal_auth_pauses_instead_of_fake_three_failures(tmp_path):
    def send(u): raise RequestFailure("auth", status=403, fatal=True)
    ledger, clock, q = queue(tmp_path, send)
    try:
        with pytest.raises(PausedRun): q.run([unit()])
        assert ledger.state(unit(), "choice")[0] == 1
    finally: ledger.close()


def test_validation_only_and_complete_selection():
    p = {"split": "validation", "units": [], "questions": []}
    data = {}
    for index, event in enumerate(EVENTS[:4]):
        for i in range(6):
            uid = f"{event}/{i}"; p["units"].append(unit(uid, [event]))
            p["questions"].append({"number": index * 6 + i, "event": event, "reference": "yes" if i < 3 else "no", "unit_ids": [uid]})
            data[uid] = {event: {"score": .8 if i < 3 else .2}}
    result = select_thresholds(p, data, "probability", "fake")
    assert result["thresholds"]["bag_knife"] == .5
    margin = select_thresholds(p, data, "margin", "fake")
    assert margin["thresholds"]["fight"] == .5 and margin["thresholds"]["bag_knife"] == 0
    with pytest.raises(ValueError): select_thresholds(p | {"split": "test"}, data, "probability", "fake")
    data.pop("fight/0")
    with pytest.raises(ValueError): select_thresholds(p, data, "probability", "fake")


def response(content):
    return {"choices": [{"finish_reason": "stop", "message": {"content": content}}]}


def test_bag_unknown_only_explicit_no_refusal_coercion():
    config = {"backend": "language"}; u = unit(targets=["bag_knife"])
    assert a.parse_response(config, u, response('{"answer":"unknown"}'))["bag_knife"]["answer"] == "unknown"
    for bad in ['{"answer":null}', '{"answer":1}', '{"answer":false,"answer":true}']:
        with pytest.raises(ValueError): a.parse_response(config, u, response(bad))
    with pytest.raises(ValueError): a.parse_response(config, u, response('{"answer":"unknown"}') | {"error": "refusal"})


def test_native_unknown_requires_probability_contract_and_identity():
    config = {"backend": "decision3", "model": "m", "revision": "r"}
    u = unit()
    payload = {"request_id": "u", "model": "m", "revision": "r", "raw": {"answers": {
        "fight": {"type": "choice", "choice": a.D3_OPTIONS[2], "probabilities": dict(zip(a.D3_OPTIONS, [.1, .2, .7]))}}}}
    assert a.parse_response(config, u, payload)["fight"]["answer"] == "unknown"
    with pytest.raises(ValueError): a.parse_response(config, u, payload | {"revision": "other"})


def test_boolean_and_nonfinite_scores_rejected():
    for v in (True, float("nan"), float("inf"), 1.1):
        r = score(plan(), {i: {"fight": {"score": v}} for i in ("u", "v")}, "probability")
        assert r["complete_valid_questions"] == 0


def test_api_key_quoted_dotenv_and_retry_after(tmp_path):
    from tools.unified_benchmark.api_run import load_explicit_key, retry_after
    p = tmp_path / "env"
    p.write_text('OPENROUTER_API_KEY="synthetic-test-only"\n')
    assert load_explicit_key(p) == "synthetic-test-only"
    assert retry_after("45") == 45
    assert retry_after("Thu, 01 Jan 1970 00:01:30 GMT", now=30) == 60
    assert retry_after("nan") == 0


def test_api_transport_saved_response_mock_no_real_http(tmp_path):
    import threading
    from tools.unified_benchmark.api_run import Transport
    from tools.unified_benchmark.contract import file_hash
    # A JPEG signature is sufficient here: this test validates envelope/ledger integration, not decoding.
    (tmp_path / "f.jpg").write_bytes(b"mock-image")
    config = {"model": "qwen/test", "backend": "language", "kind": "choice", "max_tokens": 30, "temperature": 0,
              "provider_context": {"config": {"expected_provider_name": "Mock", "provider": {"only": ["mock"], "allow_fallbacks": False}}}}
    u = unit() | {"frames": [{"path": "f.jpg", "sha256": file_hash(tmp_path / "f.jpg"), "time": 0}], "question": "Is there a fight?"}
    url, body = a.request_body(config, u, tmp_path)
    assert body["provider"] == {"only": ["mock"], "allow_fallbacks": False}
    assert 'reference' not in str(body) and 'proposed_answer' not in str(body)
    u["request_sha256"] = digest(body)
    ledger = Ledger(tmp_path / "attempts.jsonl", {"test": True})
    transport = Transport(tmp_path, config, "not-a-real-key", tmp_path / "out", ledger, threading.Event())
    transport.call = lambda url, body=None: response('{"answer":true}') | {"provider": "Mock", "model": "qwen/test", "id": "mock-generation", "usage": {"cost": .01}}
    try:
        observed = Queue(ledger, "choice", transport, slots=1).run([u])
        assert observed["u"]["fight"]["answer"] == "yes"
        assert len(list((tmp_path / "out/receipts").glob('*.json'))) == 1
    finally: ledger.close()


@pytest.mark.parametrize("bad_item", [None, [], "invalid", {"type": "choice", "probabilities": None},
    {"type": "choice", "choice": a.D3_OPTIONS[0], "probabilities": dict(zip(a.D3_OPTIONS, [10 ** 400, 0, 0]))}])
def test_decision3_retries_bad_target_without_overwriting_valid_sibling(tmp_path, bad_item):
    config = {"backend": "decision3", "model": "m", "revision": "r"}
    u = unit(targets=["fight", "fall"])
    def item(index):
        return {"type": "choice", "choice": a.D3_OPTIONS[index],
                "probabilities": dict(zip(a.D3_OPTIONS, [int(i == index) for i in range(3)]))}
    payloads = iter([
        {"fight": item(0), "fall": bad_item},
        {"fight": item(1), "fall": item(1)},
    ])
    def send(_):
        return a.parse_response(config, u, {"request_id": u["id"], "model": "m", "revision": "r",
                                           "raw": {"answers": next(payloads)}}), {}
    ledger, clock, q = queue(tmp_path, send)
    try:
        _, answers = q.one(u)
        assert answers["fight"]["answer"] == "yes"
        assert answers["fall"]["answer"] == "no"
        assert ledger.state(u, "choice")[0] == 2
    finally:
        ledger.close()


@pytest.mark.parametrize("choices", [None, [], [None], [{"finish_reason": "stop", "message": None}],
    [{"finish_reason": "stop", "message": {"content": None}}]])
@pytest.mark.parametrize("recover", [False, True])
def test_api_malformed_envelope_uses_retry_budget(tmp_path, choices, recover):
    import threading
    from tools.unified_benchmark.api_run import Transport
    from tools.unified_benchmark.contract import file_hash
    (tmp_path / "f.jpg").write_bytes(b"synthetic-image")
    config = {"model": "qwen/test", "backend": "language", "kind": "choice", "max_tokens": 30, "temperature": 0,
              "provider_context": {"config": {"expected_provider_name": "Mock", "provider": {"only": ["mock"]}}}}
    u = unit() | {"frames": [{"path": "f.jpg", "sha256": file_hash(tmp_path / "f.jpg"), "time": 0}], "question": "Fight?"}
    u["request_sha256"] = digest(a.request_body(config, u, tmp_path)[1])
    ledger = Ledger(tmp_path / "attempts.jsonl", {"test": True})
    transport = Transport(tmp_path, config, "synthetic-key", tmp_path / "out", ledger, threading.Event())
    calls = []
    def call(url, body=None):
        calls.append(body)
        payload = response('{"answer":true}') if recover and len(calls) > 1 else {"choices": choices}
        return payload | {"provider": "Mock", "model": "qwen/test", "usage": {"cost": .01}}
    transport.call = call
    tick = [0.0]
    q = Queue(ledger, "choice", transport, clock=lambda: tick[0], sleep=lambda seconds: tick.__setitem__(0, tick[0] + seconds))
    try:
        _, observed = q.one(u)
        assert observed == ({"fight": {"answer": "yes"}} if recover else {})
        assert len(calls) == (2 if recover else 3)
        assert len(list((tmp_path / "out/receipts").glob("*.json"))) == len(calls)
        assert all(row.get("metadata", {}).get("answer_error") for row in ledger.rows
                   if row["type"] == "result" and not row["complete"])
        q.one(u)
        assert len(calls) == (2 if recover else 3)
    finally:
        ledger.close()


@pytest.mark.parametrize("kind", ["margin", "probability"])
def test_oversized_ledger_score_is_invalid_evidence(kind):
    from tools.unified_benchmark.contract import valid_value
    assert not valid_value({"score": 10 ** 400}, kind)
    assert not valid_value({"score": -(10 ** 400)}, kind)


@pytest.mark.parametrize("backend", ["decision", "language"])
@pytest.mark.parametrize("field", ["score", "metadata"])
@pytest.mark.parametrize("huge", [10 ** 400, float("inf"), float("nan")])
def test_api_oversized_score_retries_but_optional_metadata_does_not(tmp_path, backend, field, huge):
    import threading
    from tools.unified_benchmark.api_run import Transport, accounting
    from tools.unified_benchmark.contract import file_hash
    (tmp_path / "f.jpg").write_bytes(b"synthetic-image")
    config = {"model": "test/model", "backend": backend, "kind": "probability" if backend == "decision" else "choice",
              "max_tokens": 30, "temperature": 0,
              "provider_context": {"config": {"expected_provider_name": "Mock", "provider": {"only": ["mock"]}}}}
    u = unit() | {"frames": [{"path": "f.jpg", "sha256": file_hash(tmp_path / "f.jpg"), "time": 0}], "question": "Fight?"}
    u["request_sha256"] = digest(a.request_body(config, u, tmp_path)[1])
    ledger = Ledger(tmp_path / "attempts.jsonl", {"test": True})
    transport = Transport(tmp_path, config, "synthetic-key", tmp_path / "out", ledger, threading.Event())
    calls = []
    def call(url, body=None):
        calls.append(body)
        payload = ({"answers": {"answer": {"type": "noul", "noul": .8}}}
                   if backend == "decision" else response('{"answer":true}'))
        payload.update(provider="Mock", model="test/model")
        if field == "score" and len(calls) == 1:
            if backend == "decision":
                payload["answers"]["answer"]["noul"] = huge
            else:
                payload = response(json.dumps({"answer": huge})) | {"provider": "Mock", "model": "test/model"}
        elif field == "metadata":
            payload.update(usage={"cost": huge, "prompt_tokens": huge, "completion_tokens": 5},
                           total_cost=huge, native_tokens_prompt=huge)
        return payload
    transport.call = call
    tick = [0.0]
    q = Queue(ledger, config["kind"], transport, clock=lambda: tick[0], sleep=lambda seconds: tick.__setitem__(0, tick[0] + seconds))
    try:
        _, answers = q.one(u)
        assert answers == {"fight": {"score": .8} if backend == "decision" else {"answer": "yes"}}
        assert len(calls) == (2 if field == "score" else 1)
        if field == "metadata":
            row = ledger.rows[-1]
            assert row["metadata"]["usage"] == {"completion_tokens": 5}
            assert "total_cost" not in row["metadata"] and "native_tokens_prompt" not in row["metadata"]
            from tools.unified_benchmark.contract import read_receipt
            receipt_path = tmp_path / "out" / row["metadata"]["receipt"]
            receipt = read_receipt(receipt_path)
            assert str(receipt["response"]["usage"]["cost"]) == str(huge)
            # The enclosing receipt is strict JSON even for Infinity/NaN raw fields.
            json.loads(receipt_path.read_text(), parse_constant=lambda x: pytest.fail(x))
            assert accounting(ledger)["attempts_without_cost_evidence"] == 1
    finally:
        ledger.close()


def test_nonfinite_raw_receipt_roundtrip_and_failed_write_leave_no_empty_file(tmp_path):
    from tools.unified_benchmark.contract import read_receipt, write_new, write_receipt
    path = tmp_path / "receipt.json"
    raw = json.loads('{"score":1e400}')
    write_receipt(path, {"response": raw, "generation_metadata": {"total_cost": float("nan")}})
    decoded = read_receipt(path)
    assert decoded["response"]["score"] == float("inf")
    assert str(decoded["generation_metadata"]["total_cost"]) == "nan"
    json.loads(path.read_text(), parse_constant=lambda x: pytest.fail(x))
    invalid = tmp_path / "invalid.json"
    with pytest.raises(ValueError):
        write_new(invalid, {"score": float("inf")})
    assert not invalid.exists()


@pytest.mark.parametrize("backend", ["language", "decision"])
@pytest.mark.parametrize("generation_data", [None, [], "invalid", {"id": "other"},
    {"id": "synthetic-generation", "provider_name": "Mock", "model": "test/model"}])
def test_api_generation_metadata_shape_preserves_inference_receipt(tmp_path, backend, generation_data):
    import threading
    from tools.unified_benchmark.api_run import Transport
    from tools.unified_benchmark.contract import file_hash, read_receipt
    (tmp_path / "f.jpg").write_bytes(b"synthetic-image")
    config = {"model": "test/model", "backend": backend, "kind": "probability" if backend == "decision" else "choice",
              "max_tokens": 30, "temperature": 0,
              "provider_context": {"config": {"expected_provider_name": "Mock", "provider": {"only": ["mock"]}}}}
    u = unit() | {"frames": [{"path": "f.jpg", "sha256": file_hash(tmp_path / "f.jpg"), "time": 0}], "question": "Fight?"}
    u["request_sha256"] = digest(a.request_body(config, u, tmp_path)[1])
    ledger = Ledger(tmp_path / "attempts.jsonl", {"test": True})
    transport = Transport(tmp_path, config, "synthetic-key", tmp_path / "out", ledger, threading.Event())
    raw = ({"answers": {"answer": {"type": "noul", "noul": .8}}}
           if backend == "decision" else response('{"answer":true}')) | {"id": "synthetic-generation", "model": "test/model"}
    calls = []
    def call(url, body=None):
        calls.append(body is not None)
        return raw if body is not None else {"data": generation_data}
    transport.call = call
    try:
        q = Queue(ledger, config["kind"], transport)
        if isinstance(generation_data, dict) and generation_data.get("provider_name") == "Mock":
            _, answers = q.one(u)
            assert answers == {"fight": {"score": .8} if backend == "decision" else {"answer": "yes"}}
        else:
            with pytest.raises(PausedRun, match="provider/model receipt identity unavailable"):
                q.one(u)
            assert ledger.rows[-1]["metadata"]["metadata_lookup_error"] == "unavailable"
        assert calls == [True, False]  # One inference POST, then one metadata GET.
        receipt_path = tmp_path / "out" / ledger.rows[-1]["metadata"]["receipt"]
        receipt = read_receipt(receipt_path)
        assert receipt["response"] == raw and receipt["generation_metadata"] == generation_data
        assert ledger.rows[-1]["metadata"]["receipt_sha256"] == file_hash(receipt_path)
    finally:
        ledger.close()


@pytest.mark.parametrize("envelope", [None, [], "invalid", {}, {"answer": []}, {"answer": None},
                                      {"answer": "invalid"}, {"answer": 1}],
                         ids=["null", "array", "string", "missing", "answer-array", "answer-null", "answer-string", "answer-number"])
def test_malformed_luna_envelope_retains_receipt_and_retries_in_transport_queue(tmp_path, envelope):
    import threading
    from tools.unified_benchmark.api_run import Transport
    from tools.unified_benchmark.contract import file_hash
    (tmp_path / "f.jpg").write_bytes(b"synthetic-image")
    config = {"model": "test/model", "backend": "decision", "kind": "probability",
              "provider_context": {"config": {"expected_provider_name": "Mock", "provider": {"only": ["mock"]}}}}
    u = unit() | {"frames": [{"path": "f.jpg", "sha256": file_hash(tmp_path / "f.jpg"), "time": 0}], "question": "Fight?"}
    u["request_sha256"] = digest(a.request_body(config, u, tmp_path)[1])
    ledger = Ledger(tmp_path / "attempts.jsonl", {"test": True})
    transport = Transport(tmp_path, config, "synthetic-key", tmp_path / "out", ledger, threading.Event())
    calls = []
    def call(url, body=None):
        calls.append(body)
        return {"answers": envelope if len(calls) == 1 else {"answer": {"type": "noul", "noul": .7}},
                "model": "test/model", "provider": "Mock"}
    transport.call = call
    clock = Clock()
    queue = Queue(ledger, "probability", transport, clock=clock.now, sleep=clock.sleep)
    try:
        _, answers = queue.one(u)
        assert answers == {"fight": {"score": .7}} and len(calls) == 2
        results = [r for r in ledger.rows if r["type"] == "result"]
        assert results[0]["answers"] == {} and not results[0]["complete"]
        assert "decision answer envelope" in results[0]["metadata"]["answer_error"]
        assert [r["attempt"] for r in results] == [1, 2]
        for row in results:
            path = tmp_path / "out" / row["metadata"]["receipt"]
            assert file_hash(path) == row["metadata"]["receipt_sha256"]
        assert len(list((tmp_path / "out/receipts").glob("*.json"))) == 2
    finally:
        ledger.close()


@pytest.mark.parametrize("bad_attempt", [True, 1.0, "1", 0, 4])
def test_ledger_rejects_invalid_result_attempt_identity(tmp_path, bad_attempt):
    ledger, _, _ = queue(tmp_path, lambda u: ({}, {}))
    try:
        base = {"unit": "u", "request_sha256": "frozen"}
        ledger.append(base | {"type": "intent", "attempt": 1})
        ledger.append(base | {"type": "result", "attempt": bad_attempt, "answers": {"fight": {"answer": "yes"}}})
        with pytest.raises(ValueError, match="invalid ledger attempt|result.*preceding intent"):
            ledger.state(unit(), "choice")
    finally:
        ledger.close()


def test_late_receipt_result_for_preceding_crash_intent_keeps_first_answer(tmp_path):
    ledger, _, _ = queue(tmp_path, lambda u: ({}, {}))
    try:
        base = {"unit": "u", "request_sha256": "frozen"}
        ledger.append(base | {"type": "intent", "attempt": 1})
        ledger.append(base | {"type": "intent", "attempt": 2})
        ledger.append(base | {"type": "result", "attempt": 1, "answers": {"fight": {"answer": "yes"}}})
        assert ledger.state(unit(), "choice") == (2, {"fight": {"answer": "yes"}})
    finally:
        ledger.close()


@pytest.mark.parametrize("order", [(1, 2, 3), (1, 3, 2), (2, 1, 3), (2, 3, 1), (3, 1, 2), (3, 2, 1)])
@pytest.mark.parametrize("kind", ["choice", "probability", "margin"])
@pytest.mark.parametrize("invalid_first", [False, True])
def test_first_valid_answers_follow_attempt_order_not_append_order(tmp_path, order, kind, invalid_first):
    ledger, _, _ = queue(tmp_path, lambda u: ({}, {}), kind)
    u = unit(targets=["fight", "smoking"])
    if kind == "choice":
        first, later, smoke, changed_smoke = ({"answer": value} for value in ("yes", "no", "unknown", "yes"))
        invalid = {"answer": "malformed"}
    else:
        first, later, smoke, changed_smoke = ({"score": value} for value in (.7, .2, .8, .1))
        invalid = {"score": True}
    answers = {1: {"fight": invalid if invalid_first else first},
               2: {"fight": later, "smoking": smoke}, 3: {"smoking": changed_smoke}}
    try:
        base = {"unit": u["id"], "request_sha256": u["request_sha256"]}
        for attempt in (1, 2, 3):
            ledger.append(base | {"type": "intent", "attempt": attempt})
        for attempt in order:
            ledger.append(base | {"type": "result", "attempt": attempt, "answers": answers[attempt]})
        before = ledger.path.read_bytes()
        assert ledger.state(u, kind) == (3, {"fight": later if invalid_first else first, "smoking": smoke})
        assert ledger.path.read_bytes() == before
    finally:
        ledger.close()


def test_offline_report_keeps_inference_date_separate_from_report_date(tmp_path, monkeypatch):
    from datetime import datetime, timezone
    from tools.unified_benchmark import retry, workflow

    ledger = Ledger(tmp_path / "attempts.jsonl", {"test": True})
    config = {"kind": "choice", "model": "synthetic"}
    try:
        for timestamp, uid in ((1700000000, "u"), (1700086400, "v")):
            monkeypatch.setattr(retry.time, "time", lambda: timestamp)
            ledger.append({"type": "intent", "unit": uid, "attempt": 1, "request_sha256": "frozen"})
            ledger.append({"type": "result", "unit": uid, "attempt": 1, "request_sha256": "frozen",
                           "answers": {"fight": {"answer": "yes"}}, "complete": True, "elapsed_seconds": .5})
        before = ledger.path.read_bytes()
        first = workflow.report(plan(), config, ledger, None)
        again = workflow.report(plan(), config, ledger, None)
        expected = datetime.fromtimestamp(1700000000, timezone.utc).isoformat()
        assert first["run_date_utc"] == again["run_date_utc"] == expected
        assert first["report_generated_at_utc"] != expected
        assert first["accuracy"] == again["accuracy"] == 1
        assert ledger.path.read_bytes() == before
    finally:
        ledger.close()


def test_report_without_attempts_does_not_invent_an_inference_date(tmp_path):
    from tools.unified_benchmark.workflow import report
    ledger = Ledger(tmp_path / "attempts.jsonl", {"test": True})
    try:
        result = report(plan(), {"kind": "choice", "model": "synthetic"}, ledger, None)
        assert result["run_date_utc"] is None
        assert result["report_generated_at_utc"]
        assert result["unattempted_unique_units"] == 2
    finally:
        ledger.close()


@pytest.mark.parametrize("response_case", ["invalid_json", "invalid_utf8", "oversized"])
def test_invalid_http_json_retains_bounded_raw_receipt_and_three_attempts(tmp_path, monkeypatch, response_case):
    import base64
    import io
    import threading
    from tools.unified_benchmark import api_run
    from tools.unified_benchmark.contract import file_hash

    key = "synthetic-secret"
    limit = 2 * 1024 * 1024
    raw = b"<html>upstream error " + key.encode() + b"</html>"
    oversized = response_case == "oversized"
    if response_case == "invalid_utf8":
        raw += b"\xff"
    if oversized:
        raw += b"x" * limit
    calls = []
    class Response(io.BytesIO):
        status = 200
    def urlopen(*args, **kwargs):
        calls.append(1)
        return Response(raw)
    monkeypatch.setattr(api_run.urllib.request, "urlopen", urlopen)
    (tmp_path / "f.jpg").write_bytes(b"synthetic-image")
    config = {"model": "qwen/test", "backend": "language", "kind": "choice", "max_tokens": 30, "temperature": 0,
              "provider_context": {"config": {"expected_provider_name": "Mock", "provider": {"only": ["mock"], "allow_fallbacks": False}}}}
    u = unit() | {"frames": [{"path": "f.jpg", "sha256": file_hash(tmp_path / "f.jpg"), "time": 0}], "question": "Is there a fight?"}
    u["request_sha256"] = digest(a.request_body(config, u, tmp_path)[1])
    ledger = Ledger(tmp_path / "attempts.jsonl", {"test": True})
    clock = Clock()
    client = api_run.Transport(tmp_path, config, key, tmp_path / "output", ledger, threading.Event())
    try:
        observed = Queue(ledger, "choice", client, slots=1, clock=clock.now, sleep=clock.sleep).run([u])
        assert observed["u"] == {} and len(calls) == 3
        receipts = list((tmp_path / "output/receipts").glob("*.json"))
        assert len(receipts) == 3
        for path in receipts:
            saved = json.loads(path.read_text())
            assert saved["http_status"] == 200
            assert saved["body_truncated"] is oversized
            assert base64.b64decode(saved["body_base64"]) == raw[:limit].replace(key.encode(), b"[REDACTED]")
            assert key not in path.read_text()
        assert api_run.accounting(ledger)["attempts_without_cost_evidence"] == 3
    finally:
        ledger.close()
