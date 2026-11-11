"""Synthetic Luna receipts through the real adapter/queue; no model or network."""
import json
import socket
import threading
import urllib.request

import pytest


from tools.unified_benchmark.adapters import parse_response, request_body
from tools.unified_benchmark.api_run import Transport
from tools.unified_benchmark.contract import EVENTS, digest, file_hash, score, select_thresholds
from tools.unified_benchmark.preflight import model_configs
from tools.unified_benchmark.retry import Ledger, Queue


@pytest.fixture(autouse=True)
def forbid_network(monkeypatch):
    def blocked(*args, **kwargs):
        pytest.fail("real network/model calls forbidden in Luna offline tests")
    monkeypatch.setattr(urllib.request, "urlopen", blocked)
    monkeypatch.setattr(socket, "create_connection", blocked)


@pytest.fixture
def config():
    return model_configs(["luna"])["luna"]


def receipt(value):
    return {"answers": {"answer": {"type": "noul", "noul": value}}}


@pytest.mark.parametrize("event", ["fight", "bag_knife"])
@pytest.mark.parametrize("value", [0, 1, 0.12345678901234568])
def test_luna_keeps_full_probability_without_applying_a_threshold(config, event, value):
    assert parse_response(config, {"targets": [event]}, receipt(value)) == {event: {"score": value}}


@pytest.mark.parametrize("payload", [
    None, [], {}, {"answers": None}, {"answers": []}, {"answers": {}},
    {"answers": {"answer": None}}, {"answers": {"answer": []}},
    {"answers": {"answer": {}}}, {"answers": {"answer": {"type": "noul"}}},
    {"answers": {"answer": {"type": "refusal", "message": "refused"}}},
    receipt(.8) | {"error": {"message": "refused"}},
    {"answers": {"answer": {"type": "noul", "noul": .8, "error": "refused"}}},
] + [receipt(value) for value in (True, False, "0.8", "false", None, -0.01, 1.01, 10**400,
                                   float("nan"), float("inf"), -float("inf"))])
def test_luna_rejects_missing_malformed_refused_or_nonprobability_answers(config, payload):
    with pytest.raises(ValueError):
        parse_response(config, {"targets": ["fight"]}, payload)


class Clock:
    def __init__(self):
        self.time = 0

    def now(self):
        return self.time

    def sleep(self, seconds):
        self.time += seconds


def planned_unit(root, config, uid="u", event="fight"):
    # Transport tests need frozen bytes, not a model-decodable image.
    frame = root / "synthetic.jpg"
    if not frame.exists():
        frame.write_bytes(b"synthetic offline image")
    unit = {"id": uid, "targets": [event], "question": "Synthetic visible-event question?",
            "frames": [{"path": frame.name, "sha256": file_hash(frame), "time": 0}]}
    _, body = request_body(config, unit, root)
    assert "reference" not in body and "proposed_answer" not in body
    return unit | {"request_sha256": digest(body)}


def transport(root, config, ledger, payloads, monkeypatch):
    saved = iter(payloads)
    client = Transport(root, config, "synthetic-not-a-key", root / "output", ledger, threading.Event())
    def call(url, body=None):
        assert url.endswith("/api/alpha/decisions")
        return next(saved) | {"provider": "OpenAI", "model": config["model"], "usage": {"cost": 0}}
    monkeypatch.setattr(client, "call", call)
    return client


@pytest.mark.parametrize("recover", [True, False])
def test_luna_transport_retry_receipts_and_final_missing(tmp_path, config, monkeypatch, recover):
    unit = planned_unit(tmp_path, config)
    plan = {"split": "test", "units": [unit], "questions": [
        {"number": 1, "event": "fight", "reference": "yes", "unit_ids": [unit["id"]]}]}
    # A valid but wrong answer must also stop retries; malformed is not a negative.
    payloads = [receipt("false"), receipt(.2)] if recover else [receipt("false")] * 3
    ledger, clock = Ledger(tmp_path / "attempts.jsonl", {"synthetic": True}), Clock()
    try:
        client = transport(tmp_path, config, ledger, payloads, monkeypatch)
        queue = Queue(ledger, "probability", client, slots=1, clock=clock.now, sleep=clock.sleep)
        observed = queue.run([unit])
        queue.run([unit])  # No resend after success or after exhausting the budget.
        used, answers = ledger.state(unit, "probability")
        assert used == (2 if recover else 3)
        assert clock.time == (30 if recover else 90)
        assert answers == ({"fight": {"score": .2}} if recover else {})
        result = score(plan, observed, "probability")
        assert result["accuracy"] == 0 and result["accuracy_denominator"] == 1
        assert result["complete_valid_questions"] == int(recover)
        assert result["final_missing_unique_units"] == int(not recover)
        assert result["incomplete_questions"] == int(not recover)
        receipts = list((tmp_path / "output/receipts").glob("*.json"))
        assert len(receipts) == used
        assert all(json.loads(path.read_text())["request_sha256"] == unit["request_sha256"] for path in receipts)
        assert any(json.loads(path.read_text())["response"]["answers"]["answer"]["noul"] == "false" for path in receipts)
    finally:
        ledger.close()


def test_luna_complete_validation_receipts_reach_threshold_selection(tmp_path, config, monkeypatch):
    plan = {"split": "validation", "units": [], "questions": []}
    payloads = []
    for event in EVENTS[:4]:
        for index in range(6):
            unit = planned_unit(tmp_path, config, f"{event}/{index}", event)
            positive = index < 3
            plan["units"].append(unit)
            plan["questions"].append({"number": len(plan["questions"]) + 1, "event": event,
                "reference": "yes" if positive else "no", "unit_ids": [unit["id"]]})
            payloads.append(receipt(.8 if positive else .2))
    ledger, clock = Ledger(tmp_path / "attempts.jsonl", {"synthetic": True}), Clock()
    try:
        client = transport(tmp_path, config, ledger, payloads, monkeypatch)
        observed = Queue(ledger, "probability", client, slots=1, clock=clock.now, sleep=clock.sleep).run(plan["units"])
        selected = select_thresholds(plan, observed, "probability", config["model"])
        assert selected["validation_summary"]["complete_valid_questions"] == 24
        assert selected["comparator"] == ">=" and selected["validation_only"] is True
        assert selected["thresholds"]["bag_knife"] == selected["thresholds"]["bag_alcohol_bottle"] == .5
        assert score(plan, observed, "probability", selected["thresholds"])["correct"] == 24
        assert len(list((tmp_path / "output/receipts").glob("*.json"))) == 24
        assert all(ledger.state(unit, "probability")[0] == 1 for unit in plan["units"])
        observed.pop(plan["units"][0]["id"])
        with pytest.raises(ValueError, match="incomplete validation"):
            select_thresholds(plan, observed, "probability", config["model"])
    finally:
        ledger.close()
