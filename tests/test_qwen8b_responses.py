"""Offline response contracts: synthetic tokens, images and receipts; no model."""
from contextlib import nullcontext
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

pytest.importorskip("fcntl", reason="local evaluation ledger requires POSIX file locking")

from tools.qwen8b import eval_local, serve
from tools.unified_benchmark.adapters import parse_response, request_body
from tools.unified_benchmark.contract import digest, file_hash
from tools.unified_benchmark.retry import RequestFailure


@pytest.fixture(autouse=True)
def forbid_network(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("response regression must not access a real endpoint")
    monkeypatch.setattr("urllib.request.urlopen", forbidden)
    monkeypatch.setattr("socket.create_connection", forbidden)


@pytest.mark.parametrize("tokens,eos_ids,finish", [
    ([7] * 30, [98, 99], "length"),
    ([7] * 30, None, "length"),
    ([7] * 29 + [99], [98, 99], "stop"),
    ([7] * 29 + [99], 99, "stop"),
    ([7, 98], [98, 99], "stop"),
])
def test_generation_reports_budget_exhaustion_to_real_parser(monkeypatch, tokens, eos_ids, finish):
    class Processor:
        def apply_chat_template(self, *args, **kwargs):
            return "synthetic prompt"

        def __call__(self, **kwargs):
            return {"input_ids": np.array([[1, 2]])}

        def batch_decode(self, actual, **kwargs):
            assert actual.tolist() == [tokens]
            # A complete JSON prefix must not disguise token exhaustion.
            return ['{"answer":true} This explanation was cut mid-sentence']

    class Model:
        device = "synthetic"
        generation_config = SimpleNamespace(eos_token_id=eos_ids)

        def generate(self, **kwargs):
            assert kwargs["max_new_tokens"] == 30
            assert kwargs["do_sample"] is False
            return np.array([[1, 2] + tokens])

    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(inference_mode=nullcontext))
    monkeypatch.setattr(serve, "_messages", lambda body: [
        {"role": "user", "content": [{"type": "image", "image": None}]}])
    receipt = serve.generate(Processor(), Model(), {"max_tokens": 30})
    assert receipt["choices"][0]["finish_reason"] == finish
    if finish == "length":
        with pytest.raises(ValueError):
            parse_response({"backend": "language"}, {"targets": ["fight"]}, receipt)
    else:
        assert parse_response({"backend": "language"}, {"targets": ["fight"]}, receipt) == {
            "fight": {"answer": "yes"}}


@pytest.fixture
def local_transport(tmp_path):
    profile = Path(__file__).resolve().parents[1] / "tools/unified_benchmark/profiles/qwen8b_r2_host.json"
    config = json.loads(profile.read_text())
    frame = tmp_path / "frame.jpg"
    Image.new("RGB", (8, 8)).save(frame)
    unit = {"id": "synthetic", "targets": ["fight"], "question": "Is fighting visible?",
            "frames": [{"path": frame.name, "sha256": file_hash(frame)}]}
    unit["request_sha256"] = digest(request_body(config, unit, tmp_path)[1])
    ledger = SimpleNamespace(state=lambda *args: (1, {}))
    transport = eval_local.LocalTransport(tmp_path, config, "http://unused.invalid", tmp_path, ledger)
    return transport, unit, tmp_path


@pytest.mark.parametrize("identity", [None, "", "  ", 42, {}, [], "different/model"])
def test_local_transport_rejects_unverified_model_and_keeps_receipt(local_transport, monkeypatch, identity):
    transport, unit, output = local_transport
    payload = {"choices": [{"finish_reason": "stop", "message": {"content": '{"answer":true}'}}]}
    if identity is not None:
        payload["model"] = identity
    monkeypatch.setattr(transport, "call", lambda body: payload)
    with pytest.raises(RequestFailure, match="actual model") as failure:
        transport(unit)
    assert failure.value.fatal
    receipts = list((output / "receipts").glob("*.json"))
    assert len(receipts) == 1
    assert json.loads(receipts[0].read_text())["response"] == payload


@pytest.mark.parametrize("payload", [None, [], "not an object"])
def test_local_transport_rejects_nonobject_receipt(local_transport, monkeypatch, payload):
    transport, unit, _output = local_transport
    monkeypatch.setattr(transport, "call", lambda body: payload)
    with pytest.raises(RequestFailure, match="actual model identity") as failure:
        transport(unit)
    assert failure.value.fatal


def test_local_transport_accepts_receipted_expected_identity(local_transport, monkeypatch):
    transport, unit, output = local_transport
    payload = {"model": transport.expected_model,
               "choices": [{"finish_reason": "stop", "message": {"content": '{"answer":true}'}}]}
    monkeypatch.setattr(transport, "call", lambda body: payload)
    answers, metadata = transport(unit)
    assert answers == {"fight": {"answer": "yes"}}
    assert metadata["actual_model"] == payload["model"]
    assert json.loads((output / metadata["receipt"]).read_text())["response"] == payload
