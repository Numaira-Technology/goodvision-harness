"""Offline producer/consumer regressions found in independent stack review."""

import io
import shlex
import subprocess
import sys
import threading
import urllib.error
import urllib.request

import pytest

from tools import diag_qwen8b_vllm_import as diag
from tools.unified_benchmark import adapters, api_run
from tools.unified_benchmark.contract import digest, file_hash
from tools.unified_benchmark.preflight import model_configs
from tools.unified_benchmark.retry import Ledger, Queue


def unit(root, config):
    image = root / "synthetic.jpg"
    image.write_bytes(b"synthetic body-only test fixture")
    result = {
        "id": "synthetic/0",
        "targets": ["fight"],
        "question": "Is fighting visible?",
        "frames": [{"path": image.name, "sha256": file_hash(image), "time": 0}],
    }
    result["request_sha256"] = digest(adapters.request_body(config, result, root)[1])
    return result


@pytest.mark.parametrize("score", [0, 0.123456789, 0.75, 1])
def test_luna_real_adapter_and_transport_queue_preserve_probability(tmp_path, score):
    config = model_configs(["luna"])["luna"]
    request = unit(tmp_path, config)
    ledger = Ledger(tmp_path / "ledger.jsonl", {"synthetic": True})
    transport = api_run.Transport(
        tmp_path, config, "SYNTHETIC", tmp_path / "out", ledger, threading.Event()
    )
    transport.call = lambda *a: {
        "id": "synthetic",
        "provider": "OpenAI",
        "model": config["model"],
        "answers": {"answer": {"type": "noul", "noul": score}},
    }
    try:
        observed = Queue(ledger, "probability", transport, slots=1).run([request])
        assert observed[request["id"]]["fight"]["score"] == score
        assert ledger.state(request, "probability")[0] == 1
        assert len(list((tmp_path / "out/receipts").glob("*.json"))) == 1
    finally:
        ledger.close()


@pytest.mark.parametrize(
    "answer",
    [
        None,
        [],
        {"type": "bool", "noul": 0.5},
        {"type": "noul", "error": "refused", "noul": 0.5},
        *[
            {"type": "noul", "noul": value}
            for value in (
                True,
                False,
                "0.5",
                None,
                float("nan"),
                float("inf"),
                -0.1,
                1.1,
                10**1000,
            )
        ],
    ],
)
def test_luna_malformed_probability_is_not_a_valid_answer(answer):
    with pytest.raises(ValueError):
        adapters.parse_response(
            {"backend": "decision"},
            {"targets": ["fight"]},
            {"answers": {"answer": answer}},
        )


def test_ambient_url_cannot_redirect_selected_cloud_credential(tmp_path, monkeypatch):
    config = model_configs(["qwen8b"])["qwen8b"]
    request = unit(tmp_path, config)
    monkeypatch.setenv("GOODVISION_CHAT_URL", "https://synthetic.invalid/unapproved")
    uri, body = adapters.request_body(config, request, tmp_path)
    assert uri == "https://openrouter.ai/api/v1/chat/completions"
    assert digest(body) == request["request_sha256"]
    transport = api_run.Transport(
        tmp_path, config, "SYNTHETIC", tmp_path, None, threading.Event()
    )
    sent = []

    def send(req, **kwargs):
        sent.append(req)
        return io.BytesIO(b"{}")

    monkeypatch.setattr(transport.opener, "open", send)
    assert transport.call(uri, body) == {}
    assert sent[0].full_url == uri
    assert sent[0].get_header("Authorization") == "Bearer SYNTHETIC"


@pytest.mark.parametrize(
    "url",
    [
        "https://synthetic.invalid/api/v1/chat/completions",
        "http://openrouter.ai/api/v1/chat/completions",
        "https://openrouter.ai@synthetic.invalid/api/v1/chat/completions",
        "https://openrouter.ai:444/api/v1/chat/completions",
        "https://openrouter.ai/unknown",
        "https://openrouter.ai/api/v1/chat/completions#fragment",
    ],
)
def test_cloud_transport_refuses_unapproved_endpoint_before_network(
    tmp_path, monkeypatch, url
):
    transport = api_run.Transport(
        tmp_path, {}, "SYNTHETIC", tmp_path, None, threading.Event()
    )
    monkeypatch.setattr(
        transport.opener, "open", lambda *a, **k: pytest.fail("network")
    )
    with pytest.raises(ValueError, match="unapproved endpoint"):
        transport.call(url, {})


def test_cloud_redirect_never_forwards_bearer_key():
    req = urllib.request.Request(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={"Authorization": "Bearer SYNTHETIC"},
    )
    with pytest.raises(urllib.error.HTTPError, match="redirects are refused"):
        api_run._NoRedirect().redirect_request(
            req, None, 302, "redirect", {}, "https://synthetic.invalid/steal"
        )


def test_diagnostic_uses_activate_sibling_and_stops_if_activation_fails(tmp_path):
    packed_python = tmp_path / "prefix/bin/python"
    remote = diag._old_verify_remote(str(packed_python))
    inner = shlex.split(remote)[2]
    assert f"source {packed_python.with_name('activate')}" in inner
    marker = tmp_path / "fell-through"
    script = f"python() {{ touch {shlex.quote(str(marker))}; }}; " + inner
    result = subprocess.run(["bash", "-c", script], capture_output=True, text=True)
    assert result.returncode != 0 and not marker.exists()


def test_new_diagnostic_uses_isolated_packed_prefix(monkeypatch):
    monkeypatch.setenv("PYTHONPATH", "/synthetic/injection")
    remote = diag._isolated_probe_remote(sys.executable, "print('synthetic-probe-ok')")
    result = subprocess.run(
        ["bash", "-c", remote], capture_output=True, text=True, check=True
    )
    assert result.stdout.strip() == "synthetic-probe-ok"
