"""Synthetic latency regressions: no credentials, paid requests or model loading."""

import importlib.util
import io
import json
import shutil
import subprocess
import sys
import urllib.error
from pathlib import Path
from types import SimpleNamespace

import pytest

SCRIPT = (
    Path(__file__).resolve().parents[1]
    / ".claude/skills/event-benchmark/scripts/latency_bench.py"
)
spec = importlib.util.spec_from_file_location("skill_latency", SCRIPT)
latency = importlib.util.module_from_spec(spec)
spec.loader.exec_module(latency)


@pytest.fixture(autouse=True)
def forbid_real_network(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("latency regressions must not contact a real endpoint")
    monkeypatch.setattr(latency.urllib.request, "urlopen", forbidden)
    monkeypatch.setattr("socket.create_connection", forbidden)


@pytest.fixture
def provider_config(tmp_path):
    path = tmp_path / "provider.json"
    path.write_text(json.dumps({
        "schema_version": 1, "model": latency.CANDIDATES["qwen"][0],
        "backend": "language", "expected_provider_name": "Synthetic Provider",
        "provider": {"only": ["synthetic-endpoint"], "allow_fallbacks": False},
        "selection": "offline test fixture", "selected_at": "2026-11-10T00:00:00Z",
    }))
    return path


def chat(content='{"answer":false}', finish="stop"):
    return json.dumps(
        {"choices": [{"finish_reason": finish, "message": {"content": content}}]}
    ).encode()


def decision(score=0.5, **extra):
    return json.dumps(
        {"answers": {"answer": {"type": "noul", "noul": score, **extra}}}
    ).encode()


@pytest.mark.parametrize(
    "backend,body",
    [
        ("language", b"not JSON"),
        ("language", b'{"error":{"message":"provider failed"}}'),
        ("language", chat('{"answer":"false"}')),
        ("language", chat('{"answer":0}')),
        ("language", chat('{"answer":null}')),
        ("language", chat('{"answer":false,"answer":true}')),
        ("language", chat('{"answer":false,"other":true}')),
        ("language", chat(finish="length")),
        ("language", b'{"choices":[],"choices":[]}'),
        ("language", b"null"),
        ("decision", decision(True)),
        ("decision", decision("0.5")),
        ("decision", decision(float("nan"))),
        ("decision", decision(float("inf"))),
        ("decision", decision(-0.1)),
        ("decision", decision(1.1)),
        ("decision", decision(0.5, error="provider failed")),
        ("decision", b'{"answers":{"answer":{"type":"noul","noul":0.2,"noul":0.7}}}'),
        (
            "decision",
            b'{"error":"provider failed",'
            b'"answers":{"answer":{"type":"noul","noul":0.2}}}',
        ),
        pytest.param(
            "language",
            b"x" * (latency.MAX_RESPONSE_BYTES + 1),
            id="oversized-response",
        ),
    ],
)
def test_http_200_invalid_answers_fail_without_retries(monkeypatch, backend, body):
    requests = []

    def respond(request, **kwargs):
        requests.append(request)
        return io.BytesIO(body)

    monkeypatch.setattr(latency.urllib.request, "urlopen", respond)
    result = latency.one_call("https://example.invalid", {}, "synthetic-key", backend)
    assert result["ok"] is False
    assert result["error"] == "invalid_response"
    assert result["elapsed_s"] >= 0
    assert len(requests) == 1
    assert "synthetic-key" not in json.dumps(result)


@pytest.mark.parametrize(
    "backend,body",
    [
        ("language", chat()),
        ("language", chat('{"answer":true}')),
        ("decision", decision(0)),
        ("decision", decision(0.5)),
        ("decision", decision(1)),
    ],
)
def test_valid_answers_are_timed(monkeypatch, backend, body):
    monkeypatch.setattr(
        latency.urllib.request, "urlopen", lambda *a, **k: io.BytesIO(body)
    )
    result = latency.one_call("https://example.invalid", {}, "synthetic-key", backend)
    assert result["ok"] is True and result["elapsed_s"] >= 0


@pytest.mark.parametrize(
    "exception,error",
    [
        (
            urllib.error.HTTPError(
                "https://example.invalid", 429, "secret body", {}, None
            ),
            "http_429",
        ),
        (urllib.error.URLError("secret body"), "transport_error"),
        (TimeoutError("secret body"), "transport_error"),
    ],
)
def test_transport_errors_are_counted_and_sanitized(monkeypatch, exception, error):
    def fail(*args, **kwargs):
        raise exception

    monkeypatch.setattr(latency.urllib.request, "urlopen", fail)
    result = latency.one_call(
        "https://example.invalid", {}, "synthetic-key", "language"
    )
    assert result["ok"] is False and result["error"] == error
    assert "secret" not in json.dumps(result)


def test_http_failure_keeps_safe_diagnostics_without_retrying(monkeypatch):
    from email.message import Message

    headers = Message()
    headers["Retry-After"] = "60"
    headers["X-Request-Id"] = "request-synthetic-key"
    headers["Authorization"] = "Bearer synthetic-key"
    body = json.dumps({"error": {
        "code": 429, "message": "Limited synthetic-key", "type": "rate_limit",
        "metadata": {"provider_name": "Provider synthetic-key", "raw": "omit this"},
    }}).encode()
    attempts = []

    def fail(request, **kwargs):
        attempts.append(request)
        raise urllib.error.HTTPError(request.full_url, 429, "limited", headers, io.BytesIO(body))

    monkeypatch.setattr(latency.urllib.request, "urlopen", fail)
    result = latency.one_call("https://example.invalid", {}, "synthetic-key", "language")
    assert len(attempts) == 1
    assert result["ok"] is False and result["error"] == "http_429"
    assert result["http_error_headers"]["retry-after"] == "60"
    assert result["http_error_detail"]["message"] == "Limited [REDACTED]"
    assert result["http_error_detail"]["provider_name"] == "Provider [REDACTED]"
    assert "authorization" not in result["http_error_headers"]
    assert "synthetic-key" not in json.dumps(result)
    assert "omit this" not in json.dumps(result)


@pytest.mark.parametrize("body", [b"not JSON", b"null", b"x" * (latency.MAX_RESPONSE_BYTES + 1)])
def test_unreadable_http_error_body_preserves_original_status(monkeypatch, body):
    def fail(request, **kwargs):
        raise urllib.error.HTTPError(request.full_url, 429, "limited", None, io.BytesIO(body))

    monkeypatch.setattr(latency.urllib.request, "urlopen", fail)
    result = latency.one_call("https://example.invalid", {}, "synthetic-key", "language")
    assert result["ok"] is False and result["error"] == "http_429"
    assert result["http_error_headers"] == {}
    assert "http_error_detail" not in result


@pytest.mark.parametrize("backend", ["language", "decision"])
def test_request_envelope_matches_canonical_runner(monkeypatch, backend):
    captured = []
    response = json.loads(chat() if backend == "language" else decision())
    monkeypatch.setattr(
        latency.runner, "_post", lambda *a: captured.append(a) or response
    )
    latency.runner.frame_score(
        backend, "model", latency.QCORE, "image", "synthetic-key"
    )
    url, body = latency.build_body(backend, "model", "image")
    assert captured == [(url, body, "synthetic-key")]
    assert set(latency.CANDIDATES) == {"qwen", "gemini", "luna"}


def test_failures_remain_in_attempt_counts_but_not_conditional_percentiles():
    result = latency.summarize(
        [
            {"ok": True, "elapsed_s": 2},
            {"ok": False, "elapsed_s": 90, "error": "transport_error"},
        ],
        [
            {"ok": True, "elapsed_s": 1},
            {"ok": True, "elapsed_s": 3},
            {"ok": False, "elapsed_s": 90, "error": "transport_error"},
        ],
    )
    assert result["complete"] is False
    assert (result["single_n"], result["single_n_ok"], result["single_n_failed"]) == (
        2,
        1,
        1,
    )
    assert (result["conc_n"], result["conc_n_ok"], result["conc_n_failed"]) == (3, 2, 1)
    assert result["single_inference_s"] == 2
    assert result["p50_s"] == 2 and result["p95_s"] == pytest.approx(2.9)
    assert len(result["concurrent_attempts"]) == 3


def test_all_failures_produce_null_statistics_and_strict_json():
    failure = {"ok": False, "elapsed_s": 90, "error": "transport_error"}
    result = latency.summarize([failure], [failure])
    assert result["complete"] is False
    assert (
        result["single_inference_s"] is None
        and result["p50_s"] is None
        and result["p95_s"] is None
    )
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("flag", ["--single-n", "--concurrency", "--rounds"])
@pytest.mark.parametrize("value", ["0", "-1"])
def test_invalid_counts_stop_before_credentials_or_network(monkeypatch, flag, value):
    def forbidden(*args, **kwargs):
        pytest.fail("input/key/network called for invalid counts")

    monkeypatch.setattr(latency, "prepare_frame", forbidden)
    monkeypatch.setattr(latency, "load_key", forbidden)
    monkeypatch.setattr(latency.urllib.request, "urlopen", forbidden)
    with pytest.raises(SystemExit) as error:
        latency.main(["--candidate", "qwen", "--frame", "fixture.jpg", flag, value])
    assert error.value.code == 2


@pytest.mark.parametrize("video", [False, True])
def test_both_input_modes_normalize_and_verify_jpeg(monkeypatch, video):
    calls = []
    jpeg = b"\xff\xd8synthetic\xff\xd9"

    def run(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(
            stdout=jpeg
            if command[0] == "ffmpeg"
            else json.dumps(
                {"streams": [{"codec_name": "mjpeg", "width": 720, "height": 404}]}
            ).encode()
        )

    monkeypatch.setattr(latency.subprocess, "run", run)
    assert latency.prepare_frame(Path("fixture.png"), video=video) == jpeg
    command, kwargs = calls[0]
    assert "scale=720:-2" in command
    assert ("-ss" in command) == video
    assert kwargs["check"] is True and kwargs["timeout"] == latency.TIMEOUT
    assert calls[1][1]["input"] == jpeg


@pytest.mark.parametrize(
    "stream",
    [
        {"codec_name": "png", "width": 720, "height": 404},
        {"codec_name": "mjpeg", "width": 640, "height": 404},
        {"codec_name": "mjpeg", "width": 720, "height": 405},
        {"codec_name": "mjpeg", "width": 720, "height": 0},
    ],
)
def test_invalid_prepared_dimensions_or_codec_fail(monkeypatch, stream):
    def run(command, **kwargs):
        return SimpleNamespace(
            stdout=b"\xff\xd8synthetic\xff\xd9"
            if command[0] == "ffmpeg"
            else json.dumps({"streams": [stream]}).encode()
        )

    monkeypatch.setattr(latency.subprocess, "run", run)
    with pytest.raises(ValueError):
        latency.prepare_frame(Path("fixture.jpg"))


@pytest.mark.parametrize("complete", [False, True])
def test_cli_writes_provenance_and_returns_failure_for_incomplete_run(
    tmp_path, monkeypatch, complete, provider_config
):
    jpeg = b"synthetic prepared image"
    monkeypatch.setattr(latency, "prepare_frame", lambda *a, **k: jpeg)
    monkeypatch.setattr(latency, "load_key", lambda *a: "synthetic-key")
    metadata = {"actual_provider": "Synthetic Provider", "actual_model": latency.CANDIDATES["qwen"][0]}
    attempts = iter([{"ok": complete, "elapsed_s": 1, "generation_id": "single-1", **metadata},
                     {"ok": True, "elapsed_s": 2, "generation_id": "concurrent-1", **metadata}])
    monkeypatch.setattr(latency, "one_call", lambda *a: next(attempts))
    output = tmp_path / "nested" / "result.json"
    exit_code = latency.main(
        [
            "--candidate",
            "qwen",
            "--frame",
            "synthetic.jpg",
            "--provider-config",
            str(provider_config),
            "--single-n",
            "1",
            "--concurrency",
            "1",
            "--rounds",
            "1",
            "--out",
            str(output),
        ]
    )
    result = json.loads(output.read_text())
    assert exit_code == (0 if complete else 1)
    assert result["complete"] is complete
    assert result["provider_observed"]["complete"] is True
    assert result["provider_context"]["config"]["provider"]["allow_fallbacks"] is False
    assert result["frame_sha256"] == latency.sha256_bytes(jpeg)
    assert result["runner_sha256"] == latency.sha256_bytes(
        latency.RUNNER_PATH.read_bytes()
    )
    assert result["script_sha256"] == latency.sha256_bytes(SCRIPT.read_bytes())
    assert all(len(result[key]) == 64 for key in ["request_sha256", "spec_sha256"])
    assert result["config"]["request_retries"] == 0
    assert result["spec_version"] == latency.runner.SPEC_VERSION
    assert result["run_at"]
    assert "synthetic-key" not in output.read_text()


@pytest.mark.skipif(
    not shutil.which("ffmpeg") or not shutil.which("ffprobe"),
    reason="FFmpeg/ffprobe required for synthetic image/video integration",
)
def test_real_ffmpeg_normalizes_synthetic_image_and_video(tmp_path):
    from PIL import Image

    # Plain portable pixels and a built-in lossless codec require no downloads,
    # private media, credentials or model calls.
    frame = tmp_path / "tiny.ppm"
    frame.write_bytes(b"P6\n32 18\n255\n" + b"\xff\x00\x00" * (32 * 18))
    video = tmp_path / "tiny.mkv"
    subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=c=black:s=32x18:r=2:d=1",
            "-c:v",
            "ffv1",
            str(video),
        ],
        capture_output=True,
        check=True,
        timeout=30,
    )
    for source, is_video in ((frame, False), (video, True)):
        jpeg = latency.prepare_frame(source, video=is_video)
        with Image.open(io.BytesIO(jpeg)) as image:
            image.load()
            assert image.format == "JPEG"
            assert image.width == 720
            assert image.height > 0 and image.height % 2 == 0


@pytest.mark.parametrize("with_provider", [False, True])
def test_offline_preflight_never_loads_credentials_or_calls_provider(
    tmp_path, monkeypatch, provider_config, with_provider
):
    monkeypatch.setattr(latency, "prepare_frame", lambda *a, **k: b"synthetic frame")
    monkeypatch.setattr(latency, "load_key", lambda *a: pytest.fail("preflight read credentials"))
    output = tmp_path / "preflight.json"
    args = ["--candidate", "qwen", "--frame", "synthetic.jpg", "--preflight", "--out", str(output)]
    if with_provider:
        args += ["--provider-config", str(provider_config)]
    assert latency.main(args) == 0
    result = json.loads(output.read_text())
    assert result["inference_calls"] == 0
    assert result["artifact_type"] == "offline-latency-preflight"
    assert bool(result["provider_context"]) is with_provider
    assert result["evidence_sha256"] == latency.sha256_bytes(
        Path(latency.runner.evidence.__file__).read_bytes())


def test_live_cli_requires_frozen_provider_before_credentials(monkeypatch):
    monkeypatch.setattr(latency, "prepare_frame", lambda *a, **k: b"synthetic frame")
    monkeypatch.setattr(latency, "load_key", lambda *a: pytest.fail("read credentials"))
    with pytest.raises(SystemExit) as error:
        latency.main(["--candidate", "qwen", "--frame", "synthetic.jpg"])
    assert error.value.code == 2


@pytest.mark.parametrize("mismatch", ["provider", "model"])
def test_identity_mismatch_keeps_successful_inference_but_fails_controlled_run(
    tmp_path, monkeypatch, provider_config, mismatch
):
    monkeypatch.setattr(latency, "prepare_frame", lambda *a, **k: b"synthetic frame")
    monkeypatch.setattr(latency, "load_key", lambda *a: "synthetic-key")
    requests = []

    def respond(request, **kwargs):
        body = json.loads(request.data)
        assert body["provider"] == {"only": ["synthetic-endpoint"], "allow_fallbacks": False}
        requests.append(request)
        receipt = json.loads(chat())
        receipt.update(id="synthetic-" + str(len(requests)),
                       provider="Wrong Provider" if mismatch == "provider" else "Synthetic Provider",
                       model="other/model" if mismatch == "model" else body["model"])
        return io.BytesIO(json.dumps(receipt).encode())

    monkeypatch.setattr(latency.urllib.request, "urlopen", respond)
    output = tmp_path / "mismatch.json"
    assert latency.main(["--candidate", "qwen", "--frame", "synthetic.jpg",
                         "--provider-config", str(provider_config), "--single-n", "1",
                         "--concurrency", "1", "--rounds", "1", "--out", str(output)]) == 1
    result = json.loads(output.read_text())
    assert result["inference_complete"] is True and result["complete"] is False
    assert result["provider_observed"]["provider_mismatch"] is (mismatch == "provider")
    if mismatch == "model":
        assert result["provider_observed"]["actual_models"] == ["other/model"]
    assert len(requests) == 2  # no retries or unnecessary metadata requests


def test_standalone_isolated_import_supports_controlled_request(tmp_path, provider_config):
    # -I excludes the repository from sys.path, just like a standalone skill CLI.
    code = """
import json, runpy, sys
module = runpy.run_path(sys.argv[1])
model = module["CANDIDATES"]["qwen"][0]
context = module["runner"].evidence.load_provider(sys.argv[2], model, "language")
url, body = module["build_body"]("language", model, "synthetic-uri", context)
assert body["provider"] == {"only": ["synthetic-endpoint"], "allow_fallbacks": False}
"""
    subprocess.run([sys.executable, "-I", "-c", code, str(SCRIPT), str(provider_config)],
                   cwd=tmp_path, check=True, capture_output=True, timeout=30)
