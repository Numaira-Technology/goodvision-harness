"""Synthetic latency regressions: no credentials, paid requests or model loading."""

import importlib.util
import io
import json
import shutil
import subprocess
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
    assert latency.CANDIDATES == {
        "qwen": ("qwen/qwen3-vl-8b-instruct", "language"),
        "gemini": ("google/gemini-2.5-flash", "language"),
        "luna": ("openai/gpt-6-luna-decisions", "decision"),
        "qwen-235b": ("qwen/qwen3-vl-235b-a22b-instruct", "language"),
        "maverick": ("meta-llama/llama-4-maverick", "language"),
    }


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
    tmp_path, monkeypatch, complete
):
    jpeg = b"synthetic prepared image"
    monkeypatch.setattr(latency, "prepare_frame", lambda *a, **k: jpeg)
    monkeypatch.setattr(latency, "load_key", lambda *a: "synthetic-key")
    attempts = iter([{"ok": complete, "elapsed_s": 1}, {"ok": True, "elapsed_s": 2}])
    monkeypatch.setattr(latency, "one_call", lambda *a: next(attempts))
    output = tmp_path / "nested" / "result.json"
    exit_code = latency.main(
        [
            "--candidate",
            "qwen",
            "--frame",
            "synthetic.jpg",
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
