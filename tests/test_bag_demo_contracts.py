"""Offline HTTP, media, hosted-reply and overlay contracts; no weights or services."""

import json
import subprocess
from types import SimpleNamespace
from unittest.mock import Mock

import httpx
import pytest
from fastapi.testclient import TestClient
from test_bag_demo import WINDOW, _synth_raw

from tools.bag_demo import alcohol_vlm as alcohol
from tools.bag_demo import media, server, yolo_boxes
from tools.bag_demo import videojev_bag as vb


@pytest.fixture
def client(tmp_path, monkeypatch):
    (tmp_path / "clip.MOV").write_bytes(b"synthetic media")
    monkeypatch.setattr(server, "MEDIA_DIR", tmp_path)
    monkeypatch.setattr(server, "duration", lambda _: 3.0)
    with TestClient(server.app, base_url="http://127.0.0.1") as client:
        yield client


@pytest.mark.parametrize("t", ["NaN", "inf", "-inf", "-1", "3", "999999"])
def test_invalid_time_never_samples_or_scores(client, monkeypatch, t):
    sample = Mock(side_effect=AssertionError("must not sample"))
    monkeypatch.setattr(vb, "sample_frames", sample)
    assert (
        client.get("/api/verdict", params={"clip": "clip.MOV", "t": t}).status_code
        == 422
    )
    sample.assert_not_called()


def test_start_of_clip_does_not_read_future_frames(client, monkeypatch):
    sample = Mock(side_effect=AssertionError("must not sample"))
    monkeypatch.setattr(vb, "sample_frames", sample)
    response = client.get("/api/verdict?clip=clip.MOV&t=0.1")
    assert response.status_code == 200
    assert response.json()["status"] == "insufficient_frames"
    assert response.json()["alcohol_status"] == "not_integrated"
    assert response.json()["knife"] is None
    sample.assert_not_called()


def test_server_passes_causal_window_and_marks_unfinished_axis(client, monkeypatch):
    captured = []

    def frames(path, times):
        captured.extend(times)
        return ["image"] * len(times)

    monkeypatch.setattr(vb, "sample_frames", frames)
    monkeypatch.setattr(vb, "predict", lambda *a, **k: {"label": "未检出刀/武器"})
    response = client.get("/api/verdict?clip=clip.MOV&t=0.95")
    assert response.status_code == 200
    assert captured == pytest.approx([0.15, 0.55, 0.95])
    assert response.json()["alcohol_status"] == "not_integrated"
    assert response.json()["clip"] == "clip.MOV"


def test_server_failure_redacts_details_and_releases_slot(client, monkeypatch):
    monkeypatch.setattr(
        vb, "sample_frames", Mock(side_effect=RuntimeError("secret@example.com"))
    )
    for _ in range(2):
        response = client.get("/api/verdict?clip=clip.MOV&t=1")
        assert response.status_code == 502
        assert response.json() == {"error": "RuntimeError", "status": "unavailable"}
    assert "endpoint" not in client.get("/api/clips").json()


def test_server_backpressure_precedes_sampling(client, monkeypatch):
    sample = Mock()
    monkeypatch.setattr(vb, "sample_frames", sample)
    with server.SCORING:
        assert client.get("/api/verdict?clip=clip.MOV&t=1").status_code == 503
    sample.assert_not_called()


def test_media_and_boxes_cannot_escape_configured_directory(client, tmp_path):
    outside = tmp_path.parent / (tmp_path.name + "-private")
    outside.write_text("synthetic private content")
    try:
        (tmp_path / "leak.MOV").symlink_to(outside)
        (tmp_path / "clip.MOV.boxes.json").symlink_to(outside)
    except OSError:
        pytest.skip("symlinks unavailable")
    assert client.get("/media/leak.MOV").status_code == 404
    assert client.get("/api/boxes/clip.MOV").status_code == 404
    assert [c["name"] for c in client.get("/api/clips").json()["clips"]] == ["clip.MOV"]
    assert client.get("/media/clip.MOV").content == b"synthetic media"


@pytest.mark.parametrize(
    "data",
    [
        [],
        {"clip": "wrong", "frames": []},
        {"clip": "clip.MOV", "frames": [{"t": float("nan"), "boxes": []}]},
        {"clip": "clip.MOV", "frames": [{"t": 0, "boxes": [{"xyxy": [0, 0, 2, 1]}]}]},
    ],
)
def test_invalid_overlay_is_an_explicit_empty_error(client, tmp_path, data):
    (tmp_path / "clip.MOV.boxes.json").write_text(json.dumps(data))
    response = client.get("/api/boxes/clip.MOV")
    assert response.status_code == 422
    assert response.json() == {"frames": [], "error": "invalid_boxes"}


def test_absent_and_valid_overlay(client, tmp_path):
    assert client.get("/api/boxes/clip.MOV").json()["frames"] == []
    data = {
        "clip": "clip.MOV",
        "frames": [
            {
                "t": 0.2,
                "boxes": [
                    {"cls": "bottle", "conf": 0.7, "xyxy": [0, 0, 1, 1], "alert": True}
                ],
            }
        ],
    }
    (tmp_path / "clip.MOV.boxes.json").write_text(json.dumps(data))
    assert client.get("/api/boxes/clip.MOV").json() == data


@pytest.mark.parametrize(
    "text",
    [
        "no answer",
        "{}",
        "[]",
        '{"answer":"false","conf":0.9}',
        '{"answer":0,"conf":0.9}',
        '{"answer":true,"conf":"0.9"}',
        '{"answer":false,"conf":NaN}',
        '{"answer":false,"conf":Infinity}',
        '{"answer":true,"conf":1.01}',
        '{"answer":false,"conf":-1}',
        '{"answer":true,"conf":true}',
        '{"answer":false}',
        None,
    ],
)
def test_malformed_alcohol_reply_never_becomes_negative(text):
    with pytest.raises(ValueError, match="invalid alcohol response"):
        alcohol._parse_answer(text)


@pytest.mark.parametrize("answer", [True, False])
def test_hosted_request_with_mock_transport_only(answer):
    def respond(request):
        assert request.headers["authorization"] == "Bearer synthetic-key"
        assert json.loads(request.content)["model"] == "synthetic-model"
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": json.dumps({"answer": answer, "conf": 0.9})
                        }
                    }
                ]
            },
        )

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        assert alcohol.ask_frame(
            "data:image/jpeg;base64,c3ludGhldGlj",
            client=client,
            endpoint="https://provider.invalid/chat",
            api_key="synthetic-key",
            model="synthetic-model",
        ) == (answer, 0.9)
        assert not client.is_closed


@pytest.mark.parametrize(
    "answers,expected,status",
    [
        ([(False, 0.9), ValueError("secret")], None, "partial"),
        ([(True, 0.9), ValueError("secret")], True, "partial"),
        ([(False, 0.9), (True, 0.2)], None, "partial"),
        ([(False, 0.9), (False, 0.8)], False, "ok"),
    ],
)
def test_alcohol_pool_preserves_partial_unknowns(
    monkeypatch, answers, expected, status
):
    monkeypatch.setattr(
        alcohol, "_provider", lambda: ("https://provider.invalid", "key", "model")
    )
    monkeypatch.setattr(alcohol, "_frame_data_uri", lambda *a: "synthetic")
    monkeypatch.setattr(alcohol, "ask_frame", Mock(side_effect=answers))
    result = alcohol.detect_alcohol("synthetic", [0, 0.4], client=Mock())
    assert result["present"] is expected
    assert result["status"] == status
    assert len(result["per_frame"]) == 2
    assert "secret" not in json.dumps(result)


@pytest.mark.parametrize("times", [[], [float("inf")], [-1], [True]])
def test_bad_samples_rejected_before_provider_access(monkeypatch, times):
    provider = Mock(side_effect=AssertionError("provider accessed"))
    monkeypatch.setattr(alcohol, "_provider", provider)
    with pytest.raises(ValueError):
        alcohol.detect_alcohol("synthetic", times)
    provider.assert_not_called()


def test_scorer_response_identity_and_alignment(monkeypatch):
    qs = vb.build_questions(WINDOW)
    payload = {
        "model": vb.MODEL,
        "revision": vb.REVISION,
        "request_id": "r",
        "raw": _synth_raw(qs, {}),
        "timings": {},
    }

    def post(*args, **kwargs):
        assert kwargs["trust_env"] is False
        return httpx.Response(200, json=payload, request=httpx.Request("POST", args[0]))

    monkeypatch.setattr(httpx, "post", post)
    assert (
        vb.predict(["i"] * 3, [0, 0.4, 0.8], request_id="r", session="s")["label"]
        == "未检出刀/武器"
    )
    for key in ("request_id", "model", "revision"):
        original = payload[key]
        payload[key] = "wrong"
        with pytest.raises(ValueError, match="identity mismatch"):
            vb.predict(["i"] * 3, [0, 0.4, 0.8], request_id="r", session="s")
        payload[key] = original
    with pytest.raises(ValueError, match="align"):
        vb.predict(["i"] * 2, [0, 0.4, 0.8], request_id="r", session="s")


def test_native_unknown_cannot_become_clean():
    qs = vb.build_questions(WINDOW)
    result = vb.parse_bag(_synth_raw(qs, {"scene/weapon": 2}), qs)
    assert vb.demo_label(result) == "存疑"


@pytest.mark.parametrize("output", [b"", b"not an image"])
def test_empty_ffmpeg_success_is_not_a_frame(monkeypatch, output):
    run = Mock(return_value=SimpleNamespace(stdout=output))
    monkeypatch.setattr(subprocess, "run", run)
    with pytest.raises(ValueError, match="no decodable frame"):
        media.frame_jpeg("local.MOV", 1)
    assert run.call_args.kwargs["timeout"] == 15


def test_local_weights_required_before_ultralytics_import(tmp_path):
    with pytest.raises(FileNotFoundError, match="already exist"):
        yolo_boxes.BagYolo(tmp_path / "not-downloaded.pt")


@pytest.mark.parametrize(
    "kwargs", [{"fps": 0}, {"fps": float("nan")}, {"conf": float("inf")}]
)
def test_yolo_rejects_invalid_settings_before_model_load(kwargs):
    with pytest.raises(ValueError):
        yolo_boxes.precompute("not-opened.MOV", **kwargs)


def test_page_rejects_late_responses_and_never_shows_clean_for_unavailable():
    import shutil
    from pathlib import Path

    node = shutil.which("node")
    if not node:
        pytest.skip("Node is required for offline page contracts")
    result = subprocess.run(
        [node, "tests/bag_demo_page_contract.cjs"],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize(
    "headers,code",
    [
        ({"host": "attacker.invalid"}, 400),
        ({"origin": "https://attacker.invalid"}, 403),
        ({"origin": "http://[invalid"}, 403),
        ({"origin": "https://127.0.0.1"}, 403),
        ({"sec-fetch-site": "cross-site"}, 403),
    ],
)
def test_local_demo_rejects_cross_site_and_rebinding(client, headers, code):
    assert client.get("/media/clip.MOV", headers=headers).status_code == code
    assert client.get("/api/clips").headers["cache-control"] == "no-store"


def _synthetic_video(path):
    import cv2
    import numpy as np

    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (120, 240))
    if not writer.isOpened():
        pytest.skip("local synthetic video encoder unavailable")
    try:
        for i in range(20):
            frame = np.full((240, 120, 3), 20 + i * 5, dtype=np.uint8)
            frame[:, :12] = (0, 0, 255)
            frame[:, -12:] = (0, 255, 0)
            writer.write(frame)
    finally:
        writer.release()


def test_real_synthetic_frame_preserves_aspect_and_rejects_eof(tmp_path):
    import shutil

    import cv2
    import numpy as np

    if not shutil.which("ffmpeg"):
        pytest.skip("ffmpeg unavailable")
    path = tmp_path / "synthetic.MOV"
    _synthetic_video(path)
    jpeg = media.frame_jpeg(path, 0.4, width=720)
    decoded = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
    assert decoded.shape[:2] == (720, 360)
    assert decoded[360, 10, 2] > 180  # left red edge still visible
    assert decoded[360, -10, 1] > 180  # right green edge still visible
    # ffmpeg versions either fail at EOF or succeed with an empty packet.
    with pytest.raises((ValueError, subprocess.CalledProcessError)):
        media.frame_jpeg(path, 9, width=720)


def test_synthetic_media_through_api_and_native_scorer_contract(tmp_path, monkeypatch):
    import shutil

    from tools.videojev_benchmark.server import Scorer

    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        pytest.skip("ffmpeg and ffprobe required")
    path = tmp_path / "synthetic.MOV"
    _synthetic_video(path)
    captured = []

    class Runtime:
        def predict(self, images, fps, questions):
            assert fps == 2.5 and all(i.startswith(b"\xff\xd8") for i in images)
            captured.extend(images)
            return _synth_raw(
                questions, {vb.BAG_KNIFE_ID: 0}, {vb.BAG_KNIFE_ID: 0.99}
            ), {}

    scorer = Scorer(Runtime(), tmp_path / "synthetic-audit.jsonl")

    def offline_post(url, **kwargs):
        result = scorer.submit(kwargs["json"]).result(timeout=2)
        return httpx.Response(200, json=result, request=httpx.Request("POST", url))

    monkeypatch.setattr(httpx, "post", offline_post)
    monkeypatch.setattr(server, "MEDIA_DIR", tmp_path)
    with TestClient(server.app, base_url="http://127.0.0.1") as client:
        response = client.get("/api/verdict?clip=synthetic.MOV&t=1.3")
    assert response.status_code == 200, response.text
    assert len(captured) == 4
    assert response.json()["label"] == "刀/武器候选"
    assert response.json()["alcohol_status"] == "not_integrated"
    assert (
        json.loads((tmp_path / "synthetic-audit.jsonl").read_text())["status"]
        == "completed"
    )
