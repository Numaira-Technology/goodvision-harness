"""Offline validation of the frozen exam, without media or provider calls."""

import json
from types import SimpleNamespace

import pytest

from tools.event_benchmark import run_eval as runner


def language(content, finish="stop"):
    return {"choices": [{"message": {"content": content}, "finish_reason": finish}]}


@pytest.mark.parametrize("content", [
    "no", "{}", '{"answer":"false"}', '{"answer":0}', '{"answer":null}',
    '{"answer":false,"answer":true}',
])
def test_malformed_chat_answer_is_an_error_not_a_negative(monkeypatch, content):
    calls = []
    monkeypatch.setattr(runner, "_post", lambda *args: calls.append(args) or language(content))
    monkeypatch.setattr(runner.time, "sleep", lambda _: None)
    assert runner.frame_score("language", "model", "question", "image", "key") is None
    assert len(calls) == runner.RETRIES


@pytest.mark.parametrize("answer", [True, False])
def test_language_boolean_and_fixed_envelope(monkeypatch, answer):
    calls = []
    monkeypatch.setattr(runner, "_post", lambda *args: calls.append(args) or language(json.dumps({"answer": answer})))
    assert runner.frame_score("language", "model", "question", "image", "key") == float(answer)
    assert calls[0][1]["temperature"] == 0
    assert calls[0][1]["max_tokens"] == 30


@pytest.mark.parametrize("score", [None, True, "0.8", float("nan"), float("inf"), -0.1, 1.1])
def test_invalid_decision_score_is_not_scored(monkeypatch, score):
    monkeypatch.setattr(runner, "_post", lambda *args: {"answers": {"answer": {"type": "noul", "noul": score}}})
    monkeypatch.setattr(runner.time, "sleep", lambda _: None)
    assert runner.frame_score("decision", "model", "question", "image", "key") is None


@pytest.mark.parametrize("failure", ["decode", "provider", "probe"])
def test_partial_video_failure_is_reported_without_a_prediction(tmp_path, monkeypatch, failure):
    path = tmp_path / "clip.mp4"
    path.write_bytes(b"fixture")
    sample = {"filename": path.name, "question": "Fight? Answer yes or no.", "proposed_answer": "no"}
    monkeypatch.setattr(runner, "probe_duration", lambda _: float("nan") if failure == "probe" else 2.0)
    monkeypatch.setattr(runner, "frame_uri", lambda _, t: None if failure == "decode" and t > 1 else "image")
    scores = iter([0.0, None] if failure == "provider" else [0.0, 0.0])
    monkeypatch.setattr(runner, "frame_score", lambda *args: next(scores))
    _, result = runner.eval_video("language", "model", sample, {path.name: path}, "key", 1.0)
    assert "error" in result and "pred" not in result
    if failure != "probe":
        assert result["frames_failed"] == 1
        assert result["frames_ok"] + result["frames_failed"] == result["frames_planned"] == 2


def fixture_dataset(tmp_path):
    media = tmp_path / "media"
    media.mkdir()
    clip = media / "clip.mp4"
    clip.write_bytes(b"fixture")
    sample = {"number": 1, "filename": clip.name, "event": "fight",
              "question": "Fight? Answer yes or no.", "proposed_answer": "no"}
    samples = tmp_path / "samples.json"
    samples.write_text(json.dumps({"samples": [sample]}))
    manifest = tmp_path / "manifest.json"
    data = {"status": "human_reviewed", "samples_json_sha256": runner.sha256(samples),
            "files": [{**sample, "sha256": runner.sha256(clip)}]}
    manifest.write_text(json.dumps(data))
    return SimpleNamespace(media=media, clip=clip, sample=sample, samples=samples,
                           manifest=manifest, data=data)


@pytest.mark.parametrize("change", ["answers", "media", "draft", "duplicate", "missing"])
def test_frozen_inputs_reject_changed_or_ambiguous_dataset(tmp_path, change):
    dataset = fixture_dataset(tmp_path)
    dirs = [dataset.media]
    if change == "answers":
        dataset.samples.write_text(dataset.samples.read_text() + " ")
    elif change == "media":
        dataset.clip.write_bytes(b"changed")
    elif change == "draft":
        dataset.data["status"] = "draft_pending_user_review"
        dataset.manifest.write_text(json.dumps(dataset.data))
    elif change == "duplicate":
        other = tmp_path / "other"
        other.mkdir()
        (other / dataset.clip.name).write_bytes(b"fixture")
        dirs.append(other)
    else:
        dataset.clip.unlink()
    with pytest.raises(ValueError):
        runner.frozen_inputs(dataset.samples, dataset.manifest, dirs)


def test_complete_run_records_frozen_provenance_and_errors(tmp_path, monkeypatch):
    dataset = fixture_dataset(tmp_path)
    output = tmp_path / "outputs" / "run.json"
    monkeypatch.setattr(runner.sys, "argv", [
        "run_eval.py", "--candidate", "qwen", "--samples", str(dataset.samples),
        "--manifest", str(dataset.manifest), "--media", str(dataset.media), "--out", str(output),
    ])
    monkeypatch.setattr(runner, "load_key", lambda _: "offline")
    monkeypatch.setattr(runner, "eval_video", lambda backend, model, sample, *args: (sample, {"error": "offline failure"}))
    result = runner.main()
    assert result["spec_version"] == "v2"
    assert result["overall"] == {"correct": 0, "total": 0, "attempted": 1, "errors": 1}
    assert result["per_event"]["fight"]["err"] == 1
    assert result["samples_json_sha256"] == runner.sha256(dataset.samples)
    assert result["per_video"][0]["sha256"] == runner.sha256(dataset.clip)
    assert result["config"]["frame_rule"].startswith("n=max(1,round(dur/interval))")
    assert json.loads(output.read_text()) == result


@pytest.mark.parametrize("interval", [0, -1, float("nan"), float("inf")])
def test_invalid_interval_is_rejected(interval):
    with pytest.raises(ValueError):
        runner.frame_times(2, interval)


def test_frozen_sampling_times():
    assert runner.frame_times(2, 1) == [0.5, 1.5]
    assert runner.frame_times(0.2, 1) == [0.1]
    assert len(runner.frame_times(20)) == 14
