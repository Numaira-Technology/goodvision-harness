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
        "--provider-config", "synthetic-provider.json", "--split-manifest", "synthetic-split.json", "--split", "test",
    ])
    monkeypatch.setattr(runner.evidence, "load_provider", lambda *args: {
        "sha256": "a" * 64, "config": {"expected_provider_name": "Provider A"}})
    monkeypatch.setattr(runner.evidence, "split_groups", lambda *args: ({1: "source-a"}, {"sha256": "b" * 64}))
    monkeypatch.setattr(runner, "load_key", lambda _: "offline")
    monkeypatch.setattr(runner, "eval_video", lambda backend, model, sample, *args: (sample, {"error": "offline failure"}))
    result = runner.main()
    assert result["spec_version"] == "v2.1"
    assert result["score_precision"] == "lossless"
    assert result["overall"] == {"correct": 0, "total": 1, "attempted": 1, "scored": 0, "errors": 1}
    assert result["per_event"]["fight"]["err"] == 1
    assert result["per_event"]["fight"]["total"] == 1
    assert result["samples_json_sha256"] == runner.sha256(dataset.samples)
    assert result["per_video"][0]["sha256"] == runner.sha256(dataset.clip)
    assert result["config"]["frame_rule"].startswith("n=max(1,round(dur/interval))")
    assert json.loads(output.read_text()) == result


@pytest.mark.parametrize("score", [0.49996, 0.5, 0.30496, 0.305])
def test_saved_decision_score_replays_exactly_at_fixed_and_tuned_gates(tmp_path, monkeypatch, score):
    from tools.event_benchmark import tune_threshold as tuner

    dataset = fixture_dataset(tmp_path)
    monkeypatch.setattr(runner, "probe_duration", lambda _: 1.0)
    monkeypatch.setattr(runner, "frame_uri", lambda *args: "offline-image")
    monkeypatch.setattr(runner, "frame_score", lambda *args: score)
    _, result = runner.eval_video("decision", "model", dataset.sample,
                                  {dataset.clip.name: dataset.clip}, "offline", 1.0)
    saved = json.loads(json.dumps(result))
    assert saved["score"] == score
    for gate in (0.5, 0.305):
        replay = tuner.confusion([(saved["score"], False)], gate)
        assert replay["FP"] == int(score >= gate)
    assert saved["pred"] == ("yes" if score >= 0.5 else "no")
    assert saved["correct"] == (score < 0.5)


@pytest.mark.parametrize("interval", [0, -1, float("nan"), float("inf")])
def test_invalid_interval_is_rejected(interval):
    with pytest.raises(ValueError):
        runner.frame_times(2, interval)


def test_frozen_sampling_times():
    assert runner.frame_times(2, 1) == [0.5, 1.5]
    assert runner.frame_times(0.2, 1) == [0.1]
    assert len(runner.frame_times(20)) == 14


@pytest.mark.parametrize("backend", ["language", "decision"])
def test_frozen_request_budget_is_three_total_attempts(monkeypatch, backend):
    calls, delays = [], []
    monkeypatch.setattr(runner, "_post", lambda *args: calls.append(args) or {})
    monkeypatch.setattr(runner.time, "sleep", delays.append)
    assert runner.frame_score(backend, "model", "question", "image", "offline") is None
    # Literal protocol limits, deliberately independent of the runner constants.
    assert len(calls) == 3
    assert delays == [1, 1, 1]


def test_provisional_report_has_separate_protocol_and_recorded_not_verified_hashes(tmp_path, monkeypatch, capsys):
    dataset = fixture_dataset(tmp_path)
    dataset.manifest.unlink()  # There is no reviewed manifest for these diagnostics.
    output = tmp_path / "run.json"
    monkeypatch.setattr(runner.sys, "argv", [
        "run_eval.py", "--candidate", "maverick", "--samples", str(dataset.samples),
        "--media", str(dataset.media), "--provisional", "--events", "fight", "--out", str(output),
    ])
    monkeypatch.setattr(runner, "load_key", lambda _: "offline")
    monkeypatch.setattr(runner, "eval_video", lambda backend, model, sample, *args: (sample, {"error": "offline failure"}))
    result = runner.main()
    assert result["spec_version"] == "base4-provisional-v2"
    assert result["frozen"] is False and result["provisional"] is True
    assert result["manifest_sha256"] is None
    assert result["scored_events"] == ["fight"]
    assert result["spec_sha256"] == runner.sha256(runner.Path(runner.__file__).with_name("PROVISIONAL.md"))
    assert result["samples_json_sha256"] == runner.sha256(dataset.samples)
    assert result["per_video"][0]["sha256"] == runner.sha256(dataset.clip)
    assert result["overall"] == {"correct": 0, "total": 1, "attempted": 1, "scored": 0, "errors": 1}
    assert result["config"]["retries"] == 3
    assert "PROVISIONAL" in capsys.readouterr().out
    assert json.loads(output.read_text()) == result


@pytest.mark.parametrize("options", [
    ["--provisional", "--events", "bag_knife"],
    ["--events", "bag_alcohol_bottle"],
    ["--provisional", "--manifest", "unverified.json"],
    ["--events", "fight"],
    ["--provisional", "--events", "fight", "fight"],
    ["--provisional", "--legacy-baseline"],
    ["--provisional", "--provider-config", "provider.json"],
    ["--provisional", "--split-manifest", "split.json"],
    ["--provisional", "--split", "test"],
])
def test_incompatible_modes_fail_before_credentials_or_provider_access(tmp_path, monkeypatch, options):
    dataset = fixture_dataset(tmp_path)
    output = tmp_path / "run.json"
    monkeypatch.setattr(runner.sys, "argv", [
        "run_eval.py", "--candidate", "qwen", "--samples", str(dataset.samples),
        "--media", str(dataset.media), "--out", str(output), *options,
    ])
    monkeypatch.setattr(runner, "load_key", lambda _: pytest.fail("credentials must not be accessed"))
    with pytest.raises(SystemExit) as error:
        runner.main()
    assert error.value.code == 2
    assert not output.exists()


def test_provisional_preflight_records_hashes_without_credentials(tmp_path, monkeypatch):
    dataset = fixture_dataset(tmp_path)
    dataset.manifest.unlink()
    monkeypatch.setattr(runner.sys, "argv", [
        "run_eval.py", "--candidate", "luna", "--samples", str(dataset.samples),
        "--media", str(dataset.media), "--provisional", "--preflight",
    ])
    monkeypatch.setattr(runner, "load_key", lambda _: pytest.fail("credentials must not be accessed"))
    result = runner.main()
    assert result["protocol_variant"] == "base4-provisional-v2"
    assert result["manifest_sha256"] is None
    assert result["frozen"] is False and result["hashes_verified"] is False
    assert result["inference_calls"] == 0
    assert "frozen provider configuration" not in result["pending"]


@pytest.mark.parametrize("provisional", [False, True])
@pytest.mark.parametrize("change", ["unknown_label", "ternary_prompt"])
def test_unknown_contract_is_not_coerced_into_binary_answers(tmp_path, monkeypatch, provisional, change):
    dataset = fixture_dataset(tmp_path)
    if change == "unknown_label":
        dataset.sample["proposed_answer"] = "unknown"
    else:
        dataset.sample["question"] = "Knife visible? Answer yes, no, or unknown."
    dataset.samples.write_text(json.dumps({"samples": [dataset.sample]}))
    dataset.data["files"][0].update(dataset.sample)
    dataset.data["samples_json_sha256"] = runner.sha256(dataset.samples)
    dataset.manifest.write_text(json.dumps(dataset.data))
    original = dataset.samples.read_bytes()
    options = ["--provisional"] if provisional else ["--manifest", str(dataset.manifest)]
    monkeypatch.setattr(runner.sys, "argv", [
        "run_eval.py", "--candidate", "qwen-235b", "--samples", str(dataset.samples),
        "--media", str(dataset.media), *options,
    ])
    monkeypatch.setattr(runner, "load_key", lambda _: pytest.fail("credentials must not be accessed"))
    with pytest.raises(SystemExit) as error:
        runner.main()
    assert error.value.code == 2
    assert dataset.samples.read_bytes() == original


@pytest.mark.parametrize("change", ["duplicate", "missing"])
def test_provisional_media_is_still_required_and_unambiguous(tmp_path, change):
    dataset = fixture_dataset(tmp_path)
    directories = [dataset.media]
    if change == "duplicate":
        other = tmp_path / "other"
        other.mkdir()
        (other / dataset.clip.name).write_bytes(b"different media")
        directories.append(other)
    else:
        dataset.clip.unlink()
    with pytest.raises(ValueError, match="ambiguous|missing media"):
        runner.provisional_inputs(dataset.samples, directories, ["fight"])
