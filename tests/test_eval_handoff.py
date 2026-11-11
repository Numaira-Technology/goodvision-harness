"""Portable handoff behavior, using synthetic replies and no model/network calls."""
import json
import subprocess
import sys
from pathlib import Path

import pytest
pytest.importorskip("fcntl", reason="Durable benchmark execution requires Linux/macOS")

from tools.unified_benchmark import adapters, preflight
from tools.unified_benchmark.contract import EVENTS, file_hash, score
from tools.unified_benchmark.workflow import verify_bundle


def response(value, finish="stop"):
    return {"choices": [{"finish_reason": finish, "message": {"content": json.dumps({"answer": value})}}]}


@pytest.mark.parametrize("value,expected", [(False, "no"), (True, "yes"), ("false", "no"), ("true", "yes")])
def test_explicit_boolean_strings(value, expected):
    config = {"backend": "language", "answer_parser": "boolean-string-v1"}
    assert adapters.parse_response(config, {"targets": ["fight"]}, response(value)) == {"fight": {"answer": expected}}


@pytest.mark.parametrize("value", [None, 0, 1, "False", "FALSE", " false ", "no", "yes", "", [], {}])
def test_no_truthiness_or_extra_string_coercion(value):
    with pytest.raises(ValueError):
        adapters.parse_response({"backend": "language", "answer_parser": "boolean-string-v1"},
                                {"targets": ["fight"]}, response(value))


def test_refusal_or_truncation_is_missing_even_with_false():
    cfg = {"backend": "language", "answer_parser": "boolean-string-v1"}
    for r in [response("false", "length"), response("false")]:
        if r["choices"][0]["finish_reason"] == "stop":
            r["choices"][0]["message"]["refusal"] = "refused"
        with pytest.raises(ValueError): adapters.parse_response(cfg, {"targets": ["fight"]}, r)


def test_missing_stays_in_overall_and_event_denominator():
    plan = {"units": [{"id": "u", "targets": ["fight"]}, {"id": "v", "targets": ["fight"]}],
            "questions": [{"number": 1, "event": "fight", "reference": "yes", "unit_ids": ["u"]},
                          {"number": 2, "event": "fight", "reference": "no", "unit_ids": ["v"]}]}
    r = score(plan, {"u": {"fight": {"answer": "yes"}}}, "choice")
    assert r["accuracy"] == .5 and r["complete_only_accuracy"] == 1
    assert r["accuracy_denominator"] == 2 and r["complete_valid_questions"] == 1
    assert r["per_event"]["fight"]["accuracy"] == .5


def test_explicit_private_inputs_are_hashed_and_joined(tmp_path, monkeypatch):
    source = tmp_path / "private"
    source.mkdir()
    sidecar = {"schema_version": 1,
               "status": "validation_membership_and_labels_frozen_with_recorded_limits"}
    for split, count, manifest_name in [("validation", 24, "manifest_validation_24.json"),
                                        ("test", 120, "manifest_120.json")]:
        rows = [{"number": i, "filename": f"{split}-{i}.mp4", "event": "fight",
                 "question": "Synthetic question", "proposed_answer": "yes", "human_review": True,
                 "sha256": f"{i + (1000 if split == 'test' else 0):064x}"}
                for i in range(1, count + 1)]
        samples_name = f"{split}-samples.json"
        (source / samples_name).write_text(json.dumps({"status": "human_reviewed", "samples": rows}))
        (source / manifest_name).write_text(json.dumps({"status": "human_reviewed", "files": rows,
                                                       "samples_json_sha256": file_hash(source / samples_name)}))
        sidecar[split] = {"questions": count, "videos": count,
                          "manifest_sha256": file_hash(source / manifest_name),
                          "samples_json_sha256": file_hash(source / samples_name),
                          "members": [{"filename": r["filename"], "sha256": r["sha256"],
                                       "source_video_group": r["filename"], "question_numbers": [r["number"]]}
                                      for r in rows]}
    (source / "validation_split.json").write_text(json.dumps(sidecar))
    monkeypatch.setattr(preflight, "PINNED_INPUTS", {p.name: file_hash(p) for p in source.iterdir()})
    parts = preflight.load_inputs(tmp_path / "bundle", source)
    assert [len(parts[s]["rows"]) for s in ("validation", "test")] == [24, 120]
    assert all(file_hash(tmp_path / "bundle" / "inputs" / name) == expected for name, expected in preflight.PINNED_INPUTS.items())
    (source / "test-samples.json").write_text("{}")
    with pytest.raises(ValueError, match="frozen input changed"):
        preflight.load_inputs(tmp_path / "changed-bundle", source)


def test_profiles_freeze_effort_and_native_windows():
    c = preflight.model_configs(["gemini25_off", "gemini25_low", "gemini38_low", "videojev", "decision3"])
    assert c["gemini25_off"]["reasoning"] == {"enabled": False}
    assert c["gemini25_low"]["reasoning"] == {"max_tokens": 1024}
    assert c["gemini38_low"]["reasoning"] == {"effort": "low"}
    assert all(c[a]["max_tokens"] == 2048 for a in ("gemini25_off", "gemini25_low", "gemini38_low"))
    assert c["videojev"]["window_frames"] == c["decision3"]["window_frames"] == 2


def test_plan_uses_current_frame_and_keeps_ground_truth_out_of_request(tmp_path):
    cfg = preflight.model_configs(["gemini25_low"])["gemini25_low"]
    (tmp_path / "frame.jpg").write_bytes(b"synthetic")
    frame = {"path": "frame.jpg", "sha256": file_hash(tmp_path / "frame.jpg"), "time": 0}
    rows = [{"number": 1, "filename": "a.mp4", "sha256": "a" * 64, "event": "fight",
             "question": "Fighting in this video?", "proposed_answer": "yes", "source_video_group": "g"}]
    p = preflight.build_plan("test", rows, {"a.mp4": {"frames": [frame]}}, cfg, tmp_path)
    _, body = adapters.request_body(cfg, p["units"][0], tmp_path)
    assert p["questions"][0]["reference"] == "yes"
    assert "current image" in body["messages"][0]["content"][0]["text"]
    assert "this video" not in body["messages"][0]["content"][0]["text"]
    assert body["reasoning"] == {"max_tokens": 1024}
    assert "proposed_answer" not in json.dumps(body) and "reference" not in json.dumps(body)


def test_historical_bundle_is_not_silently_reinterpreted(tmp_path):
    (tmp_path / "rules.json").write_text(json.dumps({"accuracy": "old"}))
    (tmp_path / "preflight.json").write_text(json.dumps({"offline_passed": True}))
    (tmp_path / "bundle-files-sha256.json").write_text(json.dumps({n: file_hash(tmp_path / n) for n in ("rules.json", "preflight.json")}))
    with pytest.raises(ValueError, match="historical replay"): verify_bundle(tmp_path)


def test_api_without_execute_does_not_read_credentials(tmp_path):
    from tools.unified_benchmark.api_run import main
    with pytest.raises(SystemExit):
        main(["--bundle", str(tmp_path / "absent"), "--output", str(tmp_path / "out"),
              "--models", "luna", "--env-file", str(tmp_path / "absent-key"), "--authorization-note", "synthetic"])
    assert not (tmp_path / "out").exists()
