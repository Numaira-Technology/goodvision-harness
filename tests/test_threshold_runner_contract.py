"""Exercise saved runner output through the tuner, without credentials or inference."""

import json

import pytest

from tools.event_benchmark import run_eval as runner
from tools.event_benchmark import tune_threshold as tuner


def prepare(tmp_path, prefix):
    directory = tmp_path / prefix
    directory.mkdir()
    samples = []
    for event in tuner.EVENTS:
        for index in range(6):
            name = f"{prefix}-{event}-{index}.mp4"
            (directory / name).write_bytes(name.encode())
            samples.append({"number": len(samples) + 1, "filename": name,
                            "event": event, "proposed_answer": "yes" if index % 2 == 0 else "no",
                            "question": f"Does this video show {event}? Answer yes or no."})
    samples_path, manifest_path = [directory / f"{kind}.json"
                                   for kind in ("samples", "manifest")]
    samples_path.write_text(json.dumps({"samples": samples}))
    manifest_path.write_text(json.dumps({
        "status": "human_reviewed", "samples_json_sha256": runner.sha256(samples_path),
        "files": [{**sample, "human_review": "user_reviewed",
                   "source_video_group": sample["filename"],
                   "sha256": runner.sha256(directory / sample["filename"])} for sample in samples],
    }))
    return directory


def prepare_controls(tmp_path, directories):
    provider = tmp_path / "provider.json"
    provider.write_text(json.dumps({
        "schema_version": 1, "model": runner.CANDIDATES["luna"][0], "backend": "decision",
        "provider": {"only": ["provider-a"], "allow_fallbacks": False},
        "expected_provider_name": "Provider A", "selection": "synthetic offline fixture",
        "selected_at": "2026-11-13T00:00:00Z",
    }))
    split = {"schema_version": 1,
             "status": "validation_membership_and_labels_frozen_with_recorded_limits",
             "train": {"membership": None}, "development": {"membership": None},
             "limits": ["synthetic offline fixture; no real dataset acceptance evidence"]}
    for directory in directories:
        manifest = directory / "manifest.json"
        rows = json.loads(manifest.read_text())["files"]
        split[directory.name] = {
            "questions": 24, "videos": 24,
            "manifest_sha256": runner.sha256(manifest),
            "samples_json_sha256": runner.sha256(directory / "samples.json"),
            "members": [{**{k: row[k] for k in ("filename", "sha256", "source_video_group")},
                         "question_numbers": [row["number"]]} for row in rows],
        }
    sidecar = tmp_path / "split.json"
    sidecar.write_text(json.dumps(split))
    return provider, sidecar


def produce(directory, monkeypatch, provider, sidecar, *, errors=0, wrong=False):
    samples_path, manifest_path, output = [directory / f"{kind}.json"
                                           for kind in ("samples", "manifest", "result")]

    def predict(backend, model, sample, *args):
        correct = not (wrong and sample["number"] == errors + 1)
        positive = (sample["proposed_answer"] == "yes") == correct
        score = 0.8 if positive else 0.2
        frames = [{"time_s": t, "score": score, "attempts": [
            {"ok": True, "score": score, "actual_provider": "Provider A", "actual_model": model,
             "generation_id": f"synthetic-{sample['filename']}-{t}"}]} for t in (0.5, 1.5)]
        if sample["number"] <= errors:
            frames[1] = {"time_s": 1.5, "error": "synthetic decode failure", "attempts": []}
            return sample, {"error": "synthetic second frame failed", "frames": frames,
                            "frames_planned": 2, "frames_ok": 1, "frames_failed": 1}
        return sample, {"score": score, "frames": frames,
                        "pred": "yes" if positive else "no", "correct": correct,
                        "frames_planned": 2, "frames_ok": 2, "frames_failed": 0}

    monkeypatch.setattr(runner, "load_key", lambda _: "offline")
    monkeypatch.setattr(runner, "eval_video", predict)
    monkeypatch.setattr(runner.sys, "argv", [
        "run_eval.py", "--candidate", "luna", "--samples", str(samples_path),
        "--manifest", str(manifest_path), "--media", str(directory), "--out", str(output),
        "--provider-config", str(provider), "--split-manifest", str(sidecar),
        "--split", directory.name,
    ])
    return output, manifest_path, runner.main()


@pytest.mark.parametrize(("errors", "wrong", "accuracy", "fight_accuracy"), [
    (0, False, 100.0, 100.0), (1, True, 91.7, 66.7),
    (6, False, 75.0, 0.0), (24, False, 0.0, 0.0),
])
def test_current_runner_can_select_and_evaluate_without_dropping_errors(
        tmp_path, monkeypatch, errors, wrong, accuracy, fight_accuracy):
    val_dir, test_dir = [prepare(tmp_path, prefix) for prefix in ("validation", "test")]
    provider, sidecar = prepare_controls(tmp_path, [val_dir, test_dir])
    val, val_manifest, _ = produce(val_dir, monkeypatch, provider, sidecar)
    selection = tuner.select_thresholds(val, val_manifest, sidecar)
    artifact = tmp_path / "thresholds.json"
    digest = tuner.write_new_json(artifact, selection)
    test, test_manifest, result = produce(test_dir, monkeypatch, provider, sidecar, errors=errors, wrong=wrong)
    report = tuner.evaluate_thresholds(artifact, digest, [test], [test_manifest], sidecar)["test"][str(test)]

    assert result["overall"]["scored"] == 24 - errors
    assert result["per_event"]["fight"]["scored"] == 6 - min(errors, 6)
    assert report["spec_version"] == "v2.1"
    assert report["accuracy_denominator"] == "attempted"
    assert report["counts"] == {"attempted": 24, "scored": 24 - errors,
                                "total": 24, "errors": errors}
    assert report["confusion_denominator"] == "scored"
    for key in ("at_0.5", "at_tau"):
        metrics = report[f"all_questions_{key}"]
        assert metrics["total"] == 24
        assert metrics["accuracy"] == accuracy
        confusion = report[f"overall_{key}"]
        assert confusion["n"] == 24 - errors
        assert sum(confusion[name] for name in ("TP", "FN", "TN", "FP")) == 24 - errors
    assert report["per_event"]["fight"]["all_questions_at_tau"]["accuracy"] == fight_accuracy
    assert report["per_event"]["fight"]["counts"]["errors"] == min(errors, 6)
