"""Exercise saved runner output through the tuner, without credentials or inference."""

import json

import pytest

from tools.event_benchmark import run_eval as runner
from tools.event_benchmark import tune_threshold as tuner


def produce(tmp_path, monkeypatch, prefix, *, errors=0, wrong=False):
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
    samples_path, manifest_path, output = [directory / f"{kind}.json"
                                           for kind in ("samples", "manifest", "result")]
    samples_path.write_text(json.dumps({"samples": samples}))
    manifest_path.write_text(json.dumps({
        "status": "human_reviewed", "samples_json_sha256": runner.sha256(samples_path),
        "files": [{**sample, "human_review": "user_reviewed",
                   "source_video_group": sample["filename"],
                   "sha256": runner.sha256(directory / sample["filename"])} for sample in samples],
    }))

    def predict(backend, model, sample, *args):
        if sample["number"] <= errors:
            return sample, {"error": "synthetic offline failure"}
        correct = not (wrong and sample["number"] == errors + 1)
        positive = (sample["proposed_answer"] == "yes") == correct
        return sample, {"score": 0.8 if positive else 0.2,
                        "pred": "yes" if positive else "no", "correct": correct,
                        "frames_planned": 2, "frames_ok": 2, "frames_failed": 0}

    monkeypatch.setattr(runner, "load_key", lambda _: "offline")
    monkeypatch.setattr(runner, "eval_video", predict)
    monkeypatch.setattr(runner.sys, "argv", [
        "run_eval.py", "--candidate", "luna", "--samples", str(samples_path),
        "--manifest", str(manifest_path), "--media", str(directory), "--out", str(output),
    ])
    return output, manifest_path, runner.main()


@pytest.mark.parametrize(("errors", "wrong", "accuracy", "fight_accuracy"), [
    (0, False, 100.0, 100.0), (1, True, 91.7, 66.7),
    (6, False, 75.0, 0.0), (24, False, 0.0, 0.0),
])
def test_current_runner_can_select_and_evaluate_without_dropping_errors(
        tmp_path, monkeypatch, errors, wrong, accuracy, fight_accuracy):
    val, val_manifest, _ = produce(tmp_path, monkeypatch, "val")
    selection = tuner.select_thresholds(val, val_manifest)
    artifact = tmp_path / "thresholds.json"
    digest = tuner.write_new_json(artifact, selection)
    test, test_manifest, result = produce(tmp_path, monkeypatch, "test", errors=errors, wrong=wrong)
    report = tuner.evaluate_thresholds(artifact, digest, [test], [test_manifest])["test"][str(test)]

    assert result["overall"]["scored"] == 24 - errors
    assert result["per_event"]["fight"]["scored"] == 6 - min(errors, 6)
    assert report["spec_version"] == "v2.1"
    assert report["accuracy_denominator"] == "attempted"
    assert report["counts"] == {"attempted": 24, "scored": 24 - errors,
                                "total": 24, "errors": errors}
    for key in ("overall_at_0.5", "overall_at_tau"):
        metrics = report[key]
        assert metrics["n"] == 24
        assert metrics["accuracy"] == accuracy
        assert sum(metrics[name] for name in ("TP", "FN", "TN", "FP")) == 24 - errors
    assert report["per_event"]["fight"]["at_tau"]["accuracy"] == fight_accuracy
    assert report["per_event"]["fight"]["counts"]["errors"] == min(errors, 6)
