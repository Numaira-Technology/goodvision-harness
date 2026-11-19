"""Threshold analysis must preserve v2 eligibility and a held-out test set."""

import copy
import hashlib
import json
import sys

import pytest

from tools.event_benchmark import tune_threshold as tuner


def run(prefix):
    digest = lambda text: hashlib.sha256(text.encode()).hexdigest()
    questions = []
    for event in tuner.EVENTS:
        for index, (gt, score) in enumerate((("yes", 0.8), ("no", 0.2)) * 3):
            name = f"{prefix}-{event}-{gt}-{index}.mp4"
            questions.append({"number": len(questions) + 1, "filename": name,
                              "sha256": digest(name), "event": event, "gt": gt,
                              "score": score, "frames_planned": 2,
                              "pred": gt, "correct": True,
                              "frames_ok": 2, "frames_failed": 0,
                              "frames": [{"time_s": t, "score": score, "attempts": [
                                  {"ok": True, "score": score, "actual_provider": "Provider A", "actual_model": "model-a", "generation_id": f"gen-{name}-{t}"}
                              ]} for t in (0.5, 1.5)]})
    data = {"spec_version": "v2", "score_precision": "lossless",
            "backend": "decision", "model": "model-a",
            "samples_json_sha256": digest(prefix + "samples"),
            "manifest_sha256": digest(prefix + "manifest"),
            "runner_sha256": digest("runner"), "spec_sha256": digest("spec"),
            "config": {"gate": 0.5, "interval": 1.0, "frame_rule": "fixed",
                       "frame_width": 720, "temperature": 0, "retries": 3,
                       "timeout": 90, "decision": "max-frame-score >= gate"},
            "overall": {"attempted": 24, "total": 24, "errors": 0},
            "per_video": questions,
            "protocol_variant": tuner.evidence.VARIANT,
            "evidence_sha256": digest("evidence"),
            "provider_context": {"sha256": digest("provider"), "config": {
                "model": "model-a", "backend": "decision", "expected_provider_name": "Provider A",
                "provider": {"only": ["provider-a"], "allow_fallbacks": False}}}}
    refresh_provider(data)
    return data


def refresh_provider(data):
    attempts = [a for row in data["per_video"] for frame in row.get("frames", []) for a in frame["attempts"]]
    data["provider_observed"] = tuner.evidence.provider_summary(attempts, data["provider_context"])


def check(data, **kwargs):
    return tuner.validate_run(data["per_video"], data, **kwargs)


@pytest.mark.parametrize(("field", "value"), [
    ("frozen", False), ("frozen", "true"), ("provisional", True), ("provisional", None),
    ("spec_version", "base4-provisional-v1"), ("scored_events", ["fight"]),
])
def test_unfrozen_metadata_cannot_enter_calibration_even_with_valid_hashes(field, value):
    data = run("val")
    data[field] = value
    with pytest.raises(ValueError):
        check(data, validation=True)


def test_non_v2_retry_budget_cannot_be_relabelled_as_v2():
    data = run("val")
    data["config"]["retries"] = 8
    with pytest.raises(ValueError, match="three total"):
        check(data, validation=True)


@pytest.mark.parametrize("score", [None, True, "0.8", -0.1, 1.1, float("nan"), float("inf")])
def test_invalid_probability_is_rejected(score):
    data = run("val")
    data["per_video"][0]["score"] = score
    with pytest.raises(ValueError, match="probabilities"):
        check(data, validation=True)


@pytest.mark.parametrize("change", ["historical", "language", "manifest", "gt", "partial", "missing", "duplicate", "totals"])
def test_ineligible_validation_is_rejected(change):
    data = run("val")
    row = data["per_video"][0]
    if change == "historical":
        data["spec_version"] = "v1"
    elif change == "language":
        data["backend"] = "language"
    elif change == "manifest":
        data.pop("manifest_sha256")
    elif change == "gt":
        row["gt"] = "unknown"
    elif change == "partial":
        row["frames_failed"] = 1
    elif change == "missing":
        row.pop("score")
        row["error"] = "missing file"
    elif change == "duplicate":
        data["per_video"][1] = copy.deepcopy(row)
    else:
        data["overall"]["attempted"] = 25
    with pytest.raises(ValueError):
        check(data, validation=True)


@pytest.mark.parametrize("boundary", ["filename", "renamed_media", "model", "sampling", "runner"])
def test_pair_rejects_overlap_or_incompatible_contract(boundary):
    val, test = run("val"), run("test")
    if boundary == "filename":
        test["per_video"][0]["filename"] = val["per_video"][0]["filename"]
    elif boundary == "renamed_media":
        test["per_video"][0]["sha256"] = val["per_video"][0]["sha256"]
    elif boundary == "model":
        test["model"] = "model-b"
    elif boundary == "sampling":
        test["config"]["interval"] = 4.0
    else:
        test["runner_sha256"] = "f" * 64
    with pytest.raises(ValueError):
        tuner.validate_pair(val["per_video"], val, test["per_video"], test)


def test_each_validation_event_requires_both_ground_truth_classes():
    data = run("val")
    for row in data["per_video"][:6]:
        row["gt"] = "yes"
        row["correct"] = row["pred"] == "yes"
    with pytest.raises(ValueError, match="positive and negative"):
        check(data, validation=True)


def save_run(tmp_path, data, *, change_manifest=None):
    prefix = data["per_video"][0]["filename"].split("-")[0]
    manifest_path, result_path = [tmp_path / f"{prefix}-{kind}.json"
                                  for kind in ("manifest", "result")]
    manifest = {
        "status": "human_reviewed", "samples_json_sha256": data["samples_json_sha256"],
        "files": [{**{key: row[key] for key in ("number", "filename", "sha256", "event")},
                   "proposed_answer": row["gt"], "human_review": "user_reviewed",
                   "question": f"Does this video show {row['event']}? Answer yes or no.",
                   "source_video_group": row["filename"]} for row in data["per_video"]],
    }
    if change_manifest:
        change_manifest(manifest)
    manifest_path.write_text(json.dumps(manifest))
    data["manifest_sha256"] = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    refresh_provider(data)
    result_path.write_text(json.dumps(data))
    return result_path, manifest_path


def freeze(tmp_path, val=None):
    source, manifest = save_run(tmp_path, val or run("val"))
    artifact_path = tmp_path / "thresholds.json"
    digest = tuner.write_new_json(artifact_path, tuner.select_thresholds(source, manifest))
    return artifact_path, digest


def test_cli_reports_test_errors_and_keeps_fixed_gate_comparison(tmp_path, monkeypatch):
    source, manifest = save_run(tmp_path, run("val"))
    artifact, output = tmp_path / "thresholds.json", tmp_path / "report.json"
    monkeypatch.setattr(sys, "argv", ["tune_threshold", "select", "--val", str(source),
                                     "--val-manifest", str(manifest), "--out", str(artifact)])
    tuner.main()
    digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
    test = run("test")
    test["per_video"][0].pop("score")
    test["per_video"][0]["error"] = "provider failed"
    test["overall"].update(total=23, errors=1)
    test_source, test_manifest = save_run(tmp_path, test)
    monkeypatch.setattr(sys, "argv", ["tune_threshold", "evaluate", "--thresholds", str(artifact),
                                     "--thresholds-sha256", digest, "--test", str(test_source),
                                     "--test-manifest", str(test_manifest), "--out", str(output)])
    tuner.main()
    report = json.loads(output.read_text())
    result = report["test"][str(test_source)]
    assert result["counts"] == {"attempted": 24, "total": 23, "errors": 1}
    assert result["overall_at_tau"]["n"] == result["overall_at_0.5"]["n"] == 23
    assert report["val_counts"]["total"] == 24
    assert report["threshold_artifact_sha256"] == digest
    assert result["per_event"]["fight"]["counts"] == {"attempted": 6, "total": 5, "errors": 1}


def test_cli_rejects_identical_val_and_test_before_writing_report(tmp_path, monkeypatch):
    source, manifest = save_run(tmp_path, run("shared"))
    artifact, output = tmp_path / "thresholds.json", tmp_path / "report.json"
    digest = tuner.write_new_json(artifact, tuner.select_thresholds(source, manifest))
    monkeypatch.setattr(sys, "argv", ["tune_threshold", "evaluate", "--thresholds", str(artifact),
                                     "--thresholds-sha256", digest, "--test", str(source),
                                     "--test-manifest", str(manifest), "--out", str(output)])
    with pytest.raises(SystemExit) as error:
        tuner.main()
    assert error.value.code == 2
    assert not output.exists()


@pytest.mark.parametrize("precision", [None, "rounded-4dp"])
def test_rounded_v2_scores_are_not_relabelled_as_exact(precision):
    data = run("val")
    data["score_precision"] = precision
    with pytest.raises(ValueError, match="lossless"):
        check(data, validation=True)


@pytest.mark.parametrize("change", ["pred", "correct"])
def test_inconsistent_fixed_baseline_is_rejected(change):
    data = run("val")
    data["per_video"][0][change] = "no" if change == "pred" else False
    with pytest.raises(ValueError, match="baseline verdict"):
        check(data, validation=True)


def test_longest_contiguous_optimum_wins_over_median_of_disconnected_grid_points():
    items = [(0.2, False), (0.4, True), (0.8, False), (0.9, True),
             (0, False), (1, True)]
    tau, result, selection = tuner.pick_tau(items)
    assert tau == 0.305
    assert result["TP"] + result["TN"] == 5
    assert selection["chosen_run"] == [0.21, 0.4]
    assert not selection["fallback_to_grid"]


def test_subgrid_scores_cannot_make_the_selected_midpoint_suboptimal():
    items = [(0.295, False), (0.301, True), (0.309, False), (0.315, True)]
    with pytest.raises(ValueError, match="midpoint is not optimal"):
        tuner.pick_tau(items)


def test_candidate_ranking_uses_correct_counts_before_display_rounding():
    items = [(0.0, False), (1.0, True)] * 5000 + [(0.989, False)]
    assert tuner.confusion(items, 0.5)["accuracy"] == 100.0
    tau, result, selection = tuner.pick_tau(items)
    assert tau == 0.995
    assert result["TP"] + result["TN"] == 10001
    assert selection["chosen_run"] == [0.99, 1.0]


@pytest.mark.parametrize(("negative", "positive", "expected"), [
    ((0.09, 0.69), (0.19, 0.79), 0.745),
    ((0.1, 0.7), (0.29, 0.89), 0.2),
])
def test_equal_length_best_runs_have_deterministic_ties(negative, positive, expected):
    items = [(score, False) for score in negative] + [(score, True) for score in positive]
    tau, _, _ = tuner.pick_tau(items)
    assert tau == expected


def test_selection_reads_only_validation_and_its_manifest(tmp_path, monkeypatch):
    source, manifest = save_run(tmp_path, run("val"))
    original = tuner.read_json
    reads = []

    def read(path):
        reads.append(path)
        return original(path)

    monkeypatch.setattr(tuner, "read_json", read)
    result = tuner.select_thresholds(source, manifest)
    assert reads == [source, manifest]
    assert result["val_result_sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert result["per_class_val"]["fight"]["positive"] == 3
    assert result["per_class_val"]["fight"]["negative"] == 3


def test_a_complete_but_undersupported_validation_event_is_not_tuned():
    data = run("val")
    data["per_video"].pop(0)
    data["overall"].update(attempted=23, total=23)
    with pytest.raises(ValueError, match="six complete"):
        check(data, validation=True)


@pytest.mark.parametrize("change", ["draft", "unreviewed", "group", "answer", "duplicate", "extra"])
def test_selection_rejects_invalid_reviewed_manifest(tmp_path, change):
    def edit(manifest):
        row = manifest["files"][0]
        if change == "draft":
            manifest["status"] = "draft_pending_user_review"
        elif change == "unreviewed":
            row["human_review"] = "pending_user_review"
        elif change == "group":
            row.pop("source_video_group")
        elif change == "answer":
            row["proposed_answer"] = "no"
        elif change == "duplicate":
            manifest["files"].append(copy.deepcopy(row))
        else:
            manifest["files"].append({**row, "number": 1000})

    source, manifest = save_run(tmp_path, run("val"), change_manifest=edit)
    with pytest.raises(ValueError):
        tuner.select_thresholds(source, manifest)


def test_truncating_a_run_and_updating_totals_does_not_change_the_reviewed_cohort(tmp_path):
    artifact, digest = freeze(tmp_path)
    data = run("test")
    source, manifest = save_run(tmp_path, data)
    data["per_video"].pop(0)
    data["overall"].update(attempted=23, total=23)
    refresh_provider(data)
    source.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="exact reviewed manifest"):
        tuner.evaluate_thresholds(artifact, digest, [source], [manifest])


def test_changed_manifest_bytes_cannot_retain_old_run_provenance(tmp_path):
    source, manifest = save_run(tmp_path, run("val"))
    manifest.write_text(manifest.read_text() + " ")
    with pytest.raises(ValueError, match="provenance"):
        tuner.select_thresholds(source, manifest)


def test_distinct_clips_from_the_same_reviewed_source_group_cannot_cross_splits(tmp_path):
    artifact, digest = freeze(tmp_path)

    def edit(manifest):
        manifest["files"][0]["source_video_group"] = "val-fight-yes-0.mp4"

    source, manifest = save_run(tmp_path, run("test"), change_manifest=edit)
    with pytest.raises(ValueError, match="overlap by source_video_group"):
        tuner.evaluate_thresholds(artifact, digest, [source], [manifest])


def test_modified_threshold_artifact_fails_before_opening_test_predictions(tmp_path, monkeypatch):
    artifact, digest = freeze(tmp_path)
    data = json.loads(artifact.read_text())
    data["thresholds"]["fight"] = 0.0
    artifact.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        tuner.evaluate_thresholds(artifact, digest, [tmp_path / "absent.json"], ["absent-manifest"])


def test_frozen_artifacts_cannot_be_overwritten(tmp_path):
    artifact, _ = freeze(tmp_path)
    before = artifact.read_bytes()
    with pytest.raises(FileExistsError):
        tuner.write_new_json(artifact, {"different": True})
    assert artifact.read_bytes() == before


def test_all_error_event_remains_visible_without_an_accuracy(tmp_path):
    artifact, digest = freeze(tmp_path)
    data = run("test")
    for row in data["per_video"][:6]:
        row.pop("score")
        row["error"] = "provider unavailable"
    data["overall"].update(total=18, errors=6)
    source, manifest = save_run(tmp_path, data)
    report = tuner.evaluate_thresholds(artifact, digest, [source], [manifest])
    fight = report["test"][str(source)]["per_event"]["fight"]
    assert fight["counts"] == {"attempted": 6, "total": 0, "errors": 6}
    assert fight["at_tau"]["accuracy"] is None
    assert fight["at_0.5"]["accuracy"] is None


def test_different_test_prompt_is_not_a_compatible_evaluation(tmp_path):
    artifact, digest = freeze(tmp_path)

    def edit(manifest):
        for row in manifest["files"]:
            if row["event"] == "fight":
                row["question"] = "Does this video show a person standing? Answer yes or no."

    source, manifest = save_run(tmp_path, run("test"), change_manifest=edit)
    with pytest.raises(ValueError, match="question text differs"):
        tuner.evaluate_thresholds(artifact, digest, [source], [manifest])


@pytest.mark.parametrize("items", [[], [(0.8, True)], [(0.2, False)]])
def test_youden_index_requires_both_reference_classes(items):
    assert tuner.confusion(items, 0.5)["youden_j"] is None


@pytest.mark.parametrize("change", ["missing_provider", "wrong_provider", "missing_id", "frame_score", "context_model"])
def test_received_provider_and_full_precision_evidence_are_required(change):
    data = run("val")
    attempt = data["per_video"][0]["frames"][0]["attempts"][0]
    if change == "missing_provider":
        attempt.pop("actual_provider")
    elif change == "wrong_provider":
        attempt["actual_provider"] = "Provider B"
    elif change == "missing_id":
        attempt.pop("generation_id")
    elif change == "frame_score":
        attempt["score"] = 0.79999
    else:
        data["provider_context"]["config"]["model"] = "other-model"
    refresh_provider(data)
    with pytest.raises(ValueError):
        check(data, validation=True)


def test_provider_or_worker_change_cannot_reuse_frozen_validation_thresholds(tmp_path):
    artifact, digest = freeze(tmp_path)
    data = run("test")
    data["config"]["workers"] = 6
    source, manifest = save_run(tmp_path, data)
    with pytest.raises(ValueError, match="configuration differs"):
        tuner.evaluate_thresholds(artifact, digest, [source], [manifest])


@pytest.mark.parametrize("change", ["provider_config", "actual_model", "helper_code"])
def test_changed_serving_identity_or_code_cannot_reuse_thresholds(tmp_path, change):
    artifact, digest = freeze(tmp_path)
    data = run("test")
    if change == "provider_config":
        data["provider_context"]["sha256"] = "f" * 64
    elif change == "helper_code":
        data["evidence_sha256"] = "f" * 64
    else:
        for row in data["per_video"]:
            for frame in row["frames"]:
                frame["attempts"][0]["actual_model"] = "changed-returned-model"
    source, manifest = save_run(tmp_path, data)
    # Exact returned-model binding now rejects an incoherent serving identity
    # before the frozen-threshold comparison. Keep the original mutation and
    # require that stronger failure rather than normalizing the returned name.
    expected = "incomplete or inconsistent" if change == "actual_model" else "differs"
    with pytest.raises(ValueError, match=expected):
        tuner.evaluate_thresholds(artifact, digest, [source], [manifest])


def test_coherent_changed_model_cannot_reuse_frozen_thresholds(tmp_path):
    artifact, digest = freeze(tmp_path)
    data = run("test")
    data["model"] = data["provider_context"]["config"]["model"] = "model-b"
    data["provider_context"]["sha256"] = hashlib.sha256(b"provider-b").hexdigest()
    for row in data["per_video"]:
        for frame in row["frames"]:
            frame["attempts"][0]["actual_model"] = "model-b"
    refresh_provider(data)
    tuner.evidence.verify_provider_evidence(data, data["per_video"])
    source, manifest = save_run(tmp_path, data)
    with pytest.raises(ValueError, match="model differs"):
        tuner.evaluate_thresholds(artifact, digest, [source], [manifest])


def test_historical_thresholds_are_frozen_as_comparison_not_used_for_selection(tmp_path):
    source, manifest = save_run(tmp_path, run("val"))
    reference = tmp_path / "old-thresholds.json"
    reference.write_text(json.dumps({"model": "model-a", "backend": "decision",
                                     "thresholds": dict.fromkeys(tuner.EVENTS, 0.99)}))
    plain = tuner.select_thresholds(source, manifest)
    selected = tuner.select_thresholds(source, manifest, reference_path=reference)
    assert plain["thresholds"] == selected["thresholds"]
    artifact = tmp_path / "frozen.json"
    digest = tuner.write_new_json(artifact, selected)
    reference.write_text("changed after freeze")
    test, test_manifest = save_run(tmp_path, run("test"))
    result = tuner.evaluate_thresholds(artifact, digest, [test], [test_manifest])["test"][str(test)]
    assert result["overall_at_previous_tau"]["accuracy"] == 50.0
    assert result["overall_at_tau"]["accuracy"] == 100.0


def test_selection_and_test_use_the_same_frozen_source_sidecar(tmp_path):
    val, test = run("val"), run("test")
    val_path, val_manifest = save_run(tmp_path, val)
    test_path, test_manifest = save_run(tmp_path, test)
    split = {"schema_version": 1, "status": "validation_membership_and_labels_frozen_with_recorded_limits",
             "train": {"membership": None}, "development": {"membership": None}, "limits": ["scene isolation unknown"]}
    for name, data, manifest in [("validation", val, val_manifest), ("test", test, test_manifest)]:
        rows = json.loads(manifest.read_text())["files"]
        split[name] = {"questions": 24, "videos": 24, "manifest_sha256": data["manifest_sha256"],
                       "samples_json_sha256": data["samples_json_sha256"], "members": [
                           {**{k: row[k] for k in ("filename", "sha256", "source_video_group")},
                            "question_numbers": [row["number"]]} for row in rows]}
    sidecar = tmp_path / "split.json"
    sidecar.write_text(json.dumps(split))
    for name, data, manifest, result in [("validation", val, val_manifest, val_path), ("test", test, test_manifest, test_path)]:
        _, data["split_context"] = tuner.evidence.split_groups(sidecar, manifest, name)
        result.write_text(json.dumps(data))
    artifact = tmp_path / "frozen.json"
    selected = tuner.select_thresholds(val_path, val_manifest, sidecar)
    digest = tuner.write_new_json(artifact, selected)
    report = tuner.evaluate_thresholds(artifact, digest, [test_path], [test_manifest], sidecar)
    assert report["split_limits"] == ["scene isolation unknown"]
    assert report["test"][str(test_path)]["counts"]["total"] == 24
    sidecar.write_text(sidecar.read_text() + " ")
    with pytest.raises(ValueError, match="sidecar provenance differs"):
        tuner.evaluate_thresholds(artifact, digest, [test_path], [test_manifest], sidecar)


def test_v21_counts_missing_in_total_and_rejects_version_mixing():
    data = run("validation")
    data['spec_version'] = 'v2.1'
    # Same valid evidence is accepted; version difference alone blocks pairing.
    tuner.validate_run(data['per_video'], data, validation=True)
    old = run("other")
    with pytest.raises(ValueError, match='spec_version'):
        tuner.validate_pair(old['per_video'], old, data['per_video'], data)


def test_v21_keeps_error_questions_in_reported_total():
    data = run("test")
    data["spec_version"] = "v2.1"
    row = data["per_video"][0]
    for key in ("score", "pred", "correct", "frames"):
        row.pop(key)
    row["error"] = "missing answer"
    data["overall"]["errors"] = 1
    refresh_provider(data)
    assert check(data) == {"attempted": 24, "total": 24, "errors": 1}
    data["overall"]["total"] = 23
    with pytest.raises(ValueError, match="totals"):
        check(data)
