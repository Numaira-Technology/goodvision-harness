"""Synthetic dataset/plan tests, never model measurements or paid requests."""
from collections import Counter
from copy import deepcopy
import json
import threading

import pytest

from tools.unified_benchmark import dataset202 as d, preflight, workflow, api_run
from tools.unified_benchmark.contract import CURRENT_VERSION, VERSION, digest, rules_for, score
from tools.qwen8b import eval_local


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("network calls are forbidden in dataset offline tests")
    monkeypatch.setattr(api_run.urllib.request, "urlopen", forbidden)


@pytest.fixture
def inputs():
    original, validation, questions, files, videos = [], [], [], [], {}
    for event, count in zip(d.EVENTS, (32, 22, 26, 20, 10, 10)):
        for i in range(count):
            number = len(original) + 1
            filename = f"bag-{i}.mp4" if event.startswith("bag_") else f"{event}-{i}.mp4"
            row = dict(number=number, filename=filename, event=event, question=f"Is {event} visible?",
                       proposed_answer="yes" if i % 2 else "no", human_review="user_reviewed", sha256=digest(filename))
            original.append(row | {"source_video_group": filename})
            questions.append({k: v for k, v in row.items() if k != "sha256"})
            files.append(row)
            videos[filename] = {"filename": filename}
    for i in range(24):
        event = d.EVENTS[i // 6]
        filename = f"val-{i}.mp4"
        validation.append(dict(number=i + 1, filename=filename, sha256=digest(filename), source_video_group=filename,
                               event=event, question=f"Is {event} visible?", proposed_answer="yes" if i % 6 < 3 else "no",
                               human_review="user_reviewed"))
    for i in range(41):
        for view in ("left", "right"):
            number = len(questions) + 1
            event = "fall" if i < 18 else "weapon"
            filename = f"robot-{i}-{view}.mp4"
            row = dict(number=number, filename=filename, event=event, question=f"Is {event} visible?",
                       proposed_answer="yes" if i % 2 else "no", human_review="pending_view_review", sha256=digest(filename))
            files.append(row)
            source = dict(source_group=f"recording-{i % 3}", source_sample_id=f"robot-{i}", view=view)
            questions.append({k: v for k, v in row.items() if k != "sha256"} | source |
                             {"reference_review": "inherited_from_human_reviewed_source", "source_interval_seconds": [i * 2, i * 2 + 1]})
            videos[filename] = {"filename": filename} | source
    common = dict(status="draft_pending_user_review", dataset="goodvision-events-202-20261110", review_status_by_cohort=d.REVIEW_STATUS.copy())
    return ({"validation": {"rows": validation, "split_context": {}}, "test": {"rows": original, "split_context": {}}},
            common | dict(n_questions=202, n_files_hashed=192, missing_files=[], samples_json_sha256=d.PINNED_202_INPUTS["samples.json"], files=files),
            common | dict(question_count=202, video_count=192, samples=questions), {"videos": list(videos.values())})


def test_extension_keeps_validation_and_original120(inputs):
    parts = d.extend_test(*inputs)
    assert parts["validation"] is inputs[0]["validation"]
    assert parts["test"]["rows"][:120] == inputs[0]["test"]["rows"]
    assert len(parts["test"]["rows"]) == 202
    assert len({r["filename"] for r in parts["test"]["rows"]}) == 192
    assert Counter(r["event"] for r in parts["test"]["rows"]) == d.EVENT_COUNTS
    assert {r["human_review"] for r in parts["test"]["rows"][120:]} == {"pending_view_review"}
    assert parts["test"]["dataset_context"]["robot82_source_recordings"] == 3


@pytest.mark.parametrize("change", ["answer", "sha256", "question", "duplicate", "registry", "pair_group", "pair_label", "review", "overlap"])
def test_bad_extension_rejected(inputs, change):
    parts, manifest, samples, registry = deepcopy(inputs)
    if change in ("answer", "sha256", "question"):
        field = "proposed_answer" if change == "answer" else change
        manifest["files"][0][field] = "no" if field == "proposed_answer" else "changed"
        if field in samples["samples"][0]:
            samples["samples"][0][field] = manifest["files"][0][field]
        if field == "proposed_answer":
            manifest["files"][0][field] = samples["samples"][0][field] = "yes"
    elif change == "duplicate":
        samples["samples"][-1]["number"] = 121
    elif change == "registry":
        registry["videos"].pop()
    elif change == "pair_group":
        samples["samples"][120]["source_group"] = "another-group"
        registry["videos"][-82]["source_group"] = "another-group"
    elif change == "pair_label":
        samples["samples"][120]["proposed_answer"] = manifest["files"][120]["proposed_answer"] = "yes"
    elif change == "review":
        samples["samples"][120]["human_review"] = manifest["files"][120]["human_review"] = "user_reviewed"
    elif change == "overlap":
        parts["validation"]["rows"][0]["source_video_group"] = "recording-0"
    with pytest.raises(ValueError):
        d.extend_test(parts, manifest, samples, registry)


def test_202_missing_and_unknown_stay_in_denominator(inputs):
    part = d.extend_test(*inputs)["test"]
    rows = part["rows"]
    plan = {"protocol": CURRENT_VERSION, "split": "test", "dataset_context": part["dataset_context"],
            "questions": [{"number": r["number"], "event": r["event"], "reference": r["proposed_answer"],
                           "unit_ids": [str(r["number"])]} for r in rows],
            "units": [{"id": str(r["number"]), "targets": [r["event"]]} for r in rows]}
    evidence = {str(r["number"]): {r["event"]: {"answer": r["proposed_answer"]}} for r in rows}
    evidence.pop("121")
    evidence["1"][rows[0]["event"]]["answer"] = "unknown"
    result = score(plan, evidence, "choice")
    assert (result["correct"], result["accuracy_denominator"]) == (200, 202)
    assert result["incomplete_questions"] == result["final_unknown_questions"] == result["final_missing_unique_units"] == 1
    assert result["cohorts"]["original120"]["correct"] == 119
    assert result["cohorts"]["robot82"]["correct"] == 81
    assert result["cohorts"]["robot82"]["final_missing_unique_units"] == 1
    assert result["cohorts"]["original120"]["final_missing_unique_units"] == 0
    assert result["dataset_context"]["review_status_by_cohort"] == d.REVIEW_STATUS


def test_default_preflight202_all_routes_and_protocol_guard(tmp_path, inputs, monkeypatch):
    parts = d.extend_test(*inputs)
    monkeypatch.setattr(preflight, "load_inputs", lambda out, frozen, protocol: parts)
    def inspect(split, row, out, media, protocol):
        assert protocol == CURRENT_VERSION
        frames = []
        for i in range(2):
            rel = f"frames/{split}/{row['filename']}-{i}.jpg"
            p = out / rel; p.parent.mkdir(parents=True, exist_ok=True); p.write_bytes(b"synthetic-test-JPEG")
            frames.append({"path": rel, "sha256": d.file_hash(p), "time": float(i), "size": [720, 480]})
        return {"filename": row["filename"], "frames": frames, "full_decode_passed": True}
    monkeypatch.setattr(preflight, "inspect_video", inspect)
    original_check_output = preflight.subprocess.check_output
    monkeypatch.setattr(preflight.subprocess, "check_output", lambda args, **kwargs:
                        "ffmpeg synthetic offline test\n" if args == ["ffmpeg", "-version"] else original_check_output(args, **kwargs))
    out = tmp_path / "bundle"
    aliases = ["qwen8b", "qwen235b", "maverick", "luna", "gemini25_off", "gemini38_low", "clip", "siglip2", "videojev", "decision3", "qwen8b_r2_host", "qwen8b_r2_vllm"]
    assert preflight.main(["--inputs", str(tmp_path), "--media", str(tmp_path), "--output", str(out), "--models", *aliases]) == 0
    audit = json.loads((out / "preflight.json").read_text())
    assert audit["protocol"] == CURRENT_VERSION and audit["unique_videos"] == 216
    assert audit["inference_calls"] == audit["network_calls"] == 0
    workflow.verify_bundle(out)
    for alias in aliases:
        plan = json.loads((out / f"plans/{alias}-test.json").read_text())
        assert len(plan["questions"]) == 202
        assert plan["rules_sha256"] == digest(rules_for(CURRENT_VERSION))
        assert len(json.loads((out / f"plans/{alias}-validation.json").read_text())["questions"]) == 24
    class SyntheticTransport:
        def __init__(self, bundle, config, key, directory, ledger, billing_stop):
            self.config = config
            plan = json.loads((bundle / f"plans/{config['alias']}-{directory.name}.json").read_text())
            self.answers = {uid: {q["event"]: q["reference"]} for q in plan["questions"] for uid in q["unit_ids"]}

        def __call__(self, unit):
            values = self.answers[unit["id"]]
            return ({event: ({"score": .9 if answer == "yes" else .1} if self.config["kind"] == "probability"
                             else {"answer": answer}) for event, answer in values.items()}, {})
    monkeypatch.setattr(api_run, "Transport", SyntheticTransport)
    runs = tmp_path / "runs"; runs.mkdir()
    bundle_digest = workflow.verify_bundle(out)
    for alias in ("qwen8b", "luna"):
        api_run.execute_model(alias, out, runs, "synthetic-not-a-key", threading.Event(), bundle_digest)
        result = json.loads((runs / alias / "test/summary.json").read_text())
        assert result["correct"] == result["accuracy_denominator"] == 202
        assert result["base100"]["accuracy_denominator"] == 100
        assert result["bag20"]["accuracy_denominator"] == 20
        assert result["cohorts"]["original120"]["accuracy_denominator"] == 120
        assert result["cohorts"]["robot82"]["accuracy_denominator"] == 82
    artifact = json.loads((runs / "luna/thresholds.json").read_text())
    assert artifact["rules_sha256"] == digest(rules_for(CURRENT_VERSION))
    assert artifact["validation_summary"]["complete_valid_questions"] == 24
    config = json.loads((out / "configs/luna.json").read_text())
    threshold_path = runs / "luna/thresholds.json"
    assert workflow.load_threshold_artifact(
        threshold_path, d.file_hash(threshold_path), bundle=out, alias="luna",
        config=config, bundle_digest=bundle_digest,
    ) == artifact
    for split in ("validation", "test"):
        plan = json.loads((out / f"plans/luna-{split}.json").read_text())
        assert workflow.context_for(plan, config, bundle_digest)["rules_sha256"] == digest(rules_for(CURRENT_VERSION))
    selected_path, report_path = tmp_path / "selected.json", tmp_path / "report.json"
    workflow.main([
        "select", "--bundle", str(out), "--model", "luna", "--ledger",
        str(runs / "luna/validation/attempts.jsonl"), "--output", str(selected_path),
    ])
    selected = json.loads(selected_path.read_text())
    assert selected["rules_sha256"] == artifact["rules_sha256"]
    workflow.main([
        "report", "--bundle", str(out), "--model", "luna", "--ledger",
        str(runs / "luna/test/attempts.jsonl"), "--output", str(report_path),
        "--thresholds", str(selected_path), "--thresholds-sha256", d.file_hash(selected_path),
    ])
    assert json.loads(report_path.read_text())["correct"] == 202

    class SyntheticLocal(SyntheticTransport):
        def __init__(self, bundle, config, endpoint, directory, ledger):
            super().__init__(bundle, config, None, directory, ledger, None)

    monkeypatch.setattr(eval_local, "LocalTransport", SyntheticLocal)
    for alias in ("qwen8b_r2_host", "qwen8b_r2_vllm"):
        eval_local.execute(alias, out, runs, "http://unused.invalid", bundle_digest, "test")
        result = json.loads((runs / alias / "test/summary.json").read_text())
        assert result["correct"] == result["accuracy_denominator"] == 202
        assert result["base100"]["correct"] == result["base100"]["accuracy_denominator"] == 100
        assert result["bag20"]["accuracy_denominator"] == 20
        assert result["cohorts"]["original120"]["accuracy_denominator"] == 120
        assert result["cohorts"]["robot82"]["accuracy_denominator"] == 82

    # Even a resealed plan/preflight pair must reject an unsupported protocol.
    path = out / "plans/qwen8b-test.json"
    original = json.loads(path.read_text())
    bad = original | {"protocol": "unsupported-protocol"}
    path.write_text(json.dumps(bad))
    audit_path = out / "preflight.json"
    audit["models"]["qwen8b"]["test"]["plan_sha256"] = digest(bad)
    audit_path.write_text(json.dumps(audit))
    hashes = json.loads((out / "bundle-files-sha256.json").read_text())
    for changed in (path, audit_path):
        hashes[changed.relative_to(out).as_posix()] = d.file_hash(changed)
    (out / "bundle-files-sha256.json").write_text(json.dumps(hashes))
    with pytest.raises(ValueError, match="unsupported unified protocol"):
        workflow.verify_bundle(out)
    path.write_text(json.dumps(original))
    path = out / "plans/qwen8b-test.json"
    wrong = json.loads(path.read_text()) | {"protocol": VERSION, "rules_sha256": digest(rules_for(VERSION))}
    path.write_text(json.dumps(wrong))
    hashes = json.loads((out / "bundle-files-sha256.json").read_text()); hashes[path.relative_to(out).as_posix()] = d.file_hash(path)
    (out / "bundle-files-sha256.json").write_text(json.dumps(hashes))
    with pytest.raises(ValueError, match="protocol differs"):
        workflow.verify_bundle(out)


def test_frozen_extension_hash_rejected_before_plan(tmp_path, inputs):
    (tmp_path / "inputs").mkdir()
    (tmp_path / "manifest_202.json").write_text("{}")
    with pytest.raises(ValueError, match="frozen input changed"):
        d.load_extension(tmp_path, tmp_path, inputs[0])


@pytest.mark.parametrize(
    "change",
    [
        "hash",
        "traversal",
        "duplicate_name",
        "duplicate_hash",
        "test_group",
        "missing_question",
        "duplicate_question",
        "blank_review",
        "nonbinary",
        "only_yes",
    ],
)
def test_bag_supplement_rejects_invalid_manifest_before_save(tmp_path, inputs, change):
    from helpers.bag_validation import bag_manifest

    parts = d.extend_test(*inputs)
    before = deepcopy(parts)
    document = bag_manifest()
    videos = document["videos"]
    if change == "traversal":
        videos[0]["filename"] = "../bad.mp4"
    elif change == "duplicate_name":
        videos[1]["filename"] = videos[0]["filename"]
    elif change == "duplicate_hash":
        videos[1]["sha256"] = videos[0]["sha256"]
    elif change == "test_group":
        videos[0]["source_video_group"] = parts["test"]["rows"][0]["source_video_group"]
    elif change == "missing_question":
        videos[0]["questions"].pop()
    elif change == "duplicate_question":
        videos[0]["questions"].append(deepcopy(videos[0]["questions"][0]))
    elif change == "blank_review":
        videos[0]["questions"][0]["human_review"] = "  "
    elif change == "nonbinary":
        videos[0]["questions"][0]["proposed_answer"] = "unknown"
    elif change == "only_yes":
        for video in videos:
            for question in video["questions"]:
                question["proposed_answer"] = "yes"
    (tmp_path / "inputs").mkdir()
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(document), encoding="utf-8")
    expected = "0" * 64 if change == "hash" else d.file_hash(manifest)

    with pytest.raises(ValueError):
        d.add_bag_validation(tmp_path, parts, manifest, expected)

    assert not (tmp_path / "inputs/bag-validation.json").exists()
    assert parts == before


@pytest.mark.parametrize(
    "extra",
    [
        ["--bag-validation", "missing.json"],
        ["--bag-validation-sha256", "0" * 64],
        [
            "--bag-validation",
            "missing.json",
            "--bag-validation-sha256",
            "0" * 64,
            "--protocol",
            "unified-frame-120-v2",
        ],
        [
            "--bag-validation",
            "missing.json",
            "--bag-validation-sha256",
            "0" * 64,
            "--models",
            "videojev",
        ],
    ],
)
def test_bag_supplement_preflight_rejects_unsupported_scope(tmp_path, extra):
    output = tmp_path / "bundle"
    args = [
        "--inputs",
        str(tmp_path),
        "--media",
        str(tmp_path),
        "--output",
        str(output),
        "--models",
        "clip",
        *extra,
    ]
    with pytest.raises(SystemExit):
        preflight.main(args)
    assert not output.exists()


@pytest.mark.parametrize("alias", ["clip", "siglip2"])
def test_bag_supplement_preflight_preserves_cohorts_pairing_and_manifest(
    tmp_path,
    inputs,
    monkeypatch,
    alias,
):
    from helpers.bag_validation import bag_manifest

    parts = d.extend_test(*inputs)
    original = deepcopy(parts)
    manifest = tmp_path / "supplement.json"
    manifest.write_text(json.dumps(bag_manifest(), indent=2), encoding="utf-8")

    def load(out, *_args):
        (out / "inputs").mkdir()
        return parts

    def inspect(split, row, out, media, protocol):
        assert protocol == CURRENT_VERSION
        frames = []
        for index in range(2):
            relative = f"frames/{split}/{row['filename']}-{index}.jpg"
            path = out / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"synthetic offline JPEG")
            frames.append(
                {
                    "path": relative,
                    "sha256": d.file_hash(path),
                    "time": float(index),
                    "size": [720, 480],
                }
            )
        return {
            "filename": row["filename"],
            "frames": frames,
            "full_decode_passed": True,
        }

    monkeypatch.setattr(preflight, "load_inputs", load)
    monkeypatch.setattr(preflight, "inspect_video", inspect)
    original_check = preflight.subprocess.check_output
    monkeypatch.setattr(
        preflight.subprocess,
        "check_output",
        lambda args, **kwargs: (
            "ffmpeg synthetic offline test\n"
            if args == ["ffmpeg", "-version"]
            else original_check(args, **kwargs)
        ),
    )
    out = tmp_path / "bundle"
    assert (
        preflight.main(
            [
                "--inputs",
                str(tmp_path),
                "--media",
                str(tmp_path),
                "--output",
                str(out),
                "--models",
                alias,
                "--bag-validation",
                str(manifest),
                "--bag-validation-sha256",
                d.file_hash(manifest),
            ]
        )
        == 0
    )

    workflow.verify_bundle(out)
    audit = json.loads((out / "preflight.json").read_text())
    assert audit["unique_videos"] == 226
    assert audit["inference_calls"] == audit["network_calls"] == 0
    assert audit["models"][alias]["validation"]["questions"] == 44
    assert audit["models"][alias]["test"]["questions"] == 202
    assert parts["validation"]["rows"][:24] == original["validation"]["rows"]
    assert parts["test"]["rows"] == original["test"]["rows"]
    assert (out / "inputs/bag-validation.json").read_bytes() == manifest.read_bytes()

    plan = json.loads((out / f"plans/{alias}-validation.json").read_text())
    assert [q["number"] for q in plan["questions"]] == list(range(1, 45))
    assert len(plan["units"]) == 68  # Two frames per video, not per question.
    for index in range(24, 44, 2):
        knife, bottle = plan["questions"][index : index + 2]
        assert (knife["event"], bottle["event"]) == d.EVENTS[4:]
        assert knife["unit_ids"] == bottle["unit_ids"]
        video = bag_manifest()["videos"][(index - 24) // 2]
        for question, reference in zip(
            (knife, bottle), video["questions"], strict=True
        ):
            assert question["reference"] == reference["proposed_answer"]
            assert question["source_video_group"] == video["source_video_group"]
        for row, reference in zip(
            parts["validation"]["rows"][index : index + 2],
            video["questions"],
            strict=True,
        ):
            assert row["human_review"] == reference["human_review"]

    (out / "inputs/bag-validation.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="bundle file differs"):
        workflow.verify_bundle(out)
