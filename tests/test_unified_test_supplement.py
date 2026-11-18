"""Synthetic supplement checks: no private inputs, media or inference calls."""

import json
import threading
from copy import deepcopy

import pytest
from helpers.bag_validation import bag_manifest
from test_native_workflow import FakeBackend
from test_native_workflow import frozen as frozen
from test_unified_dataset202 import inputs as inputs

from tools.unified_benchmark import api_run, native_run, preflight, workflow
from tools.unified_benchmark import dataset202 as d
from tools.unified_benchmark.contract import CURRENT_VERSION, digest, score
from tools.vast_run_log import summarize_score


@pytest.fixture
def supplement():
    return {
        "schema": "event-test-supplement-v1",
        "split": "test",
        "dataset": "synthetic-test222",
        "samples": [
            {
                "number": 203 + 2 * i + j,
                "filename": f"extra-{i}.mp4",
                "sha256": digest(f"extra-{i}"),
                "source_group": f"extra-recording-{i // 2}",
                "event": event,
                "proposed_answer": "yes" if i % 2 else "no",
                "human_review": "synthetic assistant annotation; unreviewed",
            }
            for i in range(10)
            for j, event in enumerate(d.EVENTS[4:])
        ],
    }


@pytest.mark.parametrize(
    "change",
    [
        "blank_review",
        "boolean_review",
        "padded_validation_group",
        "blank_group",
        "group_alias_conflict",
        "float_number",
        "bad_hash",
        "duplicate_hash",
        "duplicate_event",
        "validation_hash",
        "validation_name",
        "validation_group",
        "original_hash",
        "original_name",
        "conflicting_identity",
        "nonbinary",
        "unsupported_event",
        "traversal",
        "windows_path",
        "empty",
        "non_test",
        "bad_schema",
        "missing_dataset",
        "blank_dataset",
    ],
)
def test_invalid_supplement_rejected_before_save(tmp_path, inputs, supplement, change):
    parts = d.extend_test(*inputs)
    before = deepcopy(parts)
    rows = supplement["samples"]
    first = rows[0]
    if change == "blank_review":
        first["human_review"] = "  "
    elif change == "boolean_review":
        first["human_review"] = True
    elif change == "padded_validation_group":
        first["source_group"] = (
            " " + parts["validation"]["rows"][0]["source_video_group"]
        )
    elif change == "blank_group":
        first["source_group"] = "  "
    elif change == "group_alias_conflict":
        first["source_video_group"] = "another-recording"
    elif change == "float_number":
        first["number"] = 203.0
    elif change == "bad_hash":
        first["sha256"] = "bad"
    elif change == "duplicate_hash":
        rows[2]["sha256"] = rows[3]["sha256"] = first["sha256"]
    elif change == "duplicate_event":
        rows[1]["event"] = first["event"]
    elif change in {"validation_hash", "original_hash"}:
        split = "validation" if change == "validation_hash" else "test"
        first["sha256"] = parts[split]["rows"][0]["sha256"]
    elif change in {"validation_name", "original_name"}:
        split = "validation" if change == "validation_name" else "test"
        first["filename"] = parts[split]["rows"][0]["filename"]
    elif change == "validation_group":
        first["source_group"] = parts["validation"]["rows"][0]["source_video_group"]
    elif change == "conflicting_identity":
        first["source_group"] = "changed-recording"
    elif change == "nonbinary":
        first["proposed_answer"] = "unknown"
    elif change == "unsupported_event":
        first["event"] = "unsupported"
    elif change == "traversal":
        first["filename"] = "../extra.mp4"
    elif change == "windows_path":
        first["filename"] = "C:\\extra.mp4"
    elif change == "empty":
        supplement["samples"] = []
    elif change == "non_test":
        supplement["split"] = "validation"
    elif change == "bad_schema":
        supplement["schema"] = "unknown"
    elif change == "missing_dataset":
        supplement.pop("dataset")
    elif change == "blank_dataset":
        supplement["dataset"] = " "
    # Both questions on a video must use the mutation so identity conflict
    # cannot hide a missing field-level check.
    if change in {"padded_validation_group", "blank_group", "bad_hash"}:
        field = "sha256" if change == "bad_hash" else "source_group"
        rows[1][field] = first[field]
    if change == "group_alias_conflict":
        rows[1]["source_video_group"] = first["source_video_group"]
    (tmp_path / "inputs").mkdir()
    path = tmp_path / "supplement.json"
    path.write_text(json.dumps(supplement), encoding="utf-8")
    with pytest.raises(ValueError):
        d.add_test_supplement(tmp_path, parts, path)
    assert parts == before
    assert not (tmp_path / "inputs/test-supplement.json").exists()


@pytest.fixture
def bundle_factory(tmp_path, inputs, supplement, monkeypatch):
    def forbidden(*_args, **_kwargs):
        pytest.fail("network/model calls are forbidden in supplement tests")

    import urllib.request

    monkeypatch.setattr(urllib.request, "urlopen", forbidden)

    def load(out, *_args):
        (out / "inputs").mkdir()
        return d.extend_test(*deepcopy(inputs))

    def inspect(split, row, out, _media, protocol):
        assert protocol == CURRENT_VERSION
        frames = []
        for index in range(2):
            relative = f"frames/{split}/{row['sha256']}/{index}.jpg"
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
    check_output = preflight.subprocess.check_output

    def ffmpeg_version(command, *args, **kwargs):
        if command == ["ffmpeg", "-version"]:
            return "synthetic ffmpeg\n" if kwargs.get("text") else b"synthetic ffmpeg\n"
        # subprocess is shared with platform.platform(), whose macOS uname
        # probe expects bytes. Do not replace unrelated subprocess results.
        return check_output(command, *args, **kwargs)

    monkeypatch.setattr(preflight.subprocess, "check_output", ffmpeg_version)

    def build(name, *, extra=True, bags=False, aliases=("clip",)):
        out = tmp_path / name
        args = [
            "--inputs",
            str(tmp_path),
            "--media",
            str(tmp_path),
            "--output",
            str(out),
            "--models",
            *aliases,
        ]
        if extra:
            path = tmp_path / f"{name}-test.json"
            path.write_text(json.dumps(supplement, indent=2), encoding="utf-8")
            args.extend(["--test-supplement", str(path)])
        if bags:
            path = tmp_path / f"{name}-validation.json"
            path.write_text(json.dumps(bag_manifest()), encoding="utf-8")
            args.extend(
                [
                    "--bag-validation",
                    str(path),
                    "--bag-validation-sha256",
                    d.file_hash(path),
                ]
            )
        assert preflight.main(args) == 0
        workflow.verify_bundle(out)
        return out

    return build


def read(bundle, name):
    return json.loads((bundle / name).read_text())


def reseal(bundle):
    """Keep generic integrity checks valid to isolate semantic validation."""
    report = read(bundle, "preflight.json")
    for path in (bundle / "plans").glob("*.json"):
        alias, split = path.stem.rsplit("-", 1)
        report["models"][alias][split]["plan_sha256"] = digest(
            json.loads(path.read_text())
        )
    (bundle / "preflight.json").write_text(json.dumps(report), encoding="utf-8")
    hashes = {
        p.relative_to(bundle).as_posix(): d.file_hash(p)
        for p in bundle.rglob("*")
        if p.is_file() and p.name != "bundle-files-sha256.json"
    }
    (bundle / "bundle-files-sha256.json").write_text(
        json.dumps(hashes), encoding="utf-8"
    )


@pytest.mark.parametrize("bags", [False, True])
def test_222_bundle_preserves_existing_plans_and_reports(
    bundle_factory, supplement, bags
):
    aliases = (
        ("clip", "siglip2")
        if bags
        else tuple(
            preflight.model_configs([p.stem for p in preflight.PROFILES.glob("*.json")])
        )
    )
    baseline = bundle_factory("baseline", extra=False, bags=bags, aliases=aliases)
    combined = bundle_factory("combined", bags=bags, aliases=aliases)
    audit = read(combined, "preflight.json")
    assert audit["unique_videos"] == (236 if bags else 226)
    assert (
        audit["inference_calls"]
        == audit["network_calls"]
        == audit["model_downloads"]
        == 0
    )
    assert audit["dataset_context"]["test_questions"] == 222
    assert audit["dataset_context"]["test_videos"] == 202
    assert read(combined, "inputs/test-supplement.json") == supplement
    assert (
        read(combined, "split-audit.json")["validation"]
        == read(baseline, "split-audit.json")["validation"]
    )
    for alias in aliases:
        assert read(combined, f"plans/{alias}-validation.json") == read(
            baseline, f"plans/{alias}-validation.json"
        )
        plan = read(combined, f"plans/{alias}-test.json")
        previous = read(baseline, f"plans/{alias}-test.json")
        assert plan["questions"][:202] == previous["questions"]
        assert plan["units"][: len(previous["units"])] == previous["units"]
        assert plan["rules_sha256"] == previous["rules_sha256"]
        assert plan["config_sha256"] == previous["config_sha256"]
        evidence = {}
        for q in plan["questions"]:
            for uid in q["unit_ids"]:
                evidence.setdefault(uid, {})[q["event"]] = {"answer": q["reference"]}
        # Missing and unknown responses must stay in the supplement denominator.
        missing, unknown = plan["questions"][-2:]
        evidence[missing["unit_ids"][0]].pop(missing["event"])
        for uid in unknown["unit_ids"]:
            evidence[uid][unknown["event"]] = {"answer": "unknown"}
        result = score(plan, evidence, "choice")
        assert (result["correct"], result["accuracy_denominator"]) == (220, 222)
        for label, count in (
            ("original120", 120),
            ("robot82", 82),
            ("test_supplement", 20),
        ):
            assert result["cohorts"][label]["accuracy_denominator"] == count
        assert result["cohorts"]["test_supplement"]["correct"] == 18
        assert result["final_unknown_questions"] == result["incomplete_questions"] == 1
        assert score(plan, {}, "choice")["accuracy"] == 0


@pytest.mark.parametrize(
    "change",
    [
        "count",
        "cohort",
        "missing_manifest",
        "changed_manifest",
        "validation_overlay",
        "reference",
    ],
)
def test_bundle_rejects_inconsistent_supplement(bundle_factory, change):
    bundle = bundle_factory("tampered")
    plan_path = bundle / "plans/clip-test.json"
    plan = read(bundle, "plans/clip-test.json")
    if change == "count":
        report = read(bundle, "preflight.json")
        report["dataset_context"]["test_questions"] = 221
        (bundle / "preflight.json").write_text(json.dumps(report))
        plan["questions"].pop()
    elif change == "cohort":
        plan["dataset_context"]["test_supplement"]["question_numbers"] = [1]
    elif change == "missing_manifest":
        (bundle / "inputs/test-supplement.json").unlink()
    elif change == "changed_manifest":
        (bundle / "inputs/test-supplement.json").write_text("{}")
    elif change == "validation_overlay":
        path = bundle / "plans/clip-validation.json"
        validation = json.loads(path.read_text())
        validation["dataset_context"] = plan["dataset_context"]
        path.write_text(json.dumps(validation))
    elif change == "reference":
        q = plan["questions"][-1]
        q["reference"] = "no" if q["reference"] == "yes" else "yes"
    plan_path.write_text(json.dumps(plan))
    reseal(bundle)
    with pytest.raises(ValueError):
        workflow.verify_bundle(bundle)


def test_baseline_count_cannot_be_reduced_without_supplement(bundle_factory):
    bundle = bundle_factory("baseline", extra=False)
    report = read(bundle, "preflight.json")
    report["dataset_context"]["test_questions"] = 120
    (bundle / "preflight.json").write_text(json.dumps(report))
    path = bundle / "plans/clip-test.json"
    plan = json.loads(path.read_text())
    plan["questions"] = plan["questions"][:120]
    path.write_text(json.dumps(plan))
    reseal(bundle)
    with pytest.raises(ValueError):
        workflow.verify_bundle(bundle)


@pytest.mark.parametrize("alias", ["qwen8b", "luna"])
def test_api_reports_222_and_keeps_original_subset_denominators(
    bundle_factory, tmp_path, monkeypatch, alias
):
    bundle = bundle_factory("api", aliases=(alias,))

    class SyntheticTransport:
        def __init__(self, bundle, config, _key, directory, _ledger, _stop):
            self.config = config
            plan = read(bundle, f"plans/{alias}-{directory.name}.json")
            self.answers = {
                uid: {q["event"]: q["reference"]}
                for q in plan["questions"]
                for uid in q["unit_ids"]
            }

        def __call__(self, unit):
            return {
                event: (
                    {"score": 0.9 if answer == "yes" else 0.1}
                    if self.config["kind"] == "probability"
                    else {"answer": answer}
                )
                for event, answer in self.answers[unit["id"]].items()
            }, {"transport_seconds": 0.25}

    monkeypatch.setattr(api_run, "Transport", SyntheticTransport)
    output = tmp_path / "api-output"
    output.mkdir()
    api_run.execute_model(
        alias,
        bundle,
        output,
        "synthetic-not-a-key",
        threading.Event(),
        workflow.verify_bundle(bundle),
    )
    result = read(output, f"{alias}/test/summary.json")
    assert result["correct"] == result["accuracy_denominator"] == 222
    assert result["base100"]["accuracy_denominator"] == 100
    assert result["bag20"]["accuracy_denominator"] == 20
    assert result["cohorts"]["test_supplement"]["correct"] == 20
    latency = result["test_attempt_latency"]
    assert latency["n"] == len(read(bundle, f"plans/{alias}-test.json")["units"])
    assert latency["p50_seconds"] == latency["p95_seconds"] == 0.25
    assert latency["attempt_elapsed_fallback"] == 0
    logged = summarize_score(output, alias)
    assert logged["test"]["planned_questions"] == 222
    assert logged["test"]["latency_n"] == latency["n"]
    assert logged["test"]["latency_details"]["p95_with_failures_seconds"] == 0.25
    assert logged["validation"]["latency_details"]["split"] == "validation"
    if alias == "luna":
        artifact = read(output, f"{alias}/thresholds.json")
        assert artifact["validation_summary"]["complete_valid_questions"] == 24


def test_test_supplement_rejects_legacy_protocol_before_creating_output(tmp_path):
    output = tmp_path / "bundle"
    with pytest.raises(SystemExit):
        preflight.main(
            [
                "--inputs",
                str(tmp_path),
                "--media",
                str(tmp_path),
                "--models",
                "clip",
                "--output",
                str(output),
                "--test-supplement",
                "missing.json",
                "--protocol",
                "unified-frame-120-v2",
            ]
        )
    assert not output.exists()


@pytest.mark.parametrize("field", ["filename", "sha256", "source_group"])
def test_supplement_cannot_overlap_optional_bag_validation(
    tmp_path, inputs, supplement, field
):
    parts = d.extend_test(*inputs)
    (tmp_path / "inputs").mkdir()
    path = tmp_path / "bag-validation.json"
    document = bag_manifest()
    path.write_text(json.dumps(document))
    d.add_bag_validation(tmp_path, parts, path, d.file_hash(path))
    source_field = "source_video_group" if field == "source_group" else field
    for row in supplement["samples"][:2]:
        row[field] = document["videos"][0][source_field]
    before = deepcopy(parts)
    path = tmp_path / "test.json"
    path.write_text(json.dumps(supplement))
    with pytest.raises(ValueError):
        d.add_test_supplement(tmp_path, parts, path)
    assert parts == before
    assert not (tmp_path / "inputs/test-supplement.json").exists()


@pytest.mark.parametrize("alias", ["clip", "siglip2", "videojev", "decision3"])
def test_native_pipeline_reports_supplement_after_unchanged_validation(
    bundle_factory, frozen, tmp_path, alias
):
    bundle = bundle_factory("native", aliases=(alias,))
    _, deployment, deployment_hash = frozen

    class SyntheticBackend(FakeBackend):
        behavior = None

        def __init__(self, alias, config, bundle, deployment, output, port):
            super().__init__(alias, config, bundle, deployment, output, port)
            self.answers = {}
            for split in ("validation", "test"):
                plan = read(bundle, f"plans/{alias}-{split}.json")
                for q in plan["questions"]:
                    for uid in q["unit_ids"]:
                        assert (
                            self.answers.setdefault(uid, q["reference"])
                            == q["reference"]
                        )

        def infer(self, unit, _body):
            return 200, self.raw(unit, self.answers[unit["id"]]), {}

    output = tmp_path / "native-output"
    native_run.execute(
        alias,
        bundle,
        deployment,
        output,
        deployment_hash,
        phase="both",
        authorization="synthetic offline regression only",
        backend_factory=SyntheticBackend,
    )
    result = read(output, "test/report.json")
    assert result["correct"] == result["accuracy_denominator"] == 222
    assert result["cohorts"]["test_supplement"]["correct"] == 20
    assert read(output, "validation/report.json")["accuracy_denominator"] == 24
    logged = summarize_score(output, alias, native=True)
    assert logged["test"]["planned_questions"] == 222
    assert logged["test"]["correct"] == 222
    assert logged["test"]["accuracy"] == 1
    assert logged["validation"]["planned_questions"] == 24
    for split in ("test", "validation"):
        latency = read(output, f"{split}/report.json")["test_attempt_latency"]
        expected_attempts = len(read(bundle, f"plans/{alias}-{split}.json")["units"])
        assert latency["n"] == latency["attempts"] == expected_attempts
        assert logged[split]["latency_n"] == expected_attempts
        details = logged[split]["latency_details"]
        assert details["split"] == split
        assert details["p95_with_failures_n"] == expected_attempts
        assert details["p95_with_failures_seconds"] == latency["p95_seconds"]
        assert details["failed_attempts"] == details["timeout_attempts"] == 0
        assert details["attempts_without_result"] == 0
        assert details["successful_attempts_without_timing"] == 0
        assert details["attempt_elapsed_fallback"] == 0
    if alias in {"clip", "siglip2"}:
        artifact = read(output, "thresholds.json")
        assert artifact["validation_summary"]["complete_valid_questions"] == 24
        assert artifact["thresholds"] == dict.fromkeys(d.EVENTS, 0)
