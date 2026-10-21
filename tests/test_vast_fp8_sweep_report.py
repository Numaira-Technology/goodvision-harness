"""Frozen comparison arms reconcile saved raw results before scoring."""

import json
from dataclasses import asdict

import pytest
from test_tracked_decisions_comparison import write_run

from tools.vast_vlm import fp8_profile, fp8_sweep, runtime
from tools.vlm_multistream_benchmark import LABELS


@pytest.fixture
def arms(tmp_path):
    roots = []
    for n in (16, 24, 32):
        root = tmp_path / str(n)
        exp = fp8_sweep.Experiment(n)
        summary = write_run(root, decisions=False)
        summary.update(fp8_sweep.plan(exp), state="completed")
        for job in summary["jobs"]:
            job["result"].update(
                input_schedule_span_seconds=55.8, drain_elapsed_seconds=4
            )
            for lane in job["result"]["lanes"].values():
                lane.update(
                    expected_observations=828, reports=2, complete_valid_results=1
                )
        runtime.write_json(root / "summary.json", summary)
        runtime.write_json(
            root / "manifest.json",
            {
                **fp8_sweep.plan(exp),
                "settings": {
                    k: v
                    for k, v in vars(
                        fp8_sweep.arguments(exp, real=True, video="output.mp4")
                    ).items()
                    if k not in ("video", "output")
                },
                "preflight_only": False,
                "decoded_evidence": [],
                "source_hashes": {"script": "test-source-hash"},
            },
        )
        runtime.write_json(
            root / "deployment.json",
            {
                "vlm": {
                    "model": fp8_profile.MODEL_REPO,
                    "revision": fp8_profile.MODEL_REVISION,
                    "vllm_version": fp8_profile.ENGINE_VERSION,
                    "image_expected": fp8_profile.IMAGE,
                    "experiment": asdict(exp),
                    "kernel_bundle": None,
                    "kernel_attestation": {"validated": True, "moe_backend": "TRITON"},
                }
            },
        )
        roots.append(root)
    reference = {
        "policy": summary["event_policy"],
        "decoded_evidence": [],
        "reference_sha256": "fixture-reference",
        "lookup": {(0,): "w"},
        "refs": {"w": {"judgments": {k: {"detected": False} for k in LABELS}}},
    }
    return roots, reference


def test_compare_keeps_missing_windows_and_per_camera_scoring(arms):
    roots, reference = arms
    result = fp8_sweep.compare(roots, reference)
    assert [v["experiment"]["sequences"] for v in result["variants"]] == [16, 24, 32]
    for variant in result["variants"]:
        assert len(variant["trials"]) == 6
        for trial in variant["trials"]:
            for lane in trial["lanes"].values():
                assert lane["sent_fraction"] == 2 / 828
                assert lane["complete_fraction"] == 1 / 828
                assert lane["timely"]["on_time_coverage"] == 0
                assert not lane["timely"]["all_cameras_pass"]
                assert lane["astra"]["binary_comparable"] == 6
                assert lane["http_status_counts"] == {"200": 1, "429": 1}
                assert lane["tokens"]["completion_tokens"]["coverage"] == 0


def test_v1_history_is_read_without_rewriting_or_changing_scores(arms):
    roots, reference = arms
    expected = fp8_sweep.compare(roots, reference)
    saved = {}
    for root in roots:
        path = root / "manifest.json"
        manifest = json.loads(path.read_text())
        manifest["version"] = "vast-qwen35-fp8-yolo26s-v1"
        manifest["settings"].pop("backend_person_tracks")
        runtime.write_json(path, manifest)
        saved[path] = path.read_bytes()
    assert fp8_sweep.compare(roots, reference) == expected
    assert all(path.read_bytes() == data for path, data in saved.items())


def test_explicit_frozen_stage_load_matches_historical_defaults(arms):
    roots, reference = arms
    expected = fp8_sweep.compare(roots, reference)
    for root in roots:
        path = root / "summary.json"
        summary = json.loads(path.read_text())
        for job in summary["jobs"]:
            job["result"].update(
                streams=12,
                duration_seconds=fp8_profile.FROZEN["duration"],
                observation_fps=fp8_profile.FROZEN["observation_fps"],
                source_sample_fps=2.5,
            )
        runtime.write_json(path, summary)
    assert fp8_sweep.compare(roots, reference) == expected


@pytest.mark.parametrize(
    "field,value",
    [
        ("streams", 1),
        ("duration_seconds", 10),
        ("observation_fps", 0.25),
        ("source_sample_fps", 1),
    ],
)
def test_comparison_rejects_stage_load_that_changes_report_denominator(
    arms, field, value
):
    roots, reference = arms
    path = roots[0] / "summary.json"
    summary = json.loads(path.read_text())
    # Keep job load and the saved 828-window lane counter unchanged. Neither
    # alone protects the denominator used by the shared report.
    summary["jobs"][0]["result"][field] = value
    runtime.write_json(path, summary)
    with pytest.raises(ValueError, match="frozen stage load"):
        fp8_sweep.compare(roots, reference)


@pytest.mark.parametrize(
    "problem",
    [
        "denominator",
        "duration",
        "missing-lane",
        "duplicate-trial",
        "missing-trial",
        "incomplete",
        "policy",
        "summary-count",
        "duplicate-window",
        "changed-source",
        "changed-kernel",
        "changed-model",
        "unattested",
        "load",
        "preflight",
    ],
)
def test_comparison_refuses_confounded_or_incomplete_arms(arms, problem):
    roots, reference = arms
    root = roots[0]
    summary = json.loads((root / "summary.json").read_text())
    manifest = json.loads((root / "manifest.json").read_text())
    deployment = json.loads((root / "deployment.json").read_text())
    job = summary["jobs"][0]
    if problem == "denominator":
        job["result"]["lanes"]["events"]["expected_observations"] = 600
    elif problem == "duration":
        job["duration"] = 10
    elif problem == "missing-lane":
        summary["jobs"][-1]["result"]["lanes"].pop("grounding")
    elif problem == "duplicate-trial":
        summary["jobs"].append(job)
    elif problem == "missing-trial":
        summary["jobs"].pop()
    elif problem == "incomplete":
        summary["state"] = "interrupted"
    elif problem == "policy":
        manifest["event_policy"]["sha256"] = "other"
    elif problem == "summary-count":
        job["result"]["lanes"]["events"]["complete_valid_results"] = 2
    elif problem == "duplicate-window":
        path = root / job["raw_reports"]
        path.write_text(path.read_text() + path.read_text().splitlines()[0] + "\n")
    elif problem == "changed-source":
        manifest["source_hashes"]["script"] = "different"
    elif problem == "changed-kernel":
        deployment["vlm"]["kernel_attestation"]["moe_backend"] = "MARLIN"
    elif problem == "changed-model":
        deployment["vlm"]["revision"] = "different"
    elif problem == "unattested":
        deployment["vlm"]["kernel_attestation"]["validated"] = False
    elif problem == "load":
        manifest["settings"]["observation_fps"] = 1
    elif problem == "preflight":
        manifest["preflight_only"] = True
    runtime.write_json(root / "summary.json", summary)
    runtime.write_json(root / "manifest.json", manifest)
    runtime.write_json(root / "deployment.json", deployment)
    with pytest.raises(ValueError):
        fp8_sweep.compare(roots, reference)
