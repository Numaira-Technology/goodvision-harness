"""Offline comparison rejects incompatible runs and reconciles native audits."""

import json

import pytest

from tools.tracked_decisions import comparison, suite
from tools.tracked_grounding import report
from tools.tracked_grounding.suite import plan as qwen_plan


def write_run(root, *, decisions):
    summary = suite.plan() if decisions else qwen_plan()
    summary.update(state="completed", jobs=[])
    for mode in ("events", "compact", "parallel"):
        lanes = (
            ["events", "grounding"]
            if mode == "parallel"
            else ["grounding" if mode == "compact" else "events"]
        )
        for repeat in (1, 2):
            identifier = f"{mode}-{repeat}"
            folder = root / identifier
            folder.mkdir(parents=True)
            rows, audits = [], []
            for lane in lanes:
                for n in (0, 1):
                    good = n == 0
                    identity = {
                        "stream_session_id": identifier,
                        "request_id": f"{lane}-{n}",
                        "vlm_lane": lane,
                    }
                    rows.append(
                        {
                            "camera": 1,
                            "report": {
                                **identity,
                                "window_id": f"w{n}",
                                "state": "completed" if good else "failed",
                                "analysis_complete": good,
                                "event_policy": summary["event_policy"],
                                "evidence": [
                                    {"frame_sequence": 0, "timestamp_seconds": 0}
                                ],
                                "judgments": [
                                    {"event_label": label, "observed_detected": False}
                                    for label in suite.decisions.LABELS
                                ],
                                "timing": {
                                    "source_to_publish_seconds": 2 if decisions else 4,
                                    "http_request_attempted": True,
                                    "http_status": 200 if good else 429,
                                },
                            },
                        }
                    )
                    audits.append(
                        {
                            **identity,
                            "complete": good,
                            "http_request_attempted": True,
                            "http_status": 200 if good else 429,
                            "input_tokens": 100 if good else None,
                        }
                    )
            (folder / "reports.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in rows)
            )
            if decisions:
                (folder / "api-audit").mkdir()
                (folder / "api-audit/native.jsonl").write_text(
                    "".join(json.dumps(row) + "\n" for row in audits)
                )
            summary["jobs"].append(
                {
                    "id": identifier,
                    "phase": "measurement",
                    "mode": mode,
                    "repeat": repeat,
                    "streams": 12,
                    "duration": 55.798021,
                    "raw_reports": f"{identifier}/reports.jsonl",
                    "detector_counts": {},
                    "recording": {"complete": True},
                    "result": {
                        "all_drained": True,
                        "input_timing_valid": True,
                        "result_records_missing": 0,
                        "result_retention": {"valid": True},
                        "lanes": {
                            lane: {
                                "http_status_counts": {"200": 1, "429": 1},
                                "request_phase_metrics": {},
                            }
                            for lane in lanes
                        },
                    },
                }
            )
    (root / "summary.json").write_text(json.dumps(summary))
    return summary


@pytest.fixture
def runs(tmp_path, monkeypatch):
    reference = {
        "policy": {"sha256": comparison.POLICY_SHA},
        "lookup": {(0,): "w"},
        "refs": {
            "w": {
                "judgments": {
                    label: {"detected": False} for label in suite.decisions.LABELS
                }
            }
        },
    }
    monkeypatch.setattr(comparison, "load_reference", lambda *a: reference)
    monkeypatch.setattr(comparison, "sha", lambda *a: comparison.VIDEO_SHA)
    old, new = tmp_path / "old", tmp_path / "new"
    write_run(old, decisions=False)
    write_run(new, decisions=True)
    return old, new, reference


def test_common_reference_and_fixed_denominators(runs, tmp_path):
    old, new, reference = runs
    result = comparison.compare(new, old, tmp_path / "compare", "packet", "video")
    assert len(result["rows"]) == 8
    assert len(result["joint_by_round"]) == 2
    assert result["http_timeouts"] == {
        "decisions": {"events": 20, "grounding": 20},
        "qwen35b": {"events": 5, "grounding": 20},
    }
    for row in result["rows"]:
        assert row["decisions"]["sent_fraction"] == 2 / 828
        assert row["decisions"]["complete_fraction"] == 1 / 828
        assert row["delta_percentage_points"]["within_3s"] == pytest.approx(100 / 828)
        assert row["decisions"]["astra"]["agreement"] == 1
        assert not row["decisions"]["timely"]["all_cameras_pass"]
    analysis = report.analyze(new, reference)
    assert analysis["all_trials_complete"]
    assert analysis["decisions_usage"]["audit_coverage"]["valid"]
    assert analysis["decisions_usage"]["usage_missing_requests"] == 8
    assert analysis["decisions_usage"]["estimated_base_cost_usd"] is None
    assert "Decisions" in (new / "report.html").read_text()


@pytest.mark.parametrize(
    "problem",
    [
        "policy",
        "source",
        "load",
        "detector",
        "unfinished",
        "missing-round",
        "duplicate-round",
        "missing-lane",
        "integrity",
        "missing-audit",
    ],
)
def test_incompatible_or_unreconciled_results_rejected(runs, tmp_path, problem):
    old, new, reference = runs
    path = new / "summary.json"
    summary = json.loads(path.read_text())
    if problem == "policy":
        summary["event_policy"]["sha256"] = "different"
    elif problem == "source":
        summary["source_sha256"] = "different"
    elif problem == "load":
        summary["observation_fps"] = 1
    elif problem == "detector":
        summary["detector"] = "other"
    elif problem == "unfinished":
        summary["state"] = "interrupted"
    elif problem == "missing-round":
        summary["jobs"].pop()
    elif problem == "duplicate-round":
        summary["jobs"].append(summary["jobs"][0])
    elif problem == "missing-lane":
        summary["jobs"][-1]["result"]["lanes"].pop("events")
    elif problem == "integrity":
        summary["jobs"][0]["result"]["result_retention"]["valid"] = False
    elif problem == "missing-audit":
        (new / "events-1/api-audit/native.jsonl").unlink()
        assert not report.analyze(new, reference)["all_trials_complete"]
    path.write_text(json.dumps(summary))
    with pytest.raises(ValueError):
        comparison.compare(new, old, tmp_path / "compare", "packet", "video")
    assert not (tmp_path / "compare").exists()
