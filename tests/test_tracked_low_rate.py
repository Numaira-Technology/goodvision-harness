"""Low-frequency cloud experiment: frozen evidence, honest API accounting."""

import json

import pytest

from tools.tracked_grounding.api_report import api_seconds, summarize
from tools.tracked_grounding.suite import arguments, main, plan


def test_low_rate_plan_is_offline_and_only_changes_load(monkeypatch, capsys):
    def forbidden(*args, **kwargs):
        pytest.fail("offline plan accessed a credential/provider")

    monkeypatch.setattr("tools.tracked_grounding.suite.read_api_key", forbidden)
    assert main(["--load-profile", "four-second"]) == 0
    low = json.loads(capsys.readouterr().out)
    old = plan()
    assert low["planned_windows_per_branch_per_round"] == 168
    assert low["formal_vlm_requests"] == 1344
    assert low["global_limit"] == 4 and low["observation_fps"] == 0.25
    for key in (
        "event_policy",
        "source_sha256",
        "model",
        "provider",
        "sample_fps",
        "frames_per_window",
        "http_timeouts",
        "detector_sampling_fps",
        "streams",
        "modes",
        "repeats",
        "person_display",
    ):
        assert low[key] == old[key]
    before, after = vars(arguments()), vars(arguments(load_profile="four-second"))
    changed = {k for k in before.keys() | after.keys() if before.get(k) != after.get(k)}
    assert changed == {"observation_fps", "global_limit", "source_sample_fps"}
    with pytest.raises(ValueError):
        arguments(load_profile="unknown")


def test_api_timer_excludes_build_detector_queue_and_parse():
    report = {
        "timing": {
            "http_timing": {
                "http_until_headers_seconds": 1.2,
                "request_build_seconds": 0.2,
                "response_read_seconds": 0.5,
                "response_json_decode_seconds": 5,
                "total_seconds": 9,
            },
            "model_seconds": 12,
            "source_to_publish_seconds": 20,
        }
    }
    assert api_seconds(report) == 1.5
    del report["timing"]["http_timing"]["response_read_seconds"]
    assert api_seconds(report) is None


def test_api_report_counts_unsent_and_exact_parallel_pairs(tmp_path):
    def row(lane, wid, seconds):
        return {
            "camera": 1,
            "report": {
                "window_id": wid,
                "vlm_lane": lane,
                "state": "completed",
                "analysis_complete": True,
                "timing": {
                    "http_timing": {
                        "http_until_headers_seconds": seconds,
                        "request_build_seconds": 0,
                        "response_read_seconds": 0,
                    }
                },
            },
        }

    rows = [
        row("information", "paired", 1),
        row("grounding", "paired", 2),
        row("information", "unpaired", 1),
    ]
    raw = tmp_path / "reports.jsonl"
    raw.write_text("\n".join(json.dumps(r) for r in rows))
    document = {
        "state": "running",
        "model": "qwen/qwen3.5-35b-a3b",
        "provider": "OpenRouter; Alibaba preferred; fallback allowed",
        "jobs": [
            {
                "mode": "parallel",
                "repeat": 1,
                "phase": "measurement",
                "raw_reports": "reports.jsonl",
                "result": {
                    "duration_seconds": 55.798021,
                    "observation_fps": 0.25,
                    "source_sample_fps": 2.5,
                    "streams": 12,
                    "lanes": {"events": {}, "grounding": {}},
                },
            }
        ],
    }
    (tmp_path / "summary.json").write_text(json.dumps(document))
    result = summarize(tmp_path)
    assert not result["all_six_trials_finished"]
    by_lane = {r["lane"]: r for r in result["rows"]}
    assert by_lane["events"]["complete_rate"] == 2 / 168
    assert by_lane["joint"]["complete_rate"] == 1 / 168
    assert by_lane["joint"]["api_le3_rate"] == 1 / 168
    assert by_lane["joint"]["api_seconds"] is None
    raw.write_text(raw.read_text() + "\n" + json.dumps(rows[0]))
    with pytest.raises(ValueError, match="duplicate"):
        summarize(tmp_path)


def test_historical_streaming_profile_uses_sse_clock(tmp_path):
    import copy

    # Empty but correctly attributed historical suite; no rewriting of raw logs.
    original = {
        "state": "completed",
        "jobs": [],
        "settings": {
            "vision_profile": "glm_low",
            "model": "z-ai/glm-5.3-flash",
            "backend": "openrouter",
        },
    }
    summary = tmp_path / "summary.json"
    summary.write_text(json.dumps(original))
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"model": "z-ai/glm-5.3-flash"}))
    result = summarize(tmp_path)
    assert result["api_transport"] == "sse"
    assert not result["all_six_trials_finished"]
    assert json.loads(summary.read_text()) == original
    wrong = copy.deepcopy(original)
    wrong["settings"]["model"] = "wrong-model"
    summary.write_text(json.dumps(wrong))
    with pytest.raises(ValueError, match="model identity"):
        summarize(tmp_path)
