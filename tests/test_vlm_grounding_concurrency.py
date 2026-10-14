import json
from dataclasses import replace

import pytest
from test_vlm_independent import config
from test_vlm_multistream_benchmark import report

from goodvision_harness.vlm_admission import AdmissionSettings
from tools import vlm_grounding_concurrency as sweep
from tools.vlm_benchmark import modes
from tools.vlm_benchmark.grounding_sweep_report import analyze
from tools.vlm_multistream_benchmark import summarize


def test_plan_freezes_workload_reverses_order_and_accounts_every_request():
    args = sweep.parser().parse_args([])
    modes.validate(args)
    jobs = list(modes.jobs(args))
    assert [j["limits"]["per_stream_limit"] for j in jobs] == [4, 6, 8, 8, 6, 4]
    assert len({j["id"] for j in jobs}) == 6
    for j in jobs:
        c = j["limits"]["per_stream_limit"]
        assert j["limits"] == dict(
            global_limit=12 * c, per_stream_limit=c, local_concurrency=c
        )
        assert j["streams"] == 12 and j["mode"] == "compact"
        assert j["duration"] == args.duration
    assert modes.request_budget(args) == dict(
        measurement_upper=4968,
        preflight_and_warmup_upper=84,
        optional_soak_upper=0,
        total_upper=5052,
    )
    assert args.http_timeout == 20 and args.grounding_format == "lean-v1"
    assert not args.real and not args.record


def test_paid_plan_does_not_read_key_or_run(monkeypatch, capsys):
    def forbidden(*a, **kw):
        pytest.fail("planning must not read secrets or run inference")

    monkeypatch.setattr(sweep, "read_api_key", forbidden)
    monkeypatch.setattr(modes, "run", forbidden)
    sweep.main(["--real", "--plan-only", "--video", "not-read.mp4"])
    assert json.loads(capsys.readouterr().out)["real_model"]


@pytest.mark.parametrize(
    "options",
    [
        ["--grounding-concurrency-sweep", "4", "4"],
        ["--grounding-concurrency-sweep", "4", "9"],
        ["--streams", "8"],
        ["--modes", "events"],
        ["--grounding-format", "legacy"],
        ["--soak-seconds", "30"],
    ],
)
def test_rejects_noncomparable_sweep(options):
    with pytest.raises(ValueError, match="grounding sweep"):
        modes.validate(sweep.parser().parse_args(options))


def test_execution_layers_accept_eight_but_stay_bounded(tmp_path):
    assert replace(config(), max_in_flight=8).max_in_flight == 8
    assert AdmissionSettings(str(tmp_path / "gate.db"), 96, 8).per_stream_limit == 8
    with pytest.raises(ValueError):
        replace(config(), max_in_flight=9)
    with pytest.raises(ValueError):
        AdmissionSettings(str(tmp_path / "gate.db"), 96, 9)


def capacity_fixture(tmp_path, *, unknown=False, active=0):
    def row(error=None):
        r = {**report(error=error), "vlm_lane": "combined"}
        r["timing"].update(
            http_timing={"total_seconds": 2.0},
            source_to_request_seconds=0.5,
            admission_wait_seconds=0.2,
            http_request_attempted=True,
            inspection_attempted=True,
        )
        return r

    records = [row(), row("timeout"), row("invalid_response"), row("admission_timeout")]
    records[-1]["timing"]["http_timing"] = {}
    records[-1]["timing"]["http_request_attempted"] = False
    records[-1]["timing"]["inspection_attempted"] = False
    if unknown:
        records[0]["timing"].pop("http_request_attempted")
    lane = summarize(records, {}, 3, 6, 3, lane="grounding")
    stage = dict(
        lanes={"grounding": lane},
        input_timing_valid=True,
        all_drained=True,
        result_records_missing=0,
        result_retention={"valid": True},
        input_schedule_span_seconds=3,
        drain_elapsed_seconds=1,
        shared_admission={"active": active, "waiting": 0},
        per_stream=[{"lanes": {"grounding": lane}}],
    )
    raw = tmp_path / "reports.jsonl"
    raw.write_text("".join(json.dumps({"report": r}) + "\n" for r in records))
    return stage, raw


def test_accounting_separates_api_failure_admission_and_unscheduled(tmp_path):
    r = analyze(*capacity_fixture(tmp_path), real=True)
    assert r["valid"] and r["planned"] == 6
    assert r["attempts"] == 3 and r["not_issued"] == 3
    assert r["complete"] == 1 and r["complete_fraction_of_attempts"] == 1 / 3
    assert r["complete_fraction_of_planned"] == 1 / 6
    assert r["attempt_errors"] == {"timeout": 1, "invalid_response": 1}
    assert r["admission_rejected"] == 1 and r["windows_without_record"] == 2
    assert r["api_all_attempts"]["count"] == 3
    assert r["api_complete"]["count"] == 1
    assert r["complete_per_wall_minute"] == 15


def test_unknown_http_records_and_residual_slots_are_not_valid(tmp_path):
    r = analyze(*capacity_fixture(tmp_path, unknown=True), real=True)
    assert not r["valid"] and not r["accounting_valid"]
    assert r["unclassified_records"] == 1 and r["not_issued"] is None
    assert not analyze(*capacity_fixture(tmp_path, active=32), real=True)["valid"]


def test_elapsed_request_preparation_is_not_a_confirmed_http_send(tmp_path):
    stage, path = capacity_fixture(tmp_path)
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    rows[1]["report"]["timing"]["http_request_attempted"] = False
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    result = analyze(stage, path, real=True)
    assert result["accounting_valid"]
    assert result["attempts"] == 2
    assert result["not_issued"] == 4
    assert result["confirmed_not_attempted_records"] == 2


@pytest.mark.parametrize("retention", [False, None])
def test_missing_retention_cannot_turn_missing_reports_into_unsent_work(
    tmp_path, retention
):
    stage, path = capacity_fixture(tmp_path)
    stage["result_retention"]["valid"] = retention
    result = analyze(stage, path, real=True)
    assert not result["valid"] and not result["accounting_valid"]
    assert result["not_issued"] is None


def test_confirmed_http_send_does_not_require_elapsed_timing(tmp_path):
    stage, path = capacity_fixture(tmp_path)
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    rows[0]["report"]["timing"]["http_timing"] = {}
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    result = analyze(stage, path, real=True)
    assert result["valid"] and result["attempts"] == 3
    assert result["api_all_attempts"]["count"] == 2


@pytest.mark.parametrize("fatal", [None, 429, 402])
def test_runner_applies_all_limits_preserves_results_and_stops(
    tmp_path, monkeypatch, fatal
):
    args = sweep.parser().parse_args(
        [
            "--duration",
            "3",
            "--warmup-seconds",
            "3",
            "--repeats",
            "1",
            "--output",
            str(tmp_path / "run"),
        ]
    )
    seen = []

    class NoTelemetry:
        def __init__(self, *args):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def summary(self):
            return {}

    def fake_stage(options, base, count, frames, location, repeat, **kwargs):
        if count == 12:
            seen.append(
                (
                    options.local_concurrency,
                    options.per_stream_limit,
                    options.global_limit,
                )
            )
        error = "http_error" if fatal and count == 12 else None
        r = {**report(error=error), "vlm_lane": "combined"}
        r["timing"].update(inspection_attempted=True, http_request_attempted=False)
        if error:
            r["timing"]["http_status"] = fatal
        data = summarize([r], {}, 3, 1, 3, lane="grounding")
        combined = summarize([r] * count, {}, 3, count, 3, lane="grounding")
        folder = location / f"streams-{count}-repeat-{repeat}"
        folder.mkdir()
        (folder / "reports.jsonl").write_text(
            "".join(json.dumps({"report": r}) + "\n" for _ in range(count))
        )
        return dict(
            streams=count,
            repeat=repeat,
            lanes={"grounding": combined},
            per_stream=[{"lanes": {"grounding": data}}] * count,
            input_timing_valid=True,
            all_drained=True,
            result_records_missing=0,
            result_retention={"valid": True},
            shared_admission={"active": 0, "waiting": 0},
            paired_windows=None,
            input_schedule_span_seconds=3,
            drain_elapsed_seconds=0.1,
            nominal_requests_per_second=15,
            rate_limit_below_offered_load=False,
        )

    monkeypatch.setattr(modes, "Telemetry", NoTelemetry)
    monkeypatch.setattr(modes, "run_stage", fake_stage)
    if fatal:
        with pytest.raises(RuntimeError, match=f"HTTP {fatal}"):
            modes.run(args)
        assert seen == [(4, 4, 48)]
    else:
        modes.run(args)
        assert seen == [(4, 4, 48), (6, 6, 72), (8, 8, 96)]
    result = json.loads(
        (args.output / "grounding-concurrency.json").read_text(encoding="utf-8")
    )
    assert result["state"] == ("failed" if fatal else "completed")
    assert len(result["trials"]) == len(seen)
    assert (args.output / "grounding-concurrency.csv").exists()
    assert "API P50" in (args.output / "grounding-concurrency.html").read_text(
        encoding="utf-8"
    )
