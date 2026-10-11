"""Fixed-evidence quality contracts; all runs use fake local inspectors."""

import copy
import io
import json
import os
from html.parser import HTMLParser

import pytest
from PIL import Image

from goodvision_harness.model_api import ModelAPIError
from goodvision_harness.vlm_event_policy import annotation_policy
from goodvision_harness.vlm_independent_model import IndependentVerdict
from tools.vlm_benchmark import behavior
from tools.vlm_benchmark.behavior_report import summarize

LABELS = ("fight", "smoking", "fall", "weapon", "ppe_violation", "ground_litter")


def save(path, data):
    path.write_text(json.dumps(data), encoding="utf-8")


@pytest.fixture
def packet(tmp_path):
    p = tmp_path / "packet"
    p.mkdir()
    policy = annotation_policy(LABELS)
    windows = []
    for i in range(3):
        wid = f"W{i + 1:03}"
        (p / wid).mkdir()
        frames = []
        for j in range(2):
            data = io.BytesIO()
            Image.new("RGB", (64, 48), (i * 60, j * 80, 100)).save(data, "JPEG")
            name = f"{wid}/frame-{j + 1}.jpg"
            (p / name).write_bytes(data.getvalue())
            frames.append(
                {
                    "path": name,
                    "media_seconds": i * 2 + j * 0.8,
                    "sha256": behavior.sha(data.getvalue()),
                }
            )
        windows.append(
            {
                "window_id": wid,
                "frames": frames,
                "judgment_time_seconds": frames[-1]["media_seconds"],
            }
        )
    manifest = {
        "source_video_sha256": "a" * 64,
        "source_duration_seconds": 6,
        "policy_version": policy["version"],
        "policy_sha256": policy["sha256"],
        "windows": windows,
        "image_side": 720,
        "jpeg_quality": 90,
        "prediction_data_included": False,
        "previous_annotations_included": False,
    }
    annotations = {
        "source_video_sha256": "a" * 64,
        "policy_sha256": policy["sha256"],
        "human_reviewed": False,
        "annotator_model": "test_reference",
        "inspected_window_ids": [w["window_id"] for w in windows],
        "uninspected_window_ids": [],
        "rows": [],
    }
    for i, w in enumerate(windows):
        annotations["rows"].append(
            {
                "window_id": w["window_id"],
                "judgment_time_seconds": w["judgment_time_seconds"],
                "evidence_timestamps_seconds": [
                    f["media_seconds"] for f in w["frames"]
                ],
                "judgments": {
                    label: {
                        "detected": [True, False, None][i],
                        "reason": "REFERENCE_SECRET_DO_NOT_SEND",
                    }
                    for label in policy["event_labels"]
                },
            }
        )
    save(p / "input-manifest.json", manifest)
    save(p / "event-policy.json", policy)
    save(p / "annotations.json", annotations)
    return p


def args(packet, tmp_path, *extra):
    return behavior.parser().parse_args(
        [
            "--packet",
            str(packet),
            "--output",
            str(tmp_path / "output"),
            *(
                [
                    "--simulation-time-scale",
                    ".1",
                    "--simulated-latency",
                    "0",
                    "--input-lag-tolerance",
                    ".5",
                    "--queue-timeout",
                    "3",
                ]
                if "--simulate" in extra
                else []
            ),
            *extra,
        ]
    )


@pytest.mark.parametrize(
    "case",
    [
        "policy",
        "video",
        "duplicate_ref",
        "missing_ref",
        "reference_time",
        "frame_time",
        "frame_hash",
        "image_path",
        "absolute_path",
        "duplicate_evidence",
        "nonboolean_ref",
        "missing_label",
        "unknown_window",
        "reference_nan",
        "reference_not_inspected",
    ],
)
def test_packet_validation_rejects_unfair_or_corrupt_inputs(packet, case):
    m = behavior.read_json(packet / "input-manifest.json")
    a = behavior.read_json(packet / "annotations.json")
    policy = behavior.read_json(packet / "event-policy.json")
    if case == "policy":
        policy["decision_rules"] += "different criteria"
    elif case == "video":
        a["source_video_sha256"] = "b" * 64
    elif case == "duplicate_ref":
        a["rows"].append(copy.deepcopy(a["rows"][0]))
    elif case == "missing_ref":
        a["rows"].pop()
    elif case == "reference_time":
        a["rows"][0]["evidence_timestamps_seconds"][0] = 0.1
    elif case == "frame_time":
        m["windows"][0]["frames"][0]["media_seconds"] = 2
    elif case == "frame_hash":
        m["windows"][0]["frames"][0]["sha256"] = "c" * 64
    elif case == "image_path":
        m["windows"][0]["frames"][0]["path"] = "../annotations.json"
    elif case == "absolute_path":
        m["windows"][0]["frames"][0]["path"] = str(packet / "W001/frame-1.jpg")
    elif case == "duplicate_evidence":
        m["windows"][1]["frames"] = m["windows"][0]["frames"]
        m["windows"][1]["judgment_time_seconds"] = 0.8
    elif case == "nonboolean_ref":
        a["rows"][0]["judgments"]["fight"]["detected"] = 1
    elif case == "missing_label":
        del a["rows"][0]["judgments"]["fight"]
    elif case == "unknown_window":
        a["rows"][0]["window_id"] = "W999"
    elif case == "reference_nan":
        a["rows"][0]["judgment_time_seconds"] = float("nan")
    elif case == "reference_not_inspected":
        a["inspected_window_ids"].pop()
    save(packet / "input-manifest.json", m)
    save(packet / "annotations.json", a)
    save(packet / "event-policy.json", policy)
    with pytest.raises(ValueError):
        behavior.load_packet(packet, packet / "annotations.json")


def test_duplicate_json_fields_rejected(packet):
    (packet / "annotations.json").write_text('{"rows":[],"rows":[]}', encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate JSON"):
        behavior.load_packet(packet, packet / "annotations.json")


def test_no_credentials_or_model_calls_during_plan(packet, tmp_path, monkeypatch):
    monkeypatch.setattr(
        behavior, "key_from_environment", lambda *_: pytest.fail("key read")
    )
    out = behavior.run(
        args(packet, tmp_path), inspector_factory=lambda *_: pytest.fail("model called")
    )
    doc = behavior.read_json(out / "manifest.json")
    assert doc["measurement_kind"] == "plan_only"
    assert doc["planned_requests"] == 144
    stages = list(dict.fromkeys((x["repeat"], x["mode"]) for x in doc["planned"]))
    assert stages == [(1, "events"), (1, "grounding"), (2, "grounding"), (2, "events")]
    assert not (out / "results.jsonl").exists()
    assert doc["mode_settings"]["events"]["timeout_seconds"] == 5
    assert doc["mode_settings"]["grounding"]["timeout_seconds"] == 20
    assert doc["reference_sha256"] == behavior.sha(
        (out / "inputs/annotations.json").read_bytes()
    )
    for d in behavior.read_json(out / "summary.json")["delivery"]:
        assert d["not_run"] == d["planned"] == 36
    with pytest.raises(FileExistsError):
        behavior.run(args(packet, tmp_path))


@pytest.mark.parametrize("wire", ["legacy", "lean-v1"])
def test_dashscope_plan_keeps_policy_and_direct_route(
    packet, tmp_path, monkeypatch, wire
):
    monkeypatch.setattr(
        behavior, "key_from_environment", lambda *_: pytest.fail("key read")
    )
    options = args(
        packet, tmp_path, "--backend", "dashscope", "--grounding-format", wire
    )
    out = behavior.run(
        options, inspector_factory=lambda *_: pytest.fail("model called")
    )
    doc = behavior.read_json(out / "manifest.json")
    assert doc["model"] == "qwen3-vl-30b-a3b-instruct"
    assert doc["provider"] == "dashscope" and doc["provider_routing"] == "direct"
    assert not doc["provider_fallbacks"] and not doc["model_fallbacks"]
    assert (
        doc["event_policy"]["sha256"]
        == behavior.read_json(packet / "event-policy.json")["sha256"]
    )
    assert doc["planned_requests"] == 144
    for cfg in doc["mode_settings"].values():
        assert cfg["parameters"]["response_format"] == {"type": "json_object"}
        assert "provider" not in cfg["parameters"]
    assert "直接调用阿里云百炼" in (out / "report.html").read_text(encoding="utf-8")


def test_same_evidence_and_no_reference_in_model_messages(packet, tmp_path):
    m, p, _reference, images = behavior.load_packet(packet, packet / "annotations.json")
    configs, bodies, prompts = [], [], []
    for mode in behavior.MODES:
        item = {"mode": mode, "repeat": 1, "request_id": "request", "window_id": "W001"}
        cfg = behavior.make_config(item, m, p, args(packet, tmp_path))
        adapter = behavior.mode_inspector(cfg)
        job = behavior.make_job(item, m["windows"][0], images["W001"], p)
        try:
            messages = adapter.build_messages(job)
        finally:
            adapter.close()
        assert "REFERENCE_SECRET" not in json.dumps(messages)
        assert [f.timestamp_seconds for f in job.frames] == [0, 0.8]
        configs.append(cfg)
        prompts.append(messages[0]["content"])
        bodies.append(messages[1])
    assert bodies[0] == bodies[1]
    assert all(s.startswith(p["decision_rules"]) for s in prompts)
    assert all(c.max_in_flight == 4 for c in configs)
    for cfg in configs:
        adapter = behavior.mode_inspector(cfg)
        try:
            assert "only" not in adapter._parameters["provider"]
            assert "order" not in adapter._parameters["provider"]
            assert adapter._parameters["provider"]["sort"] == "latency"
            assert adapter._parameters["provider"]["allow_fallbacks"] is True
            assert adapter.model == behavior.DEFAULT_MODEL
            assert "models" not in adapter._parameters
            assert "route" not in adapter._parameters
            assert adapter.settings.max_retries == 0
        finally:
            adapter.close()


def test_simulation_uses_parser_and_portable_report(packet, tmp_path):
    out = behavior.run(args(packet, tmp_path, "--simulate"))
    records = [
        json.loads(line)
        for line in (out / "results.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(records) == 144
    completed = [r for r in records if r["state"] == "complete"]
    assert completed
    assert all(r["judgments"]["fight"] is False for r in completed)
    assert all(r["state"] in {"complete", "skipped"} for r in records)
    text = (out / "report.html").read_text(encoding="utf-8")
    assert "simulation_not_model_quality" in (out / "manifest.json").read_text(
        encoding="utf-8"
    )
    assert "离线模拟" in text

    class Links(HTMLParser):
        def handle_starttag(self, tag, attrs):
            for k, v in attrs:
                if k in {"src", "href"}:
                    assert (out / v).is_file(), v

    Links().feed(text)


class ErrorInspector:
    def inspect(self, job):
        raise ModelAPIError(
            "SECRET must not appear", code="http_error", status_code=429
        )

    def close(self):
        pass


@pytest.mark.parametrize("continue_timeouts", [False, True])
def test_rate_limit_stops_and_keeps_skipped_denominator(
    packet, tmp_path, continue_timeouts
):
    out = behavior.run(
        args(
            packet,
            tmp_path,
            "--simulate",
            *(["--continue-on-timeout"] if continue_timeouts else []),
        ),
        inspector_factory=lambda cfg: ErrorInspector(),
    )
    doc = behavior.read_json(out / "manifest.json")
    assert doc["state"] == "stopped" and doc["stop_reason"] == "http_429"
    assert 1 <= doc["recorded_requests"] <= 48
    summary = behavior.read_json(out / "summary.json")
    assert sum(d["failed"] for d in summary["delivery"]) == doc["recorded_requests"]
    assert sum(d["not_run"] for d in summary["delivery"]) >= 108
    assert "SECRET must not appear" not in (out / "results.jsonl").read_text(
        encoding="utf-8"
    )


@pytest.mark.parametrize("continue_timeouts", [False, True])
def test_repeated_timeouts_respect_explicit_policy_and_stay_failed(
    packet, tmp_path, continue_timeouts
):
    class TimeoutInspector:
        def inspect(self, job):
            raise ModelAPIError("request deadline", code="timeout")

        def close(self):
            pass

    options = args(
        packet,
        tmp_path,
        "--simulate",
        *(["--continue-on-timeout"] if continue_timeouts else []),
    )
    out = behavior.run(options, inspector_factory=lambda _: TimeoutInspector())
    doc = behavior.read_json(out / "manifest.json")
    summary = behavior.read_json(out / "summary.json")
    assert doc["continue_on_timeout"] is continue_timeouts
    assert doc["state"] == ("completed" if continue_timeouts else "stopped")
    assert all(d["complete"] == 0 for d in summary["delivery"])
    assert all(m["fn"] == 0 for m in summary["metrics"])
    if continue_timeouts:
        assert doc["recorded_windows"] == doc["planned_requests"] == 144
        assert doc["stop_reason"] is None
        assert all(d["failed"] > 0 and d["not_run"] == 0 for d in summary["delivery"])
        assert "普通请求超时计为失败后继续" in (out / "report.html").read_text(
            encoding="utf-8"
        )
    else:
        assert (
            doc["stop_reason"]
            == "three_consecutive_unsuccessful_requests_on_one_stream"
        )


@pytest.mark.parametrize(
    "status,code,reason",
    [
        (401, "http_error", "http_401"),
        (402, "http_error", "http_402"),
        (403, "http_error", "http_403"),
        (429, "http_error", "http_429"),
        (None, "client_error", "client_error"),
        (None, "interrupted", "interrupted"),
        (
            None,
            "invalid_verdict",
            "three_consecutive_unsuccessful_requests_on_one_stream",
        ),
    ],
)
def test_timeout_policy_does_not_disable_other_stops(status, code, reason):
    from tools.vlm_benchmark.behavior_queue import StopControl

    control = StopControl(continue_on_timeout=True)
    for _ in range(3):
        control.observe(
            {
                "http_status": status,
                "error_code": code,
                "attempted": True,
                "stream_id": 1,
                "state": "failed",
            }
        )
    assert control.event.is_set() and control.reason == reason


def test_continue_timeouts_protects_transport_failures_but_records_response_waits():
    from tools.vlm_benchmark.behavior_queue import StopControl

    control = StopControl(continue_on_timeout=True)
    result = {
        "http_status": None,
        "error_code": "timeout",
        "attempted": True,
        "stream_id": 1,
        "state": "failed",
        "http_timing": {
            "transport_trace_observed": True,
            "http_status": 200,
            "request_body_completed": True,
        },
    }
    for _ in range(4):
        control.observe(result)
    assert not control.event.is_set()
    result["http_timing"] = {"transport_trace_observed": True, "tcp_connect_seconds": 5}
    control.observe(result)
    control.observe(result)
    assert not control.event.is_set()
    control.observe(result)
    assert control.reason == "three_consecutive_transport_setup_timeouts_on_one_stream"


class UngroundedInspector:
    def __init__(self):
        self.last_response_diagnostics = {"provider_category": "Alibaba"}

    def inspect(self, job):
        return tuple(
            IndependentVerdict(label, (), True, 0.01, ())
            for label in job.request.event_labels
        )

    def close(self):
        pass


def test_raw_positive_survives_missing_grounding(packet, tmp_path):
    out = behavior.run(
        args(packet, tmp_path, "--simulate"),
        inspector_factory=lambda cfg: UngroundedInspector(),
    )
    records = [
        json.loads(line)
        for line in (out / "results.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    r = next(r for r in records if r["mode"] == "grounding")
    assert r["judgments"]["fight"] is True
    assert r["report"]["judgments"][0]["detected"] is None
    assert not r["report"]["localization_complete"]


def test_scoring_separates_reference_uncertainty_missing_and_abstention(packet):
    m, policy, reference, _ = behavior.load_packet(packet, packet / "annotations.json")
    doc = {
        "event_policy": policy,
        "planned": behavior.schedule(m, 1, streams=1),
        "repeats": 1,
        "measurement_kind": "simulation_not_model_quality",
        "state": "stopped",
    }
    plans = doc["planned"]
    records = [
        {
            **plans[0],
            "state": "complete",
            "judgments": dict.fromkeys(LABELS, True),
            "inspect_seconds": 1,
        },
        {
            **plans[1],
            "state": "complete",
            "judgments": dict.fromkeys(LABELS, None),
            "inspect_seconds": 2,
        },
        {**plans[2], "state": "failed", "judgments": {}, "inspect_seconds": 3},
        {
            **plans[3],
            "state": "complete",
            "judgments": dict.fromkeys(LABELS, False),
            "inspect_seconds": 4,
        },
    ]
    summary, detail = summarize(doc, m, reference, records)
    a = next(
        x
        for x in summary["metrics"]
        if x["mode"] == "events" and x["event_label"] == "fight"
    )
    b = next(
        x
        for x in summary["metrics"]
        if x["mode"] == "grounding" and x["event_label"] == "fight"
    )
    assert a["tp"] == 1 and a["model_uncertain"] == 1 and a["reference_uncertain"] == 1
    assert a["f1_decided_only"] == 1 and a["binary_coverage"] == 0.5
    assert a["agreement_over_all_certain"] == 0.5
    assert b["fn"] == 1 and b["not_run"] == 1 and b["reference_uncertain"] == 1
    assert b["f1_decided_only"] == 0
    assert summary["paired"][0]["both_binary_comparable"] == 1
    assert summary["delivery"][0]["failed"] == 1
    assert any(
        d["request_state"] == "failed" and d["outcome"] == "reference_uncertain"
        for d in detail
    )


def test_zero_positive_support_does_not_report_perfect_f1(packet):
    m, policy, reference, _ = behavior.load_packet(packet, packet / "annotations.json")
    for r in reference["rows"]:
        for v in r["judgments"].values():
            v["detected"] = False
    planned = behavior.schedule(m, 1, streams=1)
    doc = {
        "event_policy": policy,
        "planned": planned,
        "repeats": 1,
        "measurement_kind": "simulation_not_model_quality",
        "state": "completed",
    }
    records = [
        {
            **p,
            "state": "complete",
            "judgments": dict.fromkeys(LABELS, False),
            "inspect_seconds": 1,
        }
        for p in planned
    ]
    summary, _ = summarize(doc, m, reference, records)
    assert all(
        r["f1_decided_only"] is None
        and r["recall_decided_only"] is None
        and not r["can_assess_false_negatives"]
        for r in summary["metrics"]
    )


def test_budget_fails_before_key_or_network(packet, tmp_path, monkeypatch):
    monkeypatch.setattr(
        behavior, "key_from_environment", lambda *_: pytest.fail("key read")
    )
    with pytest.raises(ValueError, match="budget"):
        behavior.run(args(packet, tmp_path, "--real", "--max-requests", "1"))


def test_real_path_uses_explicit_key_and_never_saves_it(packet, tmp_path, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "FAKE_KEY_FOR_LOCAL_TEST")
    monkeypatch.setattr(
        behavior,
        "acquire",
        lambda root: os.open(tmp_path / "test.lock", os.O_CREAT | os.O_RDWR, 0o600),
    )

    def factory(cfg):
        assert cfg.api_key == "FAKE_KEY_FOR_LOCAL_TEST"
        return UngroundedInspector()

    out = behavior.run(args(packet, tmp_path, "--real"), inspector_factory=factory)
    assert behavior.read_json(out / "manifest.json")["state"] == "completed"
    for file in out.glob("*.json*"):
        assert "FAKE_KEY_FOR_LOCAL_TEST" not in file.read_text(encoding="utf-8")


@pytest.mark.parametrize("failure", ["incomplete", "interrupted", "close_error"])
def test_interruption_and_bad_cleanup_preserve_reports(packet, tmp_path, failure):
    class Inspector(UngroundedInspector):
        def inspect(self, job):
            if failure == "interrupted":
                raise KeyboardInterrupt()
            if failure == "incomplete":
                return (IndependentVerdict("fight", (), False, None, ()),)
            return super().inspect(job)

        def close(self):
            if failure == "close_error":
                raise RuntimeError("SECRET_CLOSE_EXCEPTION")

    out = behavior.run(
        args(packet, tmp_path, "--simulate"), inspector_factory=lambda cfg: Inspector()
    )
    doc = behavior.read_json(out / "manifest.json")
    assert doc["state"] == "stopped"
    summary = behavior.read_json(out / "summary.json")
    if failure == "interrupted":
        assert 1 <= doc["recorded_requests"] <= 48
        assert doc["stop_reason"] == "interrupted"
        assert summary["delivery"][0]["failed"] == doc["recorded_requests"]
    elif failure == "incomplete":
        assert 3 <= doc["recorded_requests"] <= 36
        assert summary["delivery"][0]["incomplete"] == doc["recorded_requests"]
        assert summary["metrics"][0]["binary_comparable"] == 0
    else:
        assert (out / "report.html").exists()
    assert (
        sum(
            d["complete"] + d["failed"] + d["incomplete"] + d["not_run"] + d["skipped"]
            for d in summary["delivery"]
        )
        == 144
    )
    assert "SECRET_CLOSE_EXCEPTION" not in (out / "manifest.json").read_text(
        encoding="utf-8"
    )


def test_misaligned_or_duplicate_results_cannot_be_scored(packet):
    m, policy, reference, _ = behavior.load_packet(packet, packet / "annotations.json")
    planned = behavior.schedule(m, 1, streams=1)
    doc = {
        "event_policy": policy,
        "planned": planned,
        "repeats": 1,
        "measurement_kind": "simulation_not_model_quality",
        "state": "completed",
    }
    record = {
        **planned[0],
        "state": "complete",
        "judgments": dict.fromkeys(LABELS, False),
        "inspect_seconds": 1,
    }
    with pytest.raises(ValueError, match="duplicate result"):
        summarize(doc, m, reference, [record, record])
    with pytest.raises(ValueError, match="planned window"):
        summarize(doc, m, reference, [{**record, "window_id": "W002"}])
    with pytest.raises(ValueError, match="unplanned"):
        summarize(doc, m, reference, [{**record, "request_id": "unknown"}])


def test_false_positive_and_negative_metrics_have_expected_denominators(packet):
    m, policy, reference, _ = behavior.load_packet(packet, packet / "annotations.json")
    planned = behavior.schedule(m, 1, streams=1)
    doc = {
        "event_policy": policy,
        "planned": planned,
        "repeats": 1,
        "measurement_kind": "simulation_not_model_quality",
        "state": "completed",
    }
    records = [
        {
            **p,
            "state": "complete",
            "inspect_seconds": 1,
            "judgments": dict.fromkeys(LABELS, p["window_id"] != "W001"),
        }
        for p in planned
    ]
    summary, _ = summarize(doc, m, reference, records)
    for row in summary["metrics"]:
        assert row["fp"] == row["fn"] == row["reference_uncertain"] == 1
        assert row["tp"] == row["tn"] == 0
        assert row["f1_decided_only"] == row["precision_decided_only"] == 0
        assert row["recall_decided_only"] == row["agreement_over_all_certain"] == 0
        assert row["binary_coverage"] == 1


def test_packet_paths_reject_windows_drive_even_on_posix(tmp_path):
    with pytest.raises(ValueError, match="relative paths"):
        behavior.packet_file(tmp_path, "C:/source/frame.jpg")


def test_six_event_comparison_rejects_subset_policy(packet):
    save(packet / "event-policy.json", annotation_policy(("fight",)))
    with pytest.raises(ValueError, match="six event"):
        behavior.load_packet(packet, packet / "annotations.json")


def test_twelve_stream_schedule_keeps_69_unique_windows():
    manifest = {"windows": [{"window_id": f"W{i:03}"} for i in range(1, 70)]}
    planned = behavior.schedule(manifest, 2)
    assert len(planned) == len({p["request_id"] for p in planned}) == 3312
    assert {p["stream_id"] for p in planned} == set(range(1, 13))
    assert len({p["window_id"] for p in planned}) == 69
    for mode in behavior.MODES:
        stage = [p for p in planned if p["repeat"] == 1 and p["mode"] == mode]
        assert len(stage) == 828
        assert stage[12]["release_offset_seconds"] == 0.8
        assert stage[11]["release_offset_seconds"] < 0.8


@pytest.mark.parametrize("interval,selected_count", [(4, 14), (5, 11)])
def test_sparse_grounding_preserves_source_speed_and_never_uses_future_frames(
    interval, selected_count
):
    times = [0.816 + i * 0.8 for i in range(69)]
    manifest = {
        "windows": [
            {"window_id": f"W{i + 1:03}", "judgment_time_seconds": t}
            for i, t in enumerate(times)
        ]
    }
    planned = behavior.schedule(manifest, 2, grounding_interval=interval)
    assert len(planned) == 3312
    for repeat in (1, 2):
        for mode in ("events", "grounding"):
            stage = [p for p in planned if p["repeat"] == repeat and p["mode"] == mode]
            assert [p["release_offset_seconds"] for p in stage] == sorted(
                p["release_offset_seconds"] for p in stage
            )
            assert max(p["release_offset_seconds"] for p in stage) < 56
            selected = [
                p for p in stage if p["stream_id"] == 1 and p["sampling_selected"]
            ]
            assert len(selected) == (69 if mode == "events" else selected_count)
            if mode == "grounding":
                for tick, p in enumerate(selected):
                    source_at = times[0] + tick * interval
                    index = int(p["window_id"][1:]) - 1
                    assert times[index] <= source_at
                    assert index == 68 or times[index + 1] > source_at
                    assert p["release_offset_seconds"] == pytest.approx(tick * interval)
                    assert p["evidence_age_at_release_seconds"] == pytest.approx(
                        source_at - times[index]
                    )


def test_sparse_plan_budget_counts_calls_and_preserves_reference_denominator(
    packet, tmp_path
):
    options = args(
        packet, tmp_path, "--grounding-interval", "5", "--max-requests", "96"
    )
    out = behavior.run(options)
    doc = behavior.read_json(out / "manifest.json")
    assert doc["planned_requests"] == 96
    assert doc["reference_window_slots"] == 144
    assert doc["source_clock_mode"] == "original_timestamps_1x"
    summary = behavior.read_json(out / "summary.json")
    for d in summary["delivery"]:
        assert d["planned"] == 36
        assert d["selected"] == (36 if d["mode"] == "events" else 12)
        assert d["not_run"] == d["selected"]
        assert d["not_sampled"] == d["planned"] - d["selected"]
    options.output = tmp_path / "denied"
    options.max_requests = 95
    with pytest.raises(ValueError, match="budget"):
        behavior.run(options)


@pytest.mark.parametrize("interval", [4, 5])
def test_sparse_run_records_unselected_separately_without_inspector_calls(
    packet, tmp_path, interval
):
    annotations = behavior.read_json(packet / "annotations.json")
    for row in annotations["rows"]:
        for verdict in row["judgments"].values():
            verdict["detected"] = True
    save(packet / "annotations.json", annotations)
    calls = []

    class CountingInspector(UngroundedInspector):
        def inspect(self, job):
            calls.append(job.request.request_id)
            return super().inspect(job)

    out = behavior.run(
        args(packet, tmp_path, "--simulate", "--grounding-interval", str(interval)),
        inspector_factory=lambda _: CountingInspector(),
    )
    doc = behavior.read_json(out / "manifest.json")
    rows = [
        json.loads(line)
        for line in (out / "results.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    chosen_ids = {p["request_id"] for p in doc["planned"] if p["sampling_selected"]}
    assert doc["state"] == "completed"
    assert len(rows) == doc["reference_window_slots"] == 144
    assert set(calls) <= chosen_ids and len(calls) == len(set(calls))
    excluded = [r for r in rows if not r["sampling_selected"]]
    assert excluded and all(
        r["state"] == "not_sampled" and not r["attempted"] for r in excluded
    )
    summary = behavior.read_json(out / "summary.json")
    for metric in summary["metrics"]:
        assert metric["fn"] == 0
        assert (
            metric["reference_positive_selected"]
            + metric["reference_positive_not_sampled"]
            == metric["planned"]
        )
    for d in summary["delivery"]:
        assert d["not_sampled"] + d["selected"] == d["planned"]
        assert d["complete_fraction_selected"] >= d["complete_fraction"]
    assert "不能将降频后的返回率称为原负载下的性能提升" in (
        out / "report.html"
    ).read_text(encoding="utf-8")


def test_model_pin_auto_routing_and_source_isolation(packet, tmp_path):
    m, policy, _, images = behavior.load_packet(packet, packet / "annotations.json")
    sessions = []
    frames = []
    for plan in behavior.schedule(m, 1)[:12]:
        cfg = behavior.make_config(plan, m, policy, args(packet, tmp_path))
        job = behavior.make_job(plan, m["windows"][0], images["W001"], policy)
        assert cfg.parameters["provider"] == {
            "sort": "latency",
            "require_parameters": True,
            "allow_fallbacks": True,
        }
        sessions.append(job.stream_session_id)
        frames.append([f.jpeg_bytes for f in job.frames])
    assert len(set(sessions)) == 12
    assert all(f == frames[0] for f in frames)


def test_provider_pin_preserves_model_policy_and_disables_fallbacks(packet, tmp_path):
    settings = args(packet, tmp_path, "--provider-only", "deepinfra")
    m, policy, _, _ = behavior.load_packet(packet, packet / "annotations.json")
    for mode in behavior.MODES:
        cfg = behavior.make_config({"mode": mode, "repeat": 1}, m, policy, settings)
        assert cfg.model == behavior.DEFAULT_MODEL
        assert cfg.parameters["provider"] == {
            "sort": "latency",
            "require_parameters": True,
            "allow_fallbacks": False,
            "only": ["deepinfra"],
        }
    out = behavior.run(settings)
    manifest = behavior.read_json(out / "manifest.json")
    assert manifest["provider_only"] == "deepinfra"
    assert manifest["provider_fallbacks"] is False
    assert manifest["event_policy"] == policy
    page = (out / "report.html").read_text(encoding="utf-8")
    assert "固定服务商 deepinfra" in page and "禁用服务商回退" in page
    assert "自动选择低延迟服务商" not in page


def test_lean_grounding_simulation_keeps_all_windows_and_quality_accounting(
    packet,
    tmp_path,
):
    out = behavior.run(
        args(
            packet,
            tmp_path,
            "--simulate",
            "--repeats",
            "1",
            "--grounding-format",
            "lean-v1",
            "--simulation-time-scale",
            "1",
        )
    )
    doc = behavior.read_json(out / "manifest.json")
    assert doc["grounding_format"] == "lean-v1"
    assert doc["grounding_interval_seconds"] is None
    assert doc["planned_requests"] == doc["reference_window_slots"] == 72
    assert doc["mode_settings"]["grounding"]["adapter"] == "LeanGroundingInspector"
    rows = [
        json.loads(line)
        for line in (out / "results.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(rows) == 72
    assert all(r["state"] == "complete" for r in rows)
    assert all(set(r["judgments"]) == set(LABELS) for r in rows)
    assert "lean-v1" in (out / "report.html").read_text(encoding="utf-8")


def test_grounding_only_plan_and_run_has_no_phantom_events(packet, tmp_path):
    out = behavior.run(
        args(
            packet,
            tmp_path,
            "--simulate",
            "--repeats",
            "1",
            "--grounding-format",
            "lean-v1",
            "--modes",
            "grounding",
            "--simulation-time-scale",
            "1",
        )
    )
    doc = behavior.read_json(out / "manifest.json")
    summary = behavior.read_json(out / "summary.json")
    assert doc["planned_requests"] == doc["reference_window_slots"] == 36
    assert {p["mode"] for p in doc["planned"]} == {"grounding"}
    assert summary["paired"] == []
    assert len(summary["delivery"]) == 1
    assert summary["delivery"][0]["complete"] == 36
    assert "src/goodvision_harness/vlm_lean_grounding.py" in doc["source_hashes"]


@pytest.mark.parametrize("provider", ["Novita", None])
def test_auto_provider_does_not_block_or_change_model(
    packet, tmp_path, monkeypatch, provider
):
    import httpx

    monkeypatch.setattr(httpx, "get", lambda *_a, **_k: pytest.fail("catalog gate"))
    monkeypatch.setattr(behavior, "key_from_environment", lambda *_: "LOCAL_FAKE")
    monkeypatch.setattr(
        behavior,
        "acquire",
        lambda _: os.open(tmp_path / "test.lock", os.O_CREAT | os.O_RDWR, 0o600),
    )

    def factory(cfg):
        assert cfg.model == behavior.DEFAULT_MODEL
        inspector = UngroundedInspector()
        inspector.last_response_diagnostics = {"provider_category": provider}
        return inspector

    out = behavior.run(
        args(packet, tmp_path, "--real", "--repeats", "1", "--observation-fps", "3"),
        inspector_factory=factory,
    )
    doc = behavior.read_json(out / "manifest.json")
    assert doc["state"] == "completed" and doc["stop_reason"] is None
    assert doc["provider_only"] is None and doc["provider_fallbacks"] is True
    assert doc["model"] == behavior.DEFAULT_MODEL and doc["model_fallbacks"] is False
    rows = [
        json.loads(line)
        for line in (out / "results.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    attempted = [r for r in rows if r.get("attempted")]
    assert attempted and all(
        r["requested_model"] == behavior.DEFAULT_MODEL for r in attempted
    )
    assert all(r["state"] == "complete" for r in attempted)
    for d in behavior.read_json(out / "summary.json")["delivery"]:
        assert d["timing"]["provider_counts"] == {
            provider or "unreported": d["complete"]
        }


def test_wrong_model_rejected_before_credentials_or_calls(
    packet, tmp_path, monkeypatch
):
    monkeypatch.setattr(
        behavior, "key_from_environment", lambda *_: pytest.fail("key read")
    )
    options = args(packet, tmp_path, "--real")
    options.model = "qwen/qwen3-vl-8b-instruct"
    with pytest.raises(ValueError, match="requires Qwen3-VL-30B"):
        behavior.run(options, inspector_factory=lambda _: pytest.fail("model called"))


def test_streams_do_not_share_quality_pairs(packet):
    m, policy, reference, _ = behavior.load_packet(packet, packet / "annotations.json")
    plan = behavior.schedule(m, 1, streams=2)
    document = {
        "event_policy": policy,
        "planned": plan,
        "repeats": 1,
        "measurement_kind": "simulation_not_model_quality",
        "state": "stopped",
    }
    # Opposite sources return different modes. No within-source pair exists.
    records = [
        {
            **p,
            "state": "complete",
            "judgments": dict.fromkeys(LABELS, True),
            "inspect_seconds": 1,
        }
        for p in plan
        if (p["mode"], p["stream_id"]) in {("events", 1), ("grounding", 2)}
    ]
    summary, _ = summarize(document, m, reference, records)
    assert all(p["both_binary_comparable"] == 0 for p in summary["paired"])
    src1, src2 = summary["per_stream"]
    assert src1["delivery"][0]["complete"] == 3
    assert src2["delivery"][0]["complete"] == 0
    assert summary["unique_windows"] == 3


def test_skips_remain_missing_not_false_negatives(packet):
    m, policy, reference, _ = behavior.load_packet(packet, packet / "annotations.json")
    plan = behavior.schedule(m, 1, streams=1)
    doc = {
        "planned": plan,
        "event_policy": policy,
        "repeats": 1,
        "measurement_kind": "simulation_not_model_quality",
        "state": "completed",
    }
    rows = [
        {**p, "state": "skipped", "judgments": {}, "error_code": "dispatch_timeout"}
        for p in plan
    ]
    summary, _ = summarize(doc, m, reference, rows)
    assert all(d["skipped"] == d["planned"] for d in summary["delivery"])
    assert all(m["fn"] == 0 and m["binary_coverage"] == 0 for m in summary["metrics"])
    assert all(m["f1_decided_only"] is None for m in summary["metrics"])


def queue_run(packet, tmp_path, *, global_limit=48, per_stream_limit=4, latency=0.12):
    import time
    from collections import Counter
    from threading import Lock
    from types import SimpleNamespace

    from tools.vlm_benchmark.behavior_queue import run_stage

    options = args(packet, tmp_path, "--simulate")
    options.global_limit, options.per_stream_limit = global_limit, per_stream_limit
    m = behavior.read_json(packet / "input-manifest.json")
    planned = [p for p in behavior.schedule(m, 1) if p["mode"] == "events"]
    counts, peaks, starts = Counter(), Counter(), []
    guard = Lock()

    def evaluate(item, job, cfg, inspector):
        start = time.monotonic()
        with guard:
            source = item["stream_id"]
            counts[source] += 1
            counts["all"] += 1
            peaks[source] = max(peaks[source], counts[source])
            peaks["all"] = max(peaks["all"], counts["all"])
            starts.append(start)
        time.sleep(latency)
        with guard:
            counts[source] -= 1
            counts["all"] -= 1
        return {
            **item,
            "state": "complete",
            "error_code": None,
            "http_status": None,
            "inspect_seconds": time.monotonic() - start,
            "judgments": {},
            "response_diagnostics": {},
        }

    records = []
    stage = run_stage(
        planned,
        options,
        tmp_path / "admission.sqlite3",
        lambda p: SimpleNamespace(
            stream_session_id=f"s{p['stream_id']}", output_mode="events"
        ),
        lambda _: None,
        lambda _: UngroundedInspector(),
        evaluate,
        records.append,
    )
    return options, planned, stage, records, peaks, starts


def test_queue_enforces_limits_and_measures_nonoverlapping_phases(packet, tmp_path):
    _options, plan, stage, rows, peaks, starts = queue_run(packet, tmp_path)
    assert stage["stop_reason"] is None
    assert 1 < peaks["all"] <= 48
    assert max(v for k, v in peaks.items() if k != "all") <= 4
    assert len(rows) == len(plan) == 36
    assert stage["peak_pending_windows"] <= 12
    assert stage["peak_worker_requests"] <= 48
    assert stage["admission"]["active"] == stage["admission"]["waiting"] == 0
    assert sum(r["attempted"] for r in rows) == stage["admission"]["grants"]
    # Aggregate rate cannot exceed the configured gate, allowing scheduling jitter
    # between the SQLite grant and the immediately following inspect call.
    if len(starts) > 1:
        assert max(starts) - min(starts) >= (len(starts) - 1) / 200 - 0.02
    for row in rows:
        if row["state"] != "complete":
            continue
        timing = row["latency"]
        pieces = [
            "dispatch_wait_seconds",
            "admission_wait_seconds",
            "inspector_setup_seconds",
            "inspect_seconds",
            "result_finalize_seconds",
            "publication_wait_seconds",
        ]
        assert sum(timing[p] for p in pieces) == pytest.approx(
            timing["input_to_publish_seconds"]
        )
        assert timing["queue_wait_seconds"] == pytest.approx(
            timing["dispatch_wait_seconds"] + timing["admission_wait_seconds"]
        )
        assert timing["scheduled_to_publish_seconds"] == pytest.approx(
            timing["input_to_publish_seconds"] + timing["input_release_lag_seconds"]
        )


def test_overload_keeps_latest_bounded_pending_and_accounts_for_every_window(
    packet, tmp_path
):
    _, plan, stage, rows, peaks, _ = queue_run(
        packet, tmp_path, global_limit=2, per_stream_limit=1, latency=0.25
    )
    assert len(rows) == len({r["request_id"] for r in rows}) == len(plan)
    assert stage["peak_worker_requests"] <= 2
    assert stage["peak_pending_windows"] <= 12
    assert peaks["all"] <= 2
    assert max(v for k, v in peaks.items() if k != "all") == 1
    skips = [r for r in rows if r["state"] == "skipped"]
    assert skips
    assert {r["error_code"] for r in skips} <= {
        "pending_superseded",
        "dispatch_timeout",
        "admission_timeout",
        "source_missed",
    }
    assert all(
        not r["attempted"] and r["latency"]["inspect_seconds"] is None for r in skips
    )


def test_stop_drain_publishes_ready_results_before_slowest_request(packet, tmp_path):
    import time
    from threading import Barrier, Event
    from types import SimpleNamespace

    from tools.vlm_benchmark.behavior_queue import run_stage

    options = args(packet, tmp_path, "--simulate")
    started = Barrier(3)
    failed_published, fast_published = Event(), Event()
    slow_saw_fast = []
    planned = [
        {
            "request_id": f"s{s}",
            "stream_id": s,
            "repeat": 1,
            "mode": "events",
            "release_offset_seconds": 0,
        }
        for s in range(1, 4)
    ]

    def evaluate(item, *_):
        began = time.monotonic()
        started.wait(timeout=3)
        if item["stream_id"] == 2:
            assert failed_published.wait(timeout=3)
        elif item["stream_id"] == 3:
            slow_saw_fast.append(fast_published.wait(timeout=3))
        failure = item["stream_id"] == 1
        return {
            **item,
            "state": "failed" if failure else "complete",
            "error_code": "http_error" if failure else None,
            "http_status": 429 if failure else None,
            "inspect_seconds": time.monotonic() - began,
            "judgments": {},
            "response_diagnostics": {},
        }

    records = []

    def publish(result):
        records.append(result)
        if result["stream_id"] == 1:
            failed_published.set()
        elif result["stream_id"] == 2:
            fast_published.set()

    stage = run_stage(
        planned,
        options,
        tmp_path / "admission.sqlite3",
        lambda p: SimpleNamespace(
            stream_session_id=f"s{p['stream_id']}", output_mode="events"
        ),
        lambda _: None,
        lambda _: UngroundedInspector(),
        evaluate,
        publish,
    )
    assert stage["stop_reason"] == "http_429"
    assert slow_saw_fast == [True]
    assert [r["stream_id"] for r in records] == [1, 2, 3]
    assert stage["admission"]["active"] == stage["admission"]["waiting"] == 0


def test_source_stall_invalidates_load_and_fast_skips_do_not_reduce_result_latency():
    from tools.vlm_benchmark.behavior_report import timing_summary

    rows = [
        {
            "state": "complete",
            "attempted": True,
            "latency": {"input_to_publish_seconds": 4, "input_release_lag_seconds": 0},
        },
        {
            "state": "skipped",
            "error_code": "source_missed",
            "attempted": False,
            "latency": {
                "input_to_publish_seconds": 0.01,
                "input_release_lag_seconds": 0.2,
            },
        },
    ]
    result = timing_summary([{}, {}, {}], rows, {})
    assert result["input_timing_valid"] is False
    assert result["source_missed"] == 1
    assert result["complete_result_latency"]["p50"] == 4
    assert result["within_seconds"]["3"]["count"] == 0
    assert result["within_seconds"]["5"]["fraction_of_planned"] == 1 / 3
