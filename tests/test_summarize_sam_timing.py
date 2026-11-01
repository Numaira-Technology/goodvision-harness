"""Synthetic saved-log diagnostics; no model, service or socket is used."""

import importlib.util
import json
from pathlib import Path

SPEC = importlib.util.spec_from_file_location(
    "summarize_sam_timing",
    Path(__file__).parents[1] / "tools/summarize_sam_timing.py",
)
timing = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(timing)


def test_correlates_worker_and_proxy_by_request_and_keeps_attempts_separate(tmp_path):
    client = tmp_path / "lane-02.log"
    client.write_text(
        "\n".join(
            [
                (
                    'INFO: sam_client_http_timing '
                    '{"request_id":"req.1","http_attempt_id":"io.1",'
                    '"outcome":"error","error_type":"RemoteProtocolError",'
                    '"total_seconds":4.8,'
                    '"http_until_headers_seconds":4.8,'
                    '"url":"https://private-token@private.test"}'
                ),
                (
                    'INFO: sam_client_append_retry '
                    '{"request_id":"req.1","recorded_at":"2026-10-05T03:53:53+00:00"}'
                ),
                (
                    'INFO: sam_client_http_timing '
                    '{"request_id":"req.1","http_attempt_id":"io.2",'
                    '"outcome":"ok","total_seconds":5.7,'
                    '"http_until_headers_seconds":5.5,"response_read_seconds":0.2}'
                ),
                (
                    'INFO: sam_client_append_recovered '
                    '{"request_id":"req.1","recorded_at":"2026-10-05T03:53:59+00:00"}'
                ),
            ]
        )
    )
    server = tmp_path / "worker.log"
    server.write_text(
        "\n".join(
            [
                (
                    "INFO Native session append timing request_id=req.1 "
                    "native_advance_seconds=3.2 snapshot_seconds=0.3 outcome=completed"
                ),
                (
                    "INFO Native session request completed request_id=req.1 "
                    "http_attempt_id=io.2 queue_wait_seconds=2.0 processing_seconds=3.5"
                ),
                (
                    "INFO SAM cloud proxy timing transport_request_id=req.1 "
                    "http_attempt_id=io.2 upstream_headers_seconds=5.4"
                ),
            ]
        )
    )
    report = timing.summarize(
        [*timing.read_records(client), *timing.read_records(server)]
    )
    assert report["correlation_coverage"] == {
        "requests": 1,
        "with_worker_request": 1,
        "with_worker_append": 1,
        "with_proxy": 1,
    }
    (request,) = report["failed_or_retried_requests"]
    assert request["http_attempts"] == 2
    assert request["client_total_seconds"] == 10.5
    assert request["retry_to_recovered_seconds"] == 6
    assert [r["http_attempt_id"] for r in request["attempts"]] == ["io.1", "io.2"]
    stages = report["per_camera_stages"]["2"]
    assert (
        stages["Native session request completed.queue_wait_seconds"]["max_seconds"]
        == 2
    )
    assert (
        stages["Native session append timing.native_advance_seconds"]["max_seconds"]
        == 3.2
    )
    assert "private" not in json.dumps(report)
    assert any("not GPU time" in note for note in report["interpretation"])


def test_relative_window_filters_client_requests_but_keeps_correlated_remote_rows(
    tmp_path,
):
    saved = tmp_path / "parsed-events.json"
    saved.write_text(
        json.dumps(
            {
                "1": {
                    "sam_client_http_timing": [
                        {
                            "request_id": "inside",
                            "relative_spawn_s": 10,
                            "total_seconds": 2,
                            "outcome": "ok",
                        },
                        {
                            "request_id": "outside",
                            "relative_spawn_s": 151,
                            "total_seconds": 8,
                            "outcome": "ok",
                        },
                    ],
                    "sam_coordinator_timing": [
                        {
                            "request_id": "inside",
                            "source_to_result": [{"sequence": 10, "seconds": 3}],
                        },
                    ],
                }
            }
        )
    )
    remote = timing._record(
        "Native session append timing",
        {
            "request_id": "inside",
            "native_advance_seconds": 1.5,
        },
    )
    report = timing.summarize(
        [*timing.read_records(saved), remote], relative_start=0, relative_end=150
    )
    assert report["correlation_coverage"]["requests"] == 1
    assert report["correlation_coverage"]["with_worker_append"] == 1
    assert report["slowest_requests"][0]["request_id"] == "inside"
    assert (
        report["per_camera_stages"]["1"][
            "sam_coordinator_timing.source_to_result_seconds"
        ]["p95_seconds"]
        == 3
    )


def test_missing_remote_measurements_are_not_zero_or_inferred_gpu_time():
    row = timing._record(
        "sam_client_http_timing",
        {
            "request_id": "req.1",
            "outcome": "ok",
            "http_until_headers_seconds": 5,
            "total_seconds": 5.1,
        },
        "3",
    )
    report = timing.summarize([row])
    assert report["correlation_coverage"]["with_worker_append"] == 0
    stages = report["per_camera_stages"]["3"]
    assert "Native session append timing.native_advance_seconds" not in stages
    assert any("cannot establish" in note for note in report["interpretation"])


def test_nearest_rank_quantiles_are_deterministic_and_empty_is_missing():
    assert timing.distribution([]) is None
    assert timing.distribution([5, 1, 4, 3, 2]) == {
        "count": 5,
        "p50_seconds": 3,
        "p95_seconds": 5,
        "max_seconds": 5,
    }
    assert (
        timing._record(
            "sam_client_http_timing",
            {
                "request_id": "https://private.test",
                "total_seconds": float("nan"),
            },
        )["fields"]
        == {}
    )


def test_recovery_and_session_lifecycle_keep_safe_scalars_without_request_ids(tmp_path):
    saved = tmp_path / "lane-04.log"
    fields = {
        "sam_coordinator_request_failure": {
            "request_id": "req.failed",
            "generation": 2,
            "request_elapsed_seconds": 4.8,
            "outcome": "error",
        },
        "sam_transport_recovery": {
            "request_id": "req.failed",
            "generation": 3,
            "retired_generation": 2,
            "failed_append_seconds": 4.8,
        },
        "sam_session_cleanup": {
            "generation": 2,
            "cleanup_seconds": 0.2,
            "outcome": "ok",
            "recovery": True,
        },
        "sam_session_open": {
            "generation": 3,
            "stream_session_id": "source.001",
            "open_seconds": 0.4,
            "outcome": "ok",
        },
    }
    saved.write_text(
        "\n".join(
            f"INFO: {event} "
            + json.dumps(
                value
                | {
                    "message": "https://private-token@private.test",
                    "body": "private-image-body",
                    "headers": {"Authorization": "private-token"},
                }
            )
            for event, value in fields.items()
        )
    )
    rows = list(timing.read_records(saved))
    report = timing.summarize(rows)
    lifecycle = report["recovery_and_lifecycle"]
    assert lifecycle["event_counts"] == dict.fromkeys(fields, 1)
    assert len(lifecycle["events"]) == 4
    assert lifecycle["stage_distributions"]["sam_session_open.open_seconds"] == {
        "count": 1,
        "p50_seconds": 0.4,
        "p95_seconds": 0.4,
        "max_seconds": 0.4,
    }
    (cleanup,) = [r for r in lifecycle["events"] if r["event"] == "sam_session_cleanup"]
    assert (
        cleanup["fields"]["generation"] == 2 and cleanup["fields"]["recovery"] is True
    )
    (recovery,) = [
        r for r in lifecycle["events"] if r["event"] == "sam_transport_recovery"
    ]
    assert recovery["fields"]["retired_generation"] == 2
    (failed,) = report["failed_or_retried_requests"]
    assert failed["request_id"] == "req.failed" and len(failed["recovery_events"]) == 2
    assert failed["http_attempts"] == 0  # Missing HTTP records are not invented.
    assert "private" not in json.dumps(report)


def test_lifecycle_window_uses_available_offsets_but_reports_unknown_offsets():
    rows = [
        timing._record(
            "sam_session_open",
            {
                "generation": i,
                "open_seconds": 0.4,
                "relative_spawn_s": offset,
            },
        )
        for i, offset in enumerate((10, 151))
    ]
    rows.append(
        timing._record(
            "sam_session_cleanup",
            {
                "generation": 0,
                "cleanup_seconds": 0.2,
                "recovery": True,
            },
        )
    )
    report = timing.summarize(rows, relative_start=0, relative_end=150)
    lifecycle = report["recovery_and_lifecycle"]
    assert lifecycle["event_counts"] == {
        "sam_session_open": 1,
        "sam_session_cleanup": 1,
    }
    assert any("without relative offsets" in note for note in report["interpretation"])
