"""Regression + behaviour tests for the diag pipeline-log parser.

Covers the loop-lag parser bug (the harness emits `stats_loop_lag`, the old
parser read `loop_lag` and so reported 0 forever) and the per-stage / bandwidth
rollup the per-stream pressure table depends on.
"""

import importlib.util
import json
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent


def _load(module_name, relpath):
    spec = importlib.util.spec_from_file_location(module_name, _ROOT / relpath)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


pipeline = _load("parse_pipeline_log", "diag/parse_pipeline_log.py")
receive = _load("parse_receive_log", "diag/parse_receive_log.py")
from goodvision_harness import resource_trace  # noqa: E402 - needs PYTHONPATH=src


def _line(event, payload):
    return f"uvicorn.error: {event} {json.dumps(payload)}"


def _receive_window(loop_p95_seconds, *, bytes_received=0, decoded_fps=29.0):
    return {
        "recorded_at": "2026-10-03T00:00:00+00:00",
        "protocol": "webrtc",
        "window_decoded_fps": decoded_fps,
        "window_emitted_fps": decoded_fps,
        "target_fps": 30,
        "window_seconds": 5.0,
        "totals": {"decoded_gaps_over_200ms": 0, "handoff_queue_dropped_frames": 0},
        "timings": {
            "stats_loop_lag": {
                "count": 10,
                "mean_seconds": loop_p95_seconds / 2,
                "max_seconds": loop_p95_seconds,
                "sampled_p95_seconds": loop_p95_seconds,
            }
        },
        "inbound_rtp": [{"packets_lost_delta": 0}],
        "transport": [{"bytes_received": bytes_received}],
    }


def test_receive_parser_reads_stats_loop_lag(tmp_path):
    """The fixed parser must surface a non-zero loop p95 from stats_loop_lag."""
    log = tmp_path / "recv.log"
    log.write_text(_line("network_receive_timing", _receive_window(0.250)) + "\n")
    rows = receive.parse_file(str(log))
    assert len(rows) == 1
    # 0.250 s -> 250 ms; the old exact-match on "loop_lag" gave 0.
    assert rows[0]["loop_p95_ms"] == 250.0
    assert "LOOP-LAG" in receive._flag(rows[0])


def test_pipeline_loop_lag_and_bandwidth(tmp_path):
    log = tmp_path / "s1.log"
    log.write_text(
        "\n".join(
            [
                _line(
                    "network_receive_timing",
                    _receive_window(0.120, bytes_received=5_000_000),
                ),
                _line(
                    "resource_timing",
                    {
                        "recorded_at": "2026-10-03T00:00:01+00:00",
                        "interval_seconds": 5.0,
                        "camera_index": 1,
                        "cpu_percent": 42.0,
                        "num_threads": 12,
                        "rss_bytes": 300_000_000,
                        "system_cpu_percent_max": 70.0,
                        "thread_cpu_seconds": {"prepare": 1.0, "display": 0.5},
                    },
                ),
                _line(
                    "sam_client_http_timing",
                    {
                        "recorded_at": "2026-10-03T00:00:01+00:00",
                        "total_seconds": 2.4,
                        "transport_bytes": 100_000,
                    },
                ),
                _line(
                    "sam_coordinator_timing",
                    {
                        "recorded_at": "2026-10-03T00:00:01+00:00",
                        "append_seconds": 2.5,
                        "commit_seconds": 0.01,
                        "source_to_result": [1.6, 1.7],
                    },
                ),
                _line(
                    "sam_display_timing",
                    {
                        "recorded_at": "2026-10-03T00:00:01+00:00",
                        "render_seconds": 0.02,
                        "render_decode_seconds": 0.008,
                        "render_draw_seconds": 0.006,
                        "render_encode_seconds": 0.006,
                        "publish_seconds": 0.001,
                    },
                ),
            ]
        )
        + "\n"
    )
    s = pipeline.StreamSummary(str(log))
    for event, rec in pipeline._iter_events(str(log)):
        s.add(event, rec)
    m = s.metrics()
    assert s.camera_index == 1
    assert m["loop_p95_ms"] == 120.0
    assert m["cpu_proc_mean"] == 42.0
    assert m["bw_ingest_mbps"] > 0  # 5 MB / 5 s ~ 8 Mbps
    assert m["sam_http_p95_ms"] > 0
    assert m["s2r_p50_ms"] > 0
    # per-stage CPU attribution present: prepare burned 1.0 s / 5 s interval = 20%
    assert round(m["cpu_prepare"]) == 20
    assert round(m["cpu_display"]) == 10


def test_pipeline_gpu_csv(tmp_path):
    csv = tmp_path / "gpu.csv"
    csv.write_text(
        "timestamp,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw\n"
        "2026/10/03 00:00:00, 85, 40, 8000, 24000, 210\n"
        "2026/10/03 00:00:01, 92, 44, 8200, 24000, 230\n"
    )
    gpu = pipeline.parse_gpu_csv(str(csv))
    assert gpu["util_max"] == 92
    assert gpu["mem_max_mb"] == 8200


def test_resource_stage_mapping():
    assert resource_trace._stage_for("sam-prepare") == "prepare"
    assert resource_trace._stage_for("sam-ordered-commit") == "commit"
    assert resource_trace._stage_for("sam-ordered-request") == "dispatch"
    assert resource_trace._stage_for("video-connection-abcd1234") == "capture"
    assert resource_trace._stage_for("scene-display") == "display"
    assert resource_trace._stage_for("unknown-thread") == "other"


def test_percentile_edges():
    assert pipeline._percentile([], 95) == 0.0
    assert pipeline._percentile([0.1], 95) == 0.1
    assert pipeline._percentile([1, 2, 3, 4], 50) in (2, 3)


def _cumulative_window(cum_bytes, *, conn="c1", qdrop=0, gaps=0, win=5.0):
    """A receive window carrying CUMULATIVE transport/totals counters."""
    return {
        "recorded_at": "2026-10-03T00:00:00+00:00",
        "connection_id": conn,
        "protocol": "webrtc",
        "window_decoded_fps": 29.0,
        "window_seconds": win,
        "totals": {
            "decoded_gaps_over_200ms": gaps,
            "handoff_queue_dropped_frames": qdrop,
        },
        "inbound_rtp": [{"packets_lost_delta": 0}],
        "transport": [{"bytes_received": cum_bytes}],
    }


def _summary_from(records):
    s = pipeline.StreamSummary("mem")
    for rec in records:
        s.add(pipeline.RECEIVE, rec)
    return s


def test_ingest_bandwidth_uses_deltas_not_running_sum():
    # Constant 5 MB/5 s = 8 Mbps; cumulative counter climbs 5->10->15->20 MB.
    mb = 1_000_000
    recs = [_cumulative_window(n * 5 * mb) for n in range(1, 5)]
    m = _summary_from(recs).metrics()
    # Must stay ~8 Mbps regardless of how many windows elapsed (old code grew it).
    assert abs(m["bw_ingest_mbps"] - 8.0) < 0.01


def test_ingest_idle_windows_do_not_dilute():
    # One 5 MB burst then idle (cumulative stays 5 MB): rate reflects the burst
    # window only; trailing idle windows add neither bytes nor time.
    mb = 1_000_000
    recs = [_cumulative_window(5 * mb)] + [_cumulative_window(5 * mb) for _ in range(3)]
    s = _summary_from(recs)
    assert s.ingest_window_seconds == 5.0  # only the window with new bytes counts
    assert abs(s._bw_ingest() * 8 / 1e6 - 8.0) < 0.01


def test_ingest_handles_reconnect_counter_reset():
    mb = 1_000_000
    recs = [
        _cumulative_window(10 * mb, conn="c1"),            # +10 MB over 5 s
        _cumulative_window(3 * mb, conn="c2"),             # new conn: +3 MB over 5 s
    ]
    s = _summary_from(recs)
    # 13 MB over 10 s = 10.4 Mbps; no negative delta from the reset.
    assert abs(s._bw_ingest() * 8 / 1e6 - 10.4) < 0.01


def test_cumulative_qdrop_and_gaps_counted_once_not_summed():
    recs = [
        _cumulative_window(0, qdrop=0, gaps=0),
        _cumulative_window(0, qdrop=1, gaps=2),
        _cumulative_window(0, qdrop=1, gaps=2),  # unchanged
        _cumulative_window(0, qdrop=3, gaps=5),
    ]
    s = _summary_from(recs)
    # Final cumulative is 3 drops / 5 gaps -- not 0+1+1+3=5 / 0+2+2+5=9.
    assert s.qdrop == 3
    assert s.gaps200 == 5
