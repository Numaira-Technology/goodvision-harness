"""Summarize saved SAM logs without opening streams or contacting a service.

Usage: python tools/summarize_sam_timing.py lane-*.log worker.log pool.log
Also accepts the camera -> event -> records JSON produced by offline analysis.
Durations from different stages can overlap; this report never adds them or
subtracts clocks on different hosts. HTTP header wait is not GPU compute time.
Correlate by opaque request_id, and HTTP attempts by http_attempt_id when logged.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path

CLIENT_EVENTS = {
    "sam_client_http_timing",
    "sam_pipeline_prepare",
    "sam_coordinator_timing",
    "sam_client_append_retry",
    "sam_client_append_recovered",
    "sam_transport_recovery",
    "sam_coordinator_request_failure",
    "sam_session_open",
    "sam_session_cleanup",
}
LIFECYCLE_EVENTS = {
    "sam_transport_recovery",
    "sam_coordinator_request_failure",
    "sam_session_open",
    "sam_session_cleanup",
}
WORKER_EVENTS = {
    "Native session append timing",
    "Native session request completed",
    "Native session request failed",
    "Native session request rejected",
    "Native session request cancelled",
    "native session deadline exceeded",
    "Native session response timing",
    "SAM cloud proxy timing",
}
IDENTIFIER = re.compile(r"[A-Za-z0-9_.:-]{1,160}\Z")


def identifier(value):
    return value if isinstance(value, str) and IDENTIFIER.fullmatch(value) else None


def _finite_number(value):
    return type(value) in (int, float) and math.isfinite(value)


def _record(event, fields, camera=None):
    if event not in CLIENT_EVENTS | WORKER_EVENTS or not isinstance(fields, dict):
        return None
    # Retain only measurements and safe correlation; never arbitrary raw text.
    clean = {
        key: value
        for key, value in fields.items()
        if (key.endswith("_seconds") or key == "relative_spawn_s")
        and _finite_number(value)
        and (value >= 0 or key == "relative_spawn_s")
    }
    for key in (
        "request_id",
        "transport_request_id",
        "http_attempt_id",
        "stream_session_id",
        "session_ref",
        "camera_id",
        "outcome",
        "phase",
        "error_type",
        "protocol_error_reason",
    ):
        if identifier(fields.get(key)) is not None:
            clean[key] = fields[key]
    for key in (
        "input_frames",
        "output_frames",
        "http_status",
        "attempt",
        "next_attempt",
        "generation",
        "retired_generation",
    ):
        if type(fields.get(key)) is int:
            clean[key] = fields[key]
    if type(fields.get("recovery")) is bool:
        clean["recovery"] = fields["recovery"]
    if isinstance(fields.get("source_to_result"), list):
        clean["source_to_result"] = [
            row["seconds"]
            for row in fields["source_to_result"]
            if isinstance(row, dict)
            and _finite_number(row.get("seconds"))
            and row["seconds"] >= 0
        ]
    value = fields.get("recorded_at")
    try:
        if isinstance(value, str):
            datetime.fromisoformat(value)
            clean["recorded_at"] = value
    except ValueError:
        pass
    return {
        "event": event,
        "camera": identifier(str(camera)) if camera is not None else None,
        "fields": clean,
    }


def read_records(path):
    """Read JSON/JSONL or existing formatted stderr logs, ignoring other lines."""
    path = Path(path)
    camera_match = re.fullmatch(r"lane-(\d+)\.log", path.name)
    camera = str(int(camera_match[1])) if camera_match else None
    if path.suffix == ".json":
        value = json.loads(path.read_text())
        if isinstance(value, dict):
            for camera_key, events in value.items():
                if not isinstance(events, dict):
                    continue
                for event, rows in events.items():
                    if isinstance(rows, list):
                        for row in rows:
                            record = _record(event, row, camera_key)
                            if record:
                                yield record
            return
    for line in path.read_text(errors="replace").splitlines():
        match = re.search(r"\b(sam_[a-z_]+) (\{.*\})\s*$", line)
        if match:
            try:
                record = _record(match[1], json.loads(match[2]), camera)
            except ValueError:
                continue
            if record:
                yield record
            continue
        for event in WORKER_EVENTS:
            marker = event + " "
            if marker not in line:
                continue
            tail = line.split(marker, 1)[1]
            fields = {}
            for key, raw in re.findall(r"\b([a-z_]+)=([^\s]+)", tail):
                try:
                    fields[key] = json.loads(raw)
                except ValueError:
                    fields[key] = raw
            record = _record(event, fields, camera)
            if record:
                yield record
            break


def distribution(values):
    values = sorted(values)
    if not values:
        return None

    # Nearest-rank quantiles, explicit and deterministic for small samples.
    def quantile(fraction):
        return values[max(0, math.ceil(len(values) * fraction) - 1)]

    return {
        "count": len(values),
        "p50_seconds": quantile(0.5),
        "p95_seconds": quantile(0.95),
        "max_seconds": values[-1],
    }


def summarize(records, *, relative_start=None, relative_end=None, slowest=10):
    groups = defaultdict(list)
    lifecycle = []
    for record in records:
        fields = record["fields"]
        if record["event"] in LIFECYCLE_EVENTS:
            offset = fields.get("relative_spawn_s")
            if not _finite_number(offset) or (
                (relative_start is None or offset >= relative_start)
                and (relative_end is None or offset < relative_end)
            ):
                lifecycle.append(record)
        request_id = fields.get("request_id") or fields.get("transport_request_id")
        if request_id:
            groups[request_id].append(record)
    selected = {}
    for request_id, rows in groups.items():
        if relative_start is not None or relative_end is not None:
            if not any(
                row["event"] == "sam_client_http_timing"
                and _finite_number(row["fields"].get("relative_spawn_s"))
                and (
                    relative_start is None
                    or row["fields"]["relative_spawn_s"] >= relative_start
                )
                and (
                    relative_end is None
                    or row["fields"]["relative_spawn_s"] < relative_end
                )
                for row in rows
            ):
                continue
        selected[request_id] = rows
    stages = defaultdict(lambda: defaultdict(list))
    requests = []
    coverage = {
        "requests": len(selected),
        "with_worker_request": 0,
        "with_worker_append": 0,
        "with_proxy": 0,
    }
    for request_id, rows in selected.items():
        clients = [r for r in rows if r["event"] == "sam_client_http_timing"]
        camera = next((r["camera"] for r in rows if r["camera"] is not None), None)
        camera = camera or next(
            (
                r["fields"].get("camera_id")
                for r in rows
                if r["fields"].get("camera_id")
            ),
            "unknown",
        )
        events = {r["event"] for r in rows}
        coverage["with_worker_request"] += any(
            e.startswith("Native session request ") for e in events
        )
        coverage["with_worker_append"] += "Native session append timing" in events
        coverage["with_proxy"] += "SAM cloud proxy timing" in events
        for row in rows:
            for key, value in row["fields"].items():
                if key.endswith("_seconds") and _finite_number(value):
                    stages[camera][f"{row['event']}.{key}"].append(value)
            for value in row["fields"].get("source_to_result", []):
                stages[camera][
                    "sam_coordinator_timing.source_to_result_seconds"
                ].append(value)
        retries = [r["fields"] for r in rows if r["event"] == "sam_client_append_retry"]
        recovered = [
            r["fields"] for r in rows if r["event"] == "sam_client_append_recovered"
        ]
        retry_seconds = None
        if (
            retries
            and recovered
            and retries[0].get("recorded_at")
            and recovered[-1].get("recorded_at")
        ):
            retry_seconds = (
                datetime.fromisoformat(recovered[-1]["recorded_at"])
                - datetime.fromisoformat(retries[0]["recorded_at"])
            ).total_seconds()
        requests.append(
            {
                "request_id": request_id,
                "camera": camera,
                "http_attempts": len(clients),
                "client_total_seconds": sum(
                    r["fields"].get("total_seconds", 0) for r in clients
                ),
                "retry_to_recovered_seconds": retry_seconds,
                "attempts": [
                    {
                        key: value
                        for key, value in r["fields"].items()
                        if key
                        in {
                            "http_attempt_id",
                            "outcome",
                            "phase",
                            "error_type",
                            "protocol_error_reason",
                            "relative_spawn_s",
                            "total_seconds",
                            "http_until_headers_seconds",
                            "response_read_seconds",
                        }
                    }
                    for r in clients
                ],
                "remote_records": sorted(e for e in events if e in WORKER_EVENTS),
                "recovery_events": [
                    r
                    for r in rows
                    if r["event"]
                    in {
                        "sam_transport_recovery",
                        "sam_coordinator_request_failure",
                    }
                ],
            }
        )
    lifecycle_counts = defaultdict(int)
    lifecycle_stages = defaultdict(list)
    for row in lifecycle:
        lifecycle_counts[row["event"]] += 1
        for key, value in row["fields"].items():
            if key.endswith("_seconds") and _finite_number(value):
                lifecycle_stages[f"{row['event']}.{key}"].append(value)
    return {
        "scope": "offline saved logs; no inference or service calls",
        "interpretation": [
            "HTTP-until-headers includes client IO, remote queues/preparation/"
            "processing and reply headers; it is not GPU time.",
            "Stage distributions are separate and may overlap. Do not add them "
            "or subtract timestamps across hosts.",
            "Worker inference/native-advance durations are host wall time, "
            "not CUDA kernel time.",
            "Missing remote records cannot establish whether GPU compute, "
            "server waiting or transport dominates.",
            "Request IDs must be unique across the supplied runs; HTTP "
            "attempts are separately correlated where available.",
            "Lifecycle records without relative offsets cover all supplied logs; "
            "offsets, when available, obey the relative window filter.",
        ],
        "correlation_coverage": coverage,
        "per_camera_stages": {
            camera: {
                key: distribution(values) for key, values in sorted(metrics.items())
            }
            for camera, metrics in sorted(stages.items())
        },
        "slowest_requests": sorted(
            requests, key=lambda r: r["client_total_seconds"], reverse=True
        )[:slowest],
        "failed_or_retried_requests": [
            r
            for r in requests
            if r["http_attempts"] > 1
            or r["recovery_events"]
            or any(a.get("outcome") != "ok" for a in r["attempts"])
        ],
        "recovery_and_lifecycle": {
            "event_counts": dict(sorted(lifecycle_counts.items())),
            "stage_distributions": {
                key: distribution(values)
                for key, values in sorted(lifecycle_stages.items())
            },
            "events": lifecycle,
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logs", nargs="+", type=Path)
    parser.add_argument("--relative-start", type=float)
    parser.add_argument("--relative-end", type=float)
    parser.add_argument("--slowest", type=int, default=10)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = summarize(
        (row for path in args.logs for row in read_records(path)),
        relative_start=args.relative_start,
        relative_end=args.relative_end,
        slowest=max(0, args.slowest),
    )
    body = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.write_text(body)
    else:
        print(body, end="")


if __name__ == "__main__":
    main()
