"""Two production coordinators and the real FIFO gate, with synthetic model time.

Run explicitly; no CUDA, real detections or contour-quality claims are made.
Each case uses the same wall-clock source schedule and 4-second result deadline.
"""

import argparse
import asyncio
import json
import math
import tempfile
import time
from pathlib import Path

import httpx
from test_perception_client import RouteServer

from goodvision_harness.perception import PerceptionSessionHTTPClient
from goodvision_harness.sam_tracking_pipeline import (
    PipelinedVideoSessionTrackingCoordinator,
)
from workers.sam_scene.multistream import FairGPUAdmission


class TimedServer(RouteServer):
    def __init__(self, gate, origin):
        super().__init__("sam_live")
        self.gate, self.origin = gate, origin
        self.timings = []

    async def __call__(self, request):
        if request.method != "POST" or not request.url.path.endswith("/append"):
            return await super().__call__(request)
        count = len(json.loads(request.content)["frames"])
        started = time.monotonic()
        granted = await self.gate.acquire_async(started + 30)
        assert granted
        acquired = time.monotonic()
        try:
            elapsed = acquired - self.origin
            cost = 0.60 if 1 <= elapsed < 5 else 0.03
            await asyncio.sleep(cost * count)
            result = await super().__call__(request)
            self.timings.append(
                {
                    "frames": count,
                    "wait": acquired - started,
                    "execution": time.monotonic() - acquired,
                }
            )
            return result
        finally:
            self.gate.release()


def quantile(values, fraction):
    return (
        sorted(values)[max(0, math.ceil(len(values) * fraction) - 1)]
        if values
        else None
    )


def run_case(directory, policy, batch_size, *, pending_frames=4, max_age=2.0):
    origin = time.monotonic()
    gate = FairGPUAdmission(4)
    sources, ages, completed_at = [{}, {}], [[], []], [[], []]
    servers, coordinators, peak = [], [], [0, 0]
    for lane in range(2):
        server = TimedServer(gate, origin)
        servers.append(server)
        client = PerceptionSessionHTTPClient(
            "http://worker.test",
            method="sam_live",
            transport=httpx.MockTransport(server),
            timeout_seconds=30,
        )

        def commit(batch, index=lane):
            now = time.monotonic()
            for frame in batch.frames:
                ages[index].append(now - sources[index][frame.frame_sequence])
                completed_at[index].append(now - origin)

        coordinators.append(
            PipelinedVideoSessionTrackingCoordinator(
                client,
                max_fps=4,
                batch_frames=1,
                batch_seconds=0.1,
                pending_capacity=4,
                on_batch=commit,
                spool_directory=directory / str(lane),
                queue_policy=policy,
                realtime_pending_frames=pending_frames,
                realtime_append_frames=batch_size,
                realtime_max_age_seconds=max_age,
                result_deadline_seconds=4,
            )
        )
    try:
        # 7 seconds, each source independently supplies 4 FPS. This is an
        # artificial overload workload, not the deployment's 1 FPS profile.
        for sequence in range(28):
            due = origin + sequence / 4
            time.sleep(max(0, due - time.monotonic()))
            for lane, coordinator in enumerate(coordinators):
                sources[lane][sequence] = time.monotonic()
                assert coordinator.submit(
                    b"jpeg",
                    stream_session_id=f"camera-{lane}",
                    frame_sequence=sequence,
                    timestamp_seconds=sequence / 4,
                )
                peak[lane] = max(peak[lane], coordinator.stats()["pending_frames"])
        time.sleep(max(0, origin + 7 - time.monotonic()))
        live = [coordinator.stats() for coordinator in coordinators]
        for coordinator in coordinators:
            coordinator.flush()
        assert all(c.wait_until_idle(30) for c in coordinators)
        result = {"policy": policy, "batch_limit": batch_size, "lanes": []}
        for lane, coordinator in enumerate(coordinators):
            stats = coordinator.stats()
            result["lanes"].append(
                {
                    "source_frames": 28,
                    "processed_during_capture": live[lane]["processed_frames"],
                    "processed_total": stats["processed_frames"],
                    "durably_admitted_frames": stats["session_admitted_frames"],
                    "skipped": stats["skipped_by_reason"],
                    "max_sampled_pending_frames": peak[lane],
                    "result_seconds_p50": quantile(ages[lane], 0.5),
                    "result_seconds_p95": quantile(ages[lane], 0.95),
                    "result_seconds_max": max(ages[lane], default=0),
                    "results_within_4_seconds": stats["results_within_deadline"],
                    "admitted_deadline_coverage": stats["admitted_deadline_coverage"],
                    "gpu_gate_wait_max": max(t["wait"] for t in servers[lane].timings),
                    "request_hold_max": max(
                        t["execution"] for t in servers[lane].timings
                    ),
                    "longest_completion_gap": max(
                        (
                            b - a
                            for a, b in zip(
                                [0, *completed_at[lane][:-1]],
                                completed_at[lane],
                                strict=True,
                            )
                        ),
                        default=0,
                    ),
                }
            )
            assert stats["submitted"] == (
                stats["processed_frames"] + sum(stats["skipped_by_reason"].values())
            )
            if policy == "latest_only":
                # Selected sources can now expire after preparation, before send.
                assert stats["session_admitted_frames"] == (
                    stats["processed_frames"]
                    + len(coordinator._spool.skipped_records())
                )
            assert servers[lane].creates == [f"camera-{lane}"]
        return result
    finally:
        for coordinator in coordinators:
            coordinator.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pending-frames", type=int, default=4)
    parser.add_argument("--max-age-seconds", type=float, default=2.0)
    parser.add_argument("--latest-only-comparison", action="store_true",
                        help="Compare FIFO with the fixed latest candidate policy")
    args = parser.parse_args()
    report = {
        "kind": "synthetic_protocol_experiment_not_gpu_benchmark",
        "playback_buffer_unchanged_seconds": 4,
        "source_fps_per_camera": 4,
        "capture_seconds": 7,
        "simulated_native_seconds_per_frame": {"normal": 0.03, "overload": 0.60},
        "overload_window_seconds": [1, 5],
        "realtime_pending_frames": args.pending_frames,
        "realtime_max_age_seconds": args.max_age_seconds,
        "note": "Real coordinator/HTTP/FIFO gate; fake image and model output. "
        "No accuracy or actual rendered-contour coverage is measured.",
        "cases": [],
    }
    cases = [
        ("durable_ordered", 4),
        ("realtime_latest", 1),
        ("realtime_latest", 2),
        ("realtime_latest", 4),
    ]
    if args.latest_only_comparison:
        cases = [("durable_ordered", 4), ("latest_only", 1)]
    for policy, limit in cases:
        with tempfile.TemporaryDirectory(prefix="sam-realtime-") as directory:
            case = run_case(
                Path(directory),
                policy,
                limit,
                pending_frames=args.pending_frames,
                max_age=args.max_age_seconds,
            )
        report["cases"].append(case)
        print(json.dumps(case), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
