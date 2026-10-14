"""Paced fixed-window workload using production shared admission.

This is a twelve-source VLM branch workload, not camera decoding or tracking.
Only one unsent pending window per source is retained; every skipped plan is
recorded. Workers never silently accumulate in an unbounded executor queue.
"""

from __future__ import annotations

import time
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from threading import Event, Lock, local

from goodvision_harness.vlm_admission import (
    AdmissionError,
    AdmissionSettings,
    SharedAdmission,
)


class StopControl:
    def __init__(self, *, continue_on_timeout=False):
        self.event, self.lock, self.reason = Event(), Lock(), None
        self.failures = Counter()
        self.transport_failures = Counter()
        self.continue_on_timeout = continue_on_timeout

    def stop(self, reason):
        with self.lock:
            self.reason = self.reason or reason
            self.event.set()

    def observe(self, result):
        if result["http_status"] in {401, 402, 403, 429}:
            self.stop(f"http_{result['http_status']}")
        elif result["error_code"] in {
            "interrupted",
            "client_error",
        }:
            self.stop(result["error_code"])
        elif result.get("attempted"):
            with self.lock:
                source = result["stream_id"]
                transport = result.get("http_timing", {})
                setup_timeout = (
                    result["error_code"] == "timeout"
                    and transport.get("transport_trace_observed")
                    and not transport.get("request_body_completed")
                    and transport.get("http_status") is None
                )
                self.transport_failures[source] = (
                    self.transport_failures[source] + 1 if setup_timeout else 0
                )
                if self.transport_failures[source] >= 3:
                    self.reason = (
                        self.reason
                        or "three_consecutive_transport_setup_timeouts_on_one_stream"
                    )
                    self.event.set()
                    return
                # Explicit bounded measurement mode: retain timeouts as failed
                # outcomes, but do not terminate the remaining fixed plan.
                if self.continue_on_timeout and result["error_code"] == "timeout":
                    self.failures[source] = 0
                    return
                self.failures[source] = (
                    0 if result["state"] == "complete" else self.failures[source] + 1
                )
                failed = self.failures[source] >= 3
            if failed:
                self.stop("three_consecutive_unsuccessful_requests_on_one_stream")


def skipped(item, code, *, due, offered, ended):
    return {
        **item,
        "state": "skipped",
        "error_code": code,
        "http_status": None,
        "attempted": False,
        "inspect_seconds": None,
        "judgments": {},
        "timeline": {
            "planned": due,
            "offered": offered,
            "worker_started": None,
            "admitted": None,
            "inspect_started": None,
            "finished": ended,
        },
        "response_diagnostics": {},
    }


def publish(result, on_result):
    now = time.monotonic()
    t = result["timeline"]
    t["published"] = now
    offered, worker, started = t["offered"], t["worker_started"], t["inspect_started"]
    admitted = t["admitted"]
    result["latency"] = {
        "input_release_lag_seconds": max(0, offered - t["planned"]),
        "dispatch_wait_seconds": max(0, worker - offered) if worker else None,
        "admission_wait_seconds": max(0, (admitted or t["finished"]) - worker)
        if worker
        else None,
        "queue_wait_seconds": max(0, admitted - offered) if started else None,
        "queue_residence_seconds": max(0, (admitted or t["finished"]) - offered),
        "inspector_setup_seconds": max(0, started - admitted) if started else None,
        "inspect_seconds": result["inspect_seconds"],
        "result_finalize_seconds": max(
            0, t["finished"] - started - result["inspect_seconds"]
        )
        if started and result["inspect_seconds"] is not None
        else None,
        "publication_wait_seconds": max(0, now - t["finished"]),
        "input_to_publish_seconds": max(0, now - offered),
        "scheduled_to_publish_seconds": max(0, now - t["planned"]),
        "evidence_to_publish_seconds": max(0, now - t["planned"])
        + result.get("evidence_age_at_release_seconds", 0),
    }
    on_result(result)


def run_stage(
    planned, args, database, config_for, job_for, factory, evaluate, on_result
):
    scale = args.simulation_time_scale if args.simulate else 1.0
    gate = SharedAdmission(
        AdmissionSettings(
            database=str(database.resolve()),
            global_limit=args.global_limit,
            per_stream_limit=args.per_stream_limit,
            requests_per_second=args.request_rate / scale,
            queue_timeout_seconds=args.queue_timeout * scale,
        )
    )
    control = StopControl(continue_on_timeout=args.continue_on_timeout)
    thread = local()
    clients, clients_lock = [], Lock()
    # Build immutable evidence before the source clock starts. References never
    # enter this path. Materializing jobs does not dispatch requests.
    prepared = {
        p["request_id"]: (config_for(p), job_for(p))
        for p in planned
        if p.get("sampling_selected", True)
    }
    pool = ThreadPoolExecutor(
        max_workers=args.global_limit, thread_name_prefix="vlm-quality"
    )
    active, pending, inflight = {}, {}, Counter()
    position = 0
    started_at = time.monotonic()
    peak_pending, peak_workers = 0, 0

    def worker(item, due, offered):
        begun = time.monotonic()
        cfg, job = prepared[item["request_id"]]
        admitted, request_started = None, None
        try:
            with gate.slot(
                cfg.stream_session_id,
                stop=control.event,
                source_deadline=offered + 3 * scale,
                lane=cfg.output_mode,
            ):
                admitted = time.monotonic()
                if control.event.is_set():
                    raise AdmissionError("admission_cancelled")
                if not hasattr(thread, "inspector"):
                    thread.inspector = factory(cfg)
                    with clients_lock:
                        clients.append(thread.inspector)
                if control.event.is_set():
                    raise AdmissionError("admission_cancelled")
                request_started = time.monotonic()
                result = evaluate(item, job, cfg, thread.inspector)
                result["attempted"] = True
                # Signal fatal errors before the main loop can grant more work.
                control.observe(result)
        except AdmissionError as exc:
            result = skipped(
                item, exc.code, due=due, offered=offered, ended=time.monotonic()
            )
        except BaseException as exc:
            code = (
                "interrupted" if isinstance(exc, KeyboardInterrupt) else "client_error"
            )
            control.stop(code)
            result = skipped(
                item, code, due=due, offered=offered, ended=time.monotonic()
            )
        result["timeline"] = {
            "planned": due,
            "offered": offered,
            "worker_started": begun,
            "admitted": admitted,
            "inspect_started": request_started,
            "finished": time.monotonic(),
        }
        return result

    def collect():
        for future, (item, _due, _offered) in list(active.items()):
            if not future.done():
                continue
            del active[future]
            inflight[item["stream_id"]] -= 1
            publish(future.result(), on_result)

    try:
        while position < len(planned) or active or pending:
            collect()
            now = time.monotonic()
            if control.event.is_set():
                for item, due, offered in pending.values():
                    publish(
                        skipped(
                            item, "stage_stopped", due=due, offered=offered, ended=now
                        ),
                        on_result,
                    )
                pending.clear()
                break
            while position < len(planned):
                now = time.monotonic()
                item = planned[position]
                due = started_at + item["release_offset_seconds"] * scale
                if due > now:
                    break
                position += 1
                stream = item["stream_id"]
                if not item.get("sampling_selected", True):
                    result = skipped(
                        item, "sampling_interval", due=due, offered=now, ended=now
                    )
                    result["state"] = "not_sampled"
                    publish(result, on_result)
                    continue
                # A stall must not replay stale arrivals in a burst and call
                # the resulting low offered load a valid twelve-stream test.
                if now - due > args.input_lag_tolerance * scale:
                    publish(
                        skipped(item, "source_missed", due=due, offered=now, ended=now),
                        on_result,
                    )
                    continue
                if stream in pending:
                    old, old_due, old_offered = pending[stream]
                    publish(
                        skipped(
                            old,
                            "pending_superseded",
                            due=old_due,
                            offered=old_offered,
                            ended=now,
                        ),
                        on_result,
                    )
                pending[stream] = item, due, now
            peak_pending = max(peak_pending, len(pending))
            for stream, (item, due, offered) in sorted(
                list(pending.items()),
                key=lambda entry: entry[1][1],
            ):
                if now - offered >= args.queue_timeout * scale:
                    del pending[stream]
                    publish(
                        skipped(
                            item,
                            "dispatch_timeout",
                            due=due,
                            offered=offered,
                            ended=now,
                        ),
                        on_result,
                    )
                elif (
                    len(active) < args.global_limit
                    and inflight[stream] < args.per_stream_limit
                ):
                    del pending[stream]
                    if control.event.is_set():
                        publish(
                            skipped(
                                item,
                                "stage_stopped",
                                due=due,
                                offered=offered,
                                ended=now,
                            ),
                            on_result,
                        )
                        break
                    future = pool.submit(worker, item, due, offered)
                    active[future] = item, due, offered
                    inflight[stream] += 1
            peak_workers = max(peak_workers, len(active))
            time.sleep(0.002)
    except KeyboardInterrupt:
        control.stop("interrupted")
    except Exception:
        control.stop("runner_error")
    finally:
        # Waiting admissions see cancellation. Already sent HTTP calls retain
        # their results and bounded transport deadline; no fake cancellation.
        if control.reason:
            control.event.set()
        # Keep publishing individual completions while other HTTP requests
        # drain. Waiting for the entire pool first adds the slowest request's
        # remaining time to every already-completed result's publish latency.
        try:
            while active:
                wait(active, return_when=FIRST_COMPLETED)
                collect()
        finally:
            pool.shutdown(wait=True)
        for item, due, offered in pending.values():
            publish(
                skipped(
                    item,
                    "stage_stopped",
                    due=due,
                    offered=offered,
                    ended=time.monotonic(),
                ),
                on_result,
            )
        for client in clients:
            try:
                client.close()
            except Exception:
                control.stop("client_close_error")
    return {
        "repeat": planned[0]["repeat"],
        "mode": planned[0]["mode"],
        "stop_reason": control.reason,
        "peak_pending_windows": peak_pending,
        "peak_worker_requests": peak_workers,
        "admission": gate.snapshot(),
        "wall_seconds": time.monotonic() - started_at,
    }
