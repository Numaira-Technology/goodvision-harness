"""Shared original-speed input, bounded inference and offline clip scoring."""

import json
import math
import signal
import threading
import time
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

PROTOCOL = "streaming-v3"
BASE_EVENTS = ["fight", "smoking", "fall", "weapon"]
SIX_EVENTS = [*BASE_EVENTS, "bag_knife", "bag_alcohol_bottle"]


def _distribution(values):
    values = sorted(values)
    if not values:
        return {"n": 0, "p50": None, "p95": None, "max": None}
    return {
        "n": len(values),
        "p50": values[math.ceil(len(values) * 0.5) - 1],
        "p95": values[math.ceil(len(values) * 0.95) - 1],
        "max": values[-1],
    }


def _coverage(states, slots, start, end):
    changes = []
    for state in states:
        left, right = (
            state.get("input_start_monotonic"),
            state.get("input_end_monotonic"),
        )
        if left is not None and right is not None:
            left, right = max(start, left), min(end, right)
            if right > left:
                changes.extend([(left, 1), (right, -1)])
    active, previous, durations = 0, start, defaultdict(float)
    for stamp, change in sorted(changes):
        durations[active] += stamp - previous
        active += change
        previous = stamp
    durations[active] += end - previous
    span = end - start
    return {
        "measurement_seconds": span,
        "seconds_by_active_inputs": dict(sorted(durations.items())),
        "full_slots_seconds": durations[slots],
        "full_slots_fraction": durations[slots] / span if span > 0 else None,
    }


def run_streams(
    samples,
    media_index,
    questions,
    backend,
    output,
    *,
    slots=12,
    workers=12,
    interval=1.0,
    window_frames=2,
    frame_width=720,
    jpeg_quality=90,
    queue_age=2.0,
):
    """Replace input at EOF immediately; previous-source requests drain independently.

    Every source has at most one outstanding window. Global outstanding work is
    bounded by workers + slots. Busy/late/expired windows stay missing evidence;
    they neither stop a source nor become negative answers. Filler sources keep
    slots occupied until every primary input reaches EOF, and never enter scoring.
    """
    import cv2

    from goodvision_harness.video_to_stream import encode_jpeg, probe_video

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    record_path = output.with_name(output.stem + "-windows.jsonl")
    records = record_path.open("x", encoding="utf-8")
    media = list(dict.fromkeys(sample["filename"] for sample in samples))
    todo = deque(media)
    lock, io_lock, stop = threading.RLock(), threading.Lock(), threading.Event()
    available = threading.Condition(lock)
    admission = threading.BoundedSemaphore(workers + slots)
    states, reusable, shared_errors = [], [], []
    unusable = set()
    latencies, primary_latencies = [], []
    primary_remaining, serial = len(media), 0
    started = time.monotonic()
    session_prefix = f"eval-{time.time_ns():x}"
    last_primary_input = [None]
    interrupted = False

    def record(row):
        try:
            with io_lock:
                records.write(json.dumps(row, allow_nan=False) + "\n")
                records.flush()
        except Exception as error:
            with lock:
                shared_errors.append("result_storage_failed: " + str(error))
            stop.set()
            raise

    def choose(slot):
        nonlocal serial
        with available:
            while not todo and not reusable and primary_remaining and not stop.is_set():
                available.wait(0.02)
            if stop.is_set() or not primary_remaining:
                return None
            primary = bool(todo)
            serial += 1
            filename = (
                todo.popleft() if primary else reusable[(serial - 1) % len(reusable)]
            )
            state = {
                "source_id": f"{session_prefix}-{serial}",
                "filename": filename,
                "slot": slot,
                "quality_reference": primary,
                "input_start_monotonic": None,
                "input_end_monotonic": None,
                "drained_monotonic": None,
                "eof_seen": False,
                "source_error": None,
                "outstanding": False,
                "raw_frames": 0,
                "late_raw_frames": 0,
                "sampled_frames": 0,
                "planned_windows": 0,
                "unformed_windows": 0,
                "opportunities": 0,
                "submitted": 0,
                "completed": 0,
                "busy_missed": 0,
                "late_missed": 0,
                "queue_expired": 0,
                "stopped_not_sent": 0,
                "input_failures": 0,
                "evidence": {
                    label: {
                        "yes": 0,
                        "no": 0,
                        "unknown": 0,
                        "request_failed": 0,
                        "request_unknown": 0,
                        "response_mismatch": 0,
                        "not_sent": 0,
                        "format_invalid": 0,
                        "first_positive": None,
                    }
                    for label in questions
                },
            }
            states.append(state)
            return state

    def infer(state, window):
        wait = getattr(backend, "wait_cooldown", None)
        if wait is not None:
            wait(stop)
        call_started = time.monotonic()
        if stop.is_set() or call_started - window["queued_monotonic"] > queue_age:
            with lock:
                state["stopped_not_sent" if stop.is_set() else "queue_expired"] += 1
            record(
                {key: value for key, value in window.items() if key != "images"}
                | {
                    "status": "stopped_before_request"
                    if stop.is_set()
                    else "expired_before_request",
                    "request_started_monotonic": None,
                    "request_state": "not_sent",
                    "predictions": None,
                }
            )
            return
        window["request_deadline_monotonic"] = window["queued_monotonic"] + queue_age
        try:
            result = backend.predict(window, stop=stop)
        except Exception as error:
            from goodvision_harness.videojev_service import _shared_cuda_failure

            result = {
                "predictions": {
                    label: {
                        "answer": None,
                        "request_state": "request_failed",
                        "format_state": "unavailable",
                        "error": f"{type(error).__name__}: {error}",
                    }
                    for label in questions
                },
                "raw": None,
                "shared_error": "cuda_shared_failure: " + str(error)
                if _shared_cuda_failure(error)
                else None,
            }
        ready = time.monotonic()
        elapsed = ready - window["last_frame_ingress_monotonic"]
        row = (
            {key: value for key, value in window.items() if key != "images"}
            | result
            | {
                "status": "returned",
                "request_started_monotonic": call_started,
                "result_ready_monotonic": ready,
                "input_to_result_seconds": elapsed,
                "queue_seconds": call_started - window["queued_monotonic"],
            }
        )
        record(row)
        with lock:
            if result.get("expired_before_request"):
                state["queue_expired"] += 1
            if result.get("shared_error"):
                shared_errors.append(result["shared_error"])
                stop.set()
            state["completed"] += 1
            if any(
                value.get("request_state") == "completed"
                and value.get("format_state") == "valid"
                for value in result["predictions"].values()
            ):
                latencies.append(elapsed)
                if state["quality_reference"]:
                    primary_latencies.append(elapsed)
            for label, evidence in state["evidence"].items():
                value = result["predictions"].get(label, {})
                if value.get("request_state") == "outcome_unknown":
                    evidence["request_unknown"] += 1
                elif value.get("request_state") == "response_mismatch":
                    evidence["response_mismatch"] += 1
                elif value.get("request_state") == "not_sent":
                    evidence["not_sent"] += 1
                elif value.get("request_state") != "completed":
                    evidence["request_failed"] += 1
                elif value.get("format_state") != "valid":
                    evidence["format_invalid"] += 1
                elif value.get("answer") is True:
                    evidence["yes"] += 1
                    if evidence["first_positive"] is None:
                        evidence["first_positive"] = {
                            "source_times": window["source_times"],
                            "result_ready_monotonic": ready,
                            "last_frame_ingress_monotonic": window[
                                "last_frame_ingress_monotonic"
                            ],
                            "input_to_result_seconds": elapsed,
                        }
                elif value.get("answer") is False:
                    evidence["no"] += 1
                else:
                    evidence["unknown"] += 1

    def done(state, future):
        try:
            future.result()
        except Exception as error:
            with lock:
                shared_errors.append("result_worker_failed: " + str(error))
            stop.set()
        finally:
            with lock:
                state["outstanding"] = False
                if state["input_end_monotonic"] is not None:
                    state["drained_monotonic"] = time.monotonic()
            admission.release()

    def reader(slot, pool):
        nonlocal primary_remaining
        while not stop.is_set():
            state = choose(slot)
            if state is None:
                return
            capture = None
            last_timestamp = None
            sample_interval = interval
            try:
                path = media_index.get(state["filename"])
                if path is None:
                    raise FileNotFoundError(
                        "media unavailable or ambiguous: " + state["filename"]
                    )
                info = probe_video(path)
                if not math.isfinite(info.source_fps) or info.source_fps <= 0:
                    raise ValueError("invalid source frame rate")
                # A faster requested grid cannot create extra original frames.
                sample_interval = max(interval, 1 / info.source_fps)
                state["sample_interval_seconds"] = sample_interval
                capture = cv2.VideoCapture(str(path))
                if not capture.isOpened():
                    raise ValueError("source decoder could not open media")
                origin, raw_index, next_sample = time.monotonic(), 0, 0.0
                pts_origin, last_timestamp = None, None
                window = deque(maxlen=window_frames)
                while not stop.is_set():
                    with lock:
                        if not state["quality_reference"] and not primary_remaining:
                            break
                    ok, bgr = capture.read()
                    if not ok:
                        state["eof_seen"] = raw_index >= info.frame_count
                        if state["eof_seen"] and last_timestamp is not None:
                            # The final frame occupies one source frame period.
                            # Without this, one-frame fillers restart in a tight
                            # loop and every source ends one frame period early.
                            end_target = origin + last_timestamp + 1 / info.source_fps
                            while not stop.is_set() and time.monotonic() < end_target:
                                stop.wait(
                                    min(0.02, max(0, end_target - time.monotonic()))
                                )
                        if not state["eof_seen"]:
                            state["source_error"] = (
                                "decoder ended before reported original frame count"
                            )
                        break
                    decoder_msec = capture.get(cv2.CAP_PROP_POS_MSEC)
                    valid_pts = math.isfinite(decoder_msec) and decoder_msec >= 0
                    if valid_pts and pts_origin is None:
                        pts_origin = decoder_msec / 1000
                    pts = decoder_msec / 1000 - pts_origin if valid_pts else None
                    nominal = raw_index / info.source_fps
                    use_pts = pts is not None and (
                        last_timestamp is None or pts > last_timestamp
                    )
                    timestamp = (
                        pts
                        if use_pts
                        else max(
                            nominal,
                            0.0
                            if last_timestamp is None
                            else last_timestamp + 1 / info.source_fps,
                        )
                    )
                    last_timestamp = timestamp
                    target = origin + timestamp
                    while not stop.is_set() and time.monotonic() < target:
                        stop.wait(min(0.02, max(0, target - time.monotonic())))
                    if stop.is_set():
                        break
                    ingress = time.monotonic()
                    decoder_msec = decoder_msec if valid_pts else None
                    if state["input_start_monotonic"] is None:
                        state["input_start_monotonic"] = ingress
                        with available:
                            if (
                                state["filename"] not in reusable
                                and state["filename"] not in unusable
                            ):
                                reusable.append(state["filename"])
                            available.notify_all()
                    state["raw_frames"] += 1
                    index, raw_index = raw_index, raw_index + 1
                    if ingress - target > 0.25:
                        state["late_raw_frames"] += 1
                        continue
                    if timestamp + 1e-8 < next_sample:
                        continue
                    steps = (
                        math.floor((timestamp - next_sample) / sample_interval + 1e-8)
                        + 1
                    )
                    state["late_missed"] += steps - 1
                    next_sample += steps * sample_interval
                    if steps > 1:
                        window.clear()
                    try:
                        height = max(
                            2, round(bgr.shape[0] * frame_width / bgr.shape[1] / 2) * 2
                        )
                        resized = cv2.resize(
                            bgr, (frame_width, height), interpolation=cv2.INTER_AREA
                        )
                        jpeg, _width, _height = encode_jpeg(
                            resized, max_image_side=None, quality=jpeg_quality
                        )
                    except Exception as error:
                        state["input_failures"] += 1
                        window.clear()
                        record(
                            {
                                "source_id": state["source_id"],
                                "filename": state["filename"],
                                "slot": slot,
                                "quality_reference": state["quality_reference"],
                                "source_times": [timestamp],
                                "status": "input_failed",
                                "error": str(error),
                            }
                        )
                        continue
                    state["sampled_frames"] += 1
                    window.append(
                        (
                            jpeg,
                            timestamp,
                            ingress,
                            index,
                            decoder_msec,
                            "decoder_pts" if use_pts else "frame_rate_fallback",
                        )
                    )
                    if len(window) != window_frames:
                        continue
                    state["opportunities"] += 1
                    with lock:
                        if state["outstanding"] or not admission.acquire(
                            blocking=False
                        ):
                            state["busy_missed"] += 1
                            record(
                                {
                                    "source_id": state["source_id"],
                                    "filename": state["filename"],
                                    "slot": slot,
                                    "quality_reference": state["quality_reference"],
                                    "source_times": [item[1] for item in window],
                                    "frame_indices": [item[3] for item in window],
                                    "status": "busy_not_sent",
                                    "predictions": None,
                                }
                            )
                            continue
                        state["outstanding"] = True
                        state["submitted"] += 1
                    payload = {
                        "source_id": state["source_id"],
                        "filename": state["filename"],
                        "slot": slot,
                        "quality_reference": state["quality_reference"],
                        "request_id": (
                            f"{state['source_id']}-window-{state['submitted']}"
                        ),
                        "sampling_fps": 1 / sample_interval,
                        "source_times": [item[1] for item in window],
                        "frame_indices": [item[3] for item in window],
                        "decoder_timestamp_msec": [item[4] for item in window],
                        "source_time_basis": [item[5] for item in window],
                        "frame_ingress_monotonic": [item[2] for item in window],
                        "last_frame_ingress_monotonic": window[-1][2],
                        "queued_monotonic": time.monotonic(),
                        "images": [item[0] for item in window],
                    }
                    future = pool.submit(infer, state, payload)
                    future.add_done_callback(
                        lambda completed, state=state: done(state, completed)
                    )
            except Exception as error:
                state["source_error"] = f"{type(error).__name__}: {error}"
            finally:
                if capture is not None:
                    capture.release()
                if state["source_error"] is not None:
                    with available:
                        unusable.add(state["filename"])
                        if state["filename"] in reusable:
                            reusable.remove(state["filename"])
                        available.notify_all()
                    state["input_failures"] += 1
                    try:
                        record(
                            {
                                "source_id": state["source_id"],
                                "filename": state["filename"],
                                "slot": slot,
                                "quality_reference": state["quality_reference"],
                                "status": "input_failed",
                                "error": state["source_error"],
                            }
                        )
                    except Exception:
                        pass  # record() already stops unsafe shared storage.
                if last_timestamp is not None:
                    state["planned_windows"] = max(
                        0,
                        math.floor(last_timestamp / sample_interval + 1e-8)
                        - window_frames
                        + 2,
                    )
                    state["unformed_windows"] = max(
                        0, state["planned_windows"] - state["opportunities"]
                    )
                with available:
                    state["input_end_monotonic"] = time.monotonic()
                    if not state["outstanding"]:
                        state["drained_monotonic"] = state["input_end_monotonic"]
                    if state["quality_reference"]:
                        primary_remaining -= 1
                        if not primary_remaining:
                            last_primary_input[0] = state["input_end_monotonic"]
                    available.notify_all()
                # The next source starts without waiting for this source's future.

    def halt(_signum, _frame):
        nonlocal interrupted
        interrupted = True
        stop.set()

    handlers = {}
    if threading.current_thread() is threading.main_thread():
        for signum in (signal.SIGINT, signal.SIGTERM):
            handlers[signum] = signal.signal(signum, halt)
    try:
        with ThreadPoolExecutor(
            max_workers=workers, thread_name_prefix="event-infer"
        ) as inference:
            with ThreadPoolExecutor(
                max_workers=slots, thread_name_prefix="event-input"
            ) as feeders:
                readers = [
                    feeders.submit(reader, slot, inference)
                    for slot in range(1, slots + 1)
                ]
                try:
                    for future in readers:
                        future.result()
                except KeyboardInterrupt:
                    interrupted = True
                    stop.set()
                except Exception as error:
                    shared_errors.append("input_worker_failed: " + str(error))
                    stop.set()
            # Accepted prior-source work finishes in this pool after input replacement.
    finally:
        stop.set()
        records.close()
        for signum, previous in handlers.items():
            signal.signal(signum, previous)
    finished = time.monotonic()
    primary = {
        state["filename"]: state for state in states if state["quality_reference"]
    }
    per_question, counters = (
        [],
        defaultdict(
            lambda: {
                "TP": 0,
                "TN": 0,
                "FP": 0,
                "FN": 0,
                "unknown": 0,
                "unscored_reference": 0,
                "pending": 0,
            }
        ),
    )
    for sample in samples:
        state = primary.get(sample["filename"])
        evidence = state["evidence"][sample["event"]] if state is not None else None
        answer = None
        incomplete = True
        if evidence is not None:
            incomplete = (
                not state["eof_seen"]
                or state["source_error"] is not None
                or state["busy_missed"]
                or state["late_missed"]
                or state["queue_expired"]
                or state["stopped_not_sent"]
                or state["input_failures"]
                or evidence["request_failed"]
                or evidence["request_unknown"]
                or evidence["not_sent"]
                or evidence["response_mismatch"]
                or state["unformed_windows"]
                or evidence["format_invalid"]
                or evidence["unknown"]
            )
            if evidence["yes"]:
                answer = True
            elif evidence["no"]:
                answer = False
        prediction = "unknown" if answer is None else "yes" if answer else "no"
        reference = sample["proposed_answer"]
        count = counters[sample["event"]]
        if state is None:
            count["pending"] += 1
            prediction, correct = None, None
        elif reference not in ("yes", "no"):
            count["unscored_reference"] += 1
            correct = None
        elif answer is None:
            count["unknown"] += 1
            correct = False
        else:
            count[
                ("TP" if answer else "FN")
                if reference == "yes"
                else ("FP" if answer else "TN")
            ] += 1
            correct = prediction == reference
        per_question.append(
            {
                "number": sample["number"],
                "filename": sample["filename"],
                "event": sample["event"],
                "gt": reference,
                "pred": prediction,
                "execution_state": "pending" if state is None else "started",
                "correct": correct,
                "evidence_complete": not bool(incomplete),
                "evidence": evidence,
            }
        )
    for count in counters.values():
        count["correct"] = count["TP"] + count["TN"]
        count["total"] = sum(count[key] for key in ("TP", "TN", "FP", "FN", "unknown"))
        count["scored_boolean"] = count["total"] - count["unknown"]
        count["planned"] = (
            count["total"] + count["unscored_reference"] + count["pending"]
        )
    window_summary = {
        key: sum(state[key] for state in primary.values())
        for key in (
            "planned_windows",
            "unformed_windows",
            "opportunities",
            "submitted",
            "completed",
            "busy_missed",
            "late_missed",
            "queue_expired",
            "stopped_not_sent",
            "input_failures",
        )
    }
    event_evidence = {
        label: {
            key: sum(state["evidence"][label][key] for state in primary.values())
            for key in (
                "yes",
                "no",
                "unknown",
                "request_failed",
                "request_unknown",
                "response_mismatch",
                "not_sent",
                "format_invalid",
            )
        }
        for label in questions
    }
    boundary = {
        "wall_time_boundary": (
            "last evidence frame actual paced ingress to normalized result "
            "available to evaluator"
        ),
        "wall_time_excludes": [
            "natural event onset",
            "frontend delivery",
            "UI publication",
        ],
        "clock": (
            "time.monotonic on the input/evaluator host; "
            "remote clocks are not subtracted"
        ),
    }
    return {
        "spec_version": PROTOCOL,
        "model": backend.metadata,
        "questions": questions,
        "config": {
            "slots": slots,
            "workers": workers,
            "interval_seconds": interval,
            "sampling_fps": 1 / interval,
            "window_frames": window_frames,
            "window_semantics": "event present in any supplied frame",
            "sampling": (
                "original-speed source grid from zero, capped at source FPS; "
                "sliding window each sampled frame"
            ),
            "source_time_basis": (
                "relative decoder PTS when increasing; "
                "source-FPS grid fallback retained per frame"
            ),
            "frame_width": frame_width,
            "jpeg_quality": jpeg_quality,
            "queue_age_seconds": queue_age,
            "outstanding_limit": workers + slots,
            "per_source_outstanding_limit": 1,
            "replacement": (
                "input EOF replaces source immediately; old requests drain by source_id"
            ),
            "clip_aggregation": (
                "any valid yes; otherwise any valid no; otherwise unknown"
            ),
            "prediction_basis": (
                "saved observations; no does not guarantee absence over all video time"
            ),
            "unknown_scoring": (
                "kept in attempted binary-reference denominator; "
                "no false/zero conversion"
            ),
            **boundary,
        },
        "execution_complete": not interrupted
        and not shared_errors
        and len(primary) == len(media)
        and all(
            state["eof_seen"]
            and state["source_error"] is None
            and not state["outstanding"]
            for state in primary.values()
        ),
        "interrupted": interrupted,
        "shared_errors": shared_errors,
        "started_monotonic": started,
        "finished_monotonic": finished,
        "wall_seconds": finished - started,
        "input_coverage": _coverage(
            states,
            slots,
            started,
            last_primary_input[0]
            or max(
                (state["input_end_monotonic"] for state in primary.values()),
                default=started,
            ),
        ),
        "latency": {
            "primary_input_to_result_seconds": _distribution(primary_latencies),
            "all_input_to_result_seconds": _distribution(latencies),
            **boundary,
        },
        "window_records": str(record_path),
        "streams": states,
        "primary_windows": window_summary,
        "primary_event_evidence": event_evidence,
        "overall": {
            "correct": sum(count["correct"] for count in counters.values()),
            "total": sum(count["total"] for count in counters.values()),
            "planned": len(samples),
            "attempted": len(samples)
            - sum(count["pending"] for count in counters.values()),
            "pending": sum(count["pending"] for count in counters.values()),
            "unknown": sum(count["unknown"] for count in counters.values()),
            "unscored_reference": sum(
                count["unscored_reference"] for count in counters.values()
            ),
        },
        "per_event": dict(counters),
        "per_video": per_question,
    }
