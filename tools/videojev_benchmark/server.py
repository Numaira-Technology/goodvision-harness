"""Loopback-only native scoring with bounded CPU preparation and one GPU worker."""

import argparse
import base64
import binascii
import json
import math
import threading
import time
import traceback
from collections import OrderedDict, deque
from concurrent.futures import CancelledError, Future, TimeoutError
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from goodvision_harness.videojev_contract import QuestionTemplate
from tools.videojev_probe.contract import CONTEXT, MODEL, REVISION, parse_choices


MAX_BODY_BYTES = 64 * 1024 * 1024
MAX_IMAGE_BYTES = 8 * 1024 * 1024
MAX_TOTAL_IMAGE_BYTES = 48 * 1024 * 1024


class WindowError(Exception):
    """A terminal request status, without inventing an event observation."""

    def __init__(self, code, detail="", *, raw=None, timings=None):
        super().__init__(detail or code)
        self.code = code
        self.raw = raw
        self.timings = timings


@dataclass(eq=False)
class _Work:
    body: dict
    arrival: float
    question_layout: str = "repeated"
    question_template: QuestionTemplate | None = None
    future: Future = field(default_factory=Future)
    state: str = "pending"
    prepared: object = None
    preprocess_start: float | None = None
    preprocess_end: float | None = None
    gpu_start: float | None = None
    client_timed_out: bool = False
    finished: bool = False

    @property
    def key(self):
        return self.body["session"], self.body["request_id"]


def _number(value):
    return type(value) in (int, float) and math.isfinite(value)


def _json_safe(value):
    if value is None or type(value) in (str, bool, int):
        return value
    if type(value) is float:
        return value if math.isfinite(value) else repr(value)
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    return repr(value)


def _question_rows(questions, *, template=False):
    if not isinstance(questions, list) or not 1 <= len(questions) <= 64:
        raise ValueError("invalid question group size")
    ids, rows = [], []
    for question in questions:
        if (
            not isinstance(question, dict)
            or not isinstance(question.get("id"), str)
            or not 0 < len(question["id"]) <= 512
            or question.get("task") != "choice"
            or not isinstance(question.get("options"), list)
            or len(question["options"]) != 3
            or any(
                not isinstance(option, str) or not 0 < len(option) <= 4096
                for option in question["options"]
            )
        ):
            raise ValueError("invalid three-choice question")
        row = {
            "id": question["id"], "task": question["task"],
            "options": list(question["options"]),
        }
        if template:
            parts = question.get("query_parts")
            if (
                not isinstance(parts, list) or len(parts) != 2
                or any(not isinstance(part, str) for part in parts)
                or not 0 < sum(len(part) for part in parts) <= 65536
            ):
                raise ValueError("invalid source-time question template")
            row["query_parts"] = tuple(parts)
        else:
            query = question.get("query")
            if not isinstance(query, str) or not 0 < len(query) <= 65536:
                raise ValueError("invalid three-choice question")
            row["query"] = query
        ids.append(question["id"])
        rows.append(row)
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate question IDs")
    return tuple(rows) if template else rows


def _request(body):
    if not isinstance(body, dict):
        raise ValueError("request must be an object")
    if any(
        not isinstance(body.get(key), str) or not 0 < len(body[key]) <= 512
        for key in ("request_id", "session")
    ):
        raise ValueError("invalid request_id or session")
    fps = body.get("fps")
    images, times = body.get("images"), body.get("frame_times")
    if not _number(fps) or fps <= 0:
        raise ValueError("fps must be finite and positive")
    if (
        not isinstance(images, list)
        or not 2 <= len(images) <= 64
        or any(
            not isinstance(image, str)
            or not image
            or len(image) > 4 * ((MAX_IMAGE_BYTES + 2) // 3)
            for image in images
        )
        or sum(len(image) for image in images) > MAX_BODY_BYTES
    ):
        raise ValueError("invalid image count or encoded image size")
    if (
        not isinstance(times, list)
        or len(times) != len(images)
        or any(not _number(value) for value in times)
        or any(left >= right for left, right in zip(times, times[1:]))
    ):
        raise ValueError("frame_times must match images and strictly increase")
    if "questions" in body and "question_template" in body:
        raise ValueError("provide full questions or a fixed template, not both")
    questions = _question_rows(body["questions"]) if "questions" in body else None
    template = (
        _question_rows(body["question_template"], template=True)
        if "question_template" in body else None
    )
    question_layout = body.get("question_layout")
    if question_layout is not None and question_layout not in (
        "repeated", "shared", "prefix", "prefix-cache", "prefix-cache-aligned",
    ):
        raise ValueError("invalid question_layout")
    return {
        "request_id": body["request_id"],
        "session": body["session"],
        "fps": fps,
        "frame_times": list(times),
        "images": list(images),
        "question_layout": question_layout,
        "questions": questions,
        "question_template": template,
    }


def _shared_cuda_failure(exc):
    detail = str(exc).lower()
    return any(
        marker in detail
        for marker in (
            "illegal memory access",
            "device-side assert",
            "unspecified launch failure",
            "driver shutting down",
            "cuda driver error",
            "cuda driver version is insufficient",
            "cuda initialization error",
            "cuda error: initialization error",
            "cuda error: unknown error",
            "context is destroyed",
            "device has been lost",
            "device lost",
            "device is lost",
            "launch timed out and was terminated",
            "all cuda-capable devices are busy or unavailable",
            "no cuda-capable device is detected",
            "no cuda gpus are available",
        )
    )


class Scorer:
    def __init__(
        self,
        runtime,
        result_log,
        *,
        runtime_factory=None,
        request_timeout=20.0,
        queue_age=19.5,
        queue_policy="fifo",
        queue_capacity=48,
        ready_capacity=2,
        cpu_workers=1,
        gpu_batch_size=1,
        context=CONTEXT,
        max_pixels=409600,
        question_layout="repeated",
    ):
        if (
            not _number(request_timeout)
            or request_timeout <= 0
            or not _number(queue_age)
            or queue_age <= 0
            or queue_policy not in ("fifo", "latest-per-stream")
            or question_layout not in (
                "repeated", "shared", "prefix", "prefix-cache", "prefix-cache-aligned",
            )
            or gpu_batch_size not in (1, 2)
            or any(
                type(value) is not int or value <= 0
                for value in (
                    queue_capacity, ready_capacity, cpu_workers, context, max_pixels
                )
            )
            or (runtime is None and runtime_factory is None)
        ):
            raise ValueError("invalid scorer configuration")
        self.runtime = runtime
        self._runtime_factory = runtime_factory
        self.result_log = Path(result_log)
        self.result_log.parent.mkdir(parents=True, exist_ok=True)
        self.request_timeout, self.queue_age = request_timeout, queue_age
        self.queue_policy, self.queue_capacity = queue_policy, queue_capacity
        self.ready_capacity = ready_capacity
        self.context, self.max_pixels = context, max_pixels
        self.question_layout = question_layout
        self.gpu_batch_size = gpu_batch_size
        self._condition = threading.Condition()
        self._log_lock = threading.Lock()
        self._pending = deque()
        self._fair_pending = OrderedDict()
        self._pipeline = deque()
        self._active_sessions = {}
        self._jobs = {}
        self._recent = OrderedDict()
        self._question_templates = OrderedDict()
        self._last_template = None
        self._gpu_work = None
        self._loaded = False
        self._accepting = True
        self._stopping = False
        self._failure = None
        self._log_failed = False
        self._gpu_thread = threading.Thread(
            target=self._gpu_loop, name="videojev-gpu", daemon=True
        )
        self._cpu_threads = [
            threading.Thread(
                target=self._cpu_loop, name=f"videojev-cpu-{index}", daemon=True
            )
            for index in range(cpu_workers)
        ]
        self._gpu_thread.start()
        for thread in self._cpu_threads:
            thread.start()

    @property
    def healthy(self):
        with self._condition:
            return self._loaded and self._accepting and self._gpu_thread.is_alive()

    def wait_ready(self):
        with self._condition:
            self._condition.wait_for(lambda: self._loaded or self._stopping)
            if self._stopping:
                if self._failure is not None:
                    raise WindowError(self._failure.code, str(self._failure))
                raise WindowError("service_stopped")

    def snapshot(self):
        with self._condition:
            return {
                "ready": self.healthy,
                "queue": self._pending_count(),
                "preparing": sum(work.state == "preparing" for work in self._pipeline),
                "prepared": sum(work.state == "ready" for work in self._pipeline),
                "gpu_busy": self._gpu_work is not None,
                "queue_policy": self.queue_policy,
                "context": self.context,
                "max_pixels": self.max_pixels,
                "question_layout_default": self.question_layout,
                "gpu_batch_size": self.gpu_batch_size,
                "model": MODEL,
                "revision": REVISION,
                "runtime": {} if self.runtime is None else self.runtime.metadata,
                "error": None if self._failure is None else self._failure.code,
            }

    def _pending_count(self):
        if self.queue_policy == "fifo":
            return len(self._pending)
        return len(self._fair_pending)

    def submit(self, body):
        body = _request(body)
        replaced = None
        rejected = False
        with self._condition:
            key = body["session"], body["request_id"]
            existing = self._jobs.get(key)
            if existing is not None:
                return existing.future
            if key in self._recent:
                value, error = self._recent[key]
                future = Future()
                if error is None:
                    future.set_result(value)
                else:
                    future.set_exception(
                        WindowError(
                            error.code, str(error), raw=error.raw,
                            timings=dict(error.timings or {}),
                        )
                    )
                return future
            if not self._loaded or not self._accepting:
                if self._failure is not None:
                    raise WindowError(self._failure.code, str(self._failure))
                raise WindowError("service_unavailable")
            rows = body.pop("question_template")
            question_template = None
            if body["questions"] is None:
                session = body["session"]
                question_template = self._question_templates.get(session)
                if rows is not None:
                    if question_template is None or rows != question_template.rows:
                        if self._last_template is not None and rows == self._last_template.rows:
                            question_template = self._last_template
                        else:
                            question_template = QuestionTemplate(rows)
                    self._question_templates[session] = question_template
                    self._last_template = question_template
                elif question_template is None:
                    raise WindowError("template_missing")
                self._question_templates.move_to_end(session)
                while len(self._question_templates) > 64:
                    self._question_templates.popitem(last=False)
                body["questions"] = question_template.rows
            work = _Work(
                body, time.monotonic(),
                question_layout=body["question_layout"] or self.question_layout,
                question_template=question_template,
            )
            if self.queue_policy == "fifo":
                if self._pending_count() >= self.queue_capacity:
                    rejected = True
                else:
                    self._pending.append(work)
            else:
                session = body["session"]
                replaced = self._fair_pending.get(session)
                if replaced is None and self._pending_count() >= self.queue_capacity:
                    rejected = True
                else:
                    # Replacing a candidate preserves this session's waiting position.
                    self._fair_pending[session] = work
            self._jobs[key] = work
            self._condition.notify_all()
        if replaced is not None:
            self._finish(replaced, error=WindowError("superseded"))
        if rejected:
            self._finish(work, error=WindowError("queue_full"))
        return work.future

    def client_timeout(self, future):
        waiting = None
        with self._condition:
            for work in self._jobs.values():
                if work.future is future:
                    work.client_timed_out = True
                    if work.state == "pending":
                        self._remove_pending(work)
                        waiting = work
                    break
        if waiting is not None:
            self._finish(waiting, error=WindowError("client_timeout_before_start"))

    def _remove_pending(self, work):
        if self.queue_policy == "fifo":
            try:
                self._pending.remove(work)
            except ValueError:
                pass
        elif self._fair_pending.get(work.body["session"]) is work:
            del self._fair_pending[work.body["session"]]

    def _claim_pending(self):
        if self.queue_policy == "fifo":
            return self._pending.popleft() if self._pending else None
        for session, work in self._fair_pending.items():
            if session not in self._active_sessions:
                del self._fair_pending[session]
                return work
        return None

    def _cpu_loop(self):
        try:
            self._prepare_loop()
        except BaseException as exc:
            self._stop_shared(WindowError("cpu_worker_stopped", str(exc)))

    def _prepare_loop(self):
        while True:
            with self._condition:
                work = None
                while work is None:
                    if self._stopping:
                        return
                    if self._loaded and len(self._pipeline) < self.ready_capacity:
                        work = self._claim_pending()
                    if work is None:
                        self._condition.wait()
                started = time.monotonic()
                expired = started - work.arrival >= self.queue_age
                claimed = not expired and work.future.set_running_or_notify_cancel()
                if claimed:
                    work.state = "preparing"
                    work.preprocess_start = started
                    self._active_sessions[work.body["session"]] = work
                    # Reserve in claim order, including when CPU workers overlap.
                    self._pipeline.append(work)
            if expired or not claimed:
                self._finish(
                    work,
                    error=WindowError(
                        "expired_before_preprocess" if expired else "cancelled_before_start"
                    ),
                )
                continue
            try:
                images = []
                total = 0
                for encoded in work.body["images"]:
                    image = base64.b64decode(encoded, validate=True)
                    total += len(image)
                    if not image or len(image) > MAX_IMAGE_BYTES or total > MAX_TOTAL_IMAGE_BYTES:
                        raise ValueError("decoded image bytes exceed request bounds")
                    images.append(image)
                prepared = self.runtime.prepare(
                    images,
                    work.body["fps"],
                    context=self.context,
                    max_pixels=self.max_pixels,
                )
                with self._condition:
                    work.preprocess_end = time.monotonic()
                    work.body["images"] = []
                    if not self._stopping:
                        work.prepared = prepared
                        work.state = "ready"
                        self._condition.notify_all()
                        continue
                self._finish(work, error=self._failure or WindowError("service_stopped"))
            except BaseException as exc:
                work.preprocess_end = time.monotonic()
                error = WindowError("preprocess_failed", f"{type(exc).__name__}: {exc}")
                if not isinstance(exc, Exception):
                    error.code = "cpu_worker_stopped"
                    self._stop_shared(error)
                self._finish(work, error=error)
                if not isinstance(exc, Exception):
                    return

    def _gpu_loop(self):
        work = None
        try:
            if self.runtime is None:
                self.runtime = self._runtime_factory()
            with self._condition:
                self._loaded = True
                self._condition.notify_all()
            while True:
                with self._condition:
                    while not self._stopping and (
                        not self._pipeline or self._pipeline[0].state != "ready"
                    ):
                        self._condition.wait()
                    if self._stopping:
                        return
                    work = self._pipeline.popleft()
                    work.state = "gpu"
                    work.gpu_start = time.monotonic()
                    self._gpu_work = work
                    self._condition.notify_all()
                if work.gpu_start - work.arrival >= self.queue_age:
                    self._finish(work, error=WindowError("expired_before_gpu"))
                    continue
                extra = None
                if (self.gpu_batch_size == 2
                        and work.question_layout in ("repeated", "shared")
                        and work.question_template is not None
                        and self.runtime.packed_choice_supported):
                    with self._condition:
                        if (self._pipeline and self._pipeline[0].state == "ready"
                                and self._pipeline[0].question_layout in ("repeated", "shared")
                                and self._pipeline[0].question_template is not None):
                            extra = self._pipeline.popleft()
                            extra.state = "gpu"
                            extra.gpu_start = time.monotonic()
                            self._condition.notify_all()
                    if extra is not None and extra.gpu_start - extra.arrival >= self.queue_age:
                        self._finish(extra, error=WindowError("expired_before_gpu"))
                        extra = None
                if extra is not None:
                    self._score_batch([work, extra])
                    continue
                self._score_one(work)
        except BaseException as exc:
            traceback.print_exc()
            error = WindowError("gpu_worker_stopped", f"{type(exc).__name__}: {exc}")
            self._stop_shared(error)
            if work is not None:
                self._finish(work, error=error)
        finally:
            with self._condition:
                self._loaded = False
                self._condition.notify_all()
            close = getattr(self.runtime, "close", None)
            if close is not None:
                try:
                    close()
                except Exception as exc:
                    print(f"Engine shutdown failed: {type(exc).__name__}: {exc}", flush=True)

    def _score_one(self, work):
        raw = None
        timings = self._timings(work)
        try:
            raw, native_timings = self.runtime.predict_prepared(
                work.prepared, work.body["questions"],
                question_layout=work.question_layout,
                question_template=work.question_template,
                frame_times=work.body["frame_times"],
            )
            timings.update(native_timings)
            timings["gpu_batch_size"] = 1
            timings.update(self._timings(work))
            timings["server_gpu_seconds"] = time.monotonic() - work.gpu_start
            try:
                parse_choices(raw, work.body["questions"])
            except (ValueError, TypeError, KeyError, IndexError) as exc:
                raise WindowError("invalid_response", str(exc), raw=raw,
                                  timings=timings) from None
            self._finish(work, value={
                "model": MODEL, "revision": REVISION,
                "request_id": work.body["request_id"], "raw": raw, "timings": timings,
            })
        except Exception as exc:
            timings.update(self._timings(work))
            timings["server_gpu_seconds"] = time.monotonic() - work.gpu_start
            self._finish(work, error=self._gpu_error(exc, raw=raw, timings=timings))

    def _gpu_error(self, exc, *, raw=None, timings=None):
        if raw is None:
            raw = getattr(exc, "raw", None)
        error = WindowError(
            exc.code if isinstance(exc, WindowError) else "window_failed",
            f"{type(exc).__name__}: {exc}", raw=raw, timings=timings,
        )
        if _shared_cuda_failure(exc):
            error.code = "cuda_shared_failure"
            self._stop_shared(error)
        elif "out of memory" in str(exc).lower() or type(exc).__name__ == "OutOfMemoryError":
            error.code = "cuda_out_of_memory"
            try:
                self.runtime.torch.cuda.empty_cache()
                self.runtime.torch.cuda.synchronize()
            except Exception as recovery:
                self._stop_shared(WindowError("cuda_recovery_failed", str(recovery)))
        return error

    def _score_batch(self, works):
        outcomes = self.runtime.predict_prepared_batch([
            {"prepared": work.prepared, "questions": work.body["questions"],
             "question_layout": work.question_layout,
             "question_template": work.question_template,
             "frame_times": work.body["frame_times"]}
            for work in works
        ])
        try:
            for item in outcomes:
                if item is None:
                    for work in works:
                        if self._stopping:
                            break
                        if not work.finished:
                            with self._condition:
                                if work is not works[0]:
                                    work.gpu_start = time.monotonic()
                                self._gpu_work = work
                            if work.gpu_start - work.arrival >= self.queue_age:
                                self._finish(work, error=WindowError("expired_before_gpu"))
                                continue
                            self._score_one(work)
                    break
                index, outcome = item
                work = works[index]
                self._gpu_work = work
                timings = self._timings(work)
                timings["server_gpu_seconds"] = time.monotonic() - work.gpu_start
                raw = None
                try:
                    if isinstance(outcome, Exception):
                        raise outcome
                    raw, native_timings = outcome
                    timings.update(native_timings)
                    try:
                        parse_choices(raw, work.body["questions"])
                    except (ValueError, TypeError, KeyError, IndexError) as exc:
                        raise WindowError("invalid_response", str(exc), raw=raw,
                                          timings=timings) from None
                    self._finish(work, value={
                        "model": MODEL, "revision": REVISION,
                        "request_id": work.body["request_id"], "raw": raw, "timings": timings,
                    })
                except Exception as exc:
                    self._finish(work, error=self._gpu_error(exc, raw=raw, timings=timings))
                if self._stopping:
                    break
                with self._condition:
                    self._gpu_work = next((item for item in works if not item.finished), None)
        except BaseException as exc:
            error = self._gpu_error(exc)
            if not isinstance(exc, Exception):
                error = WindowError("gpu_worker_stopped", f"{type(exc).__name__}: {exc}")
                self._stop_shared(error)
            for work in works:
                if not work.finished:
                    self._finish(work, error=error)
            if not isinstance(exc, Exception):
                raise
        finally:
            outcomes.close()
            if self._stopping:
                cause = self._failure or WindowError("service_shutdown")
                for work in works:
                    if not work.finished:
                        self._finish(work, error=WindowError(
                            cause.code, f"Shared service stopped before this window completed: {cause}",
                        ))

    def _timings(self, work):
        now = time.monotonic()
        timings = {"server_elapsed_seconds": now - work.arrival}
        if work.preprocess_start is not None:
            timings["cpu_queue_seconds"] = work.preprocess_start - work.arrival
        if work.preprocess_end is not None:
            timings["server_preprocess_seconds"] = work.preprocess_end - work.preprocess_start
        if work.gpu_start is not None and work.preprocess_end is not None:
            timings["gpu_queue_seconds"] = work.gpu_start - work.preprocess_end
            timings["server_queue_seconds"] = (
                timings["cpu_queue_seconds"] + timings["gpu_queue_seconds"]
            )
        return timings

    def _finish(self, work, *, value=None, error=None, write_log=True):
        with self._condition:
            if work.finished:
                return
            work.finished = True
        if error is not None:
            error = WindowError(
                error.code, str(error), raw=error.raw, timings=dict(error.timings or {})
            )
        timings = self._timings(work)
        if value is not None:
            value["timings"].update(timings)
        elif error is not None:
            error.timings = {**(error.timings or {}), **timings}
        record = {
            "request_id": work.body["request_id"],
            "session": work.body["session"],
            "fps": work.body["fps"],
            "frame_times": work.body["frame_times"],
            "question_count": len(work.body["questions"]),
            "question_layout_requested": work.question_layout,
            "status": "completed" if error is None else error.code,
            "client_deadline_elapsed": work.client_timed_out,
            "timings": value["timings"] if value is not None else error.timings,
        }
        if value is not None:
            record["raw"] = value["raw"]
        elif error is not None:
            record.update(error=type(error).__name__, detail=str(error)[:1200])
            if error.raw is not None:
                record["raw"] = error.raw
        if write_log and not self._log_failed:
            # Invalid model values remain a window error, not a storage outage.
            try:
                line = json.dumps(record, allow_nan=False)
            except (TypeError, ValueError):
                record["raw_serialization"] = "non-JSON values represented verbatim as text"
                line = json.dumps(_json_safe(record), allow_nan=False)
            try:
                with self._log_lock:
                    with self.result_log.open("a", encoding="utf-8") as stream:
                        stream.write(line + "\n")
                        stream.flush()
            except Exception as exc:
                self._log_failed = True
                error = WindowError("result_log_failure", str(exc))
                self._stop_shared(error, write_log=False)
        with self._condition:
            self._remove_pending(work)
            try:
                self._pipeline.remove(work)
            except ValueError:
                pass
            if self._gpu_work is work:
                self._gpu_work = None
            if self._active_sessions.get(work.body["session"]) is work:
                del self._active_sessions[work.body["session"]]
            work.prepared = None
            work.body["images"] = []
            work.state = "done"
            if not work.future.done():
                if error is None:
                    work.future.set_result(value)
                else:
                    work.future.set_exception(error)
            self._jobs.pop(work.key, None)
            # Cache detached outcomes, not raised exceptions and HTTP stack frames.
            self._recent[work.key] = (
                value,
                None if error is None else WindowError(
                    error.code, str(error), raw=error.raw, timings=dict(error.timings or {})
                ),
            )
            while len(self._recent) > 256:
                self._recent.popitem(last=False)
            self._condition.notify_all()

    def _stop_shared(self, error, *, write_log=True):
        with self._condition:
            if self._stopping:
                return
            self._failure = WindowError(error.code, str(error))
            self._accepting = False
            self._stopping = True
            waiting = [
                work for work in self._jobs.values()
                if work.state in ("pending", "ready") and not work.finished
            ]
            self._pending.clear()
            self._fair_pending.clear()
            for work in self._jobs.values():
                if not work.future.done():
                    work.future.set_exception(
                        WindowError(
                            error.code,
                            str(error),
                            raw=error.raw if work is self._gpu_work else None,
                            timings=dict(error.timings or {}) if work is self._gpu_work else None,
                        )
                    )
            self._condition.notify_all()
        for work in waiting:
            self._finish(
                work, error=WindowError(error.code, str(error)), write_log=write_log
            )

    def shutdown(self, wait=True):
        self._stop_shared(WindowError("service_shutdown"))
        if wait:
            for thread in self._cpu_threads:
                thread.join()
            self._gpu_thread.join()


def handler(scorer):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def respond(self, status, value):
            try:
                data = json.dumps(value, allow_nan=False).encode()
            except (TypeError, ValueError):
                data = json.dumps(_json_safe(value), allow_nan=False).encode()
            try:
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
            except (BrokenPipeError, ConnectionResetError, TimeoutError):
                pass

        def do_HEAD(self):
            self.send_response(200 if scorer.healthy else 503)
            self.end_headers()

        def do_GET(self):
            self.respond(200 if scorer.healthy else 503, scorer.snapshot())

        def do_POST(self):
            if self.path != "/predict":
                self.respond(404, {})
                return
            future = None
            request_id = None
            started = time.monotonic()
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size <= MAX_BODY_BYTES:
                    raise ValueError("body size")
                self.connection.settimeout(scorer.request_timeout)
                payload = self.rfile.read(size)
                if len(payload) != size:
                    raise ValueError("incomplete body")
                body = json.loads(payload)
                if isinstance(body, dict):
                    request_id = body.get("request_id")
                future = scorer.submit(body)
                remaining = max(0, scorer.request_timeout - (time.monotonic() - started))
                value = future.result(timeout=remaining)
                self.respond(200, value)
            except WindowError as exc:
                status = 503 if exc.code in (
                    "queue_full", "service_unavailable", "service_shutdown",
                    "cuda_shared_failure", "cuda_recovery_failed", "result_log_failure",
                    "gpu_worker_stopped", "cpu_worker_stopped",
                ) else 409 if exc.code in ("superseded", "template_missing") else 504 if (
                    exc.code.startswith("expired_") or exc.code == "client_timeout_before_start"
                ) else 500
                value = {"request_id": request_id, "error": exc.code, "detail": str(exc)[:1200]}
                if exc.raw is not None:
                    value["raw"] = exc.raw
                if exc.timings is not None:
                    value["timings"] = exc.timings
                self.respond(status, value)
            except TimeoutError:
                if future is not None:
                    scorer.client_timeout(future)
                self.respond(504, {"request_id": request_id, "error": "request_deadline"})
            except CancelledError:
                self.respond(409, {"request_id": request_id, "error": "cancelled_before_start"})
            except (ValueError, binascii.Error):
                self.respond(400, {"request_id": request_id, "error": "invalid_request"})
            except Exception as exc:
                self.respond(500, {"request_id": request_id, "error": "scorer_failure", "detail": str(exc)[:1200]})

    return Handler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--result-log", "--audit", dest="result_log", required=True)
    parser.add_argument("--port", type=int, default=8093)
    parser.add_argument("--request-timeout", type=float, default=20.0)
    parser.add_argument("--queue-age", type=float, default=19.5)
    parser.add_argument("--queue-policy", choices=("fifo", "latest-per-stream"), default="fifo")
    parser.add_argument("--queue-capacity", type=int, default=48)
    parser.add_argument("--ready-capacity", type=int, default=2)
    parser.add_argument("--cpu-workers", type=int, default=1)
    parser.add_argument("--gpu-batch-size", type=int, choices=(1, 2), default=1,
                        help="Score up to two already-ready independent windows without waiting")
    parser.add_argument("--execution-backend", choices=("native", "vllm"), default="native",
                        help="Use native scoring or the isolated vLLM 0.30 pooling adapter")
    parser.add_argument("--single-sequence", action="store_true",
                        help="Use unpadded single-sequence scoring for fixed non-prefix windows")
    parser.add_argument("--compile-language-mlp", action="store_true",
                        help="Compile existing language MLPs with Inductor; first use includes compilation")
    parser.add_argument("--fuse-language-gate-up", action="store_true",
                        help="Merge each language MLP's original gate/up weights once at load time")
    parser.add_argument("--language-mlp-fp8", action="store_true",
                        help="Use lossy FP8 weights and dynamic activations only in language MLPs")
    parser.add_argument("--language-mlp-gemm-autotune", action="store_true",
                        help="Compile language MLPs and tune ATEN/TRITON matrix kernels at first use")
    parser.add_argument("--compile-language-norm", action="store_true",
                        help="Compile ordinary language RMSNorms without changing weights or gated norms")
    parser.add_argument("--compile-vision-blocks", action="store_true",
                        help="Compile existing visual blocks; first use includes compilation")
    parser.add_argument("--profile-request", type=int, default=0,
                        help="Profile this real single-window call once; zero disables profiling")
    parser.add_argument("--profile-output", type=Path,
                        help="Directory for the optional operator trace and table")
    parser.add_argument("--context", type=int, default=CONTEXT)
    parser.add_argument("--max-pixels", type=int, default=409600)
    parser.add_argument(
        "--question-layout",
        choices=("repeated", "shared", "prefix", "prefix-cache", "prefix-cache-aligned"),
        default="repeated",
        help="Default native question layout; a request may explicitly override it",
    )
    args = parser.parse_args()
    from tools.videojev_probe.runtime import VideoJevRuntime

    runtime_class = VideoJevRuntime
    runtime_options = {
        "single_sequence": args.single_sequence,
        "compile_language_mlp": args.compile_language_mlp,
        "fuse_language_gate_up": args.fuse_language_gate_up,
        "language_mlp_fp8": args.language_mlp_fp8,
        "language_mlp_gemm_autotune": args.language_mlp_gemm_autotune,
        "compile_language_norm": args.compile_language_norm,
        "compile_vision_blocks": args.compile_vision_blocks,
    }
    if args.execution_backend == "vllm":
        if (args.question_layout not in ("repeated", "shared")
                or any(runtime_options.values()) or args.profile_request > 0):
            parser.error("The vLLM adapter uses repeated/shared layouts without native-only optimizations")
        from tools.videojev_vllm.runtime import EngineVideoJevRuntime

        runtime_class = EngineVideoJevRuntime
        runtime_options = {"context": args.context}
    elif args.profile_request > 0:
        from tools.videojev_benchmark.profiling import ProfiledVideoJevRuntime

        runtime_class = ProfiledVideoJevRuntime
        runtime_options.update({
            "profile_request": args.profile_request,
            "profile_output": args.profile_output or Path(args.result_log).parent / "kernel-profile",
        })
    scorer = Scorer(
        None,
        args.result_log,
        runtime_factory=lambda: runtime_class(
            args.cache_dir, question_layout=args.question_layout,
            **runtime_options,
        ),
        request_timeout=args.request_timeout,
        queue_age=args.queue_age,
        queue_policy=args.queue_policy,
        queue_capacity=args.queue_capacity,
        ready_capacity=args.ready_capacity,
        cpu_workers=args.cpu_workers,
        gpu_batch_size=args.gpu_batch_size,
        context=args.context,
        max_pixels=args.max_pixels,
        question_layout=args.question_layout,
    )
    server = None
    try:
        scorer.wait_ready()
        server = ThreadingHTTPServer(("127.0.0.1", args.port), handler(scorer))
        print(json.dumps(scorer.snapshot()), flush=True)
        server.serve_forever()
    finally:
        scorer.shutdown(wait=False)
        if server is not None:
            server.server_close()
        scorer.shutdown()


if __name__ == "__main__":
    main()
