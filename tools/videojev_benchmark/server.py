"""Loopback-only native scoring server with bounded, expiring GPU work."""

import argparse
import base64
import json
import queue
import threading
import time
from concurrent.futures import Future, TimeoutError
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from tools.videojev_probe.contract import MODEL, POLICY_SHA, REVISION, parse_choices


class Scorer:
    def __init__(self, runtime, audit):
        self.runtime, self.audit = runtime, Path(audit)
        self.audit.parent.mkdir(parents=True, exist_ok=True)
        self.queue = queue.Queue(maxsize=48)
        self.healthy = True
        threading.Thread(target=self.work, daemon=True).start()

    def submit(self, body):
        if not self.healthy:
            raise RuntimeError("GPU scorer unhealthy")
        if not isinstance(body, dict):
            raise ValueError("request must be an object")
        if (
            body.get("model"),
            body.get("revision"),
            body.get("policy_sha256"),
            body.get("fps"),
        ) != (MODEL, REVISION, POLICY_SHA, 2.5):
            raise ValueError("model or policy mismatch")
        if (
            any(
                not isinstance(body.get(key), str) or not 0 < len(body[key]) <= 512
                for key in ("request_id", "session")
            )
            or body.get("lane") not in ("events", "grounding")
            or not isinstance(body.get("images"), list)
            or not 2 <= len(body["images"]) <= 4
            or any(not isinstance(image, str) or not image for image in body["images"])
        ):
            raise ValueError("invalid request")
        if (
            not isinstance(body.get("questions"), list)
            or not 6 <= len(body["questions"]) <= 326
        ):
            raise ValueError("invalid question group")
        ids = []
        for question in body["questions"]:
            if (
                not isinstance(question, dict)
                or not isinstance(question.get("id"), str)
                or not question["id"]
                or question.get("task") != "choice"
                or not isinstance(question.get("query"), str)
                or not question["query"]
                or not isinstance(question.get("options"), list)
                or len(question["options"]) != 3
                or any(
                    not isinstance(option, str) or not option
                    for option in question["options"]
                )
            ):
                raise ValueError("invalid choice question")
            ids.append(question["id"])
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate question IDs")
        future = Future()
        self.queue.put_nowait((time.monotonic(), body, future))
        return future

    def work(self):
        while True:
            arrival, body, future = self.queue.get()
            start = time.monotonic()
            audit = {k: body[k] for k in ("request_id", "session", "lane")}
            audit.update(
                question_count=len(body["questions"]), queue_seconds=start - arrival
            )
            try:
                # Atomically claim work: a client may cancel a queued request,
                # but cannot race result publication once GPU execution starts.
                if not future.set_running_or_notify_cancel():
                    audit["status"] = "cancelled_before_gpu"
                    continue
                if not self.healthy or start - arrival >= 19.5:
                    audit["status"] = "expired_before_gpu"
                    future.set_exception(TimeoutError())
                    continue
                images = [base64.b64decode(v, validate=True) for v in body["images"]]
                raw, timings = self.runtime.predict(images, 2.5, body["questions"])
                parse_choices(raw, body["questions"])
                timings["server_queue_seconds"] = start - arrival
                value = dict(
                    model=MODEL,
                    revision=REVISION,
                    request_id=body["request_id"],
                    raw=raw,
                    timings=timings,
                )
                audit.update(
                    status="completed",
                    raw=raw,
                    timings=timings,
                    questions=body["questions"],
                )
                if not future.done():
                    future.set_result(value)
            except Exception as exc:
                audit.update(
                    status="failed", error=type(exc).__name__, detail=str(exc)[:1200]
                )
                if isinstance(exc, RuntimeError):
                    self.healthy = False
                if not future.done():
                    future.set_exception(exc)
            finally:
                audit["elapsed_seconds"] = time.monotonic() - arrival
                with self.audit.open("a") as f:
                    f.write(json.dumps(audit, allow_nan=False) + "\n")
                self.queue.task_done()


def handler(scorer):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def respond(self, status, value):
            data = json.dumps(value, allow_nan=False).encode()
            try:
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def do_HEAD(self):
            self.send_response(200 if scorer.healthy else 503)
            self.end_headers()

        def do_GET(self):
            self.respond(
                200 if scorer.healthy else 503,
                dict(
                    ready=scorer.healthy,
                    queue=scorer.queue.qsize(),
                    model=MODEL,
                    revision=REVISION,
                    runtime=scorer.runtime.metadata,
                ),
            )

        def do_POST(self):
            if self.path != "/predict":
                self.respond(404, {})
                return
            future = None
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size <= 12 * 1024 * 1024:
                    raise ValueError("body size")
                self.connection.settimeout(25)
                body = json.loads(self.rfile.read(size))
                future = scorer.submit(body)
                value = future.result(timeout=19.8)
                self.respond(200, value)
            except queue.Full:
                self.respond(503, {"error": "queue_full"})
            except TimeoutError:
                if future:
                    future.cancel()
                self.respond(504, {"error": "deadline"})
            except ValueError:
                self.respond(400, {"error": "invalid_request"})
            except Exception:
                self.respond(500, {"error": "scorer_failure"})

    return Handler


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache-dir", required=True)
    p.add_argument("--audit", required=True)
    a = p.parse_args()
    from tools.videojev_probe.runtime import VideoJevRuntime

    runtime = VideoJevRuntime(a.cache_dir)
    scorer = Scorer(runtime, a.audit)
    print(json.dumps({"ready": True, **runtime.metadata}), flush=True)
    ThreadingHTTPServer(("127.0.0.1", 8093), handler(scorer)).serve_forever()


if __name__ == "__main__":
    main()
