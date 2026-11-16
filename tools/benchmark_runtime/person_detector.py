"""Bounded source-ordered detector broker, shared by spawned VLM workers.

Source enqueue never performs network or journal I/O. One consumer serializes
the existing single-GPU /v1/tracks worker; overlapping windows reuse exact frame
snapshots.
The browser only reads snapshots and cannot initiate detection.
"""

import asyncio
import hashlib
import json
import math
import re
import secrets
import threading
import time
from collections import Counter, OrderedDict, deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from queue import Empty, Full, Queue
from urllib.parse import urlsplit

import httpx

from goodvision_harness.contracts import NormalizedBBox

MODEL = "yolo26-s"
WAIT_SECONDS = 2.0


def validate_snapshot(raw, session, sequence, timestamp):
    if (
        not isinstance(raw, dict)
        or raw.get("stream_session_id") != session
        or type(raw.get("frame_sequence")) is not int
        or raw["frame_sequence"] != sequence
        or type(raw.get("timestamp_seconds")) not in (int, float)
        or not math.isfinite(raw["timestamp_seconds"])
        or abs(raw["timestamp_seconds"] - timestamp) > 1e-6
        or not isinstance(raw.get("objects"), list)
        or len(raw["objects"]) > 64
    ):
        raise ValueError("detector source mismatch")
    people, seen = [], set()
    for obj in raw["objects"]:
        if not isinstance(obj, dict):
            raise ValueError("invalid detector object")
        sid = obj.get("track_id")
        if (
            not isinstance(sid, str)
            or not re.fullmatch(r"P[0-9]{4,8}", sid)
            or sid in seen
            or obj.get("label") not in {"person", "pedestrian"}
            or obj.get("observation_state") not in {"observed", "predicted"}
        ):
            raise ValueError("invalid detector identity")
        seen.add(sid)
        bbox = obj.get("bbox")
        if not isinstance(bbox, dict) or set(bbox) != {"x1", "y1", "x2", "y2"}:
            raise ValueError("invalid detector geometry")
        if any(
            type(v) not in (int, float) or not math.isfinite(v) for v in bbox.values()
        ):
            raise ValueError("invalid detector geometry")
        box = NormalizedBBox(**bbox)
        if obj["observation_state"] == "predicted":
            continue  # Prediction is not fresh visual evidence of a person.
        if (
            obj.get("last_observed_frame_sequence") != sequence
            or obj.get("last_observed_timestamp_seconds") != timestamp
        ):
            raise ValueError("stale detector observation")
        people.append(
            {
                "track_id": sid,
                "bbox": {k: getattr(box, k) for k in ("x1", "y1", "x2", "y2")},
            }
        )
    return people


class DetectorBroker:
    def __init__(self, url, output=None, *, transport=None, capacity=48):
        parsed = urlsplit(url)
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.hostname
            or parsed.username
            or parsed.password
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError("invalid detector URL")
        self.url = url.rstrip("/")
        self.output = output
        self.transport = transport or self._request
        self.queue = Queue(maxsize=capacity)
        self.condition = threading.Condition()
        self.cache = OrderedDict()
        self.latest = {}
        self.last_sequence = {}
        self.counts = Counter()
        self.stop = threading.Event()
        self.token = secrets.token_hex(24)
        self.thread = threading.Thread(target=self._work, daemon=True)
        self.log = self.server = self.server_thread = None
        self._audit = deque()
        self._audit_pending = set()
        self._audit_limit = 512
        self._audit_broken = False
        self._error = None
        self._log_closed = False

    def check_ready(self):
        with httpx.Client(trust_env=False, timeout=WAIT_SECONDS) as client:
            response = client.get(self.url + "/health/tracking")
            response.raise_for_status()
            data = response.json()
        if (
            data.get("model") != MODEL
            or data.get("gpu_ready") is not True
            or data.get("ready") is not True
        ):
            raise ValueError("start and warm the YOLO26-S GPU worker first")
        return {"model": MODEL, "gpu_ready": True}

    def __enter__(self):
        # Raw writes have no user-space buffer for close() to replay after an
        # uncertain/partial failure. A successful flush still means OS delivery,
        # not an fsync or a power-loss durability guarantee.
        self.log = (
            (self.output / "detector.jsonl").open("wb", buffering=0)
            if self.output
            else None
        )

        class Server(ThreadingHTTPServer):
            request_queue_size = 128

        try:
            self.server = Server(("127.0.0.1", 0), self._handler())
            self.server.daemon_threads = True
            self.server_thread = threading.Thread(
                target=self.server.serve_forever, daemon=True
            )
            self.endpoint = f"http://127.0.0.1:{self.server.server_port}/{self.token}"
            self.server_thread.start()
            self.thread.start()
        except Exception:
            self.__exit__(None, None, None)
            raise
        return self

    def __exit__(self, *_):
        # Queue cancellation records before the sole writer observes stop. Its
        # final drain must include these records, not leave them after join.
        with self.condition:
            self.stop.set()
            while True:
                try:
                    _key, row, _jpeg = self.queue.get_nowait()
                except Empty:
                    break
                try:
                    self._finish(row, state="cancelled")
                except BaseException as error:
                    self._fail_locked(error)
            self.condition.notify_all()
        if self.thread.ident is not None:
            self.thread.join(timeout=4)
        alive = self.thread.is_alive()
        if self.server is not None:
            if self.server_thread is not None and self.server_thread.is_alive():
                self.server.shutdown()
                self.server_thread.join(timeout=2)
            self.server.server_close()
        if not alive:
            self._close_log()
        if alive:
            # Never close a stream underneath its still-running writer.
            error = RuntimeError("detector worker did not stop")
            with self.condition:
                self._fail_locked(error)
            raise error
        with self.condition:
            if self._audit_pending:
                self._fail_locked(RuntimeError("detector audit incomplete"))
            self._check_error()

    def _fail_locked(self, error):
        if self._error is None:
            self._error = error
        self.stop.set()
        self.condition.notify_all()

    def _check_error(self):
        if self._error is not None:
            raise RuntimeError("detector broker failed") from self._error

    def _close_log(self):
        if self.log and not self._log_closed:
            try:
                self.log.close()
            except BaseException as error:
                with self.condition:
                    self._fail_locked(error)
            finally:
                self._log_closed = True

    def submit(self, camera, session, sequence, timestamp, jpeg):
        key = (session, sequence)
        with self.condition:
            self._check_error()
            if self.stop.is_set():
                raise RuntimeError("detector broker stopped")
            if sequence <= self.last_sequence.get(session, -1):
                raise ValueError("detector source must increase")
            self.last_sequence[session] = sequence
            row = {
                "camera": camera,
                "stream_session_id": session,
                "frame_sequence": sequence,
                "timestamp_seconds": timestamp,
                "state": "pending",
                "people": [],
                "submitted": time.monotonic(),
            }
            self.cache[key] = row
            self.counts["submitted"] += 1
            # Metadata only. Images live in the bounded queue, not this cache.
            while len(self.cache) > 512:
                self.cache.popitem(last=False)
            try:
                self.queue.put_nowait((key, row, jpeg))
            except Full:
                self._finish(row, state="queue_full")
            self.condition.notify_all()

    def _request(self, row, jpeg):
        # New broker for every round: never inherit identities across replays.
        session = (
            self.token[:16]
            + "-"
            + hashlib.sha256(row["stream_session_id"].encode()).hexdigest()[:24]
        )
        headers = {
            "Content-Type": "image/jpeg",
            "X-GoodVision-Stream-Session-Id": session,
            "X-GoodVision-Frame-Sequence": str(row["frame_sequence"]),
            "X-GoodVision-Timestamp-Seconds": str(row["timestamp_seconds"]),
            "X-GoodVision-Detector-Model": MODEL,
        }

        async def post():
            body = bytearray()
            async with self.http.stream(
                "POST",
                self.url + "/v1/tracks",
                content=jpeg,
                headers=headers,
            ) as response:
                response.raise_for_status()
                async for chunk in response.aiter_bytes():
                    body.extend(chunk)
                    if len(body) > 1024 * 1024:
                        raise ValueError("detector response too large")
            return bytes(body)

        # A wall-clock deadline, not a reset-on-every-chunk socket timeout.
        data = self.loop.run_until_complete(asyncio.wait_for(post(), WAIT_SECONDS))
        raw = json.loads(data)
        return validate_snapshot(
            raw, session, row["frame_sequence"], row["timestamp_seconds"]
        )

    def _finish(self, row, **values):
        # Called with condition held. Keep counts/latest immediately visible,
        # but defer all journal I/O to the existing detector worker.
        if self.log:
            if self._log_closed or len(self._audit_pending) >= self._audit_limit:
                error = RuntimeError("detector audit unavailable or capacity exhausted")
                self._fail_locked(error)
                raise error
            key = (row["stream_session_id"], row["frame_sequence"])
            if key in self._audit_pending or row["state"] != "pending":
                raise RuntimeError("duplicate detector completion")
            # Serialize now so later cache mutations cannot change the audit.
            self._audit.append(
                (key, (json.dumps({**row, **values}) + "\n").encode("utf-8"))
            )
            self._audit_pending.add(key)
        row.update(values)
        self.counts[row["state"]] += 1
        current = self.latest.get(row["camera"])
        if current is None or row["frame_sequence"] > current["frame_sequence"]:
            self.latest[row["camera"]] = dict(row)
        self.condition.notify_all()

    def _flush_audit(self):
        while True:
            with self.condition:
                if self._audit_broken or not self._audit:
                    return
                key, payload = self._audit.popleft()
            try:
                written = self.log.write(payload)
                if written != len(payload):
                    raise OSError("short detector audit write")
                self.log.flush()
            except BaseException as error:
                with self.condition:
                    # The write may be partial. Never retry it or claim success.
                    self._audit_broken = True
                    self._fail_locked(error)
                return
            with self.condition:
                # Includes in-flight records: an empty deque is not a flush ack.
                self._audit_pending.remove(key)
                self.condition.notify_all()

    def _work(self):
        # The repository supports Python 3.10; asyncio.Runner starts at 3.11.
        self.loop = None
        try:
            self.loop = asyncio.new_event_loop()
            self.http = httpx.AsyncClient(trust_env=False, timeout=WAIT_SECONDS)
            try:
                self._consume()
            finally:
                self.loop.run_until_complete(self.http.aclose())
                self.loop.run_until_complete(self.loop.shutdown_asyncgens())
        except BaseException as error:
            with self.condition:
                self._fail_locked(error)
        finally:
            try:
                self._flush_audit()
            except BaseException as error:
                with self.condition:
                    self._fail_locked(error)
            try:
                if self.loop is not None:
                    self.loop.close()
            except BaseException as error:
                with self.condition:
                    self._fail_locked(error)
            finally:
                self._close_log()

    def _consume(self):
        while not self.stop.is_set():
            self._flush_audit()
            if self.stop.is_set():
                break
            try:
                _key, row, jpeg = self.queue.get(timeout=0.1)
            except Empty:
                continue
            waited = time.monotonic() - row["submitted"]
            started = time.monotonic()
            people, state, status = [], "ready", None
            try:
                if waited > WAIT_SECONDS:
                    state = "queue_expired"
                else:
                    people = self.transport(row, jpeg)
            except Exception as exc:
                state = "detector_failed"
                code = getattr(exc, "code", None)
                if isinstance(exc, httpx.HTTPStatusError):
                    code = exc.response.status_code
                status = code if type(code) is int and 100 <= code <= 599 else None
            with self.condition:
                self._finish(
                    row,
                    state=state,
                    people=people,
                    http_status=status,
                    queue_seconds=waited,
                    http_seconds=time.monotonic() - started,
                )
            self._flush_audit()

    def lookup(self, session, sequences):
        keys = [(session, seq) for seq in sequences]
        deadline = time.monotonic() + WAIT_SECONDS
        with self.condition:
            while True:
                self._check_error()
                rows = [self.cache.get(key) for key in keys]
                if all(
                    row and row["state"] != "pending" and key not in self._audit_pending
                    for key, row in zip(keys, rows, strict=True)
                ):
                    return [dict(row) for row in rows]
                left = deadline - time.monotonic()
                if self.stop.is_set() or left <= 0:
                    raise TimeoutError("detector evidence unavailable")
                self.condition.wait(left)

    def drain(self):
        """Account for every submitted source frame before freezing trial totals."""
        deadline = time.monotonic() + 2 * WAIT_SECONDS + 1
        with self.condition:
            self._check_error()
            while (
                self.counts["submitted"]
                != sum(
                    value for key, value in self.counts.items() if key != "submitted"
                )
                or self._audit_pending
            ):
                self._check_error()
                left = deadline - time.monotonic()
                if left <= 0:
                    raise RuntimeError("detector drain timed out")
                self.condition.wait(left)
            self._check_error()
            return dict(self.counts)

    def _handler(self):
        broker = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_):
                pass

            def do_POST(self):
                try:
                    self.connection.settimeout(3)
                    size = int(self.headers.get("Content-Length", "0"))
                    if self.path != f"/{broker.token}" or not 0 < size <= 2048:
                        raise ValueError("invalid snapshot request")
                    data = json.loads(self.rfile.read(size))
                    session, seqs = data["session"], data["sequences"]
                    if (
                        not isinstance(session, str)
                        or len(session) > 160
                        or not isinstance(seqs, list)
                        or not 2 <= len(seqs) <= 4
                        or any(type(s) is not int or s < 0 for s in seqs)
                        or seqs != sorted(set(seqs))
                    ):
                        raise ValueError("invalid evidence keys")
                    body = json.dumps({"frames": broker.lookup(session, seqs)}).encode()
                    status = 200
                except Exception:
                    body, status = b'{"error":"detector_unavailable"}', 503
                try:
                    self.send_response(status)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                except OSError:
                    pass

        return Handler
