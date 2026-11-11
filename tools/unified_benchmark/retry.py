"""Durable three-attempt queue. Transport is injected; importing does no IO."""
import fcntl
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from .contract import canonical, valid_value


class PausedRun(RuntimeError):
    pass


class RequestFailure(Exception):
    def __init__(self, reason, *, status=None, retry_after=0, fatal=False, metadata=None):
        super().__init__(reason)
        self.reason, self.status, self.retry_after, self.fatal = reason, status, retry_after, fatal
        self.metadata = metadata or {}


class Ledger:
    """Exclusive process lock + fsynced intent before send. Crash intents consume budget."""
    def __init__(self, path, context):
        self.path, self.context = Path(path), context
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.file = self.path.open("a+", encoding="utf-8")
        try:
            fcntl.flock(self.file, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.file.seek(0)
            self.rows = [json.loads(line) for line in self.file if line.strip()]
            if any(row.get("context") != context for row in self.rows):
                raise ValueError("resume context differs")
        except BaseException:
            self.file.close()
            raise
        self.lock = threading.RLock()

    def close(self):
        self.file.close()

    def append(self, row):
        with self.lock:
            row = {**row, "context": self.context, "at_unix": time.time()}
            self.file.seek(0, 2)
            self.file.write(canonical(row).decode() + "\n")
            self.file.flush()
            os.fsync(self.file.fileno())
            self.rows.append(row)

    def state(self, unit, kind):
        with self.lock:
            records = [r for r in self.rows if r["unit"] == unit["id"]]
            if any(r["request_sha256"] != unit["request_sha256"] for r in records):
                raise ValueError("cannot resume a changed request")
            if any(type(r.get("attempt")) is not int or not 1 <= r["attempt"] <= 3 for r in records):
                raise ValueError("invalid ledger attempt identity")
            intents = [r for r in records if r["type"] == "intent"]
            if [r["attempt"] for r in intents] != list(range(1, len(intents) + 1)) or len(intents) > 3:
                raise ValueError("invalid cumulative attempt sequence")
            answers, preceding, completed = {}, set(), set()
            for row in records:
                attempt = row["attempt"]
                if row["type"] == "intent":
                    preceding.add(attempt)
                    continue
                if row["type"] != "result":
                    raise ValueError("invalid ledger record type")
                if attempt not in preceding:
                    raise ValueError("result has no matching preceding intent")
                if attempt in completed:
                    raise ValueError("duplicate result for intent")
                completed.add(attempt)
                for event, value in row.get("answers", {}).items():
                    if event in unit["targets"] and valid_value(value, kind):
                        answers.setdefault(event, value)
            return len(intents), answers


class Queue:
    def __init__(self, ledger, kind, send, *, slots=12, clock=time.time, sleep=time.sleep):
        if not 1 <= slots <= 12:
            raise ValueError("slots must be 1..12")
        self.ledger, self.kind, self.send, self.slots = ledger, kind, send, slots
        self.clock, self.sleep = clock, sleep
        self.lock, self.stopped = threading.Lock(), threading.Event()
        self.cooldown_until = max([r.get("cooldown_until", 0) for r in ledger.rows] + [0])

    def wait(self):
        while not self.stopped.is_set():
            with self.lock:
                delay = self.cooldown_until - self.clock()
            if delay <= 0:
                return
            self.sleep(min(.25, delay))
        raise PausedRun("queue paused; existing valid responses retained")

    def one(self, unit):
        used, answers = self.ledger.state(unit, self.kind)
        for number in range(used + 1, 4):
            if set(answers) == set(unit["targets"]):
                break
            self.wait()
            base = {"unit": unit["id"], "request_sha256": unit["request_sha256"], "attempt": number}
            self.ledger.append({**base, "type": "intent"})
            started = self.clock()
            try:
                received, metadata = self.send(unit)
                new = {event: value for event, value in received.items()
                       if event in unit["targets"] and valid_value(value, self.kind)}
                # Preserve the first valid answer even when another field requires retry.
                answers = {**new, **answers}
                row = {**base, "type": "result", "answers": new, "metadata": metadata,
                       "elapsed_seconds": self.clock() - started, "complete": set(answers) == set(unit["targets"])}
                if not row["complete"]:
                    row["error"] = "missing_or_invalid_target_response"
            except RequestFailure as error:
                row = {**base, "type": "result", "answers": {}, "complete": False,
                       "error": error.reason, "http_status": error.status,
                       "elapsed_seconds": self.clock() - started, "fatal": error.fatal,
                       "metadata": error.metadata}
                if error.fatal:
                    self.stopped.set()
                row["retry_after"] = error.retry_after
            if not row["complete"] and not row.get("fatal") and (number < 3 or row.get("http_status") == 429):
                delay = max(row.get("retry_after", 0), 30 if number == 1 else 60)
                with self.lock:
                    self.cooldown_until = max(self.cooldown_until, self.clock() + delay)
                    row["cooldown_until"] = self.cooldown_until
            self.ledger.append(row)
            if row.get("fatal"):
                raise PausedRun(row["error"])
        return unit["id"], answers

    def run(self, units):
        if len({u["id"] for u in units}) != len(units):
            raise ValueError("duplicate unit")
        with ThreadPoolExecutor(max_workers=self.slots) as pool:
            try:
                return dict(pool.map(self.one, units))
            except BaseException:
                self.stopped.set()
                raise
