"""Durable API reservations for a sequential validation-only batch.

Reserve the full per-attempt bound before its request intent. Release unused
reservation only after a persisted outcome supplies a finite actual charge.
Unknown charges and interrupted calls retain their entire reservation.
"""
import json
from pathlib import Path
import threading
import time

from .contract import atomic_json, digest, finite_number
from .retry import PausedRun


class BudgetGuard:
    def __init__(self, root, config, task_id, directory, upper):
        self.root, self.directory = Path(root), Path(directory)
        self.path = self.root / "api-budget.json"
        self.config_sha256, self.limit = digest(config), config["budget_usd"]
        self.task_id, self.upper = task_id, upper
        self.lock = threading.Lock()
        self.wait_seconds = config.get("budget_wait_seconds", 0)
        if not finite_number(self.wait_seconds) or not 0 <= self.wait_seconds <= 120:
            raise ValueError("budget wait must be finite and between zero and 120 seconds")
        self.issued_this_session = set()
        self.cancel_event = None
        # The batch and worker lifetime locks exclude other worker processes.
        # This lock serializes the twelve request threads before their intents.
        if self.path.exists():
            self.state = json.loads(self.path.read_text())
            if self.state.get("config_sha256") != self.config_sha256:
                raise ValueError("budget configuration differs")
        else:
            if list(self.root.rglob("attempts.jsonl")):
                raise ValueError("budget state missing from an existing run")
            self.state = {"config_sha256": self.config_sha256, "limit_usd": self.limit,
                          "initial_spent_usd": config.get("initial_spent_usd", 0), "reservations": {}}
            atomic_json(self.path, self.state)

    def set_cancel_event(self, event):
        self.cancel_event = event

    def __call__(self, intent):
        started = time.monotonic()
        waited = False
        while True:
            with self.lock:
                if (self.root / "STOP").exists():
                    raise PausedRun("operator stop file: no further request intents")
                if self.cancel_event is not None and self.cancel_event.is_set():
                    raise PausedRun("request queue cancelled during budget admission")
                reservations = self.state["reservations"]
                missing_outcomes = set()
                for reservation_key, reservation in reservations.items():
                    if "actual_cost_usd" in reservation:
                        continue
                    path = self.root / reservation["outcome"]
                    if not path.resolve().is_relative_to(self.root.resolve()):
                        raise ValueError("budget outcome outside run")
                    if not path.exists():
                        missing_outcomes.add(reservation_key)
                        continue
                    record = json.loads(path.read_text())
                    if any(record.get(k) != reservation[k] for k in ("unit", "attempt", "request_sha256")):
                        raise ValueError("budget outcome identity differs")
                    metadata = record.get("metadata", {})
                    cost = metadata.get("usage", {}).get("cost", metadata.get("total_cost"))
                    if finite_number(cost) and cost >= 0:
                        reservation["actual_cost_usd"] = cost
                        if cost > reservation["upper_usd"]:
                            self.state["blocked_reason"] = "provider charge exceeded the per-attempt bound"
                charged = self.state["initial_spent_usd"] + sum(
                    r.get("actual_cost_usd", r["upper_usd"]) for r in reservations.values())
                self.state["charged_or_reserved_usd"] = charged
                if self.state.get("blocked_reason"):
                    atomic_json(self.path, self.state)
                    raise PausedRun(self.state["blocked_reason"])
                key = digest({"task_id": self.task_id, **intent})
                if key in reservations:
                    atomic_json(self.path, self.state)
                    self.issued_this_session.add(key)
                    return
                if charged + self.upper <= self.limit:
                    filename = digest(intent["unit"]) + f"-attempt-{intent['attempt']}.json"
                    relative = (self.directory / "outcomes" / filename).relative_to(self.root).as_posix()
                    reservations[key] = {**intent, "task_id": self.task_id,
                                         "outcome": relative, "upper_usd": self.upper}
                    if waited:
                        reservations[key]["budget_admission_wait_seconds"] = time.monotonic() - started
                    self.state["charged_or_reserved_usd"] = charged + self.upper
                    atomic_json(self.path, self.state)
                    self.issued_this_session.add(key)
                    return
                # Only this worker's unresolved calls can still return a receipt.
                # Prior interrupted calls and completed unknown charges stay reserved.
                live = bool(missing_outcomes & self.issued_this_session)
                if not live or time.monotonic() - started >= self.wait_seconds:
                    self.state["blocked_reason"] = "budget cap: no room for another full attempt reservation"
                    atomic_json(self.path, self.state)
                    raise PausedRun(self.state["blocked_reason"])
                waited = True
            # Release the admission lock while existing requests finish.
            time.sleep(.05)
