"""Operate only the model child on the controller-owned GPU; never rent resources."""
import json
from pathlib import Path
import httpx

from .batch import _ownership


class ExperimentResources:
    def __init__(self, *, identity_path, expected_instance_id, expected_label,
                 reset_endpoint, cancel_paths, expected_source_manifest_sha256=None):
        self.path = Path(identity_path)
        self.instance, self.label = expected_instance_id, expected_label
        self.reset_endpoint = reset_endpoint
        self.cancel_paths = tuple(Path(path) for path in cancel_paths)

    def prepare(self, config):
        if any(path.exists() for path in self.cancel_paths):
            raise RuntimeError("parent requested cancellation")
        # The parent records provider inventory and exact container ownership, not probes.
        row = _ownership(json.loads(self.path.read_bytes()))
        if (row["instance_id"] != self.instance or row["experiment_label"] != self.label
                or config.get("experiment_label", self.label) != self.label):
            raise ValueError("owned deployment differs from frozen source configuration")
        return row

    def recover(self, config, failure, cleanup):
        owner = self.prepare(config)
        # Reset the owned model child and wait for exit; do not probe or replay frames.
        with httpx.Client(timeout=45, trust_env=False, follow_redirects=False) as client:
            response = client.post(self.reset_endpoint, json={"experiment_label": self.label,
                                   "expected_model": config["tracking"]["model"],
                                   "expected_sid": config["stream_session_id"]})
            response.raise_for_status()
            receipt = response.json()
        if (receipt.get("experiment_label") != self.label or receipt.get("remaining_model_pid") is not None
                or "remaining_model_pid" not in receipt
                or receipt.get("reset_confirmed") is not True):
            raise RuntimeError("owned model reset not confirmed")
        return {**owner, "safe_to_continue": True, "model_reset": receipt}
