"""Run frozen capture, saved-evidence answers, or final-model answers as separate phases."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import re
from urllib.parse import urlparse

from ..model_api import ModelAPIClient, ModelAPISettings
from .batch import run_sources, _ownership
from .capture_source import _read_frozen
from .cohort import generate_cohort
from .model_ledger import ModelLedger
from .privacy import reject_private
from .recording import publish_once
from .resources import ExperimentResources
from .summary import execute_summary


def _path(base, value):
    return (base / value).resolve()


def _publish_or_match(path, value):
    if path.exists():
        if json.loads(path.read_bytes()) != value:
            raise ValueError(f"existing artifact identity changed: {path.name}")
    else:
        publish_once(path, value)


def _answers(base, config, output, arm):
    questions = _path(base, config["public_questions"])
    index = output / arm["model"] / "observation-index.json"
    directory = output / (arm["model"] + "-answers")
    if not index.exists():
        return None
    expected = {"questions_sha256": config["public_questions_sha256"],
                "observation_index_sha256": hashlib.sha256(index.read_bytes()).hexdigest()}
    if directory.exists():
        run = json.loads((directory / "run.json").read_bytes())
        if any(run.get(k) != v for k, v in expected.items()):
            raise ValueError("answer inputs changed; use an independent output directory")
        completion = json.loads((directory / "completion.json").read_bytes())
        for row in completion["rows"]:
            path = _path(directory, row["answer_path"])
            if not path.is_relative_to(directory.resolve()):
                raise ValueError("answer path escapes its cohort")
            artifact = _read_frozen(path, row["answer_file_sha256"])
            if artifact["qa_id"] != row["qa_id"] or artifact["video_id"] != row["video_id"]:
                raise ValueError("answer row identity mismatch")
        return completion
    return generate_cohort(questions_path=questions, index_path=index, output=directory)


def run_phase(config_path, *, config_sha256, phase, release_path=None, release_sha256=None):
    config_path = Path(config_path).resolve()
    base = config_path.parent
    config = _read_frozen(config_path, config_sha256)
    reject_private(config)
    if config.get("schema_version") != "unified-evaluation-workflow-v1":
        raise ValueError("explicit integrated workflow schema required")
    if phase not in {"capture", "answers", "summary"}:
        raise ValueError("unknown evaluation phase")
    arms = config["arms"]
    names = [arm["model"] for arm in arms]
    if (not names or len(names) != len(set(names))
            or any(not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name) for name in names)):
        raise ValueError("unique filesystem-safe model names required")
    questions = _path(base, config["public_questions"])
    if hashlib.sha256(questions.read_bytes()).hexdigest() != config["public_questions_sha256"]:
        raise ValueError("public question file hash mismatch")
    output = _path(base, config["output_directory"])
    output.mkdir(parents=True, exist_ok=True)
    # One owner lock spans all phases and recovery rounds, not just this directory.
    lock_path = _path(base, config["owner_lock_path"])
    with lock_path.open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        _publish_or_match(output / "workflow-identity.json", {
            "workflow_sha256": config_sha256, "experiment_label": config["experiment_label"],
            "frozen_runtime_sha256": config["frozen_runtime_sha256"]})
        owner_config = config["resources"]
        owner_path = _path(base, owner_config["identity_path"])
        owner = _ownership(_read_frozen(owner_path, owner_config["identity_sha256"]))
        if (owner["instance_id"] != owner_config["instance_id"]
                or owner["experiment_label"] != config["experiment_label"]
                or owner["source_manifest_sha256"] != config["frozen_runtime_sha256"]):
            raise ValueError("workflow resource identity mismatch")
        if phase != "capture":
            if release_path is None or release_sha256 is None:
                raise ValueError("postprocessing requires a hash-pinned GPU release receipt")
            release = _read_frozen(release_path, release_sha256)
            if (release.get("confirmed_released") is not True
                    or release.get("gpu_label") != owner["experiment_label"]
                    or owner["instance_id"] not in release.get("release_confirmed_instance_ids", [])
                    or owner["instance_id"] not in release.get("known_instance_ids", [])
                    or release.get("remaining") != [] or release.get("cleanup_errors") != []
                    or release.get("network_containers") != {}
                    or release.get("creation_outcome_unresolved") is not False
                    or release.get("local_confirmed") is not True
                    or release.get("remote_confirmed") is not True
                    or any(row.get("Running") is not False for row in release.get("containers", {}).values())):
                raise ValueError("release receipt does not confirm the exact owned GPU")
        if phase == "answers":
            results = {arm["model"]: _answers(base, config, output, arm) for arm in arms}
            _publish_or_match(output / "answers-completion.json", {
                "workflow_sha256": config_sha256,
                "models_with_saved_observations": [name for name, row in results.items() if row is not None],
                "new_model_calls": 0})
            return
        ledger_config = config["model_ledger"]
        ledger = ModelLedger(output / "model-ledger",
            seed_manifest_path=_path(base, ledger_config["seed_manifest"]),
            seed_manifest_sha256=ledger_config["seed_manifest_sha256"],
            owner_lock_path=output / "ledger-owner.lock")
        try:
            if phase == "capture":
                resources = ExperimentResources(identity_path=owner_path,
                    expected_instance_id=owner["instance_id"], expected_label=config["experiment_label"],
                    expected_source_manifest_sha256=config["frozen_runtime_sha256"],
                    reset_endpoint=owner_config["reset_endpoint"],
                    cancel_paths=[output / "cancel-requested.json", output / "balance-blocked.json"])
                for arm in arms:
                    directory = output / arm["model"]
                    plan = _path(base, arm["plan"])
                    _read_frozen(plan, arm["plan_sha256"])
                    if directory.exists():
                        run = json.loads((directory / "run.json").read_bytes())
                        if run["plan_sha256"] != arm["plan_sha256"]:
                            raise ValueError("capture plan changed")
                        if not (directory / "completion.json").exists():
                            raise RuntimeError("partial capture requires a new frozen recovery plan; do not replay frames")
                        continue
                    run_sources(plan, plan_sha256=arm["plan_sha256"], output=directory,
                                ledger=ledger, resources=resources)
            else:
                if config.get("defer_postprocessing", False):
                    raise RuntimeError("this workflow explicitly defers final-model answers")
                settings = ModelAPISettings.from_mapping(config["summary"]["settings"])
                endpoint = urlparse(settings.endpoint)
                if (settings.max_retries != 0 or settings.inspection_endpoint
                        or endpoint.scheme != "https" or endpoint.username or endpoint.password
                        or endpoint.query or endpoint.fragment):
                    raise ValueError("final model requires an explicit HTTPS route without hidden retries")
                client = ModelAPIClient(settings, capture_error_details=True)
                for arm in arms:
                    completion = _answers(base, config, output, arm)
                    if completion is None:
                        continue
                    answer_dir = output / (arm["model"] + "-answers")
                    for row in completion["rows"]:
                        artifact = answer_dir / row["answer_path"]
                        value = _read_frozen(artifact, row["answer_file_sha256"])
                        if "evidence_sha256" not in value:
                            continue
                        execute_summary(artifact, artifact_sha256=row["answer_file_sha256"],
                            output=output / (arm["model"] + "-summary") / artifact.stem,
                            client=client, ledger=ledger, model=config["summary"]["model"],
                            parameters=config["summary"]["parameters"],
                            max_input_bytes=config["summary"]["max_input_bytes"],
                            context_budget=config["summary"].get("context_budget"))
            _publish_or_match(output / (phase + "-completion.json"), {
                "phase": phase, "workflow_sha256": config_sha256,
                "processed_not_all_answered": True})
        finally:
            ledger.close()


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("phase", choices=("capture", "answers", "summary"))
    cli.add_argument("--config", type=Path, required=True)
    cli.add_argument("--config-sha256", required=True)
    cli.add_argument("--release", type=Path)
    cli.add_argument("--release-sha256")
    args = cli.parse_args()
    run_phase(args.config, config_sha256=args.config_sha256, phase=args.phase,
              release_path=args.release, release_sha256=args.release_sha256)


if __name__ == "__main__":
    main()
