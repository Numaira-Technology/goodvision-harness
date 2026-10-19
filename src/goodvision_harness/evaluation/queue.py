"""Run frozen capture, saved-evidence answers, or final-model answers as separate phases."""
from __future__ import annotations

import argparse
import fcntl
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
    from .cohort import observation_index_record
    questions = _path(base, config["public_questions"])
    index = output / arm["model"] / "observation-index.json"
    directory = output / (arm["model"] + "-answers")
    if not index.exists():
        return None
    expected = {"public_questions": [json.loads(line) for line in questions.read_bytes().splitlines()
                                     if line.strip()],
                "observation_index_path": str(index.resolve()),
                "observation_index": observation_index_record(index)}
    if directory.exists():
        run = json.loads((directory / "run.json").read_bytes())
        if any(run.get(k) != v for k, v in expected.items()):
            raise ValueError("answer inputs changed; use an independent output directory")
        completion = json.loads((directory / "completion.json").read_bytes())
        for row in completion["rows"]:
            path = _path(directory, row["answer_path"])
            if not path.is_relative_to(directory.resolve()):
                raise ValueError("answer path escapes its cohort")
            artifact = _read_frozen(path)
            if artifact["qa_id"] != row["qa_id"] or artifact["video_id"] != row["video_id"]:
                raise ValueError("answer row identity mismatch")
        return completion
    return generate_cohort(questions_path=questions, index_path=index, output=directory)


def run_phase(config_path, *, phase, release_path=None, config_sha256=None, release_sha256=None):
    config_path = Path(config_path).resolve()
    base = config_path.parent
    config = _read_frozen(config_path)
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
    output = _path(base, config["output_directory"])
    output.mkdir(parents=True, exist_ok=True)
    # One owner lock spans all phases and recovery rounds, not just this directory.
    lock_path = _path(base, config["owner_lock_path"])
    with lock_path.open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        _publish_or_match(output / "workflow-identity.json", {
            "config_path": str(config_path), "configuration": config,
            "experiment_label": config["experiment_label"]})
        owner_config = config["resources"]
        owner_path = _path(base, owner_config["identity_path"])
        owner = _ownership(_read_frozen(owner_path))
        if (owner["instance_id"] != owner_config["instance_id"]
                or owner["experiment_label"] != config["experiment_label"]):
            raise ValueError("workflow resource identity mismatch")
        if phase != "capture":
            if release_path is None:
                raise ValueError("postprocessing requires an exact GPU release receipt")
            release = _read_frozen(release_path)
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
                "config_path": str(config_path),
                "models_with_saved_observations": [name for name, row in results.items() if row is not None],
                "new_model_calls": 0})
            return
        ledger_config = config["model_ledger"]
        ledger = ModelLedger(output / "model-ledger",
            seed_manifest_path=_path(base, ledger_config["seed_manifest"]),
            owner_lock_path=output / "ledger-owner.lock")
        try:
            if phase == "capture":
                resources = ExperimentResources(identity_path=owner_path,
                    expected_instance_id=owner["instance_id"], expected_label=config["experiment_label"],
                    reset_endpoint=owner_config["reset_endpoint"],
                    cancel_paths=[output / "cancel-requested.json", output / "balance-blocked.json"])
                for arm in arms:
                    directory = output / arm["model"]
                    plan = _path(base, arm["plan"])
                    planned = _read_frozen(plan)
                    if directory.exists():
                        run = json.loads((directory / "run.json").read_bytes())
                        if run.get("plan") != planned:
                            raise ValueError("capture plan changed")
                        if not (directory / "completion.json").exists():
                            raise RuntimeError("partial capture requires a new frozen recovery plan; do not replay frames")
                        continue
                    run_sources(plan, output=directory,
                                ledger=ledger, resources=resources)
            else:
                # Legacy duration-review flags do not defer original-question answers.
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
                        value = _read_frozen(artifact)
                        if "messages" not in value.get("summary_request", {}):
                            continue
                        execute_summary(artifact,
                            output=output / (arm["model"] + "-summary") / artifact.stem,
                            client=client, ledger=ledger, model=config["summary"]["model"],
                            parameters=config["summary"]["parameters"],
                            max_input_bytes=config["summary"]["max_input_bytes"],
                            context_budget=config["summary"].get("context_budget"))
            _publish_or_match(output / (phase + "-completion.json"), {
                "phase": phase, "config_path": str(config_path),
                "processed_not_all_answered": True})
        finally:
            ledger.close()


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("phase", choices=("capture", "answers", "summary"))
    cli.add_argument("--config", type=Path, required=True)
    cli.add_argument("--config-sha256", help=argparse.SUPPRESS)
    cli.add_argument("--release", type=Path)
    cli.add_argument("--release-sha256", help=argparse.SUPPRESS)
    args = cli.parse_args()
    run_phase(args.config, phase=args.phase, release_path=args.release)


if __name__ == "__main__":
    main()
