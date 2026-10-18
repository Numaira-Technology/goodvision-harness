"""One answer entry: native streaming capture, then released-resource answering and scoring."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import re
from urllib.parse import urlparse

from ..model_api import ModelAPIClient, ModelAPISettings
from .batch import _ownership
from .capture_source import _read_frozen
from .cohort import generate_cohort
from .evidence import _digest
from .model_ledger import ModelLedger
from .native_stream import NativeSourceFailure, capture_native_stream
from .privacy import reject_private
from .queue import _publish_or_match
from .recording import publish_once
from .resources import ExperimentResources
from .scoring import score_cohort
from .summary import execute_summary


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _released(owner, receipt):
    if (receipt.get("confirmed_released") is not True
            or receipt.get("gpu_label") != owner["experiment_label"]
            or owner["instance_id"] not in receipt.get("release_confirmed_instance_ids", [])
            or owner["instance_id"] not in receipt.get("known_instance_ids", [])
            or receipt.get("remaining") != [] or receipt.get("cleanup_errors") != []
            or receipt.get("network_containers") != {}
            or receipt.get("creation_outcome_unresolved") is not False
            or receipt.get("local_confirmed") is not True
            or receipt.get("remote_confirmed") is not True
            or any(row.get("Running") is not False for row in receipt.get("containers", {}).values())):
        raise ValueError("postprocessing requires confirmed release of the exact owned GPU")


def _saved_capture(directory, config_sha256):
    directory = Path(directory)
    if (directory / "failure.json").exists():
        raise RuntimeError("interrupted source requires a new frozen recovery identity; preserve all existing records")
    outcome = json.loads((directory / "outcome.json").read_bytes())
    completion = json.loads((directory / "completion.json").read_bytes())
    manifest = json.loads((directory / "manifest.json").read_bytes())
    cleanup = json.loads((directory / "cleanup.json").read_bytes())
    evidence = _read_frozen(directory / "capture.json", completion["capture_file_sha256"])
    previous = None
    records = sorted((directory / "records").glob("*.json"))
    for position, record_path in enumerate(records, 1):
        record = json.loads(record_path.read_bytes())
        if record.get("sequence") != position or record.get("previous_record_sha256") != previous:
            raise ValueError("saved native record chain changed")
        previous = _sha(record_path)
    if (len(records) != completion["record_count"] or previous != completion["last_record_sha256"]
            or _digest({key: value for key, value in evidence.items() if key != "evidence_sha256"}) != evidence["evidence_sha256"]
            or completion["evidence_sha256"] != evidence["evidence_sha256"]
            or any(manifest[key] != evidence[key] for key in
                   ("video_id", "stream_session_id", "source_id", "source", "provenance"))):
        raise ValueError("saved native evidence identity or record-chain digest changed")
    if (outcome.get("status") not in {"captured", "incomplete"}
            or outcome.get("config_sha256") != config_sha256
            or manifest["provenance"].get("run_config_sha256") != config_sha256
            or outcome["evidence_sha256"] != evidence["evidence_sha256"]
            or cleanup.get("shared_failure", False) is not False
            or cleanup.get("fusion_stopped", True) is not True
            or any(cleanup.get(key) is not True for key in
                   ("source_stopped", "tracking_stopped", "vlm_stopped"))):
        raise ValueError("saved native source identity or cleanup is incomplete")
    return outcome, {"video_id": outcome["video_id"],
        "capture_path": str((directory / "capture.json").resolve()),
        "capture_file_sha256": completion["capture_file_sha256"]}


def _source_failure(directory, source, config_sha256, resources):
    """Keep a failed source in the denominator without recycling the owned GPU."""
    directory = Path(directory)
    failure = json.loads((directory / "failure.json").read_bytes())
    cleanup = json.loads((directory / "cleanup.json").read_bytes())
    if (failure.get("failure_scope") != "source"
            or failure.get("config_sha256") != config_sha256
            or failure.get("stream_session_id") != source["stream_session_id"]
            or cleanup.get("shared_failure") is not False
            or cleanup.get("tracking_session_scope") != "stream_session_id"
            or cleanup.get("tracking_requests_settled") is not True
            or any(cleanup.get(key) is not True for key in
                   ("source_stopped", "tracking_stopped", "vlm_stopped", "fusion_stopped"))):
        raise RuntimeError("source-local failure cleanup is not confirmed")
    terminal_path = directory / "source-terminal.json"
    if terminal_path.exists():
        terminal = json.loads(terminal_path.read_bytes())
        if (terminal.get("video_id") != source["video_id"]
                or terminal.get("stream_session_id") != source["stream_session_id"]
                or terminal.get("config_sha256") != config_sha256
                or terminal.get("status") != "skipped"):
            raise ValueError("saved native source failure identity changed")
        return terminal
    # The detector keeps trackers by stream_session_id. With the local request
    # thread joined and no tracking failure, the next source's fresh SID needs
    # no model reset or GPU restart. Stateful SAM is not covered by this path.
    owner = resources.prepare(source)
    terminal = {"video_id": source["video_id"], "stream_session_id": source["stream_session_id"],
        "source_id": source["source_id"], "config_sha256": config_sha256,
        "status": "skipped", "failure_policy": {"scope": "source", "reason": failure["reason"],
            "action": "preserve_and_skip_after_cleanup"},
        "capture_directory": str(directory), "resource_instance_id": owner["instance_id"],
        "continuation": "fresh_stream_session_after_confirmed_local_cleanup",
        "final_answer_generated": False, "preserve_ledger_and_raw_records": True}
    publish_once(terminal_path, terminal)
    return terminal


def run(config_path, *, config_sha256, release_path=None, release_sha256=None):
    path = Path(config_path).resolve()
    base = path.parent
    config = _read_frozen(path, config_sha256)
    reject_private(config)
    if (config.get("schema_version") != "native-answer-workflow-v1"
            or type(config.get("vlm_enabled")) is not bool or config.get("face_enabled") is not False):
        raise ValueError("native answer workflow requires explicit VLM choice and face disabled")
    arms = config["arms"]
    names = [arm["model"] for arm in arms]
    if (not names or len(set(names)) != len(names)
            or any(not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name) for name in names)):
        raise ValueError("unique filesystem-safe model identities required")
    public = base / config["public_questions"]
    if _sha(public) != config["public_questions_sha256"]:
        raise ValueError("public question hash changed")
    questions = [json.loads(line) for line in public.read_bytes().splitlines() if line.strip()]
    reject_private(questions)
    videos = {row["video_id"] for row in questions}
    ledger_config = config.get("model_ledger")
    if config["vlm_enabled"] and (base / ledger_config["owner_lock"]).resolve() == (base / config["owner_lock_path"]).resolve():
        raise ValueError("workflow and spawned model-ledger ownership locks must be distinct")
    plans = {}
    sessions = set()
    for arm in arms:
        seen, entries = set(), []
        for row in arm["sources"]:
            source_path = (base / row["config"]).resolve()
            source = _read_frozen(source_path, row["config_sha256"])
            reject_private(source, observation=True)
            if (source.get("schema_version") != "native-stream-source-v1"
                    or source.get("vlm_enabled") is not config["vlm_enabled"]
                    or source["tracking"]["model"] != arm["model"]
                    or source["stream_session_id"] in sessions
                    or source["video_id"] in seen):
                raise ValueError("source model, mode or cohort identity changed")
            if config["vlm_enabled"]:
                source_ledger = source["model_ledger"]
                for key in ("output_directory", "seed_manifest", "owner_lock"):
                    if (source_path.parent / source_ledger[key]).resolve() != (base / ledger_config[key]).resolve():
                        raise ValueError("all native workers must share the cumulative ledger identity")
                if source_ledger["seed_manifest_sha256"] != ledger_config["seed_manifest_sha256"]:
                    raise ValueError("native worker ledger seed changed")
            seen.add(source["video_id"])
            sessions.add(source["stream_session_id"])
            entries.append((row, source_path, source))
        if seen != videos:
            raise ValueError("each model arm must preserve every public video exactly once")
        plans[arm["model"]] = entries
    root = (base / config["output_directory"]).resolve()
    root.mkdir(parents=True, exist_ok=True)
    with (base / config["owner_lock_path"]).open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        _publish_or_match(root / "workflow-identity.json", {
            "config_sha256": config_sha256, "vlm_enabled": config["vlm_enabled"],
            "experiment_label": config["experiment_label"], "face_enabled": False})
        resource_config = config["resources"]
        identity = base / resource_config["identity_path"]
        owner = _ownership(_read_frozen(identity, resource_config["identity_sha256"]))
        if (owner["instance_id"] != resource_config["instance_id"]
                or owner["experiment_label"] != config["experiment_label"]
                or owner["source_manifest_sha256"] != config["frozen_runtime_sha256"]):
            raise ValueError("native workflow resource identity mismatch")
        resources = ExperimentResources(identity_path=identity,
            expected_instance_id=owner["instance_id"], expected_label=owner["experiment_label"],
            expected_source_manifest_sha256=config["frozen_runtime_sha256"],
            reset_endpoint=resource_config["reset_endpoint"], cancel_paths=[root / "cancel-requested.json"])
        if release_path is None:
            if release_sha256 is not None:
                raise ValueError("release hash supplied without receipt")
            for arm in arms:
                directory = root / arm["model"]
                directory.mkdir(exist_ok=True)
                outcomes, observations = [], []
                for position, (row, source_path, source) in enumerate(plans[arm["model"]], 1):
                    destination = directory / f"{position:04d}"
                    reuse = row.get("reuse")
                    if reuse:
                        destination = (base / reuse["directory"]).resolve()
                        _read_frozen(destination / "completion.json", reuse["completion_sha256"])
                    else:
                        if not destination.exists():
                            try:
                                capture_native_stream(source_path, config_sha256=row["config_sha256"],
                                                      output=destination, resources=resources)
                            except NativeSourceFailure:
                                outcomes.append(_source_failure(destination, source,
                                    row["config_sha256"], resources))
                                continue
                        elif (destination / "source-terminal.json").exists():
                            outcomes.append(_source_failure(destination, source,
                                row["config_sha256"], resources))
                            continue
                    outcome, observation = _saved_capture(destination, row["config_sha256"])
                    if outcome["video_id"] != source["video_id"]:
                        raise ValueError("saved source video mismatch")
                    outcomes.append(outcome)
                    observations.append(observation)
                _publish_or_match(directory / "observation-index.json", {
                    "schema_version": "unified-observation-index-v1",
                    "method": f"native_streaming_{arm['model']}_vlm_{'on' if config['vlm_enabled'] else 'off'}",
                    "observations": observations, "processed_sources": outcomes})
            _publish_or_match(root / "capture-completion.json", {
                "config_sha256": config_sha256, "models": names,
                "next_action": "outer_owner_releases_gpu_then_reenters_this_command_with_release_receipt"})
            return {"state": "capture_finished_awaiting_resource_release", "output": str(root)}
        if release_sha256 is None:
            raise ValueError("hash-pinned resource release receipt required")
        _released(owner, _read_frozen(release_path, release_sha256))
        done = json.loads((root / "capture-completion.json").read_bytes())
        if done["config_sha256"] != config_sha256:
            raise ValueError("capture completion belongs to another workflow")
        if config.get("defer_postprocessing", False):
            raise RuntimeError("this frozen workflow defers final answering and scoring")
        ledger = client = None
        try:
            if config["vlm_enabled"]:
                settings = ModelAPISettings.from_mapping(config["final_model"]["settings"])
                endpoint = urlparse(settings.endpoint)
                if (settings.max_retries != 0 or settings.inspection_endpoint
                        or endpoint.scheme != "https" or endpoint.username or endpoint.password
                        or endpoint.query or endpoint.fragment):
                    raise ValueError("explicit final-model HTTPS route without hidden retries required")
                ledger = ModelLedger(base / ledger_config["output_directory"],
                    seed_manifest_path=base / ledger_config["seed_manifest"],
                    seed_manifest_sha256=ledger_config["seed_manifest_sha256"],
                    owner_lock_path=base / ledger_config["owner_lock"])
                client = ModelAPIClient(settings)
            for arm in arms:
                directory = root / arm["model"]
                index = directory / "observation-index.json"
                answers = directory / "answers"
                if answers.exists():
                    previous = json.loads((answers / "run.json").read_bytes())
                    if (previous["questions_sha256"] != config["public_questions_sha256"]
                            or previous["observation_index_sha256"] != _sha(index)):
                        raise ValueError("answer inputs changed")
                    completion = json.loads((answers / "completion.json").read_bytes())
                else:
                    completion = generate_cohort(questions_path=public, index_path=index, output=answers)
                for row in completion["rows"]:
                    artifact_path = (answers / row["answer_path"]).resolve()
                    if not artifact_path.is_relative_to(answers):
                        raise ValueError("answer path escapes the cohort")
                    artifact = _read_frozen(artifact_path, row["answer_file_sha256"])
                    no_observation = artifact["summary_request"] == {
                        "state": "not_called", "reason": "no_saved_observation"}
                    if config["vlm_enabled"] and not no_observation:
                        final = config["final_model"]
                        execute_summary(artifact_path, artifact_sha256=row["answer_file_sha256"],
                            output=directory / "model-answers" / artifact_path.stem,
                            client=client, ledger=ledger, model=final["model"],
                            parameters=final["parameters"], max_input_bytes=final["max_input_bytes"])
                scoring = config["scoring"]
                for mode in (("rule", "model") if config["vlm_enabled"] else ("rule",)):
                    result = directory / f"scores-{mode}.json"
                    cohort = answers / "completion.json"
                    if result.exists():
                        previous = json.loads(result.read_bytes())
                        expected = {"questions_sha256": config["public_questions_sha256"],
                            "truth_sha256": scoring["truth_sha256"], "cohort_sha256": _sha(cohort), "mode": mode}
                        if any(previous.get(key) != value for key, value in expected.items()):
                            raise ValueError("existing score input identity changed")
                        continue
                    score_cohort(questions_path=public, questions_sha256=config["public_questions_sha256"],
                        truth_path=base / scoring["truth"], truth_sha256=scoring["truth_sha256"],
                        cohort_path=cohort, cohort_sha256=_sha(cohort), mode=mode,
                        summary_root=directory / "model-answers" if mode == "model" else None,
                        scorer_path=base / scoring["scorer"], output=result)
            _publish_or_match(root / "answer-completion.json", {
                "config_sha256": config_sha256, "release_sha256": release_sha256,
                "models": names, "vlm_enabled": config["vlm_enabled"],
                "all_questions_retained": True, "historical_answers_modified": False})
            return {"state": "answers_and_scores_saved", "output": str(root)}
        finally:
            if ledger:
                ledger.close()


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--config", required=True, type=Path)
    cli.add_argument("--config-sha256", required=True)
    cli.add_argument("--release", type=Path)
    cli.add_argument("--release-sha256")
    args = cli.parse_args()
    print(json.dumps(run(args.config, config_sha256=args.config_sha256,
                         release_path=args.release, release_sha256=args.release_sha256)))


if __name__ == "__main__":
    main()
