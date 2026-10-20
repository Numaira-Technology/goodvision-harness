"""Own a frozen authorized event-timing cohort with native admission policies.

Preparation is offline. Launch owns one standard 24GB RTX 4090 and releases it
before any offline timing analysis. There is no final-answer or GT input phase.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

from saved_streaming_recovery import (
    LEDGER_FILES, copy_ledger, freeze_source, git, load, require_finished, rows, save,
)


CLIPS = ("0018", "0028", "0002")
PACED_INPUT_CLIPS = ("0040", "0003", "0030")
PHYSICAL_CONFLICT_CLIPS = tuple(f"{index:04d}" for index in range(1, 51))
EVENT_TRACKING_MODES = {"physical_conflict": ["fight"],
                        "native_four_events": ["fight", "smoking", "fall", "weapon"]}
PAIRED_SOURCE_KEYS = ("schema_version", "video_id", "source_id", "source", "media_path",
    "face_enabled", "vlm_enabled", "question_conditioned", "evaluation_timing",
    "sam_internal_patch", "coverage_policy", "max_source_wall_seconds",
    "drain_timeout_seconds", "tracking", "vlm")


def import_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def resources(root):
    control = import_file("walltime_resources", root / "resource-control.py")
    control.ROOT, control.PREFIX, control.OUTPUT = root, root.name, root / "output"
    original = control.docker

    def metadata_only(*args, **kwargs):
        if args and args[0] == "inspect" and "--format" in args:
            args = list(args)
            position = args.index("--format") + 1
            if args[position] == "{{json .}}":
                args[position] = ('{"Config":{"Labels":{"experiment":'
                    '{{json (index .Config.Labels "experiment")}}}},"State":{'
                    '"Running":{{json .State.Running}},"ExitCode":{{json .State.ExitCode}},'
                    '"OOMKilled":{{json .State.OOMKilled}}}}')
        return original(*args, **kwargs)

    control.docker = metadata_only
    return control


def event_recovery(previous, *, last_answer, staging, replay_clips):
    """Reference completed recordings and replay only explicitly approved sources."""
    require_finished(previous, last_answer)
    plan, outcome = load(previous / "plan.json"), load(previous / "round-outcome.json")
    if (plan.get("capture_mode") != "native_four_events"
            or plan.get("cohort") != "physical_conflict50"
            or tuple(plan.get("clips", ())) != PHYSICAL_CONFLICT_CLIPS
            or tuple(entry["clip"] for entry in plan.get("sources", ())) != PHYSICAL_CONFLICT_CLIPS
            or outcome.get("capture_completed") is not False
            or plan.get("paired_comparison", {}).get("baseline_round")
               != staging.get("comparison_baseline_round")):
        raise ValueError("Recovery requires the stopped paired four-event cohort")
    approval = staging.get("replay_approval_quote")
    if replay_clips and (not isinstance(approval, str) or not approval.strip()):
        raise ValueError("Interrupted-source replay requires its explicit human approval")
    if len(replay_clips) != len(set(replay_clips)):
        raise ValueError("Do not authorize the same replay twice")
    started, inherited, interrupted = set(), {}, {}
    for entry, selection in zip(plan["sources"], staging["selections"]):
        clip = entry["clip"]
        if any(entry[key] != selection[key] for key in ("clip", "video_id", "media_path")):
            raise ValueError("Recovery must retain the exact fixed-cohort member mapping")
        directory = previous / "output/capture/yolo26-s" / clip
        manifest_path = directory / "manifest.json"
        if not manifest_path.exists():
            if directory.exists():
                raise ValueError("An unstarted member has an unexplained recording directory")
            continue
        source, manifest = load(entry["config"]), load(manifest_path)
        baseline = load(selection["source_configuration_reference"])
        if any(source.get(key) != baseline.get(key) for key in PAIRED_SOURCE_KEYS):
            raise ValueError("Prior event source settings differ from the paired baseline")
        if (manifest.get("video_id") != entry["video_id"]
                or manifest.get("source_id") != source["source_id"]
                or manifest.get("stream_session_id") != source["stream_session_id"]):
            raise ValueError("Recovery recording and source identities differ")
        cleanup = load(directory / "cleanup.json")
        if any(cleanup.get(key) is not True for key in
               ("source_stopped", "tracking_stopped", "vlm_stopped", "fusion_stopped")):
            raise ValueError("Prior source workers have not stopped")
        started.add(clip)
        completion_path = directory / "completion.json"
        completion = load(completion_path) if completion_path.exists() else {}
        complete = (completion.get("execution_complete") is True
            and all(completion.get(key) is True for key in
                    ("source_eof", "tracking_drained", "events_drained", "fusion_drained")))
        if complete:
            if clip in replay_clips or cleanup.get("shared_failure") is True:
                raise ValueError("A completed source cannot be selected for replay")
            inherited[clip] = {"entry": entry, "directory": str(directory),
                "outcome": load(directory / "outcome.json"),
                "source_commit": manifest.get("provenance", {}).get("harness_commit")}
        else:
            if clip not in replay_clips or not (directory / "failure.json").exists():
                raise ValueError("An interrupted source lacks an explicit replay decision")
            interrupted[clip] = {"directory": str(directory),
                "prior_session": source["stream_session_id"], "source_id": source["source_id"],
                "approval_quote": approval, "old_receipts_and_unknown_costs_preserved": True}
    if set(replay_clips) != set(interrupted):
        raise ValueError("Replay approval must identify exactly the interrupted sources")
    dispatched = {row.get("context", {}).get("source_id")
                  for row in rows(previous / "output/model-ledger/model-requests.jsonl")}
    if not dispatched <= {"clip-" + clip for clip in started}:
        raise ValueError("A paid source has no reviewed prior recording")
    if len(inherited) == len(PHYSICAL_CONFLICT_CLIPS):
        raise ValueError("There is no uncompleted source to continue")
    return {"prior_round": str(previous), "inherited": inherited,
        "replayed": interrupted, "newly_captured_clips": [clip for clip in PHYSICAL_CONFLICT_CLIPS
            if clip not in inherited], "prior_capture_commit": plan["source_commit"],
        "same_cohort_order_and_observer_settings": True}


def prepare(args):
    root, source, last = args.root.resolve(), args.source_round.resolve(), args.last_answer.resolve()
    staging = load(root / "staging.json")
    mode = staging.get("capture_mode", "physical_conflict")
    if mode not in EVENT_TRACKING_MODES:
        raise ValueError("New timing preparations require an approved native event profile")
    evaluation_timing = staging.get("evaluation_timing", False)
    if type(evaluation_timing) is not bool:
        raise ValueError("evaluation_timing must be an explicit boolean")
    cohort = staging.get("cohort", "three_video")
    if cohort not in {"three_video", "physical_conflict50"}:
        raise ValueError("Unknown authorized event-timing cohort")
    if cohort == "physical_conflict50" and not evaluation_timing:
        raise ValueError("The 50-video event cohort requires actual input timing")
    if mode == "native_four_events" and cohort != "physical_conflict50":
        raise ValueError("The four-event control is authorized only for the complete fixed 50")
    clips = (PHYSICAL_CONFLICT_CLIPS if cohort == "physical_conflict50" else
             PACED_INPUT_CLIPS if evaluation_timing else CLIPS)
    if tuple(row["clip"] for row in staging["selections"]) != clips:
        raise ValueError("The source list must match the authorized cohort in member order")
    if cohort == "physical_conflict50" and not staging.get("approval_quote"):
        raise ValueError("Preserve the explicit 50-video human approval in staging")
    prior_timing = args.prior_timing_round.resolve() if args.prior_timing_round else None
    if evaluation_timing and prior_timing is None:
        raise ValueError("The instrumented run requires the prior timing resource record")
    if prior_timing is not None:
        if (load(prior_timing / "round-outcome.json")["cleanup"] != "confirmed_released"
                or load(prior_timing / "resources-after.json")["confirmed_released"] is not True):
            raise ValueError("Release the prior timing resources before preparing another run")
    paired = None
    if mode == "native_four_events":
        baseline_plan = load(prior_timing / "plan.json")
        baseline_outcome = load(prior_timing / "round-outcome.json")
        baseline_capture = load(prior_timing / "capture-completion.json")["result"]["sources"]
        if (Path(staging["comparison_baseline_round"]).resolve() != prior_timing
                or baseline_plan.get("capture_mode") != "physical_conflict"
                or baseline_plan.get("event_labels") != ["fight"]
                or baseline_plan.get("cohort") != "physical_conflict50"
                or tuple(baseline_plan["clips"]) != PHYSICAL_CONFLICT_CLIPS
                or len(baseline_plan["sources"]) != len(PHYSICAL_CONFLICT_CLIPS)
                or baseline_outcome.get("capture_completed") is not True
                or [row["video_id"] for row in baseline_capture]
                   != [row["video_id"] for row in baseline_plan["sources"]]):
            raise ValueError("Complete the exact single-fight 50 baseline before its paired control")
        require_finished(prior_timing, last)
        for row, baseline in zip(staging["selections"], baseline_plan["sources"]):
            previous = load(baseline["config"])
            if (row["clip"] != baseline["clip"] or row["video_id"] != baseline["video_id"]
                    or row["media_path"] != baseline["media_path"]
                    or Path(row["source_configuration_reference"]).resolve() != Path(baseline["config"]).resolve()
                    or row["source"] != previous["source"]
                    or row["tracking"] != previous["tracking"]
                    or row["duration_seconds"] != previous["source"]["duration_seconds"]):
                raise ValueError("The paired control must reuse each baseline media and source configuration")
        paired = {"baseline_round": str(prior_timing), "baseline_commit": baseline_plan["source_commit"],
            "baseline_capture_completion": str(prior_timing / "capture-completion.json"),
            "baseline_resource_release": str(prior_timing / "resources-after.json"),
            "baseline_harness_config": str(prior_timing / "harness.yaml"),
            "controlled_change": {"event_labels_before": ["fight"], "event_labels_after": EVENT_TRACKING_MODES[mode]},
            "unchanged_source_keys": list(PAIRED_SOURCE_KEYS), "scored_event_labels": ["fight"],
            "actual_provider_and_native_admission_may_vary": True}
    if any((root / name).exists() for name in ("prepare-intent.json", "ready.json", "launch.json")):
        raise ValueError("This experiment already has a preparation or launch")
    resumed = args.resume_event_round.resolve() if args.resume_event_round else None
    replay_clips = args.replay_interrupted_clip or []
    if (resumed is not None and (mode != "native_four_events" or cohort != "physical_conflict50")):
        raise ValueError("Event recovery is restricted to the approved four-event cohort")
    if resumed is None and replay_clips:
        raise ValueError("Replay selection requires the prior event round")
    recovery = (event_recovery(resumed, last_answer=last, staging=staging, replay_clips=replay_clips)
                if resumed is not None else None)
    release = require_finished(source, last)
    if load(last / "answer-run-outcome.json").get("resource_state") != "exited_removed":
        raise ValueError("Release the prior CPU answer container first")
    save(root / "prepare-intent.json", {"epoch": time.time(), "clips": clips,
        "model_calls": 0, "gpu_creations": 0, "source_commit": args.commit})
    commit = freeze_source(root, args.worktree.resolve(), args.commit)
    source_privacy = import_file("walltime_source_privacy",
        root / "checkout/src/goodvision_harness/evaluation/privacy.py")
    with (root / "checkout/tools/streaming_walltime_trial.py").open("xb") as target:
        target.write(git(args.worktree, "show", commit + ":tools/streaming_walltime_trial.py"))
    copy_ledger(root, resumed if resumed is not None else last)
    settings_source = resumed if resumed is not None else prior_timing if paired is not None else source
    shutil.copytree(settings_source / "gpu/source", root / "gpu/source",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "execution-authorization.json"))
    shutil.copyfile(settings_source / "harness.yaml", root / "harness.yaml")
    (root / "cohort").mkdir()
    shutil.copyfile(settings_source / "cohort/seed-weights.json", root / "cohort/seed-weights.json")
    adapter = root.parent / "detector-rf-count36-1288-resume5-20261016"
    shutil.copyfile(adapter / "control.py", root / "resource-control.py")
    shutil.copyfile(adapter / "driver.py", root / "resource-release.py")
    for name in ("logs", "gpu/output", "output"):
        (root / name).mkdir(parents=True, exist_ok=True)
    if recovery is not None:
        live = root / "output/model-ledger"
        live.mkdir()
        for name in LEDGER_FILES:
            shutil.copyfile(resumed / "output/model-ledger" / name, live / name)
        recordings = root / "output/capture/yolo26-s"
        recordings.mkdir(parents=True)
        for clip, reference in recovery["inherited"].items():
            (recordings / clip).symlink_to(reference["directory"], target_is_directory=True)
        save(root / "event-recovery.json", recovery)
    previous_authorization = load((resumed or prior_timing) / "gpu/source/execution-authorization.json"
        if prior_timing is not None else source / "authorization.json")
    approval = {"status": "explicitly_approved", "approved_by": "human_user",
        "round": root.name, "approval_quote": staging.get("approval_quote") or
        ("按此设计实现" if evaluation_timing else
         "尝试小规模测试 wall time；实在不行可以先做physical conflict"),
        "approval_reference": "Current chat: complete the single-fight 50 first, then enable "
        "fight, smoking, fall and weapon on the same 50 with other settings unchanged; compare fight outcomes."
        if mode == "native_four_events" else "Current chat: explicitly approved the complete fixed 50-video "
        "physical-conflict cohort with the established fight-only version and actual reaction timing."
        if cohort == "physical_conflict50" else
        "Current chat: approved paced-input and readable-report timing; "
        "F_8 and two annotated negatives after prerequisite deliverables and push."
        if evaluation_timing else "Current chat: authorized small wall-time trial, physical conflict first.",
        "maximum_new_create_attempts": None, "maximum_create_attempts_this_round": 1,
        "creation_budget_id": previous_authorization["creation_budget_id"],
        "new_create_attempts_used_before_round": previous_authorization["new_create_attempts_used_before_round"] + 1,
        "required_gpu_name": "RTX 4090", "maximum_simultaneous_gpu_instances": 1,
        "bounded_mask_area": False, "deadline_epoch": None, "gpu_actual_cap_usd": None,
        "scope": "native_four_events50_paired_walltime" if mode == "native_four_events"
        else "physical_conflict50_native_streaming_walltime" if cohort == "physical_conflict50"
        else "three_video_native_streaming_walltime", "clips": clips,
        "face_enabled": False, "vlm_enabled": True,
        "capture_mode": mode, "event_labels": EVENT_TRACKING_MODES[mode]}
    save(root / "gpu/source/execution-authorization.json", approval)
    history = load((resumed or prior_timing) / "gpu/output/prelaunch-prior-resources.json"
                   if prior_timing is not None else source / "gpu/source/prior-resources.json")
    deployment, cost = load(source / "gpu/output/deployment.json"), load(source / "gpu/output/gpu-cost.json")
    if deployment["instance_id"] not in {row["instance_id"] for row in history["instances"]}:
        history["instances"].append({"instance_id": deployment["instance_id"], "label": deployment["label"],
            "started_epoch": deployment["started_epoch"], "stopped_epoch": release["checked_epoch"],
            "quoted_dph": str(deployment["offer"]["dph_total"]),
            "minimum_observed_charge_usd": str(cost["current_known_charge_floor_usd"]),
            "actual_charge_usd": cost.get("actual_charge_usd"), "released": True,
            "unknown_cost_is_not_zero": True, "final_invoice": False})
    if prior_timing is not None:
        resource_prior = resumed or prior_timing
        prior_deployment = load(resource_prior / "gpu/output/deployment.json")
        prior_cost = load(resource_prior / "gpu/output/gpu-cost.json")
        if prior_deployment["instance_id"] not in {row["instance_id"] for row in history["instances"]}:
            history["instances"].append({
                "instance_id": prior_deployment["instance_id"], "label": prior_deployment["label"],
                "started_epoch": prior_deployment["started_epoch"],
                "stopped_epoch": load(resource_prior / "round-outcome.json")["finished_epoch"],
                "quoted_dph": str(prior_deployment["offer"]["dph_total"]),
                "minimum_observed_charge_usd": str(prior_cost["current_known_charge_floor_usd"]),
                "actual_charge_usd": prior_cost.get("actual_charge_usd"), "released": True,
                "unknown_cost_is_not_zero": True, "final_invoice": False})
    save(root / "gpu/output/prelaunch-prior-resources.json", history)
    ledger = {"output_directory": str(root / "output/model-ledger"),
              "seed_manifest": str(root / "ledger-seed.json"), "owner_lock": str(root / "model-ledger.lock")}
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    entries = []
    for row in staging["selections"]:
        if recovery is not None and row["clip"] in recovery["inherited"]:
            entries.append(recovery["inherited"][row["clip"]]["entry"])
            continue
        previous = load(row["source_configuration_reference"])
        vlm = deepcopy(previous["vlm"])
        vlm["inspector_settings"]["max_tokens"] = 160
        vlm["inspector_settings"]["evaluation_transport"] = {
            "lane_max_tokens": {"events": 160},
            "lane_timeout_seconds": {"events": 5.0}, "legacy_window_budgets": {}}
        configuration = {"schema_version": "native-stream-source-v1", "video_id": row["video_id"],
            "source_id": "clip-" + row["clip"], "stream_session_id": stamp + "-" + row["clip"],
            "source": {**row["source"], "negative_coverage_domain": "first_to_last_decoded_frame_pts",
                       "sampling_limitation": "native_processed_and_documented_skips"},
            "media_path": row["media_path"], "harness_config": str(root / "harness.yaml"),
            "face_enabled": False, "vlm_enabled": True, "question_conditioned": False,
            "capture_mode": mode,
            "evaluation_timing": evaluation_timing,
            "sam_internal_patch": False, "coverage_policy": "native_processed_and_documented_skips_v1",
            "max_source_wall_seconds": max(60, row["duration_seconds"]) + 300,
            "drain_timeout_seconds": previous["drain_timeout_seconds"],
            "tracking": row["tracking"], "vlm": vlm, "model_ledger": dict(ledger),
            "provenance": {"harness_commit": commit,
                "harness_original_baseline": "6c28dddd26bed7a7be50762a4c7dd6b796b2dbe3",
                "method": staging["method"], "original_duration_metadata_preserved": True,
                "tracking_parameters": {"imgsz": 1280, "confidence": 0.1, "max_det": 300},
                "weights": {"filename": "yolo26s.pt", "upstream": "ultralytics/assets release v8.4.0"}}}
        if paired is not None and any(configuration.get(key) != previous.get(key) for key in PAIRED_SOURCE_KEYS):
            raise ValueError("An unchanged paired source setting differs from the single-fight baseline")
        if recovery is not None and row["clip"] in recovery["replayed"]:
            replay = recovery["replayed"][row["clip"]]
            configuration["model_ledger"]["window_replay"] = {
                "approval_quote": replay["approval_quote"], "source_id": configuration["source_id"],
                "prior_session": replay["prior_session"], "new_session": configuration["stream_session_id"]}
            configuration["provenance"]["approved_source_replay"] = replay
        source_privacy.reject_private(configuration, observation=True)
        destination = root / "sources" / (row["clip"] + ".json")
        save(destination, configuration)
        entries.append({"clip": row["clip"], "config": str(destination),
                        "video_id": row["video_id"], "media_path": row["media_path"]})
    plan = {"source_commit": commit, "source_round": str(source), "last_answer": str(last),
        "prior_release_record": release, "sources": entries, "model_ledger": ledger,
        "final_answer_dispatch": False, "ground_truth_to_model": False,
        "resource_adapter_origin": str(adapter), "gpu_source_origin": str(settings_source / "gpu/source"),
        "maximum_create_attempts_this_round": 1, "clips": clips,
        "evaluation_timing": evaluation_timing, "cohort": cohort,
        "total_media_seconds": sum(row["duration_seconds"] for row in staging["selections"]),
        "capture_mode": mode, "event_labels": EVENT_TRACKING_MODES[mode],
        "prior_timing_round": str(prior_timing) if prior_timing is not None else None}
    if paired is not None:
        plan["paired_comparison"] = paired
    if recovery is not None:
        plan["event_recovery"] = recovery
    save(root / "plan.json", plan)
    save(root / "ready.json", {"status": "prepared_not_started", "plan_snapshot": plan,
        "source_commit": commit, "model_calls": 0, "gpu_creations": 0})
    print(json.dumps({"prepared_sources": len(entries), "source_commit": commit,
                      "model_calls": 0, "gpu_creations": 0}), flush=True)


def capture(root):
    plan, owner = load(root / "plan.json"), load(root / "owned-resource.json")
    if plan.get("capture_mode") in EVENT_TRACKING_MODES:
        return capture_events(root, plan, owner)
    # Frozen historical plans retain their original capture workflow.
    from goodvision_harness.evaluation.answer import run
    workflow = {"schema_version": "native-answer-workflow-v1", "vlm_enabled": True,
        "face_enabled": False, "experiment_label": owner["experiment_label"],
        "public_questions": str(root / "public/videos.jsonl"),
        "output_directory": str(root / "output/capture"), "owner_lock_path": str(root / "workflow.lock"),
        "resources": {"identity_path": str(root / "owned-resource.json"),
            "instance_id": owner["instance_id"], "reset_endpoint": "http://detector-remote:8093/v1/reset"},
        "model_ledger": plan["model_ledger"],
        "arms": [{"model": "yolo26-s", "sources": [{"config": row["config"]} for row in plan["sources"]]}]}
    save(root / "workflow.json", workflow)
    result = run(root / "workflow.json")
    save(root / "capture-completion.json", {"finished_epoch": time.time(), "result": result})


def capture_events(root, plan, owner):
    """Run native event tracking directly, without questions or answer exports."""
    from goodvision_harness.evaluation.native_stream import capture_native_stream, NativeSourceFailure
    from goodvision_harness.evaluation.resources import ExperimentResources

    resource_owner = ExperimentResources(identity_path=root / "owned-resource.json",
        expected_instance_id=owner["instance_id"], expected_label=owner["experiment_label"],
        reset_endpoint="http://detector-remote:8093/v1/reset",
        cancel_paths=[root / "output/capture/cancel-requested.json"])
    directory = root / "output/capture/yolo26-s"
    directory.mkdir(parents=True, exist_ok=True)
    outcomes, sessions = [], set()
    with (root / "workflow.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for position, row in enumerate(plan["sources"], 1):
            inherited = plan.get("event_recovery", {}).get("inherited", {}).get(row["clip"])
            if inherited is not None:
                if inherited["entry"] != row:
                    raise ValueError("Inherited event member differs from the frozen recovery plan")
                outcomes.append(inherited["outcome"])
                continue
            source = load(row["config"])
            if (source.get("capture_mode") != plan["capture_mode"]
                    or source["tracking"]["model"] != "yolo26-s"
                    or source["video_id"] != row["video_id"]
                    or source["stream_session_id"] in sessions):
                raise ValueError("physical-conflict source identity differs from its plan")
            for key in ("output_directory", "seed_manifest", "owner_lock"):
                if Path(source["model_ledger"][key]).resolve() != Path(plan["model_ledger"][key]).resolve():
                    raise ValueError("physical-conflict capture must preserve its model ledger")
            sessions.add(source["stream_session_id"])
            destination = directory / (row["clip"] if plan.get("cohort") == "physical_conflict50"
                                       else f"{position:04d}")
            try:
                outcome = capture_native_stream(row["config"], output=destination, resources=resource_owner)
            except NativeSourceFailure:
                cleanup, failure = load(destination / "cleanup.json"), load(destination / "failure.json")
                if (failure.get("failure_scope") != "source"
                        or failure.get("stream_session_id") != source["stream_session_id"]
                        or cleanup.get("shared_failure") is not False
                        or cleanup.get("tracking_requests_settled") is not True
                        or cleanup.get("tracking_session_scope") != "stream_session_id"
                        or any(cleanup.get(key) is not True for key in
                               ("source_stopped", "tracking_stopped", "vlm_stopped", "fusion_stopped"))):
                    raise RuntimeError("source-local event capture cleanup is not confirmed")
                resource_owner.prepare(source)
                outcome = {"video_id": source["video_id"], "status": "skipped",
                    "failure_scope": "source", "reason": failure["reason"],
                    "capture_directory": str(destination), "final_answer_generated": False}
                save(destination / "source-terminal.json", outcome)
            outcomes.append(outcome)
        save(root / "capture-completion.json", {"finished_epoch": time.time(),
            "result": {"scope": plan["capture_mode"] + "_event_tracking", "sources": outcomes,
                       "final_answer_generated": False, "answer_evidence_exported": False}})


def start_gpu(c):
    from vastai import VastAI
    c.docker("run", "-d", "--name", c.PREFIX + "-detector", "--label", "experiment=" + c.PREFIX,
        "--network", c.NETWORK, "--network-alias", "detector-remote", "--restart", "no", "--init",
        "--read-only", "--no-healthcheck", "--tmpfs", "/tmp:size=128m", "--stop-timeout", "240",
        "-e", "VAST_API_KEY", "-e", "SAM_SSH_KEY=/run/secrets/sam_ssh_key",
        "-e", "DETECTOR_INSTANCE_LABEL=goodvision-" + c.PREFIX, "-e", "DETECTOR_DEADLINE_EPOCH=",
        "-e", "DETECTOR_GPU_PRIOR_USD=0", "-e", "PYTHONDONTWRITEBYTECODE=1",
        "-v", c.ENV["SAM_SSH_KEY"] + ":/run/secrets/sam_ssh_key:ro",
        "-v", str(c.ROOT / "gpu/source") + ":/app:ro", "-v", str(c.ROOT / "gpu/output") + ":/data",
        "-v", str(c.ROOT / "cohort/seed-weights.json") + ":/weights/seed-weights.json:ro",
        "--entrypoint", "python", c.GPU_IMAGE, "-B", "-u", "/app/tools/start_detector_eval.py")
    c.update(phase="preparing_detector")
    deadline = time.time() + 7200
    while time.time() < deadline:
        c.stop_loss()
        path = c.ROOT / "gpu/output/deployment.json"
        if path.exists() and c.load(path).get("state") == "application_started":
            deployment = c.load(path)
            vast = VastAI(api_key=c.ENV["VAST_API_KEY"], retry=1)
            vast.client.timeout = (5, 20)
            owned = [row for row in c.inventory(vast) if row.get("id") == deployment["instance_id"]
                     and row.get("label") == "goodvision-" + c.PREFIX]
            if (len(owned) != 1 or owned[0].get("gpu_name") != "RTX 4090"
                    or int(owned[0].get("num_gpus", 0)) != 1
                    or not 23000 <= float(owned[0].get("gpu_ram", 0)) <= 26000):
                raise RuntimeError("The exact owned GPU does not match the authorized single 4090")
            save(c.ROOT / "owned-resource.json", {"owned": True, "instance_id": deployment["instance_id"],
                "experiment_label": deployment["label"], "gpu_name": "RTX 4090",
                "gpu_memory_mib": int(owned[0]["gpu_ram"]), "num_gpus": 1, "sam_internal_patch": False,
                "checked_epoch": time.time(), "identity_source": "exact_provider_inventory"})
            c.update(phase="detector_application_started", instance_id=deployment["instance_id"])
            return
        time.sleep(5)
    raise RuntimeError("Deployment exceeded its existing setup time limit")


def run_capture(c, plan):
    root, name = c.ROOT, c.PREFIX + "-capture"
    args = ["run", "-d", "--name", name, "--label", "experiment=" + c.PREFIX,
        "--network", c.NETWORK, "--read-only", "--no-healthcheck", "--init", "--restart", "no",
        "--tmpfs", "/tmp:size=256m", "-w", str(root), "-e", "PYTHONDONTWRITEBYTECODE=1",
        "-e", "RICH_LABEL_ENABLED=true", "-e", "PERSON_TRACKING_ENABLED=true",
        "-e", "RICH_LABEL_BACKEND=paddle", "-e", "HARNESS_DRY_RUN=false",
        "-e", "HARNESS_AUTOSTART=false", "-e", "HARNESS_STREAM_URL=", "-e", "OPENROUTER_API_KEY",
        "-e", "INFERENCE_PROVIDER=openrouter", "-e", "INFERENCE_MODEL=qwen/qwen3-vl-8b-instruct",
        "-e", "HARNESS_INFERENCE_PROFILE=", "-e", "PYTHONPATH=" + str(root / "checkout/src"),
        "-v", str(root) + ":" + str(root)]
    items = ["checkout", "gpu/source", "seed-ledger", "sources", "harness.yaml"]
    if plan.get("capture_mode") not in EVENT_TRACKING_MODES:
        items += ["public", "event-definitions.json"]
    for item in items:
        args += ["-v", str(root / item) + ":" + str(root / item) + ":ro"]
    for row in plan["sources"]:
        args += ["-v", row["media_path"] + ":" + row["media_path"] + ":ro"]
    if plan.get("event_recovery"):
        previous = plan["event_recovery"]["prior_round"]
        args += ["-v", previous + ":" + previous + ":ro"]
    args += ["--entrypoint", "python", c.IMAGE, "-B", "-u",
             str(root / "checkout/tools/streaming_walltime_trial.py"), "capture", "--root", str(root)]
    c.docker(*args)
    c.update(phase="native_four_events50_capture" if plan.get("capture_mode") == "native_four_events"
             else "physical_conflict50_native_capture" if plan.get("cohort") == "physical_conflict50"
             else "three_video_native_capture")
    while True:
        c.stop_loss()
        state = c.owned(name)
        if state is None:
            raise RuntimeError("The exact capture container disappeared")
        if not state["State"]["Running"]:
            save(root / "capture-container-outcome.json", state["State"])
            if state["State"]["ExitCode"] != 0 or state["State"]["OOMKilled"]:
                raise RuntimeError("Capture ended unsuccessfully; preserve records without automatic replay")
            return
        time.sleep(5)


def driver(root):
    from dotenv import dotenv_values
    locks = [os.fdopen(int(fd), "a") for fd in os.environ["WALLTIME_LOCK_FDS"].split(",")]
    c, plan = resources(root), load(root / "plan.json")
    credentials = {**dotenv_values(c.WORKSPACE / "goodvision-harness/.env"),
                   **dotenv_values(Path.home() / ".config/goodvision/openrouter.env")}
    for key in ("VAST_API_KEY", "SAM_SSH_KEY", "OPENROUTER_API_KEY"):
        c.ENV[key] = credentials.get(key) or os.environ[key]
    sys.modules["control"] = c
    release_helper = import_file("walltime_release", root / "resource-release.py")

    def interrupted(signum, frame):
        raise KeyboardInterrupt("Owned timing controller interrupted")

    for signum in (signal.SIGTERM, signal.SIGINT):
        signal.signal(signum, interrupted)
    error, release = None, "release_unconfirmed"
    c.update(phase="starting", controller_pid=os.getpid(), started_epoch=time.time())
    try:
        start_gpu(c)
        run_capture(c, plan)
    except BaseException as exc:
        error = {"type": type(exc).__name__, "message": "See private controller logs and saved source failures"}
        c.save(root / "failure.json", error)
        diagnostic = str(exc)
        for key in ("VAST_API_KEY", "OPENROUTER_API_KEY", "SAM_SSH_KEY"):
            if c.ENV.get(key):
                diagnostic = diagnostic.replace(c.ENV[key], "[redacted]")
        c.save(root / "private-failure-detail.json", {"type": type(exc).__name__, "message": diagnostic})
    finally:
        for signum in (signal.SIGTERM, signal.SIGINT):
            signal.signal(signum, signal.SIG_IGN)
        try:
            release = release_helper.cleanup()
            if release == "confirmed_released":
                for suffix in ("-capture", "-detector"):
                    state = c.owned(c.PREFIX + suffix)
                    if state and not state["State"]["Running"]:
                        c.docker("rm", c.PREFIX + suffix)
        except BaseException as exc:
            c.save(root / "cleanup-error.json", {"type": type(exc).__name__})
            release = "release_unconfirmed"
        c.update(phase="stopped", error=error, cleanup=release)
        save(root / "round-outcome.json", {"finished_epoch": time.time(), "error": error,
            "cleanup": release, "capture_completed": (root / "capture-completion.json").exists(),
            "source_commit": plan["source_commit"], "clips": plan["clips"], "final_answer_dispatch": False})
        for lock in locks:
            lock.close()


def launch(root):
    from dotenv import dotenv_values
    from vastai import VastAI
    locks = []
    try:
        for name in ("benchmark200-recurrent-active.lock", "ashley-core116-active.lock"):
            handle = (root.parent / name).open("a")
            locks.append(handle)
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        plan, ready = load(root / "plan.json"), load(root / "ready.json")
        if ready["plan_snapshot"] != plan:
            raise ValueError("The prepared timing plan changed")
        if any((root / name).exists() for name in ("launch-intent.json", "launch.json", "controller.log")):
            raise ValueError("This timing experiment already has a launch record")
        require_finished(Path(plan["source_round"]), Path(plan["last_answer"]))
        c = resources(root)
        experiments = [root.name, Path(plan["last_answer"]).name, Path(plan["source_round"]).name]
        if plan.get("prior_timing_round"):
            experiments.append(Path(plan["prior_timing_round"]).name)
        if plan.get("event_recovery"):
            previous = Path(plan["event_recovery"]["prior_round"])
            require_finished(previous, Path(plan["last_answer"]))
            experiments.append(previous.name)
        for experiment in experiments:
            if c.docker("ps", "-a", "--filter", "label=experiment=" + experiment,
                        "--format", "{{.Names}}", timeout=20).stdout.strip():
                raise RuntimeError("Reconcile the existing exact experiment container before launch")
        if c.docker("network", "inspect", c.NETWORK, "--format", "{{json .Containers}}", timeout=20).stdout.strip() != "{}":
            raise RuntimeError("The dedicated experiment network is not empty")
        credentials = dotenv_values(c.WORKSPACE / "goodvision-harness/.env")
        vast = VastAI(api_key=credentials.get("VAST_API_KEY") or os.environ["VAST_API_KEY"], retry=1)
        vast.client.timeout = (5, 20)
        historical = load(root / "gpu/output/prelaunch-prior-resources.json")["instances"]
        ids, labels = {row["instance_id"] for row in historical}, {row["label"] for row in historical}
        labels.add("goodvision-" + root.name)
        if any(row.get("id") in ids or row.get("label") in labels for row in c.inventory(vast)):
            raise RuntimeError("A prior exact task GPU is present; do not create a second GPU")
        save(root / "launch-intent.json", {"started_epoch": time.time(), "deadline_epoch": None,
            "gpu_label": "goodvision-" + root.name, "maximum_create_attempts": 1,
            "prior_task_gpus_absent": True})
        environment = dict(os.environ, WALLTIME_LOCK_FDS=",".join(str(handle.fileno()) for handle in locks))
        with (root / "controller.log").open("xb", buffering=0) as log:
            child = subprocess.Popen(["/usr/bin/caffeinate", "-i", sys.executable, "-B", "-u",
                str(Path(__file__).resolve()), "driver", "--root", str(root)], cwd=root,
                env=environment, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                start_new_session=True, pass_fds=tuple(handle.fileno() for handle in locks))
        record = {"epoch": time.time(), "pid": child.pid, "source_commit": plan["source_commit"],
            "clips": plan["clips"], "status": "controller_started_gpu_creation_not_yet_confirmed"}
        save(root / "launch.json", record)
        print(json.dumps(record), flush=True)
    finally:
        for handle in locks:
            handle.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "launch", "driver", "capture"))
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--source-round", type=Path)
    parser.add_argument("--last-answer", type=Path)
    parser.add_argument("--prior-timing-round", type=Path)
    parser.add_argument("--resume-event-round", type=Path,
                        help="Stopped four-event round whose completed recordings are retained")
    parser.add_argument("--replay-interrupted-clip", action="append",
                        help="Interrupted member explicitly approved for one complete replay")
    parser.add_argument("--worktree", type=Path)
    parser.add_argument("--commit")
    args = parser.parse_args()
    os.umask(0o077)
    if args.phase == "prepare":
        prepare(args)
    else:
        root = args.root.resolve()
        if Path(__file__).resolve() != root / "checkout/tools/streaming_walltime_trial.py":
            raise ValueError("Execute only the frozen committed timing entry")
        {"launch": launch, "driver": driver, "capture": capture}[args.phase](root)


if __name__ == "__main__":
    main()
