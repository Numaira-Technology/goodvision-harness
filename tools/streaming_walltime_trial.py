"""Own a frozen authorized three-video timing run with native admission policies.

Preparation is offline. Launch owns one standard 24GB RTX 4090 and releases it
before any offline timing analysis. There is no final-answer or GT input phase.
"""
from __future__ import annotations

import argparse
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
    copy_ledger, freeze_source, git, load, require_finished, save, write_questions,
)


CLIPS = ("0018", "0028", "0002")
PACED_INPUT_CLIPS = ("0040", "0003", "0030")


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


def prepare(args):
    root, source, last = args.root.resolve(), args.source_round.resolve(), args.last_answer.resolve()
    staging = load(root / "staging.json")
    evaluation_timing = staging.get("evaluation_timing", False)
    if type(evaluation_timing) is not bool:
        raise ValueError("evaluation_timing must be an explicit boolean")
    clips = PACED_INPUT_CLIPS if evaluation_timing else CLIPS
    if tuple(row["clip"] for row in staging["selections"]) != clips:
        raise ValueError("Only the authorized three clips belong in this run")
    prior_timing = args.prior_timing_round.resolve() if args.prior_timing_round else None
    if evaluation_timing and prior_timing is None:
        raise ValueError("The instrumented run requires the prior timing resource record")
    if prior_timing is not None:
        if (load(prior_timing / "round-outcome.json")["cleanup"] != "confirmed_released"
                or load(prior_timing / "resources-after.json")["confirmed_released"] is not True):
            raise ValueError("Release the prior timing resources before preparing another run")
    if any((root / name).exists() for name in ("prepare-intent.json", "ready.json", "launch.json")):
        raise ValueError("This experiment already has a preparation or launch")
    release = require_finished(source, last)
    if load(last / "answer-run-outcome.json").get("resource_state") != "exited_removed":
        raise ValueError("Release the prior CPU answer container first")
    save(root / "prepare-intent.json", {"epoch": time.time(), "clips": clips,
        "model_calls": 0, "gpu_creations": 0, "source_commit": args.commit})
    commit = freeze_source(root, args.worktree.resolve(), args.commit)
    with (root / "checkout/tools/streaming_walltime_trial.py").open("xb") as target:
        target.write(git(args.worktree, "show", commit + ":tools/streaming_walltime_trial.py"))
    copy_ledger(root, last)
    shutil.copytree(source / "gpu/source", root / "gpu/source",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "execution-authorization.json"))
    for name in ("harness.yaml", "event-definitions.json"):
        shutil.copyfile(source / name, root / name)
    (root / "cohort").mkdir()
    shutil.copyfile(source / "cohort/seed-weights.json", root / "cohort/seed-weights.json")
    adapter = root.parent / "detector-rf-count36-1288-resume5-20261016"
    shutil.copyfile(adapter / "control.py", root / "resource-control.py")
    shutil.copyfile(adapter / "driver.py", root / "resource-release.py")
    for name in ("logs", "gpu/output", "output"):
        (root / name).mkdir(parents=True, exist_ok=True)
    previous_authorization = load(prior_timing / "gpu/source/execution-authorization.json"
        if prior_timing is not None else source / "authorization.json")
    approval = {"status": "explicitly_approved", "approved_by": "human_user",
        "round": root.name, "approval_quote": "按此设计实现" if evaluation_timing else
        "尝试小规模测试 wall time；实在不行可以先做physical conflict",
        "approval_reference": "Current chat: approved paced-input and readable-report timing; "
        "F_8 and two annotated negatives after prerequisite deliverables and push."
        if evaluation_timing else "Current chat: authorized small wall-time trial, physical conflict first.",
        "maximum_new_create_attempts": None, "maximum_create_attempts_this_round": 1,
        "creation_budget_id": previous_authorization["creation_budget_id"],
        "new_create_attempts_used_before_round": previous_authorization["new_create_attempts_used_before_round"] + 1,
        "required_gpu_name": "RTX 4090", "maximum_simultaneous_gpu_instances": 1,
        "bounded_mask_area": False, "deadline_epoch": None, "gpu_actual_cap_usd": None,
        "scope": "three_video_native_streaming_walltime", "clips": clips,
        "face_enabled": False, "vlm_enabled": True}
    save(root / "gpu/source/execution-authorization.json", approval)
    history = load(source / "gpu/source/prior-resources.json")
    deployment, cost = load(source / "gpu/output/deployment.json"), load(source / "gpu/output/gpu-cost.json")
    if deployment["instance_id"] not in {row["instance_id"] for row in history["instances"]}:
        history["instances"].append({"instance_id": deployment["instance_id"], "label": deployment["label"],
            "started_epoch": deployment["started_epoch"], "stopped_epoch": release["checked_epoch"],
            "quoted_dph": str(deployment["offer"]["dph_total"]),
            "minimum_observed_charge_usd": str(cost["current_known_charge_floor_usd"]),
            "actual_charge_usd": cost.get("actual_charge_usd"), "released": True,
            "unknown_cost_is_not_zero": True, "final_invoice": False})
    if prior_timing is not None:
        prior_deployment = load(prior_timing / "gpu/output/deployment.json")
        prior_cost = load(prior_timing / "gpu/output/gpu-cost.json")
        if prior_deployment["instance_id"] not in {row["instance_id"] for row in history["instances"]}:
            history["instances"].append({
                "instance_id": prior_deployment["instance_id"], "label": prior_deployment["label"],
                "started_epoch": prior_deployment["started_epoch"],
                "stopped_epoch": load(prior_timing / "round-outcome.json")["finished_epoch"],
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
        previous = load(row["source_configuration_reference"])
        vlm = previous["vlm"]
        vlm["inspector_settings"]["evaluation_transport"]["legacy_window_budgets"] = {}
        configuration = {"schema_version": "native-stream-source-v1", "video_id": row["video_id"],
            "source_id": "clip-" + row["clip"], "stream_session_id": stamp + "-" + row["clip"],
            "source": {**row["source"], "negative_coverage_domain": "first_to_last_decoded_frame_pts",
                       "sampling_limitation": "native_processed_and_documented_skips"},
            "media_path": row["media_path"], "harness_config": str(root / "harness.yaml"),
            "event_definitions": str(root / "event-definitions.json"),
            "face_enabled": False, "vlm_enabled": True, "question_conditioned": False,
            "evaluation_timing": evaluation_timing,
            "sam_internal_patch": False, "coverage_policy": "native_processed_and_documented_skips_v1",
            "max_source_wall_seconds": max(60, row["duration_seconds"]) + 300,
            "drain_timeout_seconds": previous["drain_timeout_seconds"],
            "tracking": row["tracking"], "vlm": vlm, "model_ledger": ledger,
            "provenance": {"harness_commit": commit,
                "harness_original_baseline": "6c28dddd26bed7a7be50762a4c7dd6b796b2dbe3",
                "method": staging["method"], "original_duration_metadata_preserved": True,
                "tracking_parameters": {"imgsz": 1280, "confidence": 0.1, "max_det": 300},
                "weights": {"filename": "yolo26s.pt", "upstream": "ultralytics/assets release v8.4.0"}}}
        destination = root / "sources" / (row["clip"] + ".json")
        save(destination, configuration)
        entries.append({"clip": row["clip"], "config": str(destination),
                        "video_id": row["video_id"], "media_path": row["media_path"]})
    # This capture-only cohort contains video identities, never questions or GT.
    write_questions(root / "public/videos.jsonl", [{"video_id": row["video_id"]} for row in entries])
    plan = {"source_commit": commit, "source_round": str(source), "last_answer": str(last),
        "prior_release_record": release, "sources": entries, "model_ledger": ledger,
        "final_answer_dispatch": False, "ground_truth_to_model": False,
        "resource_adapter_origin": str(adapter), "gpu_source_origin": str(source / "gpu/source"),
        "maximum_create_attempts_this_round": 1, "clips": clips,
        "evaluation_timing": evaluation_timing,
        "prior_timing_round": str(prior_timing) if prior_timing is not None else None}
    save(root / "plan.json", plan)
    save(root / "ready.json", {"status": "prepared_not_started", "plan_snapshot": plan,
        "source_commit": commit, "model_calls": 0, "gpu_creations": 0})
    print(json.dumps({"prepared_sources": len(entries), "source_commit": commit,
                      "model_calls": 0, "gpu_creations": 0}), flush=True)


def capture(root):
    from goodvision_harness.evaluation.answer import run
    plan, owner = load(root / "plan.json"), load(root / "owned-resource.json")
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
    for item in ("checkout", "gpu/source", "seed-ledger", "sources", "public", "harness.yaml", "event-definitions.json"):
        args += ["-v", str(root / item) + ":" + str(root / item) + ":ro"]
    for row in plan["sources"]:
        args += ["-v", row["media_path"] + ":" + row["media_path"] + ":ro"]
    args += ["--entrypoint", "python", c.IMAGE, "-B", "-u",
             str(root / "checkout/tools/streaming_walltime_trial.py"), "capture", "--root", str(root)]
    c.docker(*args)
    c.update(phase="three_video_native_capture")
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
