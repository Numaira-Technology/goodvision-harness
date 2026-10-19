"""Prepare saved-streaming answers, including explicitly approved failure recovery.

Run each phase once under the experiment controller. This tool does not rent a
GPU, capture media, reconcile unknown receipts or change historical artifacts.
The controller must confirm prior processes/resources have stopped, mount source
directories read-only, and allow network access only for the answer phase.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import fcntl
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import time


LEDGER_FILES = ("model-requests.jsonl", "model-responses.jsonl", "attempt-ledger.jsonl")
STOPPED_LANES = ("source_stopped", "tracking_stopped", "vlm_stopped")
DRAINED_LANES = ("source_eof", "tracking_drained", "events_drained", "fusion_drained")


def load(path):
    return json.loads(Path(path).read_bytes())


def rows(path):
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def save(path, value):
    """Publish once using a same-directory temporary file and exclusive link."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".pending")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.link(temporary, path)
    temporary.unlink()
    descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_questions(path, questions):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        for question in questions:
            handle.write(json.dumps(question, ensure_ascii=False, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def git(worktree, *arguments):
    return subprocess.run(["git", "-C", str(worktree), *arguments], check=True,
                          capture_output=True).stdout


def require_finished(source_round, last_answer):
    release = load(source_round / "resources-after.json")
    if (load(source_round / "run.json").get("phase") != "stopped"
            or not (source_round / "round-outcome.json").is_file()
            or release.get("confirmed_released") is not True
            or release.get("creation_outcome_unresolved") is not False
            or release.get("local_confirmed") is not True
            or release.get("remote_confirmed") is not True
            or release.get("remaining") != [] or release.get("cleanup_errors") != []
            or release.get("network_containers") != {}):
        raise ValueError("Prior capture lifecycle or resource ownership is unresolved")
    outcome = load(last_answer / "answer-run-outcome.json")
    if (outcome.get("exit_code") != 0 or outcome.get("completion_saved") is not True
            or not (last_answer / "answer-completion.json").is_file()):
        raise ValueError("The latest saved-answer controller must have finished")
    return release


def copy_ledger(root, last_answer):
    """Extend the latest cumulative seed, not an older capture-only snapshot."""
    sent = defaultdict(list)
    sources = [last_answer / "seed-ledger", last_answer / "output/model-ledger"]
    files = {}
    for name in LEDGER_FILES:
        destination = root / "seed-ledger" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        count = 0
        with destination.open("xb") as target:
            for directory in sources:
                with (directory / name).open("rb") as source:
                    for line in source:
                        if not line.endswith(b"\n"):
                            raise ValueError("A historical ledger has an incomplete line")
                        if line.strip():
                            row = json.loads(line)
                            count += 1
                            if name == "model-requests.jsonl":
                                context = row.get("context", {})
                                operation = row.get("operation") or context.get("operation") or {}
                                if (context.get("phase") == "same_evidence_summary"
                                        or operation.get("kind") == "summary"):
                                    qid = context.get("qa_id") or operation.get("question")
                                    if not qid:
                                        raise ValueError("An old final request has no question scope")
                                    sent[qid].append({"call_id": row["call_id"],
                                                      "ledger": str(directory / name)})
                        target.write(line)
            target.flush()
            os.fsync(target.fileno())
        files[name] = {"path": str(destination), "rows": count}
    save(root / "ledger-seed.json", {"schema_version": "unified-model-ledger-seed-v2",
        "files": files, "source": [str(path) for path in sources],
        "seeded_not_new_calls": True, "new_reconciliations": 0})
    return sent


def capture_catalog(source_round):
    """Collect the declared historical sources and current round, not QA subsets."""
    catalog = {}
    references = load(source_round / "completed-streaming-reference.json")["sources"]
    for reference in references:
        video = reference["video_id"]
        if video in catalog:
            raise ValueError("Duplicate completed streaming video reference")
        catalog[video] = {"capture": reference["capture"],
                          "index": str(Path(reference["directory"]) / "observation-index.json"),
                          "reference": str(source_round / "completed-streaming-reference.json")}
    for reference in load(source_round / "runtime-ready.json")["workflows"]:
        video = reference["video_id"]
        if video in catalog:
            raise ValueError("A video occurs in historical and current capture selections")
        directory = source_round / "output/answer" / reference["source_id"] / "yolo26-s"
        catalog[video] = {"capture": str(directory / "0001/capture.json"),
                          "index": str(directory / "observation-index.json"),
                          "reference": str(source_round / "runtime-ready.json")}
    failures = {}
    for filename, state in (("prior-source-failures.json", "terminal_capture_failure"),
                            ("pending-source-interruptions.json", "shared_interruption_pending")):
        for row in load(source_round / filename)["sources"]:
            if row["video_id"] in failures:
                raise ValueError("Conflicting terminal and interruption source references")
            path = Path(row["path"])
            retained = {"state": state, "source_outcome": str(path),
                        "source_reference": str(source_round / filename)}
            if not path.is_file():
                # Interrupted legacy runs can point to a partial recording
                # directory plus a separate unknown-request reconciliation.
                # This is pending evidence, never permission to answer or replay.
                reconciliation = row.get("reconciliation")
                manifest_path = path / "yolo26-s/0001/manifest.json"
                if (state != "shared_interruption_pending" or not path.is_dir()
                        or not reconciliation or not Path(reconciliation).is_file()
                        or not manifest_path.is_file()):
                    raise ValueError("A declared source outcome has no saved artifact")
                manifest = load(manifest_path)
                if (manifest.get("source_id") != row["source_id"]
                        or manifest.get("video_id") != row["video_id"]):
                    raise ValueError("The partial recording belongs to another source")
                retained.update(source_manifest=str(manifest_path),
                                reconciliation=reconciliation, automatic_replay=False)
            failures[row["video_id"]] = retained
    return catalog, failures


def saved_source(reference, video):
    capture = Path(reference["capture"]).resolve()
    index_path = Path(reference["index"]).resolve()
    required = [capture, capture.parent / "completion.json", capture.parent / "cleanup.json", index_path]
    if not all(path.is_file() for path in required):
        return {"state": "saved_capture_not_available", **reference}
    evidence, cleanup, index = load(capture), load(capture.parent / "cleanup.json"), load(index_path)
    entries = [entry for entry in index["observations"] if entry["video_id"] == video]
    processed = [row for row in index["processed_sources"] if row["video_id"] == video]
    if len(entries) != 1 or len(processed) != 1:
        raise ValueError("The saved observation index does not identify one source")
    indexed_capture = Path(entries[0]["capture_path"])
    if not indexed_capture.is_absolute():
        indexed_capture = index_path.parent / indexed_capture
    if (evidence.get("schema_version") != "unified-evidence-v2"
            or evidence.get("video_id") != video
            or evidence.get("face_enabled") is not False
            or index.get("method") != "native_streaming_yolo26-s_vlm_on"
            or indexed_capture.resolve() != capture
            or processed[0].get("scope") != "native_streaming"
            or processed[0].get("vlm_enabled") is not True
            or evidence.get("provenance", {}).get("question_conditioned") is not False
            or evidence.get("provenance", {}).get("scheduling_mode") != "native_streaming"
            or evidence.get("provenance", {}).get("tracking_model") != "yolo26-s"):
        raise ValueError("Saved observation is not the declared face-disabled streaming method")
    scope = {key: evidence[key] for key in ("video_id", "source_id", "stream_session_id")}
    # Older completed captures predate the extra cleanup flags. Their explicit
    # EOF/drained fields remain lifecycle evidence; do not invent missing flags
    # or require zero failed windows. An explicit unsafe flag still wins.
    safe = (cleanup.get("shared_failure") is not True
            and cleanup.get("fusion_stopped") is not False
            and cleanup.get("tracking_requests_settled") is not False
            and all(cleanup.get(key) is True for key in STOPPED_LANES)
            and all(evidence.get(key) is True for key in DRAINED_LANES))
    return {**reference, **scope, "capture": str(capture),
            "state": "saved_source_ended" if safe else "source_lifecycle_pending",
            "capture_state": evidence.get("observation_manifest", {}).get("complete"),
            "duration_review": "advisory_original_metadata_preserved"}


def freeze_source(root, worktree, commit):
    """Copy tracked committed files; Git revisions name releases, not data checks."""
    head = git(worktree, "rev-parse", "HEAD").decode().strip()
    remote = git(worktree, "rev-parse", "refs/remotes/origin/codex/research-testing").decode().strip()
    resolved = git(worktree, "rev-parse", commit + "^{commit}").decode().strip()
    if head != resolved or remote != resolved:
        raise ValueError("Prepare only the published team-branch commit at HEAD")
    if git(worktree, "diff", "HEAD", "--", "src", "tools/saved_streaming_recovery.py",
           "tools/saved_streaming_controller.py").strip():
        raise ValueError("Committed evaluation source differs from the working tree")
    if git(worktree, "ls-files", "--others", "--exclude-standard", "--", "src").strip():
        raise ValueError("Uncommitted source modules must not enter the frozen evaluation")
    archive = git(worktree, "archive", resolved, "src", "tools/saved_streaming_recovery.py",
                  "tools/saved_streaming_controller.py")
    destination = root / "checkout"
    destination.mkdir()
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:") as bundle:
        for member in bundle:
            path = destination / member.name
            if not path.resolve().is_relative_to(destination):
                raise ValueError("Source archive path escapes the checkout")
            if member.isdir():
                path.mkdir(parents=True, exist_ok=True)
            elif member.isfile():
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("xb") as target:
                    shutil.copyfileobj(bundle.extractfile(member), target)
            else:
                raise ValueError("Only regular files and directories belong in the source archive")
    return resolved


def failed_answer_selection(path, sent, questions, *, allow_timeouts):
    """Freeze the requested failure subset without copying GT into the plan."""
    from goodvision_harness.evaluation.model_operation import summary_operation
    from goodvision_harness.evaluation.summary_recovery import SCHEMA

    original = {row["qa_id"]: row for row in questions}
    selected = []
    report = load(path)
    candidates = [row for row in report["questions"]
                  if row["status"] in {"request_failure", "invalid_answer"}]
    historical = [request for row in candidates for request in sent[row["qa_id"]]]
    wanted = {request["call_id"] for request in historical}
    archived = {}
    for ledger_path in {request["ledger"] for request in historical}:
        for request in rows(ledger_path):
            if request["call_id"] in wanted:
                if (request["call_id"] in archived
                        and archived[request["call_id"]] != request["request"]):
                    raise ValueError("Conflicting historical request records")
                archived[request["call_id"]] = request["request"]
    for row in candidates:
        qid, video = row["qa_id"], row["video_id"]
        outcome_path = Path(row["source_answer"])
        outcome = load(outcome_path)
        if (original[qid]["video_id"] != video or outcome["qa_id"] != qid
                or outcome["video_id"] != video or outcome["status"] != row["status"]):
            raise ValueError("Historical failed-answer selection disagrees with its source")
        error = outcome.get("failure_policy") or {}
        if outcome["status"] == "invalid_answer":
            response = load(outcome_path.parent / "response.json")
            if response.get("choices", [{}])[0].get("finish_reason") != "length":
                raise ValueError("This recovery covers truncated outputs, not other invalid answers")
            failure = "output_truncated"
        elif error.get("status_code") == 400:
            failure = "http400_response"
        elif error.get("code") == "timeout" and allow_timeouts:
            failure = "timeout_without_response"
        else:
            raise ValueError("Historical failure is outside the explicit recovery scope")
        history = [request["call_id"] for request in sent[qid]]
        terminal_call = (outcome.get("ledger_receipt") or {}).get("call_id")
        if terminal_call is None and len(history) == 1:
            # Some legacy failures omitted the call receipt from outcome.json.
            # Match their saved request to the only old dispatch. Any later
            # dispatch makes this fallback ineligible, including a new timeout.
            terminal_call = history[0]
        if terminal_call not in history:
            raise ValueError("The selected old outcome must identify its final historical call")
        saved = load(outcome_path.parent / "request.json")
        expected = {"model": saved["model"], "messages": saved["messages"], **saved["parameters"]}
        if archived.get(terminal_call) != expected:
            raise ValueError("The selected old outcome does not match its archived request")
        calls = history[:history.index(terminal_call) + 1]
        if calls != history:
            raise ValueError("A later dispatch exists beyond the selected old failure; preserve it")
        if len(calls) not in {1, 2} or len(set(calls)) != len(calls):
            raise ValueError("Historical sends are missing, duplicated or exhausted")
        capture = load(row["source_capture"])
        selected.append({"qa_id": qid, "video_id": video,
            "public_question": original[qid],
            "operation": summary_operation(capture, qid), "prior_call_ids": calls,
            "source_answer": str(outcome_path), "source_capture": row["source_capture"],
            "failure_class": failure, "allow_unknown_timeout": failure == "timeout_without_response",
            "remaining_sends": 3 - len(calls)})
    if len(selected) != 41 or len({row["qa_id"] for row in selected}) != 41:
        raise ValueError("The authorized historical recovery must contain exactly 41 questions")
    return {"schema_version": SCHEMA, "authorization": "human_requested_failed41_recovery_20261019",
            "max_cumulative_sends": 3, "scope_source": str(path.resolve()), "questions": selected,
            "historical_outcomes_and_costs_unchanged": True}


def prepare(args):
    root, source_round, last_answer = args.root.resolve(), args.source_round.resolve(), args.last_answer.resolve()
    if root.exists():
        raise ValueError("Use a new independent recovery directory")
    release = require_finished(source_round, last_answer)
    questions_path = source_round / "public/questions-all182.jsonl"
    questions = list(rows(questions_path))
    ids, videos = [row["qa_id"] for row in questions], {row["video_id"] for row in questions}
    if len(ids) != 182 or len(set(ids)) != 182 or len(videos) != 157:
        raise ValueError("The original 182-question, 157-video scope must remain intact")
    catalog, failures = capture_catalog(source_round)
    if set(catalog) | set(failures) != videos:
        raise ValueError("Source references do not account for the original video scope")
    root.mkdir(parents=True)
    save(root / "prepare-intent.json", {"epoch": time.time(), "source_round": str(source_round),
        "last_answer": str(last_answer), "paid_dispatch": False, "gpu_creations": 0})
    commit = freeze_source(root, args.worktree, args.commit)
    sent = copy_ledger(root, last_answer)
    recovery = None
    if args.recover_failed_answers is not None:
        if args.partial_approval is not None:
            raise ValueError("Keep failed-answer and partial-observation recovery selections separate")
        sys.path.insert(0, str(root / "checkout/src"))
        recovery = failed_answer_selection(args.recover_failed_answers, sent, questions,
                                           allow_timeouts=args.allow_selected_timeout_recovery)
        save(root / "failed-answer-authorization.json", recovery)
    selected_ids = ({row["qa_id"] for row in recovery["questions"]} if recovery is not None
                    else {question["qa_id"] for question in questions if question["qa_id"] not in sent})
    needed_videos = {question["video_id"] for question in questions if question["qa_id"] in selected_ids}
    by_video = {video: saved_source(reference, video) for video, reference in catalog.items()
                if video in needed_videos}
    if recovery is not None:
        for entry in recovery["questions"]:
            source = by_video[entry["video_id"]]
            if (Path(source["capture"]).resolve() != Path(entry["source_capture"]).resolve()
                    or any(source[field] != entry["operation"][target] for field, target in
                           (("source_id", "source"), ("stream_session_id", "session")))):
                raise ValueError("Failed-answer recovery must use the original saved observations")
    partial_approval = None
    if args.partial_approval is not None:
        partial_approval = load(args.partial_approval)
        sys.path.insert(0, str(root / "checkout/src"))
        from goodvision_harness.evaluation.partial_recording import approved_source, export_partial_recording
        selected_videos = [row["video_id"] for row in partial_approval["sources"]]
        approved_ids = [qid for row in partial_approval["sources"] for qid in row["question_ids"]]
        expected_ids = {question["qa_id"] for question in questions
                        if question["video_id"] in selected_videos and question["qa_id"] not in sent}
        if (len(set(selected_videos)) != len(selected_videos)
                or len(set(approved_ids)) != len(approved_ids)
                or set(approved_ids) != expected_ids
                or not set(selected_videos) <= set(failures)):
            raise ValueError("Partial approval must match the never-sent interrupted question scope")
        for number, video in enumerate(selected_videos, 1):
            selection = approved_source(partial_approval, video)
            output = root / "partial-sources" / f"{number:04d}" / "evidence.json"
            output.parent.mkdir(parents=True)
            evidence = export_partial_recording(selection["record_directory"], output=output,
                video_id=video, approval=partial_approval)
            by_video[video] = {"state": "approved_partial_saved_source", "capture": str(output),
                "partial_record_directory": selection["record_directory"],
                "historical_source_state": failures[video],
                **{key: evidence[key] for key in ("video_id", "source_id", "stream_session_id")}}
        save(root / "partial-approval.json", partial_approval)
    inventory, groups_by_video = [], defaultdict(list)
    for position, question in enumerate(questions, 1):
        qid, video = question["qa_id"], question["video_id"]
        source = by_video.get(video) or catalog.get(video) or failures[video]
        state = ("prior_final_dispatch_preserved" if qid in sent and qid not in selected_ids else
                 "saved_answer_candidate" if source["state"] in {
                     "saved_source_ended", "approved_partial_saved_source"} else source["state"])
        inventory.append({"position": position, "qa_id": qid, "video_id": video,
            "capability": question["capability"], "state": state,
            "source": source, "prior_final_requests": sent.get(qid, [])})
        if state == "saved_answer_candidate":
            groups_by_video[video].append(question)
    groups = []
    for position, (video, selected) in enumerate(groups_by_video.items(), 1):
        name, source = f"{position:04d}", by_video[video]
        public = root / "public" / (name + ".jsonl")
        index = root / "indexes" / (name + ".json")
        write_questions(public, selected)
        save(index, {"schema_version": "unified-observation-index-v1",
            "method": "native_streaming_yolo26-s_vlm_on", "observations": [{
                "capture_path": source["capture"], "video_id": video,
                "source_id": source["source_id"], "stream_session_id": source["stream_session_id"]}],
            "processed_sources": []})
        groups.append({"group": name, "video_id": video, "source_id": source["source_id"],
            "capture": source["capture"], "public": str(public), "index": str(index),
            "partial_record_directory": source.get("partial_record_directory"),
            "qa_ids": [question["qa_id"] for question in selected]})
    write_questions(root / "public/questions-all182.jsonl", questions)
    shutil.copytree(last_answer / "assets", root / "assets")
    final = deepcopy(load(last_answer / "plan.json")["final_model"])
    final["context_budget"]["tokenizer_path"] = str(root / "assets/tokenizer.json")
    scoring = load(source_round / "scoring-reference.json")
    plan = {"schema_version": "saved-streaming-recovery-v1", "source_commit": commit,
        "source_round": str(source_round), "last_answer": str(last_answer),
        "original_questions": 182, "original_videos": 157, "groups": groups,
        "inventory": inventory, "states": dict(Counter(row["state"] for row in inventory)),
        "final_model": final, "scoring": {key: scoring[key] for key in ("truth", "scorer")},
        "prior_release_record": release, "legacy_duration_deferral_applied": False,
        "partial_observation_approval": partial_approval,
        "failed_answer_recovery": recovery,
        "historical_dispatches_preserved": True, "new_observation_calls": 0, "new_gpu_creations": 0}
    save(root / "plan.json", plan)
    save(root / "ready.json", {"source_commit": commit, "prepared_epoch": time.time(),
        "plan_snapshot": plan, "new_model_calls": 0, "new_gpu_creations": 0})
    print(json.dumps({"source_commit": commit, "states": plan["states"],
                      "groups": len(groups), "new_model_calls": 0}), flush=True)


def score_completed(root, group, cohort, scoring, score_cohort):
    directory = root / "output/cohorts" / group["group"]
    summary = root / "output/model-answers" / group["group"]
    selected = [row for row in cohort["rows"]
                if (summary / Path(row["answer_path"]).stem / "outcome.json").is_file()]
    if not selected:
        return
    qids = {row["qa_id"] for row in selected}
    questions = [row for row in rows(group["public"]) if row["qa_id"] in qids]
    scoring_input = root / "output/scoring-cohorts" / group["group"]
    write_questions(scoring_input / "questions.jsonl", questions)
    run = load(directory / "run.json")
    run["public_questions"], run["questions"] = questions, len(questions)
    run["questions_path"] = str(scoring_input / "questions.jsonl")
    run["videos"] = len({row["video_id"] for row in questions})
    run["source_cohort"] = str(directory / "completion.json")
    run["scope"] = "saved_outcome_subset_not_full_original182"
    save(scoring_input / "run.json", run)
    save(scoring_input / "completion.json", {**cohort, "rows": selected, "questions": len(selected),
        "states": dict(Counter(row["status"] for row in selected)),
        "capability_states": {str(capability): dict(Counter(
            row["status"] for row in selected if row["capability"] == capability))
            for capability in sorted({row["capability"] for row in selected}, key=str)},
        "scope": "saved_outcome_subset_not_full_original182"})
    for row in selected:
        destination = scoring_input / row["answer_path"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(directory / row["answer_path"], destination)
    score_cohort(questions_path=scoring_input / "questions.jsonl", truth_path=scoring["truth"],
        cohort_path=scoring_input / "completion.json", mode="model", summary_root=summary,
        scorer_path=scoring["scorer"], output=root / "output/scores" / (group["group"] + ".json"))


def run_phase(args):
    root = args.root.resolve()
    plan, ready = load(root / "plan.json"), load(root / "ready.json")
    if ready["plan_snapshot"] != plan:
        raise ValueError("The prepared recovery plan changed")
    script = Path(__file__).resolve()
    if script != root / "checkout/tools/saved_streaming_recovery.py":
        raise ValueError("Run phases from the frozen committed checkout")
    if args.phase == "answer":
        if load(root / "inputs-run-outcome.json").get("exit_code") != 0:
            raise ValueError("The input-only controller must finish before paid dispatch")
        preparation = load(root / "inputs-completion.json")
        if preparation["source_commit"] != plan["source_commit"]:
            raise ValueError("Input preparation used a different source release")
        prepared = {row["qa_id"]: row for row in preparation["results"]}
    else:
        prepared = {}
    sys.path.insert(0, str(root / "checkout/src"))
    if args.dependencies is not None:
        wheels = list((root / "assets/wheels").glob("tokenizers-0.22.2-*.whl"))
        if len(wheels) != 1:
            raise ValueError("One frozen tokenizer wheel is required")
        subprocess.run([sys.executable, "-m", "pip", "install", "--no-index", "--no-deps",
            "--no-compile", "--disable-pip-version-check", "--target", str(args.dependencies),
            str(wheels[0])], check=True)
        sys.path.insert(0, str(args.dependencies))
    from goodvision_harness.evaluation.cohort import generate_cohort
    from goodvision_harness.evaluation.model_ledger import ModelLedger
    from goodvision_harness.evaluation.scoring import score_cohort
    from goodvision_harness.evaluation.summary import execute_summary
    from goodvision_harness.model_api import ModelAPIClient, ModelAPISettings

    save(root / (args.phase + "-intent.json"), {"epoch": time.time(),
        "source_commit": plan["source_commit"], "paid_dispatch_enabled": args.phase == "answer"})
    ledger = ModelLedger(root / "output/model-ledger", seed_manifest_path=root / "ledger-seed.json",
                         owner_lock_path=root / "model-ledger.lock",
                         summary_recovery=plan.get("failed_answer_recovery"))
    results, final = [], plan["final_model"]
    try:
        client = (None if args.phase == "inputs" else ModelAPIClient(
            ModelAPISettings.from_mapping(final["settings"]), capture_error_details=True))
        for group in plan["groups"]:
            directory = root / "output/cohorts" / group["group"]
            cohort = (generate_cohort(questions_path=group["public"], index_path=group["index"],
                                      output=directory) if args.phase == "inputs"
                      else load(directory / "completion.json"))
            for row in cohort["rows"]:
                artifact = directory / row["answer_path"]
                if args.phase == "answer" and not prepared[row["qa_id"]]["ready_for_dispatch"]:
                    entry = {**prepared[row["qa_id"]], "status": "retained_not_dispatched"}
                else:
                    result = execute_summary(artifact, output=root / "output/model-answers" /
                        group["group"] / artifact.stem, client=client, ledger=ledger,
                        model=final["model"], parameters=final["parameters"],
                        max_input_bytes=final["max_input_bytes"], context_budget=final["context_budget"],
                        prepare_only=args.phase == "inputs",
                        partial_observation_approval=plan.get("partial_observation_approval"))
                    if args.phase == "inputs":
                        path = root / "output/prepared-inputs" / group["group"] / (artifact.stem + ".json")
                        save(path, result)
                        metadata = result["prepared_request"].get("context_preparation", {})
                        entry = {"qa_id": row["qa_id"], "source_id": group["source_id"],
                            "ready_for_dispatch": result["ready_for_dispatch"], "reason": result["reason"],
                            "input_bytes": result["input_bytes"], "usage": metadata.get("usage"),
                            "encoding": metadata.get("encoding"), "prepared_input": str(path),
                            "paid_dispatch": False}
                    else:
                        entry = {"qa_id": row["qa_id"], "source_id": group["source_id"],
                            "status": result["status"], "final_answer": result.get("final_answer"),
                            "reason": result.get("reason"), "failure_policy": result.get("failure_policy")}
                results.append(entry)
                print(json.dumps(entry, ensure_ascii=False), flush=True)
                if (entry.get("failure_policy") or {}).get("scope") == "shared":
                    raise RuntimeError("Shared answer state is unsafe; preserve remaining questions as pending")
            if args.phase == "answer":
                score_completed(root, group, cohort, plan["scoring"], score_cohort)
        save(root / (args.phase + "-completion.json"), {"finished_epoch": time.time(),
            "source_commit": plan["source_commit"], "results": results,
            "original_questions": 182, "original_videos": 157, "original182_complete": False,
            "ready_questions": sum(row.get("ready_for_dispatch", False) for row in results),
            "paid_dispatch_enabled": args.phase == "answer", "observations_repeated": False})
    finally:
        ledger.close()


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("phase", choices=("prepare", "inputs", "answer"))
    cli.add_argument("--root", type=Path, required=True)
    cli.add_argument("--source-round", type=Path)
    cli.add_argument("--last-answer", type=Path)
    cli.add_argument("--worktree", type=Path)
    cli.add_argument("--commit")
    cli.add_argument("--partial-approval", type=Path,
                     help="Explicit human-approved source/question list for partial saved observations")
    cli.add_argument("--recover-failed-answers", type=Path,
                     help="Explicitly recover the 41 historical failures selected from this score report")
    cli.add_argument("--allow-selected-timeout-recovery", action="store_true",
                     help="Human authorization includes the selected six unknown timeout receipts")
    cli.add_argument("--dependencies", type=Path,
                     help="Container-only temporary dependency directory for the frozen tokenizer wheel")
    args = cli.parse_args()
    os.umask(0o077)
    if args.phase == "prepare":
        if any(getattr(args, key) is None for key in ("source_round", "last_answer", "worktree", "commit")):
            cli.error("prepare requires --source-round, --last-answer, --worktree and --commit")
        prepare(args)
    else:
        # The outer controller also owns the host-wide task locks and resource checks.
        with (args.root / "phase-writer.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            try:
                run_phase(args)
            except BaseException as exc:
                path = args.root / (args.phase + "-failure.json")
                if not path.exists():
                    save(path, {"epoch": time.time(), "type": type(exc).__name__,
                                "message": "Phase stopped; inspect private controller logs and atomic outputs"})
                raise


if __name__ == "__main__":
    main()
