"""Native, deterministic research evaluation on current main or frozen Ashley.

Neither mode sends the question text to observation windows or calls a final
answer LLM. The frozen51 profile selects an immutable historical runtime; this
new orchestration is not a rerun of the original 182-question publication.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import platform
import re
import subprocess
import sys
from importlib import metadata as package_metadata
from pathlib import Path
from urllib.parse import quote
from uuid import uuid4

from .extract import answer_question, parse_count_temporal
from .schema import EVENT_TARGET_TO_HARNESS_LABEL, Capture, load_media, load_questions

ROOT = Path(__file__).resolve().parents[2]
ARCHIVE = ROOT / "experiments/ashley-detector182-20261012"
FROZEN = ARCHIVE / "frozen_runtime"
NONE_OPTION = "none of these events or actions"


def frozen_source_inventory():
    """Read historical revisions and safe file inventories, not content equality."""
    manifest = json.loads((ARCHIVE / "archive-source.json").read_text())
    publication = json.loads((ARCHIVE / "published-source.json").read_text())
    inventories = {}
    for name, document in (("archive", manifest), ("publication", publication)):
        files = []
        for row in document["files"]:
            relative = Path(row["path"])
            base = (
                ROOT
                if name == "publication" and relative.parts[:1] == ("experiments",)
                else FROZEN
            )
            path = base / relative
            if (
                relative.is_absolute()
                or ".." in relative.parts
                or path.is_symlink()
                or not path.resolve().is_relative_to(base.resolve())
                or not path.is_file()
            ):
                raise ValueError("invalid historical source inventory path: " + str(relative))
            record = {"root": str(base), "path": relative.as_posix()}
            if record in files:
                raise ValueError("duplicate historical source inventory path: " + str(relative))
            files.append(record)
        inventories[name] = files
    return {
        "source_commit": manifest["source_commit"],
        "archive_manifest": str(ARCHIVE / "archive-source.json"),
        "publication_manifest": str(ARCHIVE / "published-source.json"),
        "publication_method": publication.get("method"),
        "publication_branch": publication.get("branch"),
        "publication_native_base_commit": publication.get("native_base_commit"),
        "publication_previous_native_commit": publication.get("previous_native_commit"),
        "publication_detector_reference_commit": publication.get("detector_reference_commit"),
        "files": inventories,
        "content_equality_verified": False,
    }


def verify_frozen_source():
    """Return the declared commit after inventory checks, without content attestation.

    Keep the existing caller interface; the legacy name now means inventory only.
    """
    return frozen_source_inventory()["source_commit"]


def _load_research_helper(name, filename):
    # These adapters stay current even when the engine is frozen. They import
    # the explicitly selected runtime contracts, never another engine's classes.
    spec = importlib.util.spec_from_file_location(
        "goodvision_harness." + name, ROOT / "src/goodvision_harness" / filename
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(spec.name, None)
        raise
    return module


def select_runtime(profile):
    if any(
        name == "goodvision_harness" or name.startswith("goodvision_harness.")
        for name in sys.modules
    ):
        if profile != "current-main":
            raise ValueError("frozen51 must start in a fresh process")
        return
    source = FROZEN / "src" if profile == "frozen51" else ROOT / "src"
    if profile == "frozen51":
        frozen_source_inventory()
        os.environ["ASHLEY_EVENT_DEFINITIONS"] = str(FROZEN / "event-definitions.json")
    sys.path.insert(0, str(source))


def available_events(profile):
    if profile == "frozen51":
        from goodvision_harness.event_session import SAFETY_LABELS

        return tuple(SAFETY_LABELS)
    from goodvision_harness.grounding import SAFETY_EVENT_LABELS

    return tuple(SAFETY_EVENT_LABELS)


def canonical_label(value):
    label = "_".join(str(value).split(":")[-1].lower().split())
    return EVENT_TARGET_TO_HARNESS_LABEL.get(label, label)


def labels_for_question(question):
    raw, context = question.raw, question.public_context
    if question.capability in {"counting", "location_tracking"}:
        return ()
    if (
        question.capability == "temporal_questions"
        and parse_count_temporal(question.question) is not None
    ):
        return ()
    target = (
        context.get("queried_event_label")
        or context.get("event_label")
        or raw.get("target_label")
    )
    if (
        question.capability == "event_recognition"
        and question.response_options
        and raw.get("question_form") != "event_presence_binary"
        and set(question.response_options) != {"yes", "no"}
        and question.answer_type == "single_choice"
    ):
        labels = context.get("event_labels") or [
            option for option in question.response_options if option != NONE_OPTION
        ]
    else:
        labels = (target,) if target else ()
    return tuple(dict.fromkeys(canonical_label(label) for label in labels))


class ModelCallBudget:
    def __init__(self, limit):
        if type(limit) is not int or limit < 0:
            raise ValueError("max-model-calls must be a non-negative integer")
        self.limit, self.calls = limit, 0

    def reserve(self):
        if self.calls >= self.limit:
            raise RuntimeError("native evaluation model-call budget exhausted")
        self.calls += 1


def make_inspector(config, audit, budget):
    from goodvision_harness.model_api import environment_api_key
    from goodvision_harness.scene_model import TemporalVLMInspector

    redact = _load_research_helper(
        "_research_audit", "native_event_audit.py"
    ).redact_images

    inspector = TemporalVLMInspector(
        config.model_api,
        model=config.inference.model,
        api_key=environment_api_key(config.model_api),
        request_parameters=config.inference.request_parameters,
        max_tokens=config.runtime.scene_max_tokens,
        max_image_side=config.runtime.scene_max_image_side,
        jpeg_quality=config.runtime.scene_jpeg_quality,
    )
    original = inspector._request

    async def audited_request(payload, headers, **kwargs):
        budget.reserve()
        audit.record("model_request", call=budget.calls, request=redact(payload))
        response = await original(payload, headers, **kwargs)
        audit.record("model_response", call=budget.calls, response=response)
        return response

    inspector._request = audited_request
    return inspector


def evaluate_frames(
    frames,
    tracker,
    *,
    video_id,
    output,
    metadata,
    labels=(),
    inspector=None,
    observation_fps=None,
    source_recorded=False,
):
    """Run native scene/identity contracts with injectable inputs for CPU tests."""
    from goodvision_harness.person_registry import PersonRegistry
    from goodvision_harness.scene_runtime import SceneRuntime
    from goodvision_harness.scene_worker import _validate_job, _validate_verdicts

    Audit = _load_research_helper(
        "_research_audit", "native_event_audit.py"
    ).NativeAudit
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    audit = metadata.pop("_audit", None) or Audit(output / "records.jsonl")
    session_id = str(uuid4())
    audit.record(
        "metadata",
        **{
            **metadata,
            "video_id": video_id,
            "stream_session_id": session_id,
            "requested_event_labels": list(labels),
            "observation_reads_questions": False,
            "final_llm": False,
            "execution_policy": "sequential_native_source_time",
        },
    )
    scene = SceneRuntime(
        session_id,
        "native-research",
        event_labels=tuple(labels),
        tracking_mode="batch",
        native_memory_scope="session",
        temporal_observation_fps=observation_fps if labels else None,
        max_pending_inspections=8,
        tracking_history_seconds=40,
    )
    registry = PersonRegistry(stream_session_id=session_id, source_id="native-research")
    pending, captured, events, observations = {}, [], {}, []
    eof, last_source, tracking_finalized = False, None, False

    def capture_scene():
        snapshot = scene.snapshot(include_history=True)
        for event in snapshot["scene"]["events"]:
            identifier = event.get("event_id") or str(
                (
                    event["event_label"],
                    event.get("subject_ids"),
                    event.get("first_evidence_timestamp_seconds"),
                )
            )
            row = {
                **event,
                "first_evidence": event.get("first_evidence_timestamp_seconds"),
                "confirmed_since": event.get("confirmed_since_timestamp_seconds"),
                "ended_since": event.get("ended_since_timestamp_seconds"),
            }
            if events.get(identifier) != row:
                audit.record(
                    "event",
                    stream_session_id=session_id,
                    event_id=identifier,
                    event=row,
                )
                events[identifier] = row
        return snapshot

    def drain_inspections():
        while jobs := scene.take_inspections(limit=4):
            if inspector is None:
                raise ValueError(
                    "event evaluation requires a configured native inspector"
                )
            for job in jobs:
                audit.record(
                    "window_start",
                    request_id=job.request.request_id,
                    requested_event_labels=list(job.request.event_labels),
                    start_seconds=job.request.start_seconds,
                    end_seconds=job.request.end_seconds,
                    source_frames=[
                        [
                            f.frame_sequence,
                            f.timestamp_seconds,
                        ]
                        for f in job.frames
                    ],
                )
                allowed = _validate_job(job)
                try:
                    verdicts = _validate_verdicts(job, inspector.inspect(job), allowed)
                except Exception as error:
                    scene.complete_inspection(job.request.request_id, (), success=False)
                    audit.record(
                        "window_failed",
                        request_id=job.request.request_id,
                        error_type=type(error).__name__,
                    )
                    raise
                completion_counts = {}
                accepted = scene.complete_inspection(
                    job.request.request_id,
                    verdicts,
                    success=True,
                    completion_counts=completion_counts,
                )
                record = audit.record(
                    "window_complete",
                    request_id=job.request.request_id,
                    accepted=accepted,
                    completion_counts=completion_counts,
                    allowed_subject_ids=sorted(allowed),
                    requested_event_labels=list(job.request.event_labels),
                    verdicts=[
                        {
                            "event_label": v.event_label,
                            "subject_ids": list(v.subject_ids),
                            "detected": v.detected,
                            "confidence": v.confidence,
                        }
                        for v in verdicts
                    ],
                )
                if not accepted:
                    raise ValueError("native observation completion was rejected")
                observations.append(record)
                capture_scene()

    def commit(result):
        if not result.frames:
            return

        def canonicalize(snapshots):
            if not registry.begin_native_epoch(result.backend, result.predictor_epoch):
                raise ValueError("native epoch rejected")
            mapped = []
            for raw in snapshots:
                registry.observe_snapshot(raw)
                item = registry.snapshot_at(raw.timestamp_seconds, raw.frame_sequence)
                if item is None or (item.frame_sequence, item.timestamp_seconds) != (
                    raw.frame_sequence,
                    raw.timestamp_seconds,
                ):
                    raise ValueError("canonical snapshot lost source frame")
                mapped.append(item)
            scene.apply_subject_merges(registry.drain_subject_merges())
            return tuple(mapped)

        sources = tuple(pending[s.frame_sequence] for s in result.frames)
        if not scene.commit_native_tracking_batch(
            result.frames,
            completed_timestamp_seconds=result.completed_timestamp_seconds,
            completed_frame_sequence=result.completed_frame_sequence,
            canonicalize=canonicalize,
            retained_sources=sources,
        ):
            raise ValueError(
                "native tracking commit rejected: " + str(scene.tracking_last_rejection)
            )
        for index, raw in enumerate(result.frames):
            mapped = registry.snapshot_at(raw.timestamp_seconds, raw.frame_sequence)
            frame = {
                "t": mapped.timestamp_seconds,
                "frame": mapped.frame_sequence,
                "subjects": [
                    {
                        "id": obj.track_id,
                        "bbox": [obj.bbox.x1, obj.bbox.y1, obj.bbox.x2, obj.bbox.y2],
                        "native_track": dict(obj.attributes["native_track"]),
                        "observation_state": obj.observation_state,
                        "tracking_confirmation": obj.attributes.get(
                            "tracking_confirmation", "confirmed"
                        ),
                    }
                    for obj in mapped.objects
                ],
            }
            captured.append(frame)
            extra = {}
            if getattr(result, "backend", None) == "detector_stream":
                if len(result.detector_provenance) != len(result.frames):
                    raise ValueError("detector frame lacks provenance")
                extra = {
                    "detector_model": result.detector_model,
                    "detector_provenance": result.detector_provenance[index],
                    "predicted_frame": result.predicted_frames[index],
                }
            audit.record("frame", stream_session_id=session_id, frame=frame, **extra)
            pending.pop(raw.frame_sequence)
        capture_scene()
        drain_inspections()

    try:
        tracking_session = tracker.open_session(session_id)
        limits = getattr(tracking_session, "limits", {})
        startup_frames = limits.get(
            "startup_frames", getattr(tracker, "startup_frames", 1)
        )
        append_frames = limits.get(
            "max_append_frames", getattr(tracker, "max_append_frames", 1)
        )
        if (
            type(startup_frames) is not int
            or type(append_frames) is not int
            or not 1 <= startup_frames <= append_frames <= 128
        ):
            raise ValueError("native worker advertised invalid startup/append limits")
        buffered = []
        first_append = True
        audit.record(
            "tracking_limits",
            startup_frames=startup_frames,
            max_append_frames=append_frames,
        )
        for frame in frames:
            if last_source is not None and (
                frame.frame_sequence <= last_source.frame_sequence
                or frame.timestamp_seconds <= last_source.timestamp_seconds
            ):
                raise ValueError("sampled source frame order must strictly increase")
            if len(pending) >= 128:
                raise ValueError(
                    "native worker did not emit bounded pending source frames"
                )
            pending[frame.frame_sequence] = frame
            scene.observe_frame(
                frame.jpeg_bytes, frame.timestamp_seconds, frame.frame_sequence
            )
            audit.record(
                "source_frame",
                frame=frame.frame_sequence,
                t=frame.timestamp_seconds,
            )
            buffered.append(frame)
            threshold = startup_frames if first_append else append_frames
            if len(buffered) >= threshold:
                commit(tracker.append(tuple(buffered), stream_session_id=session_id))
                buffered.clear()
                first_append = False
            last_source = frame
        if last_source is None:
            raise ValueError("native file decoded no sampled frames")
        eof = not source_recorded or any(
            row["kind"] == "source_eof" for row in audit.rows
        )
        if not eof:
            raise ValueError("source ended without natural EOF")
        if not source_recorded:
            audit.record(
                "source_eof",
                last_timestamp_seconds=last_source.timestamp_seconds,
                sampled_frames=len(captured) + len(pending),
                timestamp_basis="supplied_source_frames",
            )
        # A short file may begin and end in the same final append; negotiated
        # clients explicitly permit this even below the startup-frame threshold.
        commit(
            tracker.append(
                tuple(buffered), stream_session_id=session_id, is_last_batch=True
            )
        )
        if pending:
            raise ValueError("native worker left source frames unprocessed at EOF")
        finish_tracking = getattr(scene, "finish_tracking_sources", None)
        if finish_tracking is not None:
            finish_tracking()
        scene.finish_inspection_sources()
        drain_inspections()
        if scene.has_pending_inspections():
            raise ValueError("native observation work remains after EOF")
        final = capture_scene()
        audit.record("tracking_finished", stream_session_id=session_id)
        audit.record("final_scene", stream_session_id=session_id, scene=final)
        tracking_finalized = True
        clocks = [row for row in audit.rows if row["kind"] == "source_clock"]
        end = [row for row in audit.rows if row["kind"] == "source_eof"][-1]
        source_meta = clocks[-1] if clocks else metadata
        result = {
            "video_id": video_id,
            "fps": source_meta.get("file_fps", metadata.get("fps", 0)),
            "width": source_meta.get("width", metadata.get("width")),
            "height": source_meta.get("height", metadata.get("height")),
            "duration_seconds": end.get("last_timestamp_seconds"),
            "frames": captured,
            "events": list(events.values()),
            "observations": observations,
            "complete": True,
            "stream_session_id": session_id,
            "provenance": metadata,
        }
        (output / "capture.json").write_text(
            json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2) + "\n"
        )
        return result
    except BaseException as error:
        if not audit.failed:
            audit.record(
                "evaluation_failed",
                error_type=type(error).__name__,
                natural_eof=eof,
                tracking_finalized=tracking_finalized,
            )
        raise
    finally:
        scene.close()
        tracker.close()
        if inspector is not None:
            inspector.close()
        audit.close()


def _event_state(capture, label):
    episodes = [event for event in capture["events"] if event["event_label"] == label]
    if any(
        event.get("confirmed_since") is not None
        and event.get("phase") in {"confirmed", "ended"}
        for event in episodes
    ):
        return "present"
    if any(event.get("phase") == "suspected" for event in episodes):
        return "unknown"
    windows = capture.get("observations", ())
    if not windows:
        return "unknown"
    scene_scoped = label == "ground_litter"
    if capture.get("provenance", {}).get("runtime_profile") == "frozen51":
        from goodvision_harness.event_session import requires_subject

        scene_scoped = not requires_subject(label)
    for window in windows:
        verdicts = window.get("verdicts", ())
        matches = [row for row in verdicts if row["event_label"] == label]
        counts = window.get("completion_counts", {})
        if (
            not window.get("accepted")
            or label not in window.get("requested_event_labels", ())
            or not matches
            or counts.get("accepted_verdicts") != len(verdicts)
            or any(row["detected"] is not False for row in matches)
        ):
            return "unknown"
        allowed = set(window.get("allowed_subject_ids", ()))
        covered = {subject for row in matches for subject in row["subject_ids"]}
        if (scene_scoped and covered) or (
            not scene_scoped and (not allowed or not allowed <= covered)
        ):
            return "unknown"
    return "absent"


def deterministic_answer(question, capture):
    """Keep absent, omitted and unconfirmed native evidence distinguishable."""
    if not capture.get("complete"):
        raise ValueError("incomplete native capture cannot produce a prediction")
    if question.capability != "event_recognition":
        return answer_question(question, Capture.from_json(capture))
    labels = labels_for_question(question)
    states = [_event_state(capture, label) for label in labels]
    binary = (
        question.raw.get("question_form") == "event_presence_binary"
        or set(question.response_options or ()) == {"yes", "no"}
        or question.answer_type != "single_choice"
    )
    if binary:
        if len(states) != 1 or states[0] == "unknown":
            return ""
        return "yes" if states[0] == "present" else "no"
    options = question.response_options or ()
    if question.public_context.get("response_format") == "event_presence_combination":
        if "unknown" in states:
            return ""
        matched = []
        for option in options:
            tokens = [token.strip() for token in option.split(";")]
            parsed = [
                re.fullmatch(r"([A-Z]):\s*(present|absent)", token) for token in tokens
            ]
            if (
                len(tokens) == len(states)
                and all(parsed)
                and all(
                    match.group(1) == chr(ord("A") + index)
                    for index, match in enumerate(parsed)
                )
                and [match.group(2) for match in parsed] == states
            ):
                matched.append(option)
        return matched[0] if len(matched) == 1 else ""
    present = [
        label for label, state in zip(labels, states, strict=True) if state == "present"
    ]
    unresolved_positive = any(
        event.get("phase") == "suspected" and event["event_label"] in labels
        for event in capture["events"]
    )
    unresolved_positive = unresolved_positive or any(
        window.get("accepted")
        and window.get("completion_counts", {}).get("accepted_verdicts")
        == len(window.get("verdicts", ()))
        and any(
            row["detected"] is True
            and row["event_label"] in labels
            and states[labels.index(row["event_label"])] != "present"
            for row in window.get("verdicts", ())
        )
        for window in capture.get("observations", ())
    )
    if len(present) == 1 and not unresolved_positive:
        return next(
            (option for option in options if canonical_label(option) == present[0]), ""
        )
    if (
        not present
        and states
        and all(state == "absent" for state in states)
        and NONE_OPTION in options
    ):
        return NONE_OPTION
    return ""


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--media", type=Path, required=True)
    parser.add_argument("--media-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--runtime-profile",
        choices=("current-main", "frozen51"),
        default="current-main",
    )
    parser.add_argument("--backend", choices=("detector", "sam"), default="detector")
    parser.add_argument(
        "--detector-model", choices=("yolo26-s", "rfdetr-l"), default="yolo26-s"
    )
    parser.add_argument("--tracking-endpoint", required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument(
        "--tracking-key-env",
        help="environment variable containing the worker token; never logged",
    )
    parser.add_argument("--geometry-only", action="store_true")
    parser.add_argument("--max-model-calls", type=int, default=1000)
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args(argv)
    budget = ModelCallBudget(args.max_model_calls)
    select_runtime(args.runtime_profile)
    from goodvision_harness.config import load_config
    from goodvision_harness.detector_native_session import DetectorNativeSessionAdapter
    from goodvision_harness.perception import (
        create_tracking_client,
        is_contour_endpoint,
    )

    allowed = set(available_events(args.runtime_profile))
    questions, media = load_questions(args.questions), load_media(args.media)
    if len({q.qa_id for q in questions}) != len(questions):
        raise ValueError("question IDs must be unique")
    plan = []
    for question in questions:
        labels = labels_for_question(question)
        if set(labels) - allowed:
            raise ValueError(
                "unsupported labels in selected runtime profile: "
                + ", ".join(sorted(set(labels) - allowed))
            )
        if args.geometry_only and labels:
            raise ValueError("geometry-only cannot evaluate event questions")
        path = Path(media[question.video_id]["path"])
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("media paths must be relative to media-root")
        path = (args.media_root / path).resolve()
        if not path.is_relative_to(args.media_root.resolve()) or not path.is_file():
            raise ValueError("media file is missing or outside media-root")
        source_record = {
            "video_id": question.video_id,
            "media_file": str(path),
            "manifest_relative_path": media[question.video_id]["path"],
        }
        plan.append((question, path, source_record, labels))
    frozen_record = frozen_source_inventory()
    frozen_commit = frozen_record["source_commit"]
    if args.plan_only:
        print(
            json.dumps(
                {
                    "runtime_profile": args.runtime_profile,
                    "frozen_source_commit": frozen_commit,
                    "questions": len(plan),
                    "event_questions": sum(bool(item[3]) for item in plan),
                    "max_model_calls": budget.limit,
                    "makes_model_calls": False,
                }
            )
        )
        return 0
    # Resolve configuration before creating output or contacting a worker.
    config = load_config(
        args.config
        or (FROZEN if args.runtime_profile == "frozen51" else ROOT) / "config.yaml"
    )
    tracking_key = None
    if args.tracking_key_env:
        if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", args.tracking_key_env) is None:
            raise ValueError("tracking-key-env must be an environment variable name")
        tracking_key = os.environ.get(args.tracking_key_env)
        if not tracking_key:
            raise ValueError("tracking credential environment variable is unavailable")
    if (
        args.backend == "sam"
        and is_contour_endpoint(args.tracking_endpoint)
        and not tracking_key
    ):
        raise ValueError("authenticated tracking requires --tracking-key-env")
    if args.backend == "sam" and config.rich_labeling.backend != "sam31_stream":
        raise ValueError("SAM native research requires sam31_stream configuration")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    try:
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        revision = None
    changed_files = subprocess.check_output(
        ["git", "diff", "HEAD", "--name-only", "--",
         "src/goodvision_harness", "benchmark"],
        cwd=ROOT, text=True,
    ).splitlines()
    runtime_root = FROZEN if args.runtime_profile == "frozen51" else ROOT
    provenance = {
        "schema": 2,
        "runtime_profile": args.runtime_profile,
        "runtime_source_commit": frozen_commit
        if args.runtime_profile == "frozen51"
        else revision,
        "orchestrator_commit": revision,
        "orchestrator_has_uncommitted_changes": bool(changed_files),
        "uncommitted_source_files": changed_files,
        "extractor_file": str(Path(__file__).with_name("extract.py")),
        "native_runner_file": str(Path(__file__).resolve()),
        "runtime_source_inventory": {
            "root": str(runtime_root),
            "files": [
                str(path.relative_to(runtime_root))
                for path in sorted((runtime_root / "src/goodvision_harness").rglob("*.py"))
            ],
        },
        "frozen_source_record": frozen_record,
        "questions_file": str(args.questions.resolve()),
        "media_manifest_file": str(args.media.resolve()),
        "config_file": str((
            args.config
            or (FROZEN if args.runtime_profile == "frozen51" else ROOT) / "config.yaml"
        ).resolve()),
        "backend": args.backend,
        "detector_model": args.detector_model if args.backend == "detector" else None,
        "historical_publication_reproduction": False,
        "observation_reads_questions": False,
        "final_llm": False,
    }
    from urllib.parse import urlsplit

    settings = {
        "model": config.inference.model,
        "provider": config.model_api.provider,
        "model_host": urlsplit(config.model_api.endpoint).hostname,
        "inspection_host": urlsplit(config.model_api.inspection_endpoint).hostname
        if getattr(config.model_api, "inspection_endpoint", None)
        else None,
        "request_parameters": config.inference.request_parameters,
        "tracking_sample_fps": config.rich_labeling.max_fps,
        "observation_fps": config.runtime.scene_observation_fps,
        "max_tokens": config.runtime.scene_max_tokens,
        "max_image_side": config.runtime.scene_max_image_side,
        "jpeg_quality": config.runtime.scene_jpeg_quality,
        "timeout_seconds": config.model_api.timeout_seconds,
    }
    packages = {}
    for name in ("av", "httpx", "Pillow", "numpy", "opencv-python-headless"):
        try:
            packages[name] = package_metadata.version(name)
        except package_metadata.PackageNotFoundError:
            packages[name] = None
    provenance.update(
        effective_settings=settings,
        python=platform.python_version(),
        dependencies=packages,
    )
    (args.output_dir / "manifest.json").write_text(
        json.dumps(provenance, indent=2) + "\n"
    )
    failures = 0
    aborted = False
    with (
        (args.output_dir / "predictions.jsonl").open("x") as predictions,
        (args.output_dir / "details.jsonl").open("x") as details,
    ):
        for question, path, source_record, labels in plan:
            directory = args.output_dir / "questions" / quote(question.qa_id, safe="")
            if aborted:
                predictions.write(
                    json.dumps({"qa_id": question.qa_id, "prediction": ""}) + "\n"
                )
                details.write(
                    json.dumps(
                        {
                            "qa_id": question.qa_id,
                            "prediction": "",
                            "status": "not_attempted_after_failure",
                        }
                    )
                    + "\n"
                )
                continue
            Audit = _load_research_helper(
                "_research_audit", "native_event_audit.py"
            ).NativeAudit
            audit = Audit(directory / "records.jsonl")
            tracker = (
                DetectorNativeSessionAdapter(
                    args.tracking_endpoint, model=args.detector_model
                )
                if args.backend == "detector"
                else create_tracking_client(
                    __import__("dataclasses").replace(
                        config.rich_labeling, endpoint=args.tracking_endpoint
                    ),
                    api_key=tracking_key,
                )
            )
            inspector = make_inspector(config, audit, budget) if labels else None
            source = _load_research_helper(
                "_research_source", "offline_benchmark.py"
            ).iter_native_frames
            try:
                capture = evaluate_frames(
                    source(
                        path,
                        sample_fps=config.rich_labeling.max_fps,
                        max_image_side=config.runtime.scene_max_image_side,
                        jpeg_quality=config.runtime.scene_jpeg_quality,
                        record=audit.record,
                    ),
                    tracker,
                    video_id=question.video_id,
                    output=directory,
                    metadata={**provenance, "media_source": source_record, "_audit": audit},
                    labels=labels,
                    inspector=inspector,
                    observation_fps=config.runtime.scene_observation_fps,
                    source_recorded=True,
                )
                prediction = deterministic_answer(question, capture)
                row = {
                    "qa_id": question.qa_id,
                    "prediction": prediction,
                    "status": "answered" if prediction else "unknown_or_ambiguous",
                }
            except Exception as error:
                failures += 1
                row = {
                    "qa_id": question.qa_id,
                    "prediction": "",
                    "status": "failed",
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
            predictions.write(
                json.dumps(
                    {"qa_id": question.qa_id, "prediction": row["prediction"]},
                    ensure_ascii=False,
                )
                + "\n"
            )
            details.write(json.dumps(row, ensure_ascii=False) + "\n")
            predictions.flush()
            details.flush()
            print(json.dumps(row, ensure_ascii=False), flush=True)
            if failures:
                aborted = True  # Emit remaining blank rows without further requests.
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
