"""Capture a new session from frozen source frames through native lanes into immutable evidence.

The outer controller owns configuration freezing, exclusive 4090 allocation,
ledger checks, retries, source skips, and shared identity/I/O/connection recovery.
This entry point sends no health or warmup requests.
"""
from __future__ import annotations

import argparse
from dataclasses import fields
import json
import os
import re
from pathlib import Path
import time
from urllib.parse import urlparse
import httpx

from ..contracts import json_mapping_as_dict
from ..model_api import ModelAPISettings
from ..perception import PerceptionSessionHTTPClient
from ..rich_labeling import RichLabelHTTPClient
from ..video_tracking import VideoFrame
from ..video_tracking_session import VideoSessionTrackingHTTPClient
from .offline_observer import fixed_window_plan
from .failures import failure_policy
from .model_ledger import ModelLedger
from .privacy import reject_private
from .profile import load_profile
from .recording import publish_once
from .session import UnifiedCaptureSession
from .tracking_adapter import CountProtocolTracker


def _read_frozen(path, expected=None):
    """Read a saved input; legacy checksum arguments are not used."""
    return json.loads(Path(path).read_bytes())


def _tracking_result(result):
    # Native attributes are deeply frozen; asdict/deepcopy cannot copy mappingproxy values.
    return {field.name: ([frame.as_dict() for frame in result.frames]
                         if field.name == "frames" else getattr(result, field.name))
            for field in fields(result)}


def _failure_detail(error, *, credential=""):
    detail = {"error_type": type(error).__name__, "error_code": getattr(error, "code", None)}
    if isinstance(error, httpx.HTTPStatusError):
        body = error.response.content
        text = body[:16384].decode("utf-8", errors="replace")
        detail.update(http_status=error.response.status_code, response_bytes=len(body),
                      detail_truncated=len(body) > 16384)
    elif isinstance(error, (ValueError, RuntimeError)):
        text = str(error)[:2048]
    else:
        text = type(error).__name__
    if credential:
        text = text.replace(credential, "[REDACTED]")
    text = re.sub(r"(?i)\bbearer\s+[^\s\"']+", "Bearer [REDACTED]", text)
    text = re.sub(r"(?i)((?:api[_-]?key|access[_-]?token|authorization|password)[\"']?\s*[:=]\s*[\"']?)[^\s\"',}]+",
                  r"\1[REDACTED]", text)
    detail["detail_redacted"] = text
    return detail


def capture_source(config_path, *, output, ledger, config_sha256=None):
    from .configuration import source_settings
    config_path = Path(config_path).resolve()
    config = _read_frozen(config_path)
    reject_private(config, observation=True)
    if (config.get("schema_version") != "unified-source-run-v1"
            or config.get("gpu_policy") != "RTX4090_24GB_only"
            or config.get("sam_internal_patch") is not False
            or config.get("face_enabled") is not False
            or config.get("question_conditioned") is not False):
        raise ValueError("explicit standard4090/stockSAM/no-face/unconditioned run required")
    base = config_path.parent
    manifest = _read_frozen(base / config["source_manifest"])
    reject_private(manifest, observation=True)
    if manifest.get("source_eof") is not True or not manifest["frames"]:
        raise ValueError("prepared source requires original decoder EOF and nonempty frame plan")
    if manifest["video_id"] != config["video_id"]:
        raise ValueError("source manifest video identity mismatch")
    frame_plan = [{"frame": row["source_frame_index"], "t": row["timestamp_seconds"]}
                  for row in manifest["frames"]]
    profile = load_profile(base / config["event_definitions"])
    window_plan = fixed_window_plan(frame_plan, profile,
                                    observation_fps=config["observation_fps"])
    # Wrap this inspector's transport with a ledger; leave the global production client untouched.
    settings = ModelAPISettings.from_mapping(config["vlm"]["settings"])
    endpoint = urlparse(settings.endpoint)
    if (settings.inspection_endpoint or settings.max_retries != 0
            or endpoint.scheme != "https" or endpoint.username or endpoint.password
            or endpoint.query or endpoint.fragment or not isinstance(ledger, ModelLedger)):
        raise ValueError("frozen HTTPS model route, no hidden retry and active ledger required")
    if ledger.seed_manifest_path != (base / config["model_ledger"]["seed_manifest"]).resolve():
        raise ValueError("active model ledger differs from frozen source configuration")
    credential = os.environ.get(settings.api_key_env, "")
    vlm_config = profile.vlm_config(
        stream_session_id=config["stream_session_id"], source_id=config["source_id"], generation=1,
        model_settings=settings, model=config["vlm"]["model"], api_key=credential,
        parameters=config["vlm"]["parameters"], options=config["vlm"]["inspector_settings"],
        observation_fps=config["observation_fps"], max_in_flight=1,
        route_id=config["vlm"]["route_id"])
    tracking = config["tracking"]
    if tracking["mode"] not in {"native_session", "immediate", "count_protocol"}:
        raise ValueError("unknown formal tracking mode")
    count_protocol = tracking["mode"] == "count_protocol"
    native_batches = tracking["mode"] == "native_session" or (count_protocol and tracking["model"] == "sam3.1")
    session = UnifiedCaptureSession(
        output, video_id=config["video_id"], stream_session_id=config["stream_session_id"],
        source_id=config["source_id"], source=manifest["source"], frame_plan=frame_plan,
        window_plan=window_plan, provenance={**config["provenance"], "run_config_path": str(config_path),
                                           "source_configuration": source_settings(config),
                                           "source_manifest_path": str((base / config["source_manifest"]).resolve()),
                                           "ledger_manifest": str(ledger.seed_manifest_path)},
        profile=profile, vlm_config=vlm_config,
        tracking_mode="batch" if native_batches else "immediate",
        source_fps=manifest["sampling_fps"], max_entities=config["scene_max_entities"], ledger=ledger)
    root = session.recorder.root
    client = None
    opened = None
    successful = False
    started = time.monotonic()
    try:
        if tracking["mode"] == "native_session":
            native_options = dict(timeout_seconds=75, transport_codec="jpeg",
                                  max_response_objects=tracking["expected_native_limits"]["max_objects"])
            if tracking["processing_method"] == "sam_live":
                client = PerceptionSessionHTTPClient(tracking["endpoint"], method="sam_live", **native_options)
            elif tracking["processing_method"] is None:
                client = VideoSessionTrackingHTTPClient(tracking["endpoint"], **native_options)
            else:
                raise ValueError("unknown frozen native SAM route")
            opened = client.open_session(config["stream_session_id"])
            session.recorder.operational_record("tracking-session", {
                "video_session_id": opened.video_session_id, "stream_session_id": opened.stream_session_id,
                "predictor_epoch": opened.predictor_epoch, "limits": json_mapping_as_dict(opened.limits)})
            if opened.limits != tracking["expected_native_limits"]:
                raise ValueError("native SAM limits differ from frozen configuration")
        elif count_protocol:
            client = CountProtocolTracker(tracking["endpoint"], model=tracking["model"],
                    sid=config["stream_session_id"], parameters=tracking["parameters"],
                    expected_weights_path=tracking.get("weights_path"))
            opened = client.open()
            session.recorder.operational_record("tracking-session", opened)
        else:
            client = RichLabelHTTPClient(tracking["endpoint"], timeout_seconds=75,
                                         mode="tracks", include_attributes=False)
        session.recorder.timing({"kind": "tracking_initialization", "clock_domain": session.recorder.clock_domain,
                                 "started_monotonic": started, "completed_monotonic": time.monotonic(),
                                 "includes_actual_session_open": opened is not None,
                                 "cold_model_load_or_reuse_not_separately_observed": True})
        pending = []
        for index, row in enumerate(manifest["frames"]):
            data = (base / manifest["frames_directory"] / row["file"]).read_bytes()
            frame = VideoFrame(row["source_frame_index"], row["timestamp_seconds"], data)
            session.source_frame(frame)
            last = index + 1 == len(manifest["frames"])
            if tracking["mode"] == "native_session":
                pending.append(frame)
                if index + 1 < client.startup_frames and not last:
                    continue
                request_frames = tuple(pending)
            else:
                request_frames = (frame,)
            request_id = f"{config['stream_session_id']}:tracking:{frame.frame_sequence}"
            request_start = time.monotonic()
            session.recorder.operational_record("tracking-request", {
                "request_id": request_id, "stream_session_id": config["stream_session_id"],
                "frame_sequences": [value.frame_sequence for value in request_frames],
                "state": "dispatching", "is_last_batch": last,
                "request_start": request_start, "clock_domain": session.recorder.clock_domain})
            response_complete = None
            try:
                if tracking["mode"] == "native_session":
                    result = client.append(request_frames, stream_session_id=config["stream_session_id"],
                                           is_last_batch=last, request_id=request_id, retry_transport=False)
                    response_complete = time.monotonic()
                    session.recorder.operational_record("tracking-receipt", {
                        "request_id": request_id, "state": "received", "response_complete": response_complete,
                        "result": _tracking_result(result)})
                    session.tracking_batch(result)
                    pending.clear()
                else:
                    result = client.infer(data, stream_session_id=config["stream_session_id"],
                                          frame_sequence=frame.frame_sequence,
                                          timestamp_seconds=frame.timestamp_seconds)
                    response_complete = time.monotonic()
                    session.recorder.operational_record("tracking-receipt", {
                        "request_id": request_id, "state": "received", "response_complete": response_complete,
                        "result": _tracking_result(result) if native_batches else result.as_dict(),
                        "raw_count_protocol_response": client.last_response if count_protocol else None})
                    if native_batches:
                        session.tracking_batch(result)
                    else:
                        session.tracking_snapshot(result)
            except Exception as exc:
                ended = time.monotonic()
                terminal = {"request_id": request_id, "state": "failed",
                            "stage": "request" if response_complete is None else "local_response_contract",
                            "request_start": request_start, "response_complete": response_complete,
                            "ended_monotonic": ended, "clock_domain": session.recorder.clock_domain,
                            "failure_policy": failure_policy(exc), **_failure_detail(exc, credential=credential)}
                # Persist the terminal failure even if the adapter is poisoned, independent of the VLM ledger.
                publish_once(root / f"tracking-failure-{frame.frame_sequence:08d}.json", terminal)
                raise
            session.recorder.timing({"clock_domain": session.recorder.clock_domain,
                                      "kind": "tracking_request", "request_id": request_id,
                                      "frame_sequences": [value.frame_sequence for value in request_frames],
                                      "request_start": request_start, "response_complete": response_complete,
                                      "status": "succeeded", "cache_replay": False})
        if count_protocol:
            session.recorder.operational_record("tracking-receipt", {
                "kind": "actual_eof_drain", "result": client.finish()})
        drained = (client.stats()["state"] == "closed" and client.stats()["pending_frame_count"] == 0
                   if tracking["mode"] == "native_session" else True)
        session.recorder.operational_record("source-eof", {
            "all_prepared_frames_consumed": True, "original_decoder_source_eof": True,
            "source_manifest_path": str((base / config["source_manifest"]).resolve()),
            "source_eof_monotonic": time.monotonic()})
        evidence = session.finish(source_eof=True, tracking_drained=drained)
        outcome = {"schema_version": "unified-source-outcome-v1", "video_id": config["video_id"],
                   "stream_session_id": config["stream_session_id"],
                   "status": "captured" if evidence["observation_manifest"]["complete"] else "incomplete",
                   "source_id": config["source_id"], "failures": evidence["failures"],
                   "processing_wall_seconds": time.monotonic() - started,
                   "scope": "offline_native_components_without_face", "final_answer_generated": False}
        publish_once(root / "outcome.json", outcome)
        successful = True
        return outcome
    except Exception as exc:
        # Save detector failure details and body hashes here; keep VLM receipts in the cumulative ledger.
        publish_once(root / "outcome.json", {
            "schema_version": "unified-source-outcome-v1", "video_id": config["video_id"],
            "stream_session_id": config["stream_session_id"], "status": "interrupted",
            "failure_policy": failure_policy(exc),
            "error_type": type(exc).__name__, "error_code": getattr(exc, "code", None),
            "errno": getattr(exc, "errno", None), "final_answer_generated": False,
            "failure_detail": _failure_detail(exc, credential=credential),
            "source_eof": False, "tracking_drained": False,
            "next_action": "outer_controller_classifies_model_skip_or_shared_resource_recovery"})
        raise
    finally:
        cleanup = {"stream_session_id": config["stream_session_id"],
                   "native_session_opened": opened is not None,
                   "remote_cleanup_confirmed": False, "local_cleanup_confirmed": False,
                   "requires_worker_reset": not successful, "errors": []}
        try:
            try:
                session.close()
                cleanup["local_cleanup_confirmed"] = True
            except Exception as exc:
                cleanup["errors"].append({"stage": "local_runtime", "type": type(exc).__name__})
        finally:
            if client is not None:
                if count_protocol:
                    try:
                        client.discard()
                        cleanup["remote_cleanup_confirmed"] = client.drained
                    except Exception as exc:
                        cleanup["errors"].append({"stage": "count_session_close", "type": type(exc).__name__})
                if tracking["mode"] == "native_session" and opened is not None:
                    try:
                        client.discard_session(timeout_seconds=15)
                        cleanup["remote_cleanup_confirmed"] = True
                    except httpx.HTTPStatusError as exc:
                        cleanup["remote_cleanup_confirmed"] = exc.response.status_code in (404, 410)
                        if not cleanup["remote_cleanup_confirmed"]:
                            cleanup["errors"].append({"stage": "native_delete", "status": exc.response.status_code})
                    except Exception as exc:
                        cleanup["errors"].append({"stage": "native_delete", "type": type(exc).__name__})
                try:
                    client.close()
                except Exception as exc:
                    cleanup["errors"].append({"stage": "client_close", "type": type(exc).__name__})
            if (tracking["mode"] == "native_session" or count_protocol) and not cleanup["remote_cleanup_confirmed"]:
                cleanup["requires_worker_reset"] = True
            # The immediate service uses native SID isolation/LRU cleanup; no remote DELETE occurred.
            cleanup["immediate_session_policy"] = "new_sid_native_lru" if tracking["mode"] == "immediate" else None
            publish_once(root / "cleanup.json", cleanup)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--config-sha256", help=argparse.SUPPRESS)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = _read_frozen(args.config)
    ledger_config = config["model_ledger"]
    ledger = ModelLedger(args.config.parent / ledger_config["output_directory"],
                         seed_manifest_path=args.config.parent / ledger_config["seed_manifest"],
                         owner_lock_path=args.config.parent / ledger_config["owner_lock"])
    try:
        capture_source(args.config, output=args.output, ledger=ledger)
    finally:
        ledger.close()


if __name__ == "__main__":
    main()
