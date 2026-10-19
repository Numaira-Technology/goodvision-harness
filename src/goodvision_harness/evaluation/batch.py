"""Run a frozen video batch and import terminal sources without new inference.

The exclusive outer controller owns GPU allocation and release. Recovery uses a
new frozen plan and output directory; completed sources keep their original IDs.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

from .capture_source import capture_source, _read_frozen
from .failures import failure_policy
from .privacy import reject_private
from .recording import publish_once
from .capture_recovery import import_reused_source


def _ownership(receipt):
    if (receipt.get("owned") is not True or receipt.get("gpu_name") != "RTX 4090"
            or type(receipt.get("gpu_memory_mib")) is not int
            or not 23000 <= receipt["gpu_memory_mib"] <= 26000
            or receipt.get("num_gpus") != 1 or not receipt.get("instance_id")
            or not receipt.get("experiment_label")
            or receipt.get("sam_internal_patch") is not False):
        raise ValueError("actual owned standard24GB single4090 and stockSAM receipt required")
    return receipt


def run_sources(plan_path, *, output, ledger, resources, plan_sha256=None):
    """Use owned resource receipts; stop shared failures before the next source.

    prepare(config) and recover(config, failure, cleanup) perform resource actions,
    not health probes. Recovery must confirm cleanup on the same owned instance.
    """
    plan_path = Path(plan_path).resolve()
    plan = _read_frozen(plan_path)
    reject_private(plan, observation=True)
    if plan.get("schema_version") != "unified-capture-batch-v1":
        raise ValueError("explicit formal source batch required")
    entries, sids, videos = [], set(), set()
    for row in plan["sources"]:
        path = plan_path.parent / row["config"]
        config = _read_frozen(path)
        reject_private(config, observation=True)
        sid, vid = config["stream_session_id"], config["video_id"]
        if sid in sids or vid in videos:
            raise ValueError("one fresh SID and capture per video per method required")
        sids.add(sid)
        videos.add(vid)
        entries.append((row, path, config))
    if not entries:
        raise ValueError("empty formal source batch")
    skips = list(plan.get("input_skips", []))
    skip_ids = [row["video_id"] for row in skips]
    if len(skip_ids) != len(set(skip_ids)) or set(skip_ids) & videos:
        raise ValueError("input skips must be unique and disjoint from scheduled sources")
    if any(row.get("status") != "skipped" for row in skips):
        raise ValueError("only explicit terminal input skips belong in input_skips")
    root = Path(output).resolve()
    root.mkdir(parents=True, exist_ok=False)
    (root / "sources").mkdir()
    (root / "progress").mkdir()
    publish_once(root / "run.json", {"method": plan["method"], "plan_path": str(plan_path), "plan": plan,
                                     "sources": len(entries), "started_at_utc": datetime.now(timezone.utc).isoformat()})
    outcomes, observations = list(plan.get("input_skips", [])), []
    try:
        for position, (row, path, config) in enumerate(entries, 1):
            source_root = root / "sources" / f"{position:04d}"
            if row.get('reuse'):
                reused = row['reuse']
                owner = _ownership(reused['owner'])
                terminal = import_reused_source(reused, config, source_root)
                publish_once(root / 'progress' / f'{position:04d}-resource.json', owner)
                publish_once(root / 'progress' / f'{position:04d}-reuse.json', {
                    'origin_round': reused['origin_round'], 'source_directory': reused['source_directory'],
                    'original_instance_id': owner['instance_id'], 'new_inference_performed': False})
                if reused.get('recovery'):
                    publish_once(root / 'progress' / f'{position:04d}-recovery.json', reused['recovery'])
                publish_once(root / 'progress' / f'{position:04d}-outcome.json', terminal)
                outcomes.append(terminal)
                capture = source_root / 'capture.json'
                if capture.exists():
                    observations.append({'video_id': config['video_id'], 'capture_path': str(capture)})
                continue
            owner = _ownership(resources.prepare(config))
            publish_once(root / "progress" / f"{position:04d}-resource.json", owner)
            failure, outcome = None, None
            try:
                outcome = capture_source(path, output=source_root, ledger=ledger)
            except Exception as exc:
                failure = failure_policy(exc)
                # Shared I/O and identity errors stop the batch without failing unattempted sources.
                if failure["scope"] == "shared":
                    raise
            cleanup_path = source_root / "cleanup.json"
            cleanup = json.loads(cleanup_path.read_bytes()) if cleanup_path.exists() else {}
            if (not cleanup.get("local_cleanup_confirmed") or cleanup.get("errors")
                    or cleanup.get("requires_worker_reset") or failure):
                recovery = _ownership(resources.recover(config, failure, cleanup))
                if (recovery["instance_id"] != owner["instance_id"]
                        or recovery["experiment_label"] != owner["experiment_label"]
                        or recovery.get("safe_to_continue") is not True):
                    raise RuntimeError("source_cleanup_not_confirmed")
                publish_once(root / "progress" / f"{position:04d}-recovery.json", recovery)
            status = "skipped" if failure else outcome["status"]
            terminal = {"video_id": config["video_id"], "stream_session_id": config["stream_session_id"],
                        "status": status, "failure_policy": failure, "capture_directory": str(source_root),
                        "resource_instance_id": owner['instance_id'], "reused_capture": False}
            publish_once(root / "progress" / f"{position:04d}-outcome.json", terminal)
            outcomes.append(terminal)
            capture = source_root / "capture.json"
            if capture.exists():
                observations.append({"video_id": config["video_id"], "capture_path": str(capture)})
    except Exception as exc:
        if not any(row["video_id"] == config["video_id"] for row in outcomes):
            outcomes.append({"video_id": config["video_id"], "stream_session_id": config["stream_session_id"],
                             "status": "shared_failure", "failure_policy": failure_policy(exc),
                             "capture_directory": str(source_root)})
        publish_once(root / "recovery-required.json", {
            "failure_policy": failure_policy(exc), "processed_sources_including_input_skips": len(outcomes),
            "attempted_capture_sources": position,
            "unattempted_sources": max(0, len(entries) - position),
            "affected_source": config["video_id"],
            "do_not_repeat_old_entrypoint": True})
        raise
    finally:
        publish_once(root / "observation-index.json", {
            "schema_version": "unified-observation-index-v1", "method": plan["method"],
            "observations": observations, "processed_sources": outcomes})
    completion = {"method": plan["method"], "sources": len(entries), "outcomes": outcomes,
                  "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                  "all_sources_processed_not_all_answered": True}
    publish_once(root / "completion.json", completion)
    return completion
