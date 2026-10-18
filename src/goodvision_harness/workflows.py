"""Bounded workflow drafts, applied group snapshots, and native execution plans.

Preparing a candidate never changes state. The fleet owner checks affected camera
lifecycles before committing that candidate under its control lock.
"""

from __future__ import annotations

import copy
import json
import math
import os
import re
from pathlib import Path
from threading import RLock
from typing import Any
from uuid import uuid4

from .event_catalog import event_catalog

SCHEMA_VERSION = 1
MAX_DOCUMENT_BYTES = 2 * 1024 * 1024
MAX_WORKFLOWS = 64
MAX_GROUPS = 128
MAX_TRASH = 100
MAX_NODES = 64
MAX_EDGES = 256
MAX_CAMERAS = 1024
MAX_PROMPT = 4000
_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,95}\Z")
_CAMERA = re.compile(r"camera(?:0[1-9]|[1-9][0-9]+)\Z")
NODE_PORTS = {
    "source": {"inputs": (), "outputs": ("frames",)},
    "face": {"inputs": ("frames",), "outputs": ("faces",)},
    "events": {"inputs": ("frames",), "outputs": ("events",)},
    "summary": {"inputs": ("frames",), "outputs": ("summary",)},
    "output": {"inputs": ("frames", "faces", "events", "summary"), "outputs": ()},
}
_NAMES = {
    "source": "Video source",
    "face": "Face recognition",
    "events": "Safety events",
    "summary": "Scene summary",
    "output": "Output",
}


class WorkflowValidationError(ValueError):
    """The proposed assets or graph cannot be represented safely."""


class WorkflowConflictError(RuntimeError):
    """An optimistic revision no longer matches the stored document."""


class WorkflowStorageError(RuntimeError):
    """Persistent state could not be read or atomically written."""


def empty_document() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "revision": 0,
        "workflows": [],
        "groups": [],
        "trash": [],
    }


def _fail(message: str) -> None:
    raise WorkflowValidationError(message)


def _object(value: Any, allowed: set[str], label: str) -> dict:
    if not isinstance(value, dict) or set(value) - allowed:
        _fail(f"invalid {label} fields")
    return value


def _identifier(value: Any, label: str = "id") -> str:
    if not isinstance(value, str) or not _ID.fullmatch(value):
        _fail(f"invalid {label}")
    return value


def _name(value: Any, label: str = "name") -> str:
    if (
        not isinstance(value, str)
        or not value.strip()
        or len(value) > 128
        or any(ord(c) < 32 or ord(c) == 127 for c in value)
    ):
        _fail(f"{label} must be nonempty text under 129 characters")
    return value.strip()


def _boolean(value: Any, label: str) -> bool:
    if type(value) is not bool:
        _fail(f"{label} must be boolean")
    return value


def _integer(value: Any, label: str) -> int:
    if type(value) is not int or not 0 <= value < 2**53:
        _fail(f"invalid {label}")
    return value


def _list(value: Any, limit: int, label: str) -> list:
    if not isinstance(value, list) or len(value) > limit:
        _fail(f"{label} must be a list with at most {limit} entries")
    return value


def _bytes(value: Any) -> bytes:
    try:
        body = json.dumps(
            value, ensure_ascii=False, allow_nan=False, separators=(",", ":")
        ).encode("utf-8")
    except (TypeError, ValueError, OverflowError, RecursionError):
        _fail("workflow assets must contain finite JSON values")
    if len(body) >= MAX_DOCUMENT_BYTES:
        _fail("workflow assets exceed the 2 MiB limit")
    return body


def _event_options() -> dict[str, dict]:
    # Detector membership and sensitivity options remain owned by the live catalog.
    return {
        item["id"]: item
        for item in event_catalog()["events"]
        if item.get("kind") != "identity"
    }


def _config(kind: str, value: Any) -> dict:
    allowed = {"name"}
    if kind == "events":
        allowed |= {"features", "event_sensitivities"}
    elif kind == "summary":
        allowed |= {"interval_seconds", "prompt"}
    raw = _object(value, allowed, "component configuration")
    config = {"name": _name(raw.get("name", _NAMES[kind]))}
    if kind == "events":
        catalog = _event_options()
        features = _list(raw.get("features", list(catalog)), 64, "event features")
        if (
            not features
            or any(not isinstance(f, str) or f not in catalog for f in features)
            or len(set(features)) != len(features)
        ):
            _fail("event features must be unique supported safety events")
        supplied = raw.get("event_sensitivities", {})
        if not isinstance(supplied, dict) or set(supplied) - set(features):
            _fail("event sensitivities must refer to selected safety events")
        sensitivities = {}
        for feature in catalog:
            if feature not in features:
                continue
            sensitivity = catalog[feature].get("sensitivity")
            if sensitivity is None:
                if feature in supplied:
                    _fail("this event has no configurable sensitivity")
                continue
            level = supplied.get(feature, sensitivity["default"])
            if level not in [option["value"] for option in sensitivity["options"]]:
                _fail("unsupported event sensitivity")
            sensitivities[feature] = level
        config.update(
            features=[f for f in catalog if f in features],
            event_sensitivities=sensitivities,
        )
    elif kind == "summary":
        interval = raw.get("interval_seconds", 60)
        if (
            type(interval) not in (int, float)
            or not math.isfinite(interval)
            or not 1 <= interval <= 3600
        ):
            _fail("summary interval must be between 1 and 3600 seconds")
        prompt = raw.get("prompt", "Describe what is visible in the current scene.")
        if (
            not isinstance(prompt, str)
            or not prompt.strip()
            or len(prompt) > MAX_PROMPT
            or any(ord(c) < 32 and c not in "\n\t\r" for c in prompt)
        ):
            _fail("summary prompt must be nonempty text under 4001 characters")
        config.update(interval_seconds=interval, prompt=prompt)
    return config


def validate_graph(graph: Any, *, for_apply: bool = False) -> dict:
    """Normalize a draft; Apply additionally requires usable source-to-output paths."""
    graph = _object(graph, {"nodes", "edges"}, "graph")
    nodes, by_id = [], {}
    counts = {kind: 0 for kind in NODE_PORTS}
    for raw in _list(graph.get("nodes", []), MAX_NODES, "nodes"):
        raw = _object(raw, {"id", "type", "x", "y", "config", "enabled"}, "node")
        nid = _identifier(raw.get("id"), "node id")
        kind = raw.get("type")
        if not isinstance(kind, str) or kind not in NODE_PORTS or nid in by_id:
            _fail("duplicate node id or unsupported component type")
        node = {
            "id": nid,
            "type": kind,
            "enabled": _boolean(raw.get("enabled", True), "node enabled"),
            "config": _config(kind, raw.get("config", {})),
        }
        for axis in ("x", "y"):
            number = raw.get(axis, 0)
            if (
                type(number) not in (int, float)
                or not math.isfinite(number)
                or abs(number) > 100_000
            ):
                _fail("node coordinates must be finite and within 100000")
            node[axis] = number
        counts[kind] += 1
        if kind in {"source", "face", "events"} and counts[kind] > 1:
            _fail(f"only one {kind} component is supported per workflow")
        by_id[nid] = node
        nodes.append(node)
    edges, edge_ids, bindings, adjacency = [], set(), set(), {nid: [] for nid in by_id}
    incoming: dict[str, list[dict]] = {nid: [] for nid in by_id}
    for raw in _list(graph.get("edges", []), MAX_EDGES, "edges"):
        edge = _object(
            raw, {"id", "from_node", "from_port", "to_node", "to_port"}, "edge"
        )
        eid = _identifier(edge.get("id"), "edge id")
        source, target = edge.get("from_node"), edge.get("to_node")
        if (
            eid in edge_ids
            or not isinstance(source, str)
            or source not in by_id
            or not isinstance(target, str)
            or target not in by_id
            or source == target
        ):
            _fail("edge must connect distinct existing nodes with a unique id")
        out_port, in_port = edge.get("from_port"), edge.get("to_port")
        if (
            out_port not in NODE_PORTS[by_id[source]["type"]]["outputs"]
            or in_port not in NODE_PORTS[by_id[target]["type"]]["inputs"]
            or out_port != in_port
        ):
            _fail("incompatible component ports")
        key = (source, out_port, target, in_port)
        if key in bindings:
            _fail("duplicate connection")
        if by_id[target]["type"] != "output" and incoming[target]:
            _fail("processing components accept only one source input")
        normalized = {
            "id": eid,
            "from_node": source,
            "from_port": out_port,
            "to_node": target,
            "to_port": in_port,
        }
        edges.append(normalized)
        incoming[target].append(normalized)
        adjacency[source].append(target)
        edge_ids.add(eid)
        bindings.add(key)
    degrees = {nid: len(incoming[nid]) for nid in by_id}
    pending = [nid for nid, degree in degrees.items() if not degree]
    visited = 0
    while pending:
        nid = pending.pop()
        visited += 1
        for target in adjacency[nid]:
            degrees[target] -= 1
            if not degrees[target]:
                pending.append(target)
    if visited != len(nodes):
        _fail("workflow connections must not contain a cycle")
    normalized = {"nodes": nodes, "edges": edges}
    if for_apply:
        active = {n["id"] for n in nodes if n["enabled"]}
        sources = {n["id"] for n in nodes if n["type"] == "source"} & active
        outputs = {n["id"] for n in nodes if n["type"] == "output"} & active
        if not sources or not outputs:
            _fail("Apply requires an enabled source and output")
        forward = set(sources)
        for _ in nodes:
            forward.update(
                e["to_node"]
                for e in edges
                if e["from_node"] in forward and e["to_node"] in active
            )
        backward = set(outputs)
        for _ in nodes:
            backward.update(
                e["from_node"]
                for e in edges
                if e["to_node"] in backward and e["from_node"] in active
            )
        if not active <= forward & backward or not any(
            incoming[nid] for nid in outputs
        ):
            _fail("enabled components require connected source-to-output paths")
    return normalized


def _gates(value: Any, node_ids: set[str], *, nullable: bool = False) -> dict:
    if not isinstance(value, dict) or set(value) - node_ids:
        _fail("block overrides must refer to existing applied component ids")
    result = {}
    for nid, enabled in value.items():
        result[nid] = (
            None
            if nullable and enabled is None
            else _boolean(enabled, "block override")
        )
    return result


def _camera_override(value: Any, node_ids: set[str]) -> dict:
    value = _object(value, {"enabled", "block_enabled"}, "camera override")
    enabled = value.get("enabled")
    return {
        "enabled": None if enabled is None else _boolean(enabled, "camera enabled"),
        "block_enabled": _gates(
            value.get("block_enabled", {}), node_ids, nullable=True
        ),
    }


def compile_graph(
    graph: Any,
    *,
    enabled: bool = True,
    block_enabled: dict | None = None,
    camera_override: dict | None = None,
) -> dict:
    """Compile only connected, effectively enabled branches with output consumers."""
    graph = validate_graph(graph, for_apply=True)
    nodes = {n["id"]: n for n in graph["nodes"]}
    gates = _gates({} if block_enabled is None else block_enabled, set(nodes))
    override = _camera_override(
        {} if camera_override is None else camera_override, set(nodes)
    )
    hard_enabled = (
        _boolean(enabled, "group enabled") and override["enabled"] is not False
    )
    active = {
        nid
        for nid, node in nodes.items()
        if hard_enabled
        and node["enabled"]
        and gates.get(nid, True)
        and override["block_enabled"].get(nid) is not False
    }
    sources = {nid for nid in active if nodes[nid]["type"] == "source"}
    reachable = set(sources)
    for _ in nodes:
        reachable.update(
            e["to_node"]
            for e in graph["edges"]
            if e["from_node"] in reachable and e["to_node"] in active
        )
    outputs, consumed = [], set()
    for nid, node in nodes.items():
        if node["type"] != "output" or nid not in reachable:
            continue
        inputs = [
            {"node_id": e["from_node"], "port": e["from_port"]}
            for e in graph["edges"]
            if e["to_node"] == nid and e["from_node"] in reachable
        ]
        if inputs:
            inputs.sort(key=lambda binding: (binding["node_id"], binding["port"]))
            outputs.append({"node_id": nid, "inputs": inputs})
            consumed.update(binding["node_id"] for binding in inputs)
    features, sensitivities, summaries = set(), {}, []
    face_id = event_id = None
    for nid in sorted(consumed):
        node, kind = nodes[nid], nodes[nid]["type"]
        if kind == "face":
            face_id = nid
            features.add("stranger")
        elif kind == "events":
            event_id = nid
            features.update(node["config"]["features"])
            sensitivities.update(node["config"]["event_sensitivities"])
        elif kind == "summary":
            summaries.append(
                {
                    "id": nid,
                    **{
                        key: node["config"][key]
                        for key in ("interval_seconds", "prompt")
                    },
                }
            )
    ordered_features = [
        item["id"] for item in event_catalog()["events"] if item["id"] in features
    ]
    return {
        "features": ordered_features,
        "event_sensitivities": sensitivities,
        "summaries": summaries,
        "outputs": sorted(outputs, key=lambda o: o["node_id"]),
        "enabled": bool(outputs),
        "face_node_id": face_id,
        "events_node_id": event_id,
    }


def _workflow(value: Any) -> dict:
    value = _object(value, {"id", "name", "graph"}, "workflow")
    return {
        "id": _identifier(value.get("id")),
        "name": _name(value.get("name")),
        "graph": validate_graph(value.get("graph", {})),
    }


def _group(value: Any) -> dict:
    fields = {
        "id",
        "name",
        "camera_ids",
        "workflow_id",
        "applied_graph",
        "enabled",
        "block_enabled",
        "camera_overrides",
        "assignment_generation",
    }
    value = _object(value, fields, "source group")
    cameras = _list(value.get("camera_ids", []), MAX_CAMERAS, "group cameras")
    if any(
        not isinstance(cid, str) or not _CAMERA.fullmatch(cid) for cid in cameras
    ) or len(set(cameras)) != len(cameras):
        _fail("group camera ids must be unique provisioned camera identifiers")
    graph = value.get("applied_graph")
    graph = validate_graph(graph, for_apply=True) if graph is not None else None
    ids = {node["id"] for node in graph["nodes"]} if graph is not None else set()
    overrides = value.get("camera_overrides", {})
    if not isinstance(overrides, dict) or set(overrides) - set(cameras):
        _fail("camera overrides must belong to this group")
    workflow_id = value.get("workflow_id")
    if workflow_id is not None:
        _identifier(workflow_id, "workflow id")
    if graph is not None and workflow_id is None:
        _fail("an applied snapshot requires a workflow assignment")
    return {
        "id": _identifier(value.get("id")),
        "name": _name(value.get("name")),
        "camera_ids": list(cameras),
        "workflow_id": workflow_id,
        "applied_graph": graph,
        "enabled": _boolean(value.get("enabled", True), "group enabled"),
        "block_enabled": _gates(value.get("block_enabled", {}), ids),
        "camera_overrides": {
            cid: _camera_override(ov, ids) for cid, ov in overrides.items()
        },
        "assignment_generation": _integer(
            value.get("assignment_generation", 0), "assignment generation"
        ),
    }


def require_exclusive_groups(groups: list[dict]) -> None:
    assigned = set()
    for group in groups:
        if assigned.intersection(group["camera_ids"]):
            _fail("a camera can belong to only one source group")
        assigned.update(group["camera_ids"])


def validate_document(value: Any) -> dict:
    _bytes(value)
    value = _object(
        value,
        {"schema_version", "revision", "workflows", "groups", "trash"},
        "workflow document",
    )
    if (
        type(value.get("schema_version")) is not int
        or value["schema_version"] != SCHEMA_VERSION
    ):
        _fail("unsupported workflow document schema_version")
    workflows = [
        _workflow(w) for w in _list(value.get("workflows"), MAX_WORKFLOWS, "workflows")
    ]
    groups = [_group(g) for g in _list(value.get("groups"), MAX_GROUPS, "groups")]
    workflow_ids = {w["id"] for w in workflows}
    if len(workflow_ids) != len(workflows) or len({g["id"] for g in groups}) != len(
        groups
    ):
        _fail("duplicate workflow or source group id")
    if any(
        g["workflow_id"] is not None and g["workflow_id"] not in workflow_ids
        for g in groups
    ):
        _fail("source group references an unknown workflow")
    require_exclusive_groups(groups)
    trash, trash_ids = [], set()
    for item in _list(value.get("trash"), MAX_TRASH, "trash"):
        item = _object(item, {"id", "kind", "asset", "assignments"}, "trash entry")
        tid = _identifier(item.get("id"), "trash id")
        kind = item.get("kind")
        if tid in trash_ids or kind not in {"workflow", "group"}:
            _fail("invalid trash id or kind")
        asset = (
            _workflow(item.get("asset"))
            if kind == "workflow"
            else _group(item.get("asset"))
        )
        assignments = []
        for assignment in _list(
            item.get("assignments", []), MAX_GROUPS, "saved assignments"
        ):
            assignment = _object(
                assignment,
                {"group_id", "generation_after", "replacement_id", "previous"},
                "saved assignment",
            )
            previous = _group(assignment.get("previous"))
            replacement = assignment.get("replacement_id")
            if replacement is not None:
                _identifier(replacement, "replacement workflow id")
            group_id = _identifier(assignment.get("group_id"), "group id")
            if (
                kind != "workflow"
                or previous["id"] != group_id
                or previous["workflow_id"] != asset["id"]
            ):
                _fail("invalid saved assignment provenance")
            assignments.append(
                {
                    "group_id": group_id,
                    "replacement_id": replacement,
                    "generation_after": _integer(
                        assignment.get("generation_after"), "assignment generation"
                    ),
                    "previous": previous,
                }
            )
        trash.append(
            {"id": tid, "kind": kind, "asset": asset, "assignments": assignments}
        )
        trash_ids.add(tid)
    result = {
        "schema_version": SCHEMA_VERSION,
        "revision": _integer(value.get("revision"), "revision"),
        "workflows": workflows,
        "groups": groups,
        "trash": trash,
    }
    _bytes(result)
    return result


def compile_group(document: dict, group_id: str) -> dict[str, dict]:
    document = validate_document(document)
    group = _find(document["groups"], group_id, "source group")
    if group["applied_graph"] is None:
        return {}
    return {
        cid: compile_graph(
            group["applied_graph"],
            enabled=group["enabled"],
            block_enabled=group["block_enabled"],
            camera_override=group["camera_overrides"].get(cid),
        )
        for cid in group["camera_ids"]
    }


def compile_document(document: dict) -> dict[str, dict]:
    document = validate_document(document)
    result = {}
    for group in document["groups"]:
        if group["applied_graph"] is not None:
            for cid in group["camera_ids"]:
                result[cid] = compile_graph(
                    group["applied_graph"],
                    enabled=group["enabled"],
                    block_enabled=group["block_enabled"],
                    camera_override=group["camera_overrides"].get(cid),
                )
    return result


def changed_cameras(previous: dict, candidate: dict) -> list[str]:
    """Include applied assignment intent, even when its compiled plan is identical."""
    before, after = compile_document(previous), compile_document(candidate)

    def assignments(document: dict) -> dict:
        return {
            cid: (group["id"], group["workflow_id"], group["assignment_generation"])
            for group in document["groups"]
            if group["applied_graph"] is not None
            for cid in group["camera_ids"]
        }

    old_assignments, new_assignments = assignments(previous), assignments(candidate)
    return sorted(
        cid
        for cid in before.keys() | after.keys()
        if before.get(cid) != after.get(cid)
        or old_assignments.get(cid) != new_assignments.get(cid)
    )


def _find(items: list[dict], identifier: str, label: str) -> dict:
    _identifier(identifier, f"{label} id")
    for item in items:
        if item["id"] == identifier:
            return item
    _fail(f"unknown {label}")


def _clear_assignment(group: dict) -> None:
    group.update(workflow_id=None, applied_graph=None, block_enabled={})
    group["camera_overrides"] = {
        cid: {"enabled": ov["enabled"], "block_enabled": {}}
        for cid, ov in group["camera_overrides"].items()
    }


class WorkflowStore:
    """Private atomic control document; None explicitly selects an in-memory store."""

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path).expanduser().absolute() if path is not None else None
        self._lock = RLock()
        self._memory = empty_document()

    def load(self) -> dict:
        with self._lock:
            if self.path is None:
                return copy.deepcopy(self._memory)
            try:
                if self.path.is_symlink():
                    raise WorkflowStorageError(
                        "workflow storage must not be a symbolic link"
                    )
                try:
                    with self.path.open("rb") as handle:
                        body = handle.read(MAX_DOCUMENT_BYTES + 1)
                except FileNotFoundError:
                    return empty_document()
                if len(body) > MAX_DOCUMENT_BYTES:
                    raise WorkflowStorageError(
                        "workflow storage exceeds its size limit"
                    )
                return validate_document(json.loads(body))
            except WorkflowStorageError:
                raise
            except (OSError, ValueError, TypeError, RecursionError) as exc:
                raise WorkflowStorageError(
                    "workflow storage could not be read"
                ) from exc

    def candidate(self, action: str, payload: dict, expected_revision: int) -> dict:
        with self._lock:
            document = self.load()
            if _integer(expected_revision, "revision") != document["revision"]:
                raise WorkflowConflictError(
                    "workflow assets changed; reload before saving"
                )
            if not isinstance(payload, dict):
                _fail("workflow command must be an object")
            document = copy.deepcopy(document)
            self._command(document, action, payload)
            document["revision"] += 1
            return validate_document(document)

    def commit(self, candidate: dict, expected_revision: int) -> dict:
        with self._lock:
            current = self.load()
            if _integer(expected_revision, "revision") != current["revision"]:
                raise WorkflowConflictError(
                    "workflow assets changed; reload before saving"
                )
            document = validate_document(candidate)
            if document["revision"] != expected_revision + 1:
                raise WorkflowConflictError(
                    "candidate revision is not the next stored revision"
                )
            if self.path is None:
                self._memory = copy.deepcopy(document)
                return copy.deepcopy(document)
            temporary = self.path.with_name(f".{self.path.name}.{uuid4().hex}.tmp")
            try:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                if self.path.is_symlink():
                    raise WorkflowStorageError(
                        "workflow storage must not be a symbolic link"
                    )
                fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
                with os.fdopen(fd, "wb") as handle:
                    handle.write(_bytes(document) + b"\n")
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temporary, self.path)
            except (OSError, ValueError) as exc:
                raise WorkflowStorageError(
                    "workflow assets could not be saved"
                ) from exc
            finally:
                try:
                    temporary.unlink(missing_ok=True)
                except OSError:
                    pass  # Preserve the original storage failure if cleanup also fails.
            return copy.deepcopy(document)

    @staticmethod
    def _command(document: dict, action: str, payload: dict) -> None:
        workflows, groups, trash = (
            document["workflows"],
            document["groups"],
            document["trash"],
        )
        if action == "create_workflow":
            _object(payload, {"name", "graph"}, "create workflow")
            workflows.append(
                _workflow(
                    {
                        "id": f"workflow-{uuid4().hex}",
                        "name": payload.get("name"),
                        "graph": payload.get("graph", {"nodes": [], "edges": []}),
                    }
                )
            )
        elif action == "save_workflow":
            value = _workflow(payload)
            _find(workflows, value["id"], "workflow").update(value)
        elif action == "create_group":
            _object(payload, {"name", "camera_ids"}, "create source group")
            groups.append(
                _group(
                    {
                        "id": f"group-{uuid4().hex}",
                        "name": payload.get("name"),
                        "camera_ids": payload.get("camera_ids", []),
                    }
                )
            )
        elif action == "update_group":
            allowed = {
                "id",
                "name",
                "camera_ids",
                "enabled",
                "block_enabled",
                "camera_overrides",
            }
            _object(payload, allowed, "update source group")
            group = _find(groups, payload.get("id"), "source group")
            updated = {**group, **copy.deepcopy(payload)}
            if "camera_ids" in payload and "camera_overrides" not in payload:
                if not isinstance(payload["camera_ids"], list):
                    _fail("camera_ids must be a list")
                updated["camera_overrides"] = {
                    cid: ov
                    for cid, ov in group["camera_overrides"].items()
                    if cid in payload["camera_ids"]
                }
            updated = _group(updated)
            if any(updated[key] != group[key] for key in allowed - {"id", "name"}):
                updated["assignment_generation"] += 1
            group.update(updated)
        elif action == "apply":
            _object(payload, {"workflow_id", "group_ids"}, "Apply")
            workflow = _find(workflows, payload.get("workflow_id"), "workflow")
            graph = validate_graph(workflow["graph"], for_apply=True)
            ids = _list(payload.get("group_ids"), MAX_GROUPS, "Apply groups")
            if (
                not ids
                or any(not isinstance(gid, str) for gid in ids)
                or len(set(ids)) != len(ids)
            ):
                _fail("Apply requires unique selected source groups")
            selected = [_find(groups, gid, "source group") for gid in ids]
            nodes = {n["id"] for n in graph["nodes"]}
            for group in selected:
                # Stable ids retain gates; removed draft nodes leave no stale gates.
                group["block_enabled"] = {
                    nid: gate
                    for nid, gate in group["block_enabled"].items()
                    if nid in nodes
                }
                for override in group["camera_overrides"].values():
                    override["block_enabled"] = {
                        nid: gate
                        for nid, gate in override["block_enabled"].items()
                        if nid in nodes
                    }
                group.update(
                    workflow_id=workflow["id"], applied_graph=copy.deepcopy(graph)
                )
                group["assignment_generation"] += 1
        elif action == "delete_group":
            _object(payload, {"id"}, "delete source group")
            group = _find(groups, payload.get("id"), "source group")
            trash.append(
                {
                    "id": f"trash-{uuid4().hex}",
                    "kind": "group",
                    "asset": copy.deepcopy(group),
                    "assignments": [],
                }
            )
            groups.remove(group)
        elif action == "delete_workflow":
            _object(payload, {"id", "replacement_id", "unassign"}, "delete workflow")
            workflow = _find(workflows, payload.get("id"), "workflow")
            replacement_id = payload.get("replacement_id")
            if "unassign" in payload:
                _boolean(payload["unassign"], "unassign")
            if replacement_id is not None:
                if replacement_id == workflow["id"] or payload.get("unassign"):
                    _fail("choose a different replacement or explicit unassign")
                _find(workflows, replacement_id, "replacement workflow")
            assigned = [g for g in groups if g["workflow_id"] == workflow["id"]]
            if (
                assigned
                and replacement_id is None
                and payload.get("unassign") is not True
            ):
                _fail("workflow is assigned; choose a replacement or explicit unassign")
            assignments = []
            for group in assigned:
                previous = copy.deepcopy(group)
                if replacement_id is None:
                    _clear_assignment(group)
                else:
                    group["workflow_id"] = replacement_id
                group["assignment_generation"] += 1
                assignments.append(
                    {
                        "group_id": group["id"],
                        "generation_after": group["assignment_generation"],
                        "replacement_id": replacement_id,
                        "previous": previous,
                    }
                )
            trash.append(
                {
                    "id": f"trash-{uuid4().hex}",
                    "kind": "workflow",
                    "asset": copy.deepcopy(workflow),
                    "assignments": assignments,
                }
            )
            workflows.remove(workflow)
        elif action == "restore":
            _object(payload, {"id"}, "restore")
            item = _find(trash, payload.get("id"), "trash entry")
            asset = copy.deepcopy(item["asset"])
            collection = workflows if item["kind"] == "workflow" else groups
            if any(existing["id"] == asset["id"] for existing in collection):
                _fail("the original asset id is already present")
            collection.append(asset)
            if item["kind"] == "workflow":
                for assignment in item["assignments"]:
                    group = next(
                        (g for g in groups if g["id"] == assignment["group_id"]), None
                    )
                    if (
                        group is not None
                        and group["assignment_generation"]
                        == assignment["generation_after"]
                        and group["workflow_id"] == assignment["replacement_id"]
                    ):
                        for key in (
                            "workflow_id",
                            "applied_graph",
                            "block_enabled",
                            "camera_overrides",
                        ):
                            group[key] = copy.deepcopy(assignment["previous"][key])
                        group["assignment_generation"] += 1
            trash.remove(item)
        else:
            _fail("unsupported workflow command")
