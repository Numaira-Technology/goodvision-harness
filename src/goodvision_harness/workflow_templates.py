# ruff: noqa: RUF001 -- User-facing Chinese punctuation.
"""Starter templates: wired module groups a new workflow can begin from.

A template is a workflow graph without deployment details: video sources carry
no camera channel and modules carry no service addresses. Built-in templates
ship with the code; saved ones live in their own file beside the workflow
document, so the workflow document schema is unchanged.
"""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from threading import RLock
from typing import Any
from uuid import uuid4

from . import object_presence
from .event_registry import EventVocabulary
from .workflows import (
    MAX_DOCUMENT_BYTES,
    WorkflowStorageError,
    _bytes,
    _fail,
    _identifier,
    _name,
    _object,
    validate_graph,
)

SCHEMA_VERSION = 1
MAX_TEMPLATES = 64
MAX_DESCRIPTION = 500
_DEPLOYMENT_KEYS = ("camera_id", "service_url", "cloud_service_url")


def _edge(eid, source, port, target, to_port=None):
    return {
        "id": eid, "from_node": source, "from_port": port,
        "to_node": target, "to_port": to_port or port,
    }


BUILTIN = (
    {
        "id": "builtin-videojev-yolo",
        "name": "VideoJev + YOLO 基础盘",
        "description": (
            "视频源 → YOLO 人框（Bbox）→ 物体存在（VideoJev，按人框裁剪）→ 输出；"
            "同时 视频源 → System 1 事件（VideoJev）→ 输出。物体预设为实验性，"
            "未经验证；人框裁剪是否提升识别需在真实视频上对比后才能下结论。"
        ),
        "graph": {
            "nodes": [
                {"id": "source", "type": "source", "x": 40, "y": 60, "config": {}},
                {"id": "bbox", "type": "tracking", "x": 330, "y": 40,
                 "config": {"engine": "bbox"}},
                # Explicit experimental preset (no default object exists); the
                # description says it is not validated.
                {"id": "objects", "type": "objects", "x": 330, "y": 235,
                 "config": {"engine": "system1",
                            "objects": [object_presence.PRESETS["knife"]]}},
                {"id": "events", "type": "events", "x": 330, "y": 430,
                 "config": {"engine": "system1"}},
                {"id": "output", "type": "output", "x": 640, "y": 200,
                 "config": {}},
            ],
            "edges": [
                _edge("e-source-bbox", "source", "frames", "bbox"),
                _edge("e-source-objects", "source", "frames", "objects"),
                _edge("e-bbox-objects", "bbox", "tracks", "objects"),
                _edge("e-source-events", "source", "frames", "events"),
                _edge("e-bbox-output", "bbox", "tracks", "output"),
                _edge("e-objects-output", "objects", "objects", "output"),
                _edge("e-events-output", "events", "events", "output"),
            ],
        },
    },
)


def strip_deployment(graph: dict) -> dict:
    """Remove camera channels and service addresses; keep engines and modes."""
    graph = copy.deepcopy(graph)
    for node in graph.get("nodes", []):
        config = node.get("config")
        if isinstance(config, dict):
            for key in _DEPLOYMENT_KEYS:
                config.pop(key, None)
            # A remote mode without its address cannot apply; use local again.
            if config.get("execution_mode") in {"server", "cloud"}:
                config.pop("execution_mode")
    return graph


def template(value: Any, vocabulary: EventVocabulary | None = None) -> dict:
    value = _object(value, {"id", "name", "description", "graph"}, "template")
    description = value.get("description", "")
    if (
        not isinstance(description, str)
        or len(description) > MAX_DESCRIPTION
        or any(ord(c) < 32 and c not in "\n\t" for c in description)
    ):
        _fail(f"template description must be text under {MAX_DESCRIPTION + 1} chars")
    return {
        "id": _identifier(value.get("id"), "template id"),
        "name": _name(value.get("name"), "template name"),
        "description": description.strip(),
        "graph": validate_graph(
            strip_deployment(value.get("graph", {})), vocabulary=vocabulary
        ),
    }


def builtin_templates(vocabulary: EventVocabulary | None = None) -> list[dict]:
    return [
        {**template(item, vocabulary=vocabulary), "builtin": True}
        for item in BUILTIN
    ]


class TemplateStore:
    """Saved templates in one private JSON file; None keeps them in memory."""

    def __init__(
        self, path: str | Path | None = None, *,
        vocabulary: EventVocabulary | None = None,
    ) -> None:
        self.vocabulary = vocabulary
        self.path = Path(path).expanduser().absolute() if path is not None else None
        self._lock = RLock()
        self._memory: list[dict] = []

    def _load_saved(self) -> list[dict]:
        if self.path is None:
            return copy.deepcopy(self._memory)
        try:
            if self.path.is_symlink():
                raise WorkflowStorageError("template storage must not be a link")
            try:
                with self.path.open("rb") as handle:
                    body = handle.read(MAX_DOCUMENT_BYTES + 1)
            except FileNotFoundError:
                return []
            if len(body) > MAX_DOCUMENT_BYTES:
                raise WorkflowStorageError("template storage exceeds its size limit")
            value = _object(json.loads(body), {"schema_version", "templates"},
                            "template document")
            if value.get("schema_version") != SCHEMA_VERSION:
                _fail("unsupported template schema_version")
            items = value.get("templates")
            if not isinstance(items, list) or len(items) > MAX_TEMPLATES:
                _fail("invalid saved templates")
            return [template(item, vocabulary=self.vocabulary) for item in items]
        except WorkflowStorageError:
            raise
        except (OSError, ValueError, TypeError, RecursionError) as exc:
            raise WorkflowStorageError("template storage could not be read") from exc

    def _write(self, items: list[dict]) -> None:
        document = {"schema_version": SCHEMA_VERSION, "templates": items}
        body = _bytes(document)
        if self.path is None:
            self._memory = copy.deepcopy(items)
            return
        temporary = self.path.with_name(f".{self.path.name}.{uuid4().hex}.tmp")
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            if self.path.is_symlink():
                raise WorkflowStorageError("template storage must not be a link")
            fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(fd, "wb") as handle:
                handle.write(body + b"\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self.path)
        except OSError as exc:
            raise WorkflowStorageError("templates could not be saved") from exc
        finally:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass

    def list(self) -> list[dict]:
        with self._lock:
            return builtin_templates(self.vocabulary) + [
                {**item, "builtin": False} for item in self._load_saved()
            ]

    def save(self, payload: Any) -> dict:
        payload = _object(payload, {"name", "description", "graph"}, "save template")
        with self._lock:
            items = self._load_saved()
            if len(items) >= MAX_TEMPLATES:
                _fail(f"at most {MAX_TEMPLATES} saved templates")
            item = template(
                {"id": f"template-{uuid4().hex}", **payload},
                vocabulary=self.vocabulary,
            )
            self._write([*items, item])
            return {**item, "builtin": False}

    def delete(self, template_id: str) -> None:
        _identifier(template_id, "template id")
        if any(item["id"] == template_id for item in BUILTIN):
            _fail("built-in templates cannot be deleted")
        with self._lock:
            items = self._load_saved()
            kept = [item for item in items if item["id"] != template_id]
            if len(kept) == len(items):
                _fail("unknown template")
            self._write(kept)
