from __future__ import annotations

import copy
import json
import os
import stat
from pathlib import Path

import pytest

import goodvision_harness.workflows as model
from goodvision_harness.workflows import (
    WorkflowConflictError,
    WorkflowStorageError,
    WorkflowStore,
    WorkflowValidationError,
    changed_cameras,
    compile_document,
    compile_graph,
    validate_graph,
)


def graph() -> dict:
    nodes = [
        {"id": "source", "type": "source", "config": {"name": "Camera"}},
        {"id": "face", "type": "face", "config": {"name": "People"}},
        {
            "id": "events",
            "type": "events",
            "config": {
                "name": "Safety",
                "features": ["smoking"],
                "event_sensitivities": {"smoking": "standard"},
            },
        },
        {
            "id": "summary",
            "type": "summary",
            "config": {
                "name": "Minute summary",
                "interval_seconds": 60,
                "prompt": "Describe the scene.",
            },
        },
        {"id": "output", "type": "output", "config": {"name": "Results"}},
    ]
    edges = [
        {
            "id": f"source-{target}",
            "from_node": "source",
            "from_port": "frames",
            "to_node": target,
            "to_port": "frames",
        }
        for target in ("face", "events", "summary")
    ] + [
        {
            "id": f"{source}-output",
            "from_node": source,
            "from_port": port,
            "to_node": "output",
            "to_port": port,
        }
        for source, port in (
            ("face", "faces"),
            ("events", "events"),
            ("summary", "summary"),
        )
    ]
    return {"nodes": nodes, "edges": edges}


def command(store: WorkflowStore, action: str, payload: dict) -> dict:
    before = store.load()
    candidate = store.candidate(action, payload, before["revision"])
    assert store.load() == before
    return store.commit(candidate, before["revision"])


def setup_assets(store: WorkflowStore) -> tuple[str, str, str]:
    doc = command(store, "create_workflow", {"name": "Entrance", "graph": graph()})
    workflow_id = doc["workflows"][0]["id"]
    doc = command(store, "create_group", {"name": "East", "camera_ids": ["camera01"]})
    first = doc["groups"][0]["id"]
    doc = command(store, "create_group", {"name": "West", "camera_ids": ["camera02"]})
    return workflow_id, first, doc["groups"][1]["id"]


def test_initial_store_is_empty_and_candidates_do_not_write(tmp_path: Path) -> None:
    path = tmp_path / "workflow-assets.json"
    store = WorkflowStore(path)
    original = store.load()
    assert original == {
        "schema_version": 1,
        "revision": 0,
        "workflows": [],
        "groups": [],
        "trash": [],
    }
    assert compile_document(original) == {}
    draft = store.candidate("create_workflow", {"name": "Empty draft"}, 0)
    assert not path.exists()
    assert store.load() == original
    assert draft["workflows"][0]["graph"] == {"nodes": [], "edges": []}
    saved = store.commit(draft, 0)
    assert WorkflowStore(path).load() == saved
    if os.name != "nt":
        assert stat.S_IMODE(path.stat().st_mode) == 0o600
    saved["workflows"].clear()
    assert store.load()["workflows"]


def test_revision_checks_reject_stale_candidates_and_commands() -> None:
    store = WorkflowStore()
    first = store.candidate("create_workflow", {"name": "First"}, 0)
    competing = store.candidate("create_workflow", {"name": "Competing"}, 0)
    store.commit(first, 0)
    with pytest.raises(WorkflowConflictError):
        store.commit(competing, 0)
    with pytest.raises(WorkflowConflictError):
        store.candidate("create_group", {"name": "Stale"}, 0)
    assert store.load()["workflows"][0]["name"] == "First"


def test_drafts_and_selective_apply_preserve_other_applied_snapshots() -> None:
    store = WorkflowStore()
    workflow_id, first, second = setup_assets(store)
    doc = command(
        store, "apply", {"workflow_id": workflow_id, "group_ids": [first, second]}
    )
    before = compile_document(doc)
    updated = graph()
    updated["nodes"][3]["config"]["prompt"] = "Count visible people."
    draft = command(
        store, "save_workflow", {"id": workflow_id, "name": "Revised", "graph": updated}
    )
    assert changed_cameras(doc, draft) == []
    assert compile_document(draft) == before
    applied = command(
        store, "apply", {"workflow_id": workflow_id, "group_ids": [first]}
    )
    assert changed_cameras(draft, applied) == ["camera01"]
    plans = compile_document(applied)
    assert plans["camera01"]["summaries"][0]["prompt"] == "Count visible people."
    assert plans["camera02"] == before["camera02"]


def test_apply_validates_every_selected_group_before_candidate_can_commit() -> None:
    store = WorkflowStore()
    workflow_id, first, _ = setup_assets(store)
    before = store.load()
    with pytest.raises(WorkflowValidationError, match="unknown source group"):
        store.candidate(
            "apply",
            {"workflow_id": workflow_id, "group_ids": [first, "missing"]},
            before["revision"],
        )
    assert store.load() == before


def test_identical_reapply_still_marks_target_cameras_affected() -> None:
    store = WorkflowStore()
    workflow_id, first, second = setup_assets(store)
    before = command(
        store, "apply", {"workflow_id": workflow_id, "group_ids": [first, second]}
    )
    after = command(store, "apply", {"workflow_id": workflow_id, "group_ids": [first]})
    assert compile_document(before) == compile_document(after)
    assert changed_cameras(before, after) == ["camera01"]


def test_compiler_preserves_native_node_ids_and_actual_output_bindings() -> None:
    plan = compile_graph(graph())
    assert plan["enabled"] is True
    assert plan["features"] == ["smoking", "stranger"]
    assert plan["event_sensitivities"] == {"smoking": "standard"}
    assert plan["face_node_id"] == "face"
    assert plan["events_node_id"] == "events"
    assert plan["summaries"] == [
        {"id": "summary", "interval_seconds": 60, "prompt": "Describe the scene."}
    ]
    assert plan["outputs"] == [
        {
            "node_id": "output",
            "inputs": [
                {"node_id": "events", "port": "events"},
                {"node_id": "face", "port": "faces"},
                {"node_id": "summary", "port": "summary"},
            ],
        }
    ]


def test_independent_summaries_can_bind_to_one_output_without_merging_instances() -> (
    None
):
    value = graph()
    extra = copy.deepcopy(value["nodes"][3])
    extra["id"] = "summary-two"
    extra["config"].update(interval_seconds=15, prompt="Count vehicles.")
    value["nodes"].append(extra)
    value["edges"].extend(
        [
            {
                "id": "source-two",
                "from_node": "source",
                "from_port": "frames",
                "to_node": "summary-two",
                "to_port": "frames",
            },
            {
                "id": "two-output",
                "from_node": "summary-two",
                "from_port": "summary",
                "to_node": "output",
                "to_port": "summary",
            },
        ]
    )
    plan = compile_graph(value, camera_override={"block_enabled": {"summary": False}})
    assert plan["summaries"] == [
        {"id": "summary-two", "interval_seconds": 15, "prompt": "Count vehicles."}
    ]
    assert {binding["node_id"] for binding in plan["outputs"][0]["inputs"]} == {
        "face",
        "events",
        "summary-two",
    }


@pytest.mark.parametrize(
    "kwargs",
    [
        {"enabled": False, "camera_override": {"enabled": True}},
        {
            "block_enabled": {"source": False},
            "camera_override": {"block_enabled": {"source": True}},
        },
        {"camera_override": {"enabled": False}},
        {"block_enabled": {"output": False}},
    ],
)
def test_parent_and_source_gates_cannot_be_bypassed(kwargs: dict) -> None:
    plan = compile_graph(graph(), **kwargs)
    assert plan == {
        "features": [],
        "event_sensitivities": {},
        "summaries": [],
        "outputs": [],
        "enabled": False,
        "face_node_id": None,
        "events_node_id": None,
    }


def test_per_camera_inherit_and_component_off_are_independent() -> None:
    store = WorkflowStore()
    workflow_id, first, _ = setup_assets(store)
    command(store, "apply", {"workflow_id": workflow_id, "group_ids": [first]})
    doc = command(
        store,
        "update_group",
        {
            "id": first,
            "block_enabled": {"face": False},
            "camera_overrides": {
                "camera01": {
                    "enabled": None,
                    "block_enabled": {"face": True, "summary": False},
                }
            },
        },
    )
    plan = compile_document(doc)["camera01"]
    assert plan["features"] == ["smoking"]
    assert plan["summaries"] == []
    assert plan["face_node_id"] is None
    assert plan["events_node_id"] == "events"


def test_source_only_output_is_a_valid_plan_without_fake_detection_features() -> None:
    value = {
        "nodes": [{"id": "s", "type": "source"}, {"id": "o", "type": "output"}],
        "edges": [
            {
                "id": "e",
                "from_node": "s",
                "from_port": "frames",
                "to_node": "o",
                "to_port": "frames",
            }
        ],
    }
    plan = compile_graph(value)
    assert plan["enabled"] is True
    assert plan["features"] == []
    assert plan["outputs"] == [
        {"node_id": "o", "inputs": [{"node_id": "s", "port": "frames"}]}
    ]


def test_missing_connections_are_editable_but_not_applicable() -> None:
    value = graph()
    value["edges"] = []
    assert len(validate_graph(value)["nodes"]) == 5
    with pytest.raises(WorkflowValidationError, match="source-to-output"):
        validate_graph(value, for_apply=True)
    with pytest.raises(WorkflowValidationError, match="source and output"):
        compile_graph({"nodes": [], "edges": []})


@pytest.mark.parametrize(
    ("node_index", "patch"),
    [
        (0, {"x": float("nan")}),
        (0, {"y": True}),
        (0, {"x": 100001}),
        (1, {"type": "custom-script"}),
        (1, {"config": {"threshold": 0.1}}),
        (2, {"config": {"features": ["stranger"]}}),
        (2, {"config": {"features": ["smoking", "smoking"]}}),
        (
            2,
            {
                "config": {
                    "features": ["smoking"],
                    "event_sensitivities": {"fight": "high"},
                }
            },
        ),
        (3, {"config": {"interval_seconds": 0}}),
        (3, {"config": {"interval_seconds": True}}),
        (3, {"config": {"prompt": ""}}),
        (3, {"config": {"prompt": "x" * 4001}}),
    ],
)
def test_graph_rejects_unsupported_or_unbounded_component_settings(
    node_index: int, patch: dict
) -> None:
    value = graph()
    value["nodes"][node_index].update(patch)
    with pytest.raises(WorkflowValidationError):
        validate_graph(value)


@pytest.mark.parametrize(
    "edge_patch",
    [
        {"from_node": "missing"},
        {"to_port": "summary"},
        {"from_port": "../frames"},
        {"to_node": "source"},
    ],
)
def test_typed_connections_reject_unknown_nodes_ports_and_self_cycles(
    edge_patch: dict,
) -> None:
    value = graph()
    value["edges"][0].update(edge_patch)
    with pytest.raises(WorkflowValidationError):
        validate_graph(value)


def test_native_singletons_and_unique_connections_are_enforced() -> None:
    value = graph()
    extra = copy.deepcopy(value["nodes"][1])
    extra["id"] = "another-face"
    value["nodes"].append(extra)
    with pytest.raises(WorkflowValidationError, match="one face"):
        validate_graph(value)
    value = graph()
    value["edges"].append({**value["edges"][0], "id": "duplicate-binding"})
    with pytest.raises(WorkflowValidationError, match="duplicate connection"):
        validate_graph(value)


def test_event_membership_is_loaded_from_current_catalog(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    catalog = model.event_catalog()
    catalog["events"].append({"id": "new_condition", "kind": "detection"})
    monkeypatch.setattr(model, "event_catalog", lambda: copy.deepcopy(catalog))
    value = graph()
    value["nodes"][2]["config"] = {"features": ["new_condition"]}
    plan = compile_graph(value)
    assert "new_condition" in plan["features"]
    assert plan["event_sensitivities"] == {}


@pytest.mark.parametrize(
    "override",
    [
        {"block_enabled": {"missing": True}},
        {"camera_override": {"enabled": "inherit"}},
        {"camera_override": {"block_enabled": {"face": 1}}},
        {"camera_override": {"features": ["fall"]}},
    ],
)
def test_invalid_overrides_are_not_silently_ignored(override: dict) -> None:
    with pytest.raises(WorkflowValidationError):
        compile_graph(graph(), **override)


def test_camera_membership_is_exclusive_and_overrides_must_belong_to_group() -> None:
    store = WorkflowStore()
    _, first, _ = setup_assets(store)
    before = store.load()
    for action, payload in [
        ("create_group", {"name": "Overlap", "camera_ids": ["camera01"]}),
        ("update_group", {"id": first, "camera_ids": ["camera01", "camera01"]}),
        (
            "update_group",
            {"id": first, "camera_overrides": {"camera99": {"enabled": False}}},
        ),
    ]:
        with pytest.raises(WorkflowValidationError):
            store.candidate(action, payload, before["revision"])
        assert store.load() == before


def test_deleting_in_use_workflow_requires_explicit_unassign_and_can_restore() -> None:
    store = WorkflowStore()
    workflow_id, first, _ = setup_assets(store)
    applied = command(
        store, "apply", {"workflow_id": workflow_id, "group_ids": [first]}
    )
    with pytest.raises(WorkflowValidationError, match="assigned"):
        store.candidate("delete_workflow", {"id": workflow_id}, applied["revision"])
    deleted = command(store, "delete_workflow", {"id": workflow_id, "unassign": True})
    assert compile_document(deleted) == {}
    assert changed_cameras(applied, deleted) == ["camera01"]
    restored = command(store, "restore", {"id": deleted["trash"][0]["id"]})
    assert compile_document(restored) == compile_document(applied)


def test_workflow_restore_never_overwrites_a_newer_assignment_or_applied_graph() -> (
    None
):
    store = WorkflowStore()
    original_id, first, _ = setup_assets(store)
    command(store, "apply", {"workflow_id": original_id, "group_ids": [first]})
    replacement_graph = graph()
    replacement_graph["nodes"][3]["config"]["prompt"] = "Replacement summary."
    doc = command(
        store, "create_workflow", {"name": "Replacement", "graph": replacement_graph}
    )
    replacement_id = doc["workflows"][-1]["id"]
    deleted = command(
        store, "delete_workflow", {"id": original_id, "replacement_id": replacement_id}
    )
    trash_id = deleted["trash"][0]["id"]
    newer = command(
        store, "apply", {"workflow_id": replacement_id, "group_ids": [first]}
    )
    restored = command(store, "restore", {"id": trash_id})
    assert restored["groups"][0]["workflow_id"] == replacement_id
    assert compile_document(restored) == compile_document(newer)
    assert len(restored["workflows"]) == 2


def test_deleted_group_restore_cannot_steal_a_reassigned_camera() -> None:
    store = WorkflowStore()
    _, first, second = setup_assets(store)
    deleted = command(store, "delete_group", {"id": first})
    command(
        store, "update_group", {"id": second, "camera_ids": ["camera01", "camera02"]}
    )
    before = store.load()
    with pytest.raises(WorkflowValidationError, match="only one"):
        store.candidate(
            "restore", {"id": deleted["trash"][0]["id"]}, before["revision"]
        )
    assert store.load() == before


def test_removed_draft_nodes_prune_only_their_stale_gates_on_apply() -> None:
    store = WorkflowStore()
    workflow_id, first, _ = setup_assets(store)
    command(store, "apply", {"workflow_id": workflow_id, "group_ids": [first]})
    command(
        store,
        "update_group",
        {"id": first, "block_enabled": {"face": False, "events": False}},
    )
    value = graph()
    value["nodes"] = [n for n in value["nodes"] if n["id"] != "face"]
    value["edges"] = [
        e for e in value["edges"] if "face" not in (e["from_node"], e["to_node"])
    ]
    command(
        store, "save_workflow", {"id": workflow_id, "name": "No faces", "graph": value}
    )
    doc = command(store, "apply", {"workflow_id": workflow_id, "group_ids": [first]})
    assert doc["groups"][0]["block_enabled"] == {"events": False}


def test_corrupt_and_oversized_persistent_state_fail_closed(tmp_path: Path) -> None:
    path = tmp_path / "workflow-assets.json"
    store = WorkflowStore(path)
    for body in (b"not JSON", b"{}", b"x" * (model.MAX_DOCUMENT_BYTES + 1)):
        path.write_bytes(body)
        with pytest.raises(WorkflowStorageError):
            store.load()
        with pytest.raises(WorkflowStorageError):
            store.candidate("create_group", {"name": "No fallback"}, 0)
        assert path.read_bytes() == body


def test_failed_atomic_write_preserves_last_document(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = WorkflowStore(tmp_path / "workflow-assets.json")
    saved = command(store, "create_workflow", {"name": "Original"})
    before = store.path.read_bytes()
    candidate = store.candidate(
        "create_group", {"name": "Uncommitted"}, saved["revision"]
    )

    def fail_replace(*_args) -> None:
        raise PermissionError("disk write rejected")

    monkeypatch.setattr(model.os, "replace", fail_replace)
    with pytest.raises(WorkflowStorageError):
        store.commit(candidate, saved["revision"])
    assert store.path.read_bytes() == before
    assert store.load() == saved
    assert not list(tmp_path.glob("*.tmp"))


def test_document_and_trash_limits_reject_without_mutation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = WorkflowStore()
    doc = command(store, "create_workflow", {"name": "First"})
    first = doc["workflows"][0]["id"]
    doc = command(store, "create_workflow", {"name": "Second"})
    second = doc["workflows"][1]["id"]
    monkeypatch.setattr(model, "MAX_TRASH", 1)
    before = command(store, "delete_workflow", {"id": first})
    with pytest.raises(WorkflowValidationError, match="trash"):
        store.candidate("delete_workflow", {"id": second}, before["revision"])
    assert store.load() == before
    monkeypatch.setattr(model, "MAX_NODES", 2)
    with pytest.raises(WorkflowValidationError, match="nodes"):
        validate_graph(graph())


def test_persistent_document_does_not_accept_unknown_schema_or_revision(
    tmp_path: Path,
) -> None:
    path = tmp_path / "workflow-assets.json"
    doc = model.empty_document()
    for patch in (
        {"schema_version": 2},
        {"revision": True},
        {"secret": "not a supported field"},
    ):
        path.write_text(json.dumps({**doc, **patch}))
        with pytest.raises(WorkflowStorageError):
            WorkflowStore(path).load()
