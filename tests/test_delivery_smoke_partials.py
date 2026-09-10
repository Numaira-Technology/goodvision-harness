"""The delivery smoke accepts independent partials without weakening coverage."""

from __future__ import annotations

import asyncio
import copy
import json
import runpy
from pathlib import Path

import httpx
import pytest


@pytest.fixture(scope="module")
def smoke():
    return runpy.run_path(str(Path(__file__).parents[1] / "tools/delivery_smoke.py"))


def payload(labels, *, message="safety", timestamp=2.0):
    all_labels = ["fight", "smoking", "fall", "weapon", "stranger"]
    return {
        "type": "analysis_result", "schema_version": "2",
        "analysis_kind": "selected_features", "message_id": message,
        "frame": {
            "width": 240, "height": 160, "source_frame_index": int(timestamp * 10),
            "timestamp_seconds": timestamp,
        },
        "analysis_complete": len(labels) == 5,
        "missing_event_labels": [label for label in all_labels if label not in labels],
        "judgments": [
            {"event_label": label, "detected": True, "confidence": 0.9,
             "object_ids": [label]} for label in labels
        ],
        "objects": [
            {"object_id": label, "attributes": {"dry_run": True},
             "bbox": {"coordinate_system": "source_normalized_xyxy"}}
            for label in labels
        ],
    }


def test_smoke_requires_exact_missing_labels_and_positive_evidence(smoke):
    safety = payload(["fight", "smoking", "fall", "weapon"])
    face = payload(["stranger"], message="face", timestamp=1.8)
    for item in (safety, face):
        smoke["check_result"](item)
    for change in (
        {"missing_event_labels": []},
        {"analysis_complete": True},
        {"missing_event_labels": ["stranger", "stranger"]},
        {"judgments": safety["judgments"][:-1]},
        {"objects": []},
    ):
        with pytest.raises(RuntimeError):
            smoke["check_result"]({**safety, **change})


def test_smoke_feature_projection_requires_original_frame_and_all_metadata(smoke):
    source = payload(["fight", "smoking", "fall", "weapon"])
    projection = {
        **source, "judgments": [source["judgments"][1]],
        "objects": [source["objects"][1]],
    }
    smoke["check_result"](projection, "smoking")
    smoke["check_projection"](source, projection, "smoking")
    for key, value in (
        ("frame", {**source["frame"], "timestamp_seconds": 1.8}),
        ("message_id", "rewritten-message"),
        ("missing_event_labels", []),
        ("objects", source["objects"]),
    ):
        with pytest.raises(RuntimeError, match="original frame or fields"):
            smoke["check_projection"](source, {**projection, key: value}, "smoking")


def test_smoke_global_sse_checks_all_five_positives_and_monotonic_cursors(smoke):
    safety = payload(["fight", "smoking", "fall", "weapon"])
    face = payload(["stranger"], message="face", timestamp=1.8)

    async def check(rows):
        def response(request):
            body = "".join(
                f"id: epoch:{cursor}\nevent: analysis_result\n"
                f"data: {json.dumps(row)}\n\n"
                for cursor, row in rows
            )
            return httpx.Response(
                200, content=body.encode(), headers={
                    "Content-Type": "text/event-stream", "X-Accel-Buffering": "no",
                },
            )

        async with httpx.AsyncClient(
            base_url="http://isolated.test", transport=httpx.MockTransport(response),
        ) as client:
            return await smoke["check_sse"](
                client, "/v1/results", None, through_cursor=2,
            )

    assert asyncio.run(check([(1, safety), (2, face)])) == {
        "safety": safety, "face": face,
    }
    with pytest.raises(RuntimeError, match="missing synthetic results"):
        asyncio.run(check([(1, safety)]))
    with pytest.raises(RuntimeError, match="cursor did not increase"):
        asyncio.run(check([(2, safety), (1, face)]))
    no_face_positive = copy.deepcopy(face)
    no_face_positive["judgments"][0].update(detected=False, object_ids=[])
    no_face_positive["objects"] = []
    with pytest.raises(RuntimeError, match="missing synthetic results"):
        asyncio.run(check([(1, safety), (2, no_face_positive)]))
