"""The event registry must not change any label, policy or rendered prompt.

The registry replaced hard-coded label tuples and frozensets spread across the
grounding, sensitivity, scene and overlay modules. The fixture was generated
from the code immediately before that replacement, so a diff here means the
refactor changed observable behaviour rather than just its source of truth.

Regenerate only with a recorded reason:

    python -c "import json,sys; sys.path[:0]=['src','tests']; \
        from test_event_registry_parity import build_snapshot; \
        print(json.dumps(build_snapshot(), ensure_ascii=False, indent=2))" \
        > tests/fixtures/event_registry_parity.json
"""

from __future__ import annotations

import inspect
import io
import json
from pathlib import Path
from typing import Any

from PIL import Image

from goodvision_harness import (
    grounding,
    overlay_fusion,
    scene_inspection,
    scene_runtime,
    scene_state,
    scene_worker,
)
from goodvision_harness.config import load_config
from goodvision_harness.contracts import NormalizedBBox
from goodvision_harness.event_sensitivity import (
    EVENT_SENSITIVITY_DEFAULTS,
    EVENT_SENSITIVITY_OPTIONS,
    normalize_event_sensitivities,
    sensitivity_policy_text,
    sensitivity_ui_info,
)
from goodvision_harness.scene_evidence import EvidenceFrame, InspectionRequest
from goodvision_harness.sliding_window import FrameWindow, SampledFrame

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = Path(__file__).parent / "fixtures" / "event_registry_parity.json"

_SMOKING_STANDARD_RULE = "Smoking STANDARD: supplied by the caller."


def _window() -> FrameWindow:
    return FrameWindow(
        index=3,
        frames=(
            SampledFrame(
                sample_index=0,
                source_frame_index=0,
                timestamp_seconds=0.0,
                jpeg_bytes=b"first-frame",
                width=320,
                height=180,
            ),
            SampledFrame(
                sample_index=1,
                source_frame_index=2,
                timestamp_seconds=0.4,
                jpeg_bytes=b"second-frame",
                width=320,
                height=180,
            ),
        ),
    )


def _jpeg() -> bytes:
    with Image.new("RGB", (64, 48), "red") as image, io.BytesIO() as encoded:
        image.save(encoded, format="JPEG")
        return encoded.getvalue()


def _text_parts(messages: list[dict[str, Any]]) -> list[str]:
    """Keep only prompt text; encoded images depend on the local Pillow build."""

    parts: list[str] = []
    for message in messages:
        content = message["content"]
        if isinstance(content, str):
            parts.append(content)
            continue
        parts.extend(
            item["text"] for item in content if item.get("type") == "text"
        )
    return parts


def _multi_messages(
    event_labels: tuple[str, ...],
    *,
    append_contract: bool,
    sensitivities: dict[str, str] | None = None,
) -> list[str]:
    settings = load_config(ROOT / "config.yaml").inference
    prompt = (
        settings.multi_event_prompt
        if event_labels == grounding.SAFETY_EVENT_LABELS and not append_contract
        else settings.prompt
    )
    return _text_parts(
        grounding.build_multi_grounding_messages(
            _window(),
            system_prompt=settings.system_prompt,
            prompt=prompt,
            event_labels=event_labels,
            append_contract=append_contract,
            event_sensitivities=sensitivities,
        )
    )


def _temporal_messages(event_labels: tuple[str, ...]) -> list[str]:
    box = NormalizedBBox(0.1, 0.1, 0.4, 0.9)
    job = scene_runtime.TemporalInspection(
        "session-1",
        "source-1",
        InspectionRequest("request-1", ("person-1",), "new_person", event_labels, 0, 1),
        (
            EvidenceFrame("session-1", 10, 0, _jpeg()),
            EvidenceFrame("session-1", 11, 1, _jpeg()),
        ),
        ((10, "person-1", box), (11, "person-1", box)),
    )
    return _text_parts(scene_inspection.build_temporal_messages(job))


def build_snapshot() -> dict[str, Any]:
    """Render everything the registry now owns, as comparable JSON."""

    safety = grounding.SAFETY_EVENT_LABELS
    all_standard = dict.fromkeys(safety, "standard")
    all_high = dict.fromkeys(safety, "high")
    return {
        "labels": {
            "safety": list(safety),
            "detection": list(grounding.DETECTION_EVENT_LABELS),
            "stranger": grounding.STRANGER_EVENT_LABEL,
            "event_names_zh": dict(grounding._EVENT_NAMES_ZH),
            "scene_state": sorted(scene_state._EVENT_LABELS),
            "scene_runtime": sorted(scene_runtime._EVENTS),
            "scene_inspection": sorted(scene_inspection._EVENT_LABELS),
            "scene_runtime_default": list(
                inspect.signature(scene_runtime.SceneRuntime.__init__)
                .parameters["event_labels"]
                .default
            ),
            "overlay_person_targets": sorted(overlay_fusion._PERSON_TARGET_EVENTS),
            "overlay_object_targets": sorted(overlay_fusion._OBJECT_TARGET_EVENTS),
        },
        "sensitivity": {
            "defaults": dict(EVENT_SENSITIVITY_DEFAULTS),
            "options": {
                label: [dict(option) for option in options]
                for label, options in EVENT_SENSITIVITY_OPTIONS.items()
            },
            "ui_info": sensitivity_ui_info(),
            "normalized_defaults": normalize_event_sensitivities(None, safety),
            "normalized_subset": normalize_event_sensitivities(
                {"smoking": "standard"}, ("smoking", "fall")
            ),
            "policy_defaults": sensitivity_policy_text(
                normalize_event_sensitivities(None, safety),
                smoking_standard_rule=_SMOKING_STANDARD_RULE,
            ),
            "policy_all_standard": sensitivity_policy_text(
                normalize_event_sensitivities(all_standard, safety),
                smoking_standard_rule=_SMOKING_STANDARD_RULE,
            ),
            "policy_all_high": sensitivity_policy_text(
                normalize_event_sensitivities(all_high, safety),
                smoking_standard_rule=_SMOKING_STANDARD_RULE,
            ),
            "policy_evidence_only": sensitivity_policy_text(
                normalize_event_sensitivities(None, safety),
                smoking_standard_rule=_SMOKING_STANDARD_RULE,
                include_output_guidance=False,
            ),
        },
        "prompts": {
            "multi_fixed_four": _multi_messages(safety, append_contract=False),
            "multi_four_with_contract": _multi_messages(safety, append_contract=True),
            "multi_single": _multi_messages(("smoking",), append_contract=True),
            "multi_pair": _multi_messages(("fight", "fall"), append_contract=True),
            "multi_all_high": _multi_messages(
                safety, append_contract=True, sensitivities=all_high
            ),
            "temporal_four": _temporal_messages(safety),
            "temporal_pair": _temporal_messages(("smoking", "fight")),
        },
    }


def test_registry_preserves_labels_policies_and_prompts() -> None:
    expected = json.loads(FIXTURE.read_text(encoding="utf-8"))
    actual = build_snapshot()
    for section in expected:
        assert actual[section] == expected[section], section
    assert actual == expected


def test_verdict_log_stays_restricted_to_the_built_in_safety_labels() -> None:
    """A session-supplied label must never widen what the verdict log records."""

    assert scene_worker._PUBLIC_VERDICT_LABELS == frozenset(
        {"fight", "smoking", "fall", "weapon"}
    )
