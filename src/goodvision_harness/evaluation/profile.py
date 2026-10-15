"""Fixed, question-independent event profiles without service startup or model calls."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

from ..event_sensitivity import sensitivity_policy_text
from .vocab import FROZEN_EVENT_LABELS, OBSERVATION_EVENT_LABELS, VOCAB_VERSION

DEFINITIONS_SHA256 = "241f7243b52be5947d2f88ddbd6e3e89fcb3c793135e6bbeeac905cd5bf6aa22"


@dataclass(frozen=True)
class EvaluationEventProfile:
    labels: tuple[str, ...]
    scene_only_labels: tuple[str, ...]
    definitions: tuple[tuple[str, str], ...]
    source_sha256: str
    profile_sha256: str

    def kwargs(self):
        return {"allowed_event_labels": self.labels,
                "scene_only_event_labels": self.scene_only_labels,
                "event_definitions": self.definitions}

    def provenance(self):
        return {"vocab_version": VOCAB_VERSION, "labels": list(self.labels),
                "scene_only_labels": list(self.scene_only_labels),
                "source_sha256": self.source_sha256, "profile_sha256": self.profile_sha256,
                "question_conditioned": False, "face_enabled": False}

    def scene_runtime(self, *, stream_session_id, source_id, diagnostics=None,
                      tracking_mode="immediate", native_memory_scope=None,
                      tracking_history_seconds=12.0, max_entities=128):
        from ..runtime.scene import SceneRuntime
        return SceneRuntime(stream_session_id, source_id, event_labels=self.labels,
                            stranger_enabled=False, independent_temporal_inference=True,
                            tracking_mode=tracking_mode, native_memory_scope=native_memory_scope,
                            tracking_history_seconds=tracking_history_seconds, max_entities=max_entities,
                            diagnostics=diagnostics, **self.kwargs())

    def vlm_config(self, *, stream_session_id, source_id, generation, model_settings,
                   model, api_key, parameters=None, options=None, admission=None,
                   observation_fps=1.25, max_in_flight=4, route_id=None):
        from ..vlm_independent import VLMProcessConfig
        parameters = dict(parameters or {})
        # Events and grounding have different contracts; do not impose a shared six-label schema.
        if "response_format" in parameters:
            raise ValueError("parallel evaluation requires per-lane output contracts, not a shared schema")
        return VLMProcessConfig(
            stream_session_id=stream_session_id, source_id=source_id, generation=generation,
            labels=self.labels, model_settings=model_settings, model=model, api_key=api_key,
            parameters=parameters, options=dict(options or {}), admission=admission,
            observation_fps=observation_fps, max_in_flight=max_in_flight, route_id=route_id,
            output_mode="parallel", **self.kwargs())

    def window(self, *, window_id, start, end, request_ids=None):
        return {"window_id": window_id, "start": start, "end": end,
                "event_labels": list(self.labels), "report_lanes": ["information", "grounding"],
                "request_ids": dict(request_ids or {})}


def load_profile(definitions_path):
    """Load 51 frozen definitions plus the standard fight policy without question-based filtering."""
    raw = Path(definitions_path).read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != DEFINITIONS_SHA256:
        raise ValueError("fixed event definitions hash mismatch")
    rows = json.loads(raw)
    if not isinstance(rows, list) or tuple(row["label"] for row in rows) != FROZEN_EVENT_LABELS:
        raise ValueError("fixed 51-label order mismatch")
    definitions, scene_only = [], []
    for row in rows:
        if row.get("target") not in {"person", "object"}:
            raise ValueError("explicit event target required")
        text = json.dumps({key: row[key] for key in (
            "definition", "counter_examples", "target", "target_hint") if key in row}, ensure_ascii=False)
        definitions.append((row["label"], text))
        if row["target"] == "object":
            scene_only.append(row["label"])
    fight = sensitivity_policy_text({"fight": "standard"}, smoking_standard_rule="",
                                    include_output_guidance=False)
    definitions.append(("fight", fight))
    payload = {"version": VOCAB_VERSION, "definitions": definitions, "scene_only": scene_only}
    profile_digest = hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False,
                                               separators=(",", ":")).encode()).hexdigest()
    return EvaluationEventProfile(OBSERVATION_EVENT_LABELS, tuple(scene_only),
                                  tuple(definitions), digest, profile_digest)
