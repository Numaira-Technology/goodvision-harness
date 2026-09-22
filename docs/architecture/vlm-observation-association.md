# VLM observations and tracking-owned event state

The independent VLM branch publishes each temporal observation immediately.
It no longer compares boxes across requests, retains a person-box history, or
maintains a second person-event state machine. The former 3-second candidate
cache and 1.2-second / 0.35-IoU pose matching are removed.

SAM/native tracking supplies person continuity. Background association binds
each VLM observation to tracking at its source time, then submits its positive,
negative or uncertain evidence to the existing scene event state machine.
Confirmation, clearing and duplicate-source handling use that scene's tracking
IDs and configured evidence rules.

## Association retained for one observation

The VLM still receives original frames and returns local participant references
and boxes. Associating those participants with SAM tracks remains necessary:
the two branches do not share IDs in their model inputs.

This binding uses tracking observations within 0.55 source seconds. Matches must
be mutual best matches with IoU at least 0.5 and a 0.15 advantage over another
candidate. There is no relaxed pose threshold. Matching failure leaves the
observation unassociated; it does not create a replacement track or a clear vote.

VLM publication does not wait for SAM. Without tracking, event observations and
localization diagnostics remain available, but repeated boxes cannot confirm
or clear a person-event episode.

## Output semantics

- `vlm_report.subject_id_scope` is `observation`. Its object/subject IDs are valid
  only within that report, even if consecutive boxes or model-local IDs match.
  `objects` contains only participants localized in this observation.
- `observations[].accepted` means the observation has complete participant
  grounding and is eligible for background association. It does not mean that
  a scene vote has been accepted or that an event is confirmed.
- `judgments[].observed_detected` preserves the current model decision. Positive
  judgments remain `confirmation_state: pending` and `detected: null`; negative
  judgments have `detected: false`. These are not historical episode states.
  Unknown and omitted judgments remain distinct from negatives.
- Raw VLM reports omit `events`, `missing_event_labels`, `association` and
  `confirmation`. They describe a current observation, not a persistent episode.
  Judgment `event_ids` / `object_ids` remain empty compatibility fields.
- Requested VLM labels are session context: `/v1/vlm/status.event_labels` exposes
  the branch's configured labels alongside `stream_session_id` and `generation`.
  Retain that context for the corresponding run. Missing labels are the requested
  set minus `judgments[].event_label`; a negative or uncertain judgment is still
  a returned judgment. Do not infer missing labels from the set of all supported
  events or use a new session's configuration for an old report. `analysis_complete`
  remains the immediate completeness flag (and is false on request failure).
- `type: vlm_report` establishes the observation-only semantics. Association and
  confirmation destinations are protocol conventions rather than static objects
  repeated in every report. Background binding never rewrites the original report.
- `/v1/vlm/associations` connects report-local subject and evidence IDs with
  tracking subject IDs and scene event IDs. Its legacy `vlm_event_ids` list is
  empty because no independent VLM episodes are created.
- `/v1/scene` and scene SSE carry persistent event state. Tracking-owned safety
  results also publish as `analysis_kind: scene_state` through the normal result
  stream and webhook. They do not replace the global latest VLM observation.
  Feature-specific latest results remain source ordered; consumers must inspect
  `analysis_kind` or use `/v1/scene` when they need persistent event state.
- The camera's current-observation summary says “发现” for a raw positive.
  It recognizes `type: vlm_report`; confirmation and clearing belong to the
  continuous-scene display.

This reduction applies to `vlm_report` and its overall-description projection.
Scene snapshots retain their `events`; other analysis-result protocols retain
their existing completeness metadata. The report builder emits 25 root fields;
the published report adds `process_metrics` for 26, before camera-specific wrapping.

The VLM model output protocol is unchanged. Invalid geometry still preserves a
reliable event decision, but cannot contribute person-specific confirmation or
clearing. No approximate match between successive VLM replies is used as identity
evidence.

Offline regressions cover non-overlapping movement with a stable tracking ID,
different tracking IDs at the same position, duplicate evidence, missing/bad
localization, confirmation and clearing, publication while SAM is blocked, and
scene-result delivery. They do not establish SAM identity accuracy or model
latency on real video.
