# Live service isolation after the bcceda69 dual-camera run

Implemented directly on `codex/entity-event-engine` in the primary worktree.
Browser/fleet changes are owned by the testing task (f39777c and 681d420).

## Changes

- Display callbacks now publish source receipt/preview, capture their run and
  coordinator, and perform durable tracking admission and camera-frame archival
  outside the service state lock. A separate source/commit gate preserves the
  ordering of admitted evidence before native tracking commits. After admission,
  the callback rechecks generation, session, coordinator and scene before changing
  current presentation/scene state. Capture times and frame identities are kept.
- `/v1/status?lightweight=true` omits archive, tracking, display-worker and fusion
  statistics. `/v1/vlm/status` reads only bounded counters owned by the independent
  VLM branch. Full `/v1/status` keeps detailed diagnostics for existing clients.
- Both status modes expose `vlm`: state, running, ready, readiness_reason,
  last_source_age_seconds, last_success_age_seconds, last_attempt_state,
  last_attempt_error, analysis_complete, pending_requests, inflight_requests and
  publication/request counters. Failed responses do not refresh successful-result
  freshness. `pending_requests` counts finished requests waiting for ordered
  publication; `inflight_requests` counts unfinished dispatched requests. Counts
  are snapshots, refreshed on publication and at least once per second.
- Scene projections remain in history/SSE, webhooks and the stranger feature
  snapshot, but cannot replace the global latest independent VLM report. Legacy
  operation retains source-ordered latest-result selection.
- Display metrics use an independent lock and up to 64 source/viewer slots.
  `display_instance_id` separates report-sequence watermarks; the most recently
  accepted current-source viewer is shown intact rather than merging counters.
- VLM validation preserves safe reason codes across process boundaries, explicitly
  identifies invalid normalized boxes and strengthens the coordinate prompt.
  Invalid coordinates are still rejected; no inferred pixel-to-normalized
  conversion or relaxed positive grounding was added.
- Network decoding receives at least the browser target rate (15 FPS by default),
  including deployments still binding an old `display.fps: 5` configuration.
  Higher configured rates remain supported. This is an admission/output target,
  not proof that every source or browser actually achieves that rate.
- With tracking explicitly disabled, camera capacity comes from provisioned
  application workers. SAM-enabled configurations retain their GPU-lane limit.

## Local checks

The initial regression selection passed 38 tests. A broader selection passed 297
of 304 tests; seven fixtures needed the new status keyword/source gate or an
explicit tracking-enabled configuration. All seven then passed in a focused run.
The final targeted run passed 31 tests covering the updated network FPS path,
validation diagnostics, actual spawned VLM processes and service-level branch
isolation.

Tests exercise blocked admission, concurrent lightweight status, commit ordering,
generation replacement while admission is blocked, per-viewer telemetry, failed
result freshness, strict coordinate diagnostics and old display configuration.
Local tests use fake model responses; they establish correctness and isolation,
not real dual-camera throughput or event accuracy.

## Next experiment configuration and limits

For real Face + independent VLM without a SAM instance, use legacy tracking mode,
`rich_labeling.enabled: false`, `tracking_enabled: false`,
`processing_method: null`, and `HARNESS_FLEET_CAPACITY=2`. Keep backend
`sam31_stream` for the existing browser display contract, enable scene active mode,
`SCENE_INDEPENDENT_VLM=true`, and the browser display setting. Use CPU Face and the
authorized real VLM route. The testing task owns service startup and recording.

The four-second playback buffer, SAM temporal batching, worker inference logic
and tracking sampling policy are unchanged. Other scene/identity operations still
use the state lock; this patch does not claim that all shared-lock work has been
eliminated. A completed VLM response can remain partial/empty, so `ready` describes
transport/result freshness rather than accuracy or complete label coverage.
The next recording must validate sustained playback and complete timely judgments.
