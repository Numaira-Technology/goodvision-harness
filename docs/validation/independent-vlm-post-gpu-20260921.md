# Independent VLM and post-GPU backend repairs — 2026-09-21

Based on `a225e4433a8aba2d1b8b560c9b32a04965231423`, isolated branch `codex/sam-realtime-queue`. This change does not change SAM sampling, persistent video-session semantics, original-video transport, or the 4-second playback buffer. Browser queue repairs from the subsequent L40S report are a separate change.

## Runtime ownership

One spawned VLM process belongs to one immutable source/session/generation. It samples original source JPEGs independently, builds ordered multi-frame windows, makes bounded concurrent API requests, commits observations in source order using the existing two-positive/two-negative SceneState rules, and returns public reports. Identity processing, SAM, scene callbacks, evidence archival, and optional webhooks cannot gate report publication.

Network sessions use the shared VideoConnection's original-frame history with a private cursor. Browser-camera admission supplies validated original JPEGs before the downstream scene/SAM callback. Those callbacks have a separate bounded worker, so an upload does not hold the owner/control lock while waiting for scene state. Optional face model warmup is deferred to inference in mixed independent mode. Startup ownership/revision validation remains in place; tracker availability is not a VLM prerequisite.

The VLM parent reader publishes to its private ResultBus and the existing public ResultBus before background association. An epoch check atomically rejects publication to a retired common result bus. Old scene/finalization projections cannot overwrite independent safety reports. Stranger projections retain their existing path. No API credential or JPEG is included in public reports.

## API and compatibility

- `/v1/vlm/status`: independent process state, source/result age, successes/failures, incomplete/empty observations, queue drop counters, optional association status.
- `/v1/vlm/results/latest`: latest `vlm_report`, with source evidence IDs/times, actual request start/end and publish timestamps, measured latency, observations, confirmed events, missing labels and `analysis_complete`.
- `/v1/vlm/results/events`: SSE event `vlm_report`.
- `/v1/vlm/associations`: later links to canonical tracked people and scene event IDs, including original evidence/request IDs and the source-time delta.
- Existing `/v1/results/latest` and feature JSON/SSE routes receive independent reports; fleet forwarding permits the new routes. Legacy consumers must still distinguish `null` / missing evaluations from `false`.
- Default `runtime.scene_independent_vlm: true`; `SCENE_INDEPENDENT_VLM=false` restores the old scene-owned VLM scheduler. Passive/dry-run behavior and stranger/face contracts remain supported.

## Bounded work and lifecycle

| Work | Bound / behavior |
|---|---|
| VLM original input | 8 JPEGs, at most 4 MiB each; coalesce by source sampling cadence and reject full/old inputs before model admission |
| Child evidence window | Last 12 samples, at most 16 MiB and 3 source seconds; each request uses up to 4 frames and at least 0.75 seconds |
| Model requests | Configured cadence, 1–4 in flight; ordered commit, existing finite HTTP deadline, no unbounded retries |
| Public output | IPC queue 16 and local history 128; output pressure is counted |
| Evidence for later association | 64 source samples / 32 MiB |
| Optional association | 64 reports / 32 MiB / 90 seconds; retired/expired/missing matches counted; 256 retained links |
| Browser source callbacks in parent | One active plus 4 queued JPEGs; only not-yet-admitted callbacks may be superseded, with explicit counters |
| Live network face admission | Default 2 observations / 32 MiB, held through inference and all ordered delivery callbacks, not released at enqueue |

End-of-source freezes further VLM admission, sends a child sentinel, drains outstanding API reports, and only then finishes optional association. The camera source callbacks drain before the final tracking watermark is declared. Capture elapsed time still freezes immediately on stop; accepted upstream tracking frames remain valid. Process shutdown has bounded joins/terminate fallback and releases queues/evidence. Child failures are visible, and a new run creates a new process and independent confirmation state.

After tracking has ended, a retained original whose sequence exceeds the last completed SAM watermark is known unavailable for outlining. This fixes the independently observed 5060 Ti tail-frame wait without inventing a result or closing identity/evidence prematurely. An exceptional display-drain timeout is visible and only ends presentation work.

## Semantics learned from real API testing

Removing leftover empty SAM-box metadata and requesting explicit label assessment improved the model's usable output. Unlocalized negatives may be retained for audit alongside valid grounded positives, but cannot vote on a person, clear an event, or count as complete coverage. Ungrounded positives, invalid coordinates, malformed responses, and unknown subject references are rejected.

Matching uses reciprocal unique box overlap with an ambiguity margin. Strict matching allows brief reacquisition; a separate pose-tolerant fallback is limited to 1.2-second neighbors, mutual center containment and IoU >= 0.35. This handles observed fall/get-up box changes without lowering the stricter IoU >= 0.5 used for delayed SAM association. It does not identify people; ambiguous matching creates a fresh local actor rather than combining votes.

Full 55.8-second replays used the existing `qwen/qwen3-vl-8b-instruct:nitro` route at 640px / JPEG70 / 1600 output tokens. No SAM or newly rented GPU was used. Final prompt run: 69 reports, 68 valid, 1 invalid-coordinate response; source-ingress-to-publication P50 1.359s, P95 2.916s, max 3.469s. Actual two-observation confirmations occurred at source 1.6s (smoking-like), 8.8s (striking) and 13.6s (fall). Replaying those exact saved model responses with the actor-association correction cleared the fall at 15.2s. That last step is an offline state replay, not a new paid API performance run.

The model still returned 38 empty reports, and only 29 reports explicitly covered all requested labels. Empty/omitted replies are unknown, not verified negatives. One short staged clip does not establish general recall or precision. The VLM-owned image cache is temporary; durable later scene/identity association remains optional and can expire when SAM never catches up. Do not claim every original video frame or unmatched VLM image is durably archived.

Detailed visual reference, per-run counts and evidence: `D:/Documents/Project Bird/code/goodvision-harness/.goodvision/diagnostics/vlm-independent-20260921/COMPARISON.md`. Raw API replies and exact timing evidence are under `C:/Users/frank/.codex/visualizations/2026/09/20/01a0be90-703a-7331-b130-c82042016014/vlm-independent-20260921/`.

## Verification

102 targeted tests passed together (33.48s): actual spawned-process lock isolation, HTTP/SSE and fleet contracts, input pressure, crash/restart, source cursor and camera stop, face durable ordering, unavailable display tails, current/legacy scene delivery, event confirmation, empty/ungrounded replies, and temporal actor ambiguity. A subsequent 14-test focused rerun passed after adding independent readiness. The mixed face/tracker startup check is recorded in the final validation receipt alongside this document.

Legacy scene tests use `SCENE_INDEPENDENT_VLM=false`; new independent tests explicitly enable the production default. Always set `PYTHONPATH=<this worktree>/src` when testing: an installed editable package elsewhere otherwise imports the main checkout. Ruff checks and `git diff --check` pass for the new modules and tests. These are meaningful local regressions, not a new full dual-GPU acceptance run.

The 5060 Ti and L40S experiments both ran fixed commit `a225e44`; they diagnose the old version and do not validate these changes. The L40S measurement is a valid 300-second capture, but its cleanup script released GPU before final native EOF flush. Its resulting stop failure must not be attributed to this code or used as independent evidence of the old tail-frame deadlock. Future GPU teardown must await native final flush and terminal session state, not merely empty queue counters.
