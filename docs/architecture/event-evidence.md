# Continuous events and video evidence

The analysis JSON/SSE and webhook remain immediate, per-observation outputs.
An archive event remaining active does **not** mean the current frame is positive.
Recording, identity sidecars and archive revisions never wait in the inference path.

Mixed sessions schedule local faces and remote safety in independent bounded pools.
Each completed branch publishes its own AnalysisResult v2 message immediately;
another branch appearing in `missing_event_labels` does not imply a failure.
Source times can interleave across branches. Freshness is enforced separately for
safety and faces, and capability latest endpoints retain each capability's newest
source observation. Every valid result, including late negatives, is published
so consumers can reconstruct source-ordered state. Late positives also produce
a console event notice without replacing a newer video frame. Each message keeps its original frame and identity
time basis. The delayed display carries annotations independently on original
source frames; exact evidence keeps the analyzed frame's original verdict.

## People and continuity

New strangers are reported on the first usable unmatched face observation; no
consecutive-frame confirmation is required. Known and unknown faces use the same
`face.min_face_size` floor (20 pixels by default), with no extra stranger size gate.
Annotated video draws face and safety-event boxes independently. Known faces stay
green and strangers stay red; a safety event draws its own target boxes without
requiring a face observation, actor match or `subject_id`. Face objects are not
merged into rich body tracks. Event-to-rich-object tracking remains available.

Each safety event keeps its latest positive boxes and alert until ten consecutive
valid negative source frames clear both together. A positive updates the boxes
and resets the count. Replayed or revised observations of the same frame count
once; skipped/failed branches and other event types do not advance the counter.
Older positives cannot revive a newer confirmed clear.

The display preserves original video frames between inference results. Empty or
expired face observations remove face boxes independently of safety-event boxes;
face annotations expire after three seconds without a fresh face result. Session
reset clears all display state. Held boxes are display annotations, not new
observations: per-frame JSON, exact evidence and archive identity associations
retain their existing semantics.

`subject_id` identifies a person track within one source/session. `person_id` is
the independently enrolled identity. Frame-local `object_id` and model-local
`actor_id` are not persistent identities. Known identities survive side views and
ordinary failed matches while a reliable track continues; ambiguous crossings,
identity conflicts and more than 3 seconds without real observations require
reconfirmation. Predicted boxes cannot refresh the last real observation.

Without a GPU worker, existing event-model requests may return up to eight
compact actors on positive frames. Compact all-negative responses omit actors.
Missing auxiliary fields are
optional and do not invalidate an otherwise valid event judgment. Standard smoking
and weapon boxes identify visible items. High-sensitivity smoking may ground the
visible hand/mouth action or involved person when the item cannot be identified;
the description must disclose suspected smoking and cannot invent item details.
People are associated
separately. A pure local-face session does not acquire a new remote dependency.

The console exposes standard/high sensitivity for each safety event before a
run. HTTP and WebSocket start accept `event_sensitivities`, for example
`{"smoking":"high","fall":"standard"}`. Choices are frozen for the session and
reported in its response/status. Smoking defaults to high; other safety events
default to standard. Standard smoking requires an identifiable smoking item;
high smoking accepts plausible actions and context without that requirement.
Neither mode changes the full-scene 320 px input or local face resolution.
For display with or without face tracking, a pending clear retains both the event
title and its latest positive boxes until the tenth valid clear. Exact evidence and
per-frame JSON remain the original model observations.

With a ready GPU detector, the shared `/v1/tracks` worker path runs independently
of optional attributes and UI overlays. `/health/tracking` checks the detector;
the existing `/health/ready` retains its full rich-label readiness meaning.
`rich_labeling.tracking_enabled` defaults to true; `PERSON_TRACKING_ENABLED`
overrides it. Unavailable workers are probed asynchronously and do not delay
analysis. Turning rich labels off still permits tracking.

## Event resources

| GET endpoint | Purpose |
| --- | --- |
| `/v1/events` | Canonical event list with cursor pagination |
| `/v1/events/stream` | Independent archive lifecycle/revision SSE |
| `/v1/events/{event_id}` | Current detail, resolving aliases |
| `/v1/events/{event_id}/video` | Current available MP4, or pending state |
| `/v1/events/{event_id}/video?revision=N` | Exact immutable video revision |

List filters: `event_label`, `stream_session_id`, `subject_id`, `message_id`,
`since`, `until`, `limit` (1–200, default 50), and `before` (returned
`next_cursor`). Numeric `since`/`until` are source-video seconds, so include a
session filter when comparing a specific recording. The response contains
`events`, `next_cursor`, `epoch`, and `cursor`.

Use `GET /v1/events?message_id=<analysis message ID>` to find the current archive
association of an observation. Original AnalysisResult v2 messages are never
rewritten; optional object attributes can contain `subject_id`, `subject_ids`
and event references already available at publication time. Absence of a
reference in an immediate response does not mean that archival failed.

Each event includes its label, source/session, subject association, source-time
`start_seconds`, `last_detected_seconds`, `end_seconds`, `end_reason`, identity
information, `revision`, `finalized`, and `evidence`. Different people and labels
are independent. Uncertain ownership stays explicitly unassociated.
For `subject_type: target` or `scene`, the subject key denotes a fallback target
or scene track; it does not identify a person. `group` carries all associated
participants in `subject_ids`.

The first positive immediately enters the background event queue. Two seconds
of consecutive explicit negative judgments close an archive episode. A one-second
pause followed by another positive stays in that episode. Missing/failed/skipped
analysis is unknown, not negative; five seconds without positive support closes
with an observation-timeout reason. These intervals only govern the archive.

Out-of-order results can revise the archive for up to 65 seconds after each
window was submitted. Earlier pending windows are tracked separately by label.
Mixed sessions register separate safety/stranger window identities, including
their own dropped, failed and cancelled states. A face-only message listing
safety as missing cannot interrupt safety's explicit negative interval, and vice
versa. Reassociated observations retain their original branch/window scope.
Once the closing boundary is resolved, the ended event can be finalized early.
Late positives can merge episodes; late negatives can split them. Published
observation anchors preserve identifiers where possible, with dynamic aliases
for merged identifiers. Always resolve current associations through the API.

SSE uses `event: event_change`; its data contains complete affected `events`
and `aliases` (`event_id`, `canonical_event_id`). Apply each message atomically:
upsert events and remove their previous aliases, then apply the supplied aliases.
Video status changes use the same event. Connect using the list response's
`?after=<epoch>:<cursor>`, or reconnect with `Last-Event-ID`. On `event: reset`,
clear the cached archive list and query it again. Reset occurs after restart,
expired replay history or retention changes. This is independent of the existing
`/v1/results/events` stream, whose per-observation contract is unchanged.

## Video and resource limits

The default clip is source-time first positive minus 1 second through plus
2 seconds: approximately 3 seconds, 5 FPS, longest side at most 720 pixels,
H.264/yuv420p MP4, no audio. Source images are retained independently of the
smaller inference images. Real source gaps or beginning/end truncation are
reported as partial clips, with actual timestamps and duration.

The 5 FPS setting controls playback cadence; repeated output frames are not new
source observations. A real, contiguous successor source frame can prove coverage
up to the requested end (including for a 1 FPS source). Without that boundary,
evidence stays short and never extends across a source outage. H.264 dimensions
are rounded down to even sizes; small sources are not enlarged except for the
encoder's minimum two-pixel dimension.

`evidence.revision` and `evidence.status` describe the requested video;
`evidence.requested_window` describes the target interval. `evidence.available`
describes the already playable revision, including its URL and actual interval.
A revised interval can be encoding while an older available clip remains
playable. The old clip is never advertised as the new revision. Version URLs
never change bytes; retired versions return 404. MP4 responses support HTTP Range.
Without an available clip, the unversioned endpoint returns 202 for pending
work or 503 for a failed recording. Read the evidence state before retrying.

The `events` configuration supports:

| Setting | Default |
| --- | --- |
| `directory` | `event-evidence` beside the known-face gallery |
| `clip_seconds`, `pre_roll_seconds`, `fps` | 3, 1, 5 |
| `max_image_side`, `jpeg_quality` | 720, 70 |
| `buffer_seconds`, `buffer_max_bytes` | 90, 67108864 |
| `retention_days`, `max_storage_bytes` | 7, 10737418240 (10 GiB) |
| `revision_seconds` | 65 |
| `negative_gap_seconds`, `unknown_timeout_seconds` | 2, 5 |
| `queue_size`, `encoder_timeout_seconds` | 512, 10 |
| Video encoder | Supplied by the runtime |

`EVENT_EVIDENCE_DIRECTORY` overrides the configured directory.
Keep buffer time at least revision time plus clip time. Docker uses the
existing writable `/data` volume, normally `/data/event-evidence`; do not use the
small `/tmp` tmpfs for the archive. SQLite, staging and final clips are stored
together. Staged complete work can resume after restart; interrupted work with
missing source frames is reported as failed. Age or space limits retire old ended
event resources, including their videos and associations. Queue/storage/encoder
failures remain visible in `/v1/status` and event detail while recognition continues.

## Migration and validation

Stranger crop capture and the persistent stranger embedding gallery have been
removed. `/v1/faces/strangers` and `/v1/faces/strangers/{stranger_id}/photo` return
410 with `code: stranger_photos_removed` and `events_url: /v1/events`, without
initializing face inference. New results do not contain `stranger_id` or `photo_url`.

The enrolled gallery, model volumes and deployment settings remain in use. Existing
`face-gallery-strangers.json` and `stranger-captures/` are not loaded or served.
After backup, an operator can remove those two legacy resources from the face-data
volume; do not remove `face-gallery.json`, models, or the volume itself.

The dashboard's event history and player are independent of its live capability
cards. Synthetic dry-run sessions can generate real playable clips of synthetic
input. Such tests validate delivery and temporal behavior, not real-model accuracy.
Validate actual identity switches, duplicate events and first-result/P95 latency
on representative recordings on the target CPU and GPU deployments.
See the current [release validation](../../RELEASE.md) for completed publication,
continuity and image-delivery checks, and the remaining real-model limits.
