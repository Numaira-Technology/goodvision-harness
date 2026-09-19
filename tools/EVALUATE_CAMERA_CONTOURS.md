# Offline contour-following evaluation

`evaluate_camera_contours.cjs` runs saved camera JPEGs through the actual display
JavaScript in headless Chrome, with a real Web Worker. It does not request camera
access, call a model, or access an external website. Install or supply Playwright
through `NODE_PATH`; supply Chrome with `--chrome` if it is not at the Windows
default path.

```powershell
node tools/evaluate_camera_contours.cjs `
  --fixtures .goodvision/diagnostics/<private-run> `
  --source .goodvision/diagnostics/<private-run>/camera-display-baseline.js `
  --output .goodvision/diagnostics/<private-run>/baseline-quality.json

node tools/evaluate_camera_contours.cjs `
  --fixtures .goodvision/diagnostics/<private-run> `
  --output .goodvision/diagnostics/<private-run>/candidate-quality.json `
  --synthetic-seconds 12 --target-fps 20

node --test tests/contour_quality_metrics.cjs
```

Omit `--source` to test the current implementation. Use `--quality false` for
the synthetic throughput test alone. Keep the machine otherwise idle for timing
comparisons; do not run baseline and candidate Chrome tests concurrently.

Use `--motion-backend dense` for the production OpenCV Worker loader. The local
server serves the exact `camera_display_dense.js` and vendored OpenCV 4.13.0
assets; no external CDN or substitute implementation is used. Each replay
Worker must acknowledge `motion_ready` with backend `dense` before evaluation.
The realtime Controller starts normally and exercises the bounded initialization
queue; its report must contain the same acknowledgement. Main JavaScript,
adapter, and vendor bytes are frozen at process startup, hashed in the report,
and saved beside it. Use `--dense-source` and `--opencv-source` with those saved
files, together with `--source`, to reproduce that exact implementation.
Omitting the backend flag retains the legacy motion path for baseline comparison.

`--anchor-delay-ms 3000 --max-propagation-seconds 4` tests three-second synthetic
model delivery and an experimental four-second propagation horizon. The default
horizon remains two seconds. This only configures the test; it does not change
production defaults. `--history-seconds 8` configures both Worker retention and
synthetic metadata retention; the default is six seconds. For a five-second
propagation experiment with a one-second display buffer, retain at least seven
seconds (eight provides room for scheduling jitter). Coverage is split
between warmup and a steady interval starting at `max(2, anchor delay + 1)` source
seconds and ending two seconds before the test ends.
`--display-delay-seconds 1.7` changes only the test Controller's display buffer
(default one second). Keep model delivery fixed with `--anchor-delay-ms` when
comparing buffers; actual callback-to-canvas timing remains measured. The steady
source-time window stays the same for comparability.

For an independently verified video corpus, use `--manifest <private.json>` and
`--scenarios production`. The manifest contains `run` and `sets`; each set has a
safe `name`, `sources` (`sequence`, `timestamp`, base64 `jpeg`) and `anchors`
(`source_sequence`, `media_timestamp`, `epoch`, `subjects`). A subject contains
`track_id`, normalized `polygons`, and `tracking_confirmation`. Verify exact
source-image provenance before building this manifest. The production scenarios
use the unchanged 0.2-second frame-gap limit, including delayed anchors up to
3.5 seconds; default fixture scenarios also include explicit sparse-gap tests.
Use `--motion-width 160`, `240`, or `320` to compare display-motion image sizes
(height remains capped at 240). This is passed both to replay grayscale decoding
and the real controller configuration; use a controller version that supports
`motion_width` for the throughput test. It does not change analysis/upload image
resolution. `--delays 1,2.5` can narrow the source-delay scenarios during diagnosis.

### Buffered 1 Hz anchor comparison

```powershell
node tools/evaluate_camera_contours.cjs `
  --fixtures .goodvision/diagnostics/<private-run> `
  --output .goodvision/diagnostics/<private-run>/buffered-quality.json `
  --motion-backend dense --replay-mode buffered --anchor-cadence-hz 1 `
  --model-delays 0.8,1.2 --display-delay-seconds 1.7 `
  --max-propagation-seconds 5 --history-seconds 8
```

This mode compares `anchor_policy=past_only` and `nearest_arrived` with identical
arrived metadata. It advances three event types on the original source timeline:
frame capture, model result arrival at anchor time plus the configured model
delay, and target presentation after the configured display buffer. A future
source may be used only after it was captured and its SAM result actually
arrived; each returned person's anchor is checked. Per-person source anchors
also control the frozen-anchor comparator when different people use different
geometry sources. The policies belong to production `ContourHistory`; the tool
does not emulate nearest selection by hiding metadata.

The planned cadence selects the first *existing* SAM reference at or after each
one-second tick. Missing intervals remain missing, recorded timestamps remain
unchanged, and actual supplied cadence is reported. No reference mask or precise
1 Hz frame is invented. Other recorded SAM frames are withheld as evaluation
references. Fixed model delays are assumptions, not a replay of measured latency
variation. Use `--scenarios production` with the verified continuous 10 fps
manifest; original 3 fps JPEGs still require the explicitly labelled sparse-gap
diagnostic and cannot prove 15 fps camera quality.

For an actual recorded SAM arrival trace, build a separate supply/reference
manifest and use `--replay-mode trace`:

```powershell
node tools/build_camera_contour_trace_fixture.cjs `
  --fixtures .goodvision/diagnostics/<private-run> `
  --trace .goodvision/diagnostics/<private-run>/sam-cadence-report.json `
  --metadata-delay-seconds 0.15 `
  --output .goodvision/diagnostics/<private-run>/actual-trace-fixture.json

node tools/evaluate_camera_contours.cjs `
  --manifest .goodvision/diagnostics/<private-run>/actual-trace-fixture.json `
  --output .goodvision/diagnostics/<private-run>/actual-trace-quality.json `
  --motion-backend dense --replay-mode trace --display-delay-seconds 1.7 `
  --max-propagation-seconds 5 --history-seconds 8
```

The builder verifies exact source sequence/time and arrival-clock consistency,
keeps actual SAM masks and new native epoch/ID unchanged, and adds an explicitly
assumed metadata delivery delay. References remain old recorded SAM geometry.
It requires exactly one person in every old reference and overlapping new SAM
observation, records an explicit geometry-only correspondence across the runs,
and rejects multiple new identities. This comparison cannot establish native
identity continuity. `reference_anchors` and supply `anchors` are separate;
every supply anchor carries its actual `availability_clock_seconds`. No cadence
resampling or nominal latency override occurs in trace mode. The exact-reference
control still checks rendering against the reference itself and is reported
separately from the actual-trace comparison.

The fixture folder contains `display-metadata.json`, `early-sam-results.json`,
`early-source-images.json`, and `tail-source-images.json`. The image lists have
`sequence`, original `timestamp`, and base64 `jpeg` fields. Early SAM records
contain `result.frames`, source coordinates and native identities; the tool
extracts the same saved contours, without generating new model output. Source
timestamps must match the reference exactly. Input images, result JSON, saved
implementation snapshots, and overlay PNGs belong in a private ignored folder.

## What is measured

- The exact-source control must have IoU 1 and return every reference person.
- The production-gap scenario retains the implementation's 0.2-second limit.
  A 3 fps JPEG recording cannot supply that continuity and should be rejected.
- The explicitly named `sparse_gap_*` scenarios permit 0.5-second gaps for
  offline stress testing. They preserve original timestamps, replay every
  source frame in order, and delay anchor delivery by 0.4, 1, 1.5, 2 or 2.5 source
  seconds. The previous-anchor case also withholds the target's SAM contour.
  These delays are simulated effective anchor ages relative to the displayed
  source, not measurements of model delivery latency; the quality replay does
  not add another one-second display delay. With the evaluator's default
  two-second propagation horizon, the 2.5-second case should report expiration.
- Polygon masks are rasterized at the original JPEG resolution. Reported
  measures include mask IoU, reference recall, prediction precision, centroid
  displacement, symmetric 8-neighbour boundary chamfer mean and p95, and
  reference recall in three equal vertical sections of the person's bounds.
- Missing predictions contribute zero IoU/recall. Boundary error and precision
  are undefined for a missing person and are averaged only over visible
  predictions. Availability and no-anchor counts must be considered together
  with geometric accuracy. `recall90` and `iou80` counts are descriptive bands,
  not a claim that the user's accuracy requirement has been accepted.
  The summary's boundary p95 is the 95th percentile of per-person boundary-p95
  values, not a pooled percentile over every boundary pixel in the corpus.
- A person marked `complete:false` or without closed polygons is not counted as
  a complete prediction. Partial open lines are counted separately and contribute
  zero full-mask IoU/recall. Line precision and reference-boundary recall at 3 and
  6 original-image pixels measure only segments actually drawn; no closing chord
  is invented. Read both together: a tiny correct segment can have high precision
  while leaving almost the entire person unmarked. These boundaries remain SAM
  references, including their crop edges, rather than human-labelled truth.
- Frozen source-anchor IoU is also reported as a comparator: a motion candidate
  should improve on simply retaining old geometry. Unexpected identities and
  contours for the wrong display sequence are checked separately.
- Native-ID matching remains the primary score. An additional global one-to-one
  geometric IoU match separates geometric overlap from native-ID agreement; it
  must not be used to hide an association error or to claim human identity truth.
- The two PNGs show the largest boundary error among visible propagated results
  in each real recording segment. Green is the corresponding SAM reference,
  magenta is the propagated result, in identical source coordinates.

The reference is another recorded SAM contour, **not human-labelled truth**.
The default near-camera fixtures come from one user run. A manifest can also
provide a separately recorded continuous 10 fps scene. Neither is a diverse
validation corpus, and success on 3 fps JPEGs or a 10 fps recording does not
demonstrate accurate person propagation at 15 fps.

`node tools/summarize_contour_quality.cjs <report.json> <private-summary.json>`
aggregates an existing report by reference bounding-box width in the evaluated
grayscale image: below 4, 4–8, 8–12, and at least 12 pixels. It excludes no-anchor
warmup and the exact-source control, keeps missing predictions in the denominator,
and does not rerun Chrome. A wide box can still contain thin or untextured regions.

## Synthetic scheduling check

The optional 10–120 second test generates a moving textured silhouette into a
30 fps canvas capture stream, plays it through a real HTML video element and
uses the production `BrowserController`, video-frame callbacks and Worker.
It serves delayed synthetic anchors and locally acknowledges analysis uploads;
analysis admission remains 3 fps, independent of the display target.
An encoded corner marker is read from each captured bitmap to bind its anchor
to the exact generated pose, avoiding an off-by-one video callback assumption.

`--synthetic-anchor-hz` controls nominal synthetic anchor supply (default 2 Hz,
preserving the original half-second schedule). `--anchor-policy` selects
`past_only` (the original default) or `nearest_arrived` for the actual Controller.
`--synthetic-anchor-trace <SAM-report.json>` instead uses the report's ordered
`observations`: source intervals and measured `source_to_result_seconds`.
Each due source selects the first distinct captured bitmap at or after that time,
whose corner marker supplies its exact generated polygon. Availability is delayed
from that frame's capture callback by the recorded source age. Actual Controller
polling adds metadata waiting; this mode does not add another assumed 150 ms.
The private result records the trace hash and each requested/selected source time,
selection delay, marker, source age, and capture/availability clocks. Scheduled
anchor count and distinct anchors actually returned by metadata polling are
separate: the final scheduled result may arrive after the test ends.
This replays measured timing on synthetic
geometry, not the original camera images or a live model call. Trace-mode steady
statistics use source seconds 2 through duration minus 2; fixed-delay runs retain
their original warmup of max(2, anchor delay plus 1) seconds.

The report includes distinct presentation rate, duplicate presentations,
callback-to-canvas age, presentation gaps, Worker backlog and message latency,
missing-contour reasons, and late results. Its source is artificial and metadata
delivery is controlled: it tests browser scheduling/compute, not real-person
quality, sensor latency, network/model latency, or camera capture reliability.
The steady rate excludes warmup and the last two source seconds; the full-run rate
and raw presentation records remain available. A result below 15 fps must not
be rounded up into a passed 15 fps acceptance claim.

The formal `compute_ms` values in Worker acknowledgements measure actual
`appendFrame` and `select` execution. The realtime report prefers these values,
excluding initialization and queue waiting. Parent-side message latency includes
queue waiting. The local server also wraps `Worker.onmessage` with a timer;
quality reports label this separately as `handler_wall`, and old implementations
without formal timing use it as a documented fallback. Synchronous handler
timing cannot measure asynchronous initialization or execution of its queued
messages. The extra timing message is ignored by the production controller.
Private `worker_operation_records` correlate formal compute time with source
sequence, anchor sequence and anchor age. `worker_backlog_samples` retain source
time and pending-frame count, separating startup backlog from steady load;
these arrays are omitted from terminal summaries.
Deadline-expired drawing requests form a separate `present_expired` compute
category. Their zero compute time is never included in the valid `present`
mean. The report includes total and steady expired counts, Worker-result coverage,
and actual displayed coverage separately. Skipping an overdue drawing request
does not count as a successful contour frame, even if the Controller reports
fewer late results afterward. `worker_presentation_records` retains each result's
source timestamp and expiration flag for review.

`--capture-failures true` additionally retains the first four distinct rejection
reasons, their source anchors and received grayscale frame chains in the private
result JSON. This supports deterministic algorithm diagnosis. Capturing arrays
adds one-time work, so use the ordinary run for performance acceptance. Pixel
arrays and raw presentation records are omitted from terminal summaries.
For the dense adapter this also observes OpenCV remap/threshold outputs and the
unchanged warp APIs, preserving Embind overload dispatch metadata. It captures
up to two separated boundary/component-appearance failures with source/target
gray pixels, rejected mask, base mask, map, and diagnostics. Synthetic frame
markers bind these cases to exact generated geometry. This diagnostic wrapper
copies arrays and must not be used for performance acceptance. Render two cases
with `node tools/render_camera_contour_failure.cjs <private-report> <private-stem>`.
