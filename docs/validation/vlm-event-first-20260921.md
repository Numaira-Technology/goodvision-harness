# Preserve event observations when localization is unavailable

This update follows the user's event-first requirement. It is implemented in the
primary worktree on `codex/entity-event-engine`, after the live-service isolation
change `67d92e0`.

## Evidence and scope

The fixed-version `nonsam-v1` recording collected 327/342 reports on its two
cameras: 256/261 had `invalid_response`, and the observed validation reason was
`invalid_subject_bbox`. Those logs contain the rejected reports, not the original
30B assistant replies, so they do not establish which exact coordinate values
caused every failure.

An earlier real 8B replay saved an original reply at source time 53.6 seconds with
boxes such as `[305,219,546,568]`, alongside four valid negative event judgments.
The old normalized-0..1 parser rejected that entire reply. The response is retained
as a regression fixture with explicit provenance; it is not represented as a new
30B test. The [Qwen3-VL technical report, section 3.2.4](https://arxiv.org/html/2511.21631v1)
specifies normalized coordinates in the 0..1000 range.

The initial six-window reproduction was blocked pending destination-specific
authorization. On 2026-09-21 the user explicitly authorized OpenRouter in this
task. The six requests then completed using the current main-worktree code;
results and limits are recorded below.

## Protocol and behavior

- Qwen3-VL prompts request `bbox_coordinate_system: normalized_1000`; other models
  request `normalized_1`. A declared 1000-scale box is divided by exactly 1000.
  Older undeclared 0..1 responses remain compatible. An undeclared out-of-range
  box is not guessed to be pixels, percentages or normalized 1000 coordinates.
- Bad, missing, ambiguous or unsupported subject geometry is quarantined locally.
  Reliable event labels, booleans and confidence values remain available. Null
  confidence remains null. No replacement box or person identity is fabricated.
- Missing participant references also leave the event observation intact. Only
  fully grounded participant groups can update person-event confirmation/clear
  votes or be associated with SAM identities. One valid participant cannot stand
  in for an invalid second participant. Unlocalized negatives cannot clear an
  earlier person's confirmed event.
- Invalid JSON, incomplete completions and unreliable event fields still produce
  explicit failures. The change does not coerce strings into event booleans or
  accept unrequested labels or invalid confidence values.

Every reliable judgment now contains:

| Field | Meaning |
| --- | --- |
| `observed_detected` | This observation's model decision: true, false or null |
| `detected` | Existing person-event confirmation/clear semantics |
| `confirmation_state` | confirmed, pending, not_detected or uncertain |
| `localization_state` | available, partial or unavailable |
| `grounding_issues` | Safe reason codes; no model prose, credentials or URLs |

`analysis_complete` measures event-label coverage independently of
`localization_complete`. A report can have complete event judgments with no usable
boxes. Health counters include `unlocalized_reports` and `invalid_bbox_reports`,
so lower request failure counts cannot hide localization degradation.

Public model names retain the existing `numaira-beta-fast` alias. Public route C
maps to persisted route AA and `qwen/qwen3-vl-30b-a3b-instruct:nitro`; the service
passes that model into the independent process and inspector. Public `route_id`
is now frozen per run and included in report model metadata and branch health.

## Verification

148 local regression tests passed, followed by 38 focused route/process tests.
They cover bad/missing/duplicate boxes, explicit coordinate conversion, missing
participants, preserved positive/negative observations, no fabricated confirmation,
legacy episode semantics, the recorded reply and public-route model propagation.
Tests use local fixtures/stub transports, not new API calls.

### Authorized real 30B replay

Code: `71f18c9` (event-first implementation `c9e75c3`). Destination:
`https://openrouter.ai/api/v1/chat/completions`. Requested model:
`qwen/qwen3-vl-30b-a3b-instruct:nitro`. All six replies reported
`qwen/qwen3-vl-30b-a3b-instruct`, provider `Alibaba`. This establishes the actual
request/response model metadata separately from the public display alias.

The same SDR recording was used:
SHA-256 `dc68d2e2d3d9935f0a028799ba3220037a9c9b998a54503cb61b93353b8598fc`.
Each request contained four frames, 0.4 seconds apart, ending at the timestamp
below. All four event sensitivities were STANDARD. Settings matched the saved
test configuration: maximum image side 720, JPEG quality 90; source extraction
first encoded JPEG at quality 80. Requests were sequential with no retries.

| Window end | Positive model decisions | Event judgments | Grounded judgments | Request seconds |
| --- | --- | --- | --- | --- |
| 1.6 s | smoking | 4/4 | 4/4 | 2.750 |
| 9.6 s | fight | 4/4 | 4/4 | 2.422 |
| 13.6 s | fall | 4/4 | 4/4 | 2.844 |
| 24.0 s | none | 4/4 | 4/4 | 2.609 |
| 42.4 s | none | 4/4 | 4/4 | 2.703 |
| 51.2 s | none | 4/4 | 1/4 | 2.454 |

All six replies explicitly declared `normalized_1000`. All completed, with
24/24 event judgments retained and zero `invalid_subject_bbox` failures.
At 51.2 seconds the model returned empty participant lists for smoking, fall and
weapon. All three negative decisions survived with `unlocalized_event`, without
invented participant associations. Event completeness and localization
completeness therefore remain distinct in a real reply, not only in unit tests.
Median request duration was 2.656 seconds (range 2.422–2.844). The API reported a
total cost of USD 0.00213499. These are individual request durations, not display
latency, sustained throughput, or end-to-end confirmed-event latency.

Offline visual review overlaid the declared boxes on the last source frame and
reviewed all four frames in each positive window. The coordinate conversion aligns
the foreground participant at 1.6, 9.6, 13.6 and 51.2 seconds. The 24-second box
misses the participant's head/upper body, so valid geometry does not establish
accurate full-person localization. Small background boxes also need evaluation
under the real association workload. No geometric correction was guessed.

The visible striking gesture and descent to the floor are consistent with the
fight/fall decisions under the configured policy. At 1.6 seconds the visible
mouth-level object/action remains ambiguous: a smoking-positive response is not
proof of tobacco use or correct STANDARD-policy object identification. The three
negative windows provide useful fall false-positive spot checks. These six chosen
windows are not a general accuracy/recall benchmark, and do not test temporal
confirmation or the dual-camera playback target.

Original response choices, including returned Markdown fences and empty
participant references, are checked in at
`tests/fixtures/vlm/qwen3vl_30b_openrouter_20260921.json`; no images or credentials
are included. Six recorded-response regressions were added; all 30 tests in
`tests/test_vlm_event_first.py` passed. These regressions preserve protocol
behavior and model decisions; they do not assert that model decisions are truth.
Local raw replies, extracted frames, box overlays and statistics are retained in
`.goodvision/diagnostics/vlm-independent-20260921/coordinate-after/`.

The testing task owns the compatible UI and full dual-camera recording. Its
separate automatic approval review still requires a direct confirmation there;
this task did not run that blocked workload. No new production-code change was
needed for the six-window verification. Full playback and long-run acceptance
remain with the testing task.
