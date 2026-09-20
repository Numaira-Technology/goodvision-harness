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

The attempted six-window real 30B reproduction was rejected by automatic approval
review pending explicit authorization of the video transfer to OpenRouter. No
new real-model accuracy or localization result is claimed by this change.

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

## Verification and next step

148 local regression tests passed, followed by 38 focused route/process tests.
They cover bad/missing/duplicate boxes, explicit coordinate conversion, missing
participants, preserved positive/negative observations, no fabricated confirmation,
legacy episode semantics, the recorded reply and public-route model propagation.
Tests use local fixtures/stub transports, not new API calls.

The testing task owns the compatible UI presentation and recording environment.
Once the outstanding concrete transfer authorization is resolved, validate the
new wire format on real 30B replies and inspect actual box alignment as well as
event coverage. A parser passing a reply is not a claim of model accuracy.
