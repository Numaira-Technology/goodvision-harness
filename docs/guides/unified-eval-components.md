# Unified evaluation component scope

This integration publishes existing evaluation code and interfaces. It does not
run an evaluation or publish experiment results. The source is the local work on
`codex/harness-unified-eval-20261013`, based on
`6c28dddd26bed7a7be50762a4c7dd6b796b2dbe3`. The team integration base is
`codex/research-testing` at `ea3bdd71e4e03d8b918bfbfa2d68e538ba15a25c`.
That commit is a merge base, not the original pre-experiment Harness baseline.
No tests or evaluations have run on this integrated version; historical results
do not establish its correctness.

## Included components

| Area | Scope |
| --- | --- |
| Event profiles | Explicit fixed definitions, allowed labels, and scene-scoped labels. Production defaults retain their existing labels. The evaluation profile has 51 frozen labels plus fight, with faces disabled. |
| Three-lane observation | Person tracking, independent events, and independent compact grounding. The VLM lanes receive no questions, ground truth, or detector boxes; association follows raw report publication. |
| Input and capture | A single-source entry point reads frozen configuration, JPEGs, source timestamps, and hashes. It uses offline windows, backpressure, and terminal-frame handling through native components, not the full production HTTP service. |
| Evidence export | Raw observations, associations, source identity, input hashes, EOF, and drain states. Failed or incomplete captures remain distinct from complete evidence. |
| Request safeguards | Full-input hash caching, request/receipt accounting, unknown-outcome blocking, at most three sends per input, and 429 backoff. This integration includes ledger/cache code, not ledger/cache data. |
| Answers | Deterministic parsing for 14 question forms, saved model requests, and a separate final-answer executor that reads those observations after GPU release. |
| Batch recovery | Hash-pinned source plans, exact terminal-result reuse, original hardware provenance, and source cleanup. Shared faults stop the batch without failing unattempted sources. |
| Scoring | Local rule/model scoring through the unchanged, hash-pinned PR63 scorer. Terminal failures remain in the denominator; pending work prevents a completed full-cohort score. |
| Detector adapters | YOLO-L registration, YOLO parameter validation, and finite-value checks and clipping for YOLO/RF person boxes. RF weights and the proposed RF/YOLO equal-parameter comparison remain unchanged. |
| Loading and calls | An explicit tracking-response object limit with the existing default of 64, plus native-session and count-protocol adapters. This integration excludes SAM worker algorithm changes. |

Use `evaluation.capture_source` for configured single-source model execution.
`evaluation.answer_artifact` and `evaluation.cohort` generate answers from saved
evidence. Imports do not allocate GPUs or start evaluations. Execution requires
an explicit configuration.

Use `evaluation.queue` for batch capture and postprocessing, then
`evaluation.scoring` for local grading. See [the workflow guide](unified-eval-workflow.md)
for configuration, recovery, and release requirements. The resource adapter
operates the model child on an existing controller-owned GPU; it does not rent
instances or replace the outer controller's release responsibility.

`profile.load_profile` requires a frozen definitions file matching
`DEFINITIONS_SHA256` and the expected label order. The team's existing
`experiments/ashley-detector182-20261012/frozen_runtime/event-definitions.json`
matches that hash; this integration does not modify it. The operator supplies
media, credentials, configuration, and output paths. No new experiment data
files accompany this integration.

The model ledger uses POSIX file locks, so capture requires Linux/macOS. This
integration adds no Windows capture support and leaves the team's existing
Windows features unchanged.

## Compatibility with team workflows

Keep the existing native benchmark, VLM research, gateway, and deployment entry
points. `benchmark/run_native.py` and this three-lane capture path represent
different methods; the former does not implement this rule/same-evidence summary
ablation. Report historical native SAM-Ashley and Harness-plus-summary results
under separate names.

The team retains `research-v1`, paired scheduling, lean-v1 grounding, concurrency
up to eight, and historical-result retention settings. Fixed evaluation
definitions require the `main-baseline` dispatch path and `legacy` grounding.
Unsupported combinations raise an error rather than applying the six-label
research policy. Fixed-profile prompts use their supplied definitions and
reports carry a separate `fixed-evaluation-events-v1` fingerprint.

Without an explicit evaluation profile, events retain the team's 160-token cap
and timeout of at most five seconds. Explicit profiles use the caller's timeout
and token settings, with an events default of 1600. This does not change any
historical frozen run's budget.

The integration supplies the native 1600 default when offline observation omits
top-level `max_tokens`, avoiding a `KeyError` during override lookup. Window,
lane, and explicit top-level budget precedence stays unchanged. Historical
frozen code and requests remain untouched.

## Lossless final-answer input preparation

The final-answer executor reuses the offline benchmark's shared-value and
record-schema codecs from commits `266e1b6` and `90c2f9f`. The original
pre-experiment Harness baseline remains `6c28dddd26bed7a7be50762a4c7dd6b796b2dbe3`;
these codecs are later evaluation work, now connected to the unified answer path.

After reading the saved complete observations, `evaluation.summary` prepares
oversized text requests with `evaluation.summary_input`. It first shares repeated
values, then shared record keys and strings if needed. Each selected representation
retains every observation, timestamp, numeric value and original text. The public
question stays outside the encoding. The codec includes instructions for expanding
its notation, and the final model receives that encoded evidence. The saved
artifact and deterministic parser continue to use the original evidence.

Requests within the configured byte limit keep their existing messages. For an
oversized request, the executor selects a smaller lossless representation and
records its encoding, original/prepared byte counts and candidates under
`request.json.input_preparation`. Reserved marker collisions retain the preceding
representation. If all available representations exceed the byte limit, the
executor preserves a `not_called` outcome with the prepared size. It does not drop
windows, remove SAM details, paraphrase observations or raise the limit. This
byte-budget preparation does not measure tokens or establish a provider context
limit; the old tokenizer and lossy budget fallbacks are outside this integration.

Existing final-answer output directories retain their saved representation and
terminal outcomes. Requests already present in the cumulative ledger keep their
original messages, preserving successful-response reuse and unknown-outcome
blocks. Answering a previously uncalled oversized artifact uses an independent
output location with the same evidence and cumulative ledger. The actual selected
messages, rather than the original oversized messages, are saved and dispatched.

This change affects final-answer text preparation only. It retains the one answer
entry, VLM switch, capture scheduling, native frame skips, detector weights and
parameters, resource release ordering, prompts for observation, and PR63 scoring.
Running frozen checkouts remain unchanged. The prior benchmark's grounded-SAM
budget adapter and its observation-dropping fallbacks are not imported.

Tests were not written or run for this integration because the user did not
authorize them. Review is limited to source and Git differences; formal recovery
results must be recorded separately when the new frozen code is used.

## Token-aware final-answer recovery (2026-10-19)

Independently frozen recoveries can opt into `final_model.context_budget` with
`model`, `context_tokens`, `margin_tokens`, `tokenizer_path`, and
`tokenizer_source`. The supported route is the existing Qwen3-VL-8B-Instruct
two-message text answer, with only temperature and max_tokens parameters. A
local official tokenizer counts the text chat template, including the assistant
prefix. Output tokens and an explicit safety margin are reserved. This is a
local input count, not a claim about the provider's actual usage. The existing
byte limit remains in force. Install the evaluation extra and supply the frozen
tokenizer file during evaluation preparation; there is no runtime download.

For never-sent inputs that exceed either budget, `summary_encoding` adds a
lossless window-table candidate. Consecutive records share fixed fields without
merging rows; repeated strings and UUID fragments are defined once. The decoder
preserves row order, nulls, empty collections, numeric values, and all original
text. The new table and its legacy dictionary encoding are compared by actual
local token count and byte size. Unfittable inputs remain explicitly not_called;
no lossy filtering, truncation, larger byte cap, or multi-call summarization is
introduced. The question and skipped-time disclosure are outside the codec.

The cumulative ledger now also remembers dispatched question/evidence pairs.
Changing an encoding never creates permission to replay a paid answer: the
original dispatched messages are reused, with success caching and unknown
blocks unchanged. Historical request files and results remain immutable.

Evaluation clients opt into bounded HTTP diagnostics: at most 4096 response
bytes are decoded and credential-redacted into the private ledger error field.
The public exception remains generic. This applies to both synchronous
transports; it cannot recover bodies discarded by historical clients, and an
HTTP400 is not reclassified as a free or safe-to-retry request.

This extends the earlier final-answer input adapter, not the observation model,
weights, media, detector parameters, scheduling, or PR63 scoring. Cumulative
changes still compare to pre-task Harness `6c28dddd26bed7a7be50762a4c7dd6b796b2dbe3`,
not to a recovery copy or the current team branch. No tests were written or run
for this change under the current user restrictions. Static review and actual
authorized evaluation outcomes must be reported separately.

## Excluded work

- Unapproved SAM `mask_area.py` and `streaming_state.py` candidates.
- The unselected latency metric implementation. Literature-based metric
  selection remains a separate research task.
- The old experiment-specific controllers and their automatic provisioning,
  report generation, and duration/latency calculations. The integrated batch
  and recovery modules adapt the later frozen recovery logic, without those
  controller features.
- The old count-audit script and historical reports. The integrated final-answer
  executor and scoring adapter do not regenerate historical results.
- Local media, weights, databases, experiment configurations, results, request
  caches, cost ledgers, and secrets.

The team's existing benchmark data and frozen references remain in place, as do
the local historical worktree, successful responses, and run evidence.

The original cohort still contains 182 questions from 157 video sources. This
integration does not complete that evaluation, the RF equal-parameter attempt,
or metric selection. Future runs must record fresh source, input, and
configuration fingerprints. Do not rewrite historical answers or reuse a
historical run identity for this version.

Tests were not run because the user did not authorize them. Test and CI
configuration remain unchanged. Publication commits use `[skip ci]` to avoid
triggering repository tests.
