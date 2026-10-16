# Three-lane evaluation workflow

For the native timed-source answer entry and its VLM switch, see
[Streaming answer entry](streaming-answer-entry.md). The commands below describe
the retained historical offline workflow and remain available for its frozen
recovery records.

Capture each video once per detector method, then answer questions from the
saved observations. The Harness system has person tracking, VLM and face lanes;
face stays disabled for this evaluation. Independent events and compact grounding
are the two tasks inside the VLM lane. This workflow uses
native components offline with backpressure; it does not run the production
HTTP service or measure online response quality.

The integration reuses existing capture, batch recovery, final-answer, and PR63
scoring code. It does not establish that the original 182-question run finished.
Keep historical native SAM-Ashley, question-conditioned window summaries, and
this Harness-plus-final-answer method separate.

## Phases

Run these commands in the configured project environment. Substitute paths and
SHA-256 values from the frozen experiment record. They are execution commands,
not setup checks; capture and summary can incur costs.

```sh
python -m goodvision_harness.evaluation.queue capture --config /run/workflow.json --config-sha256 WORKFLOW_SHA256
python -m goodvision_harness.evaluation.queue answers --config /run/workflow.json --config-sha256 WORKFLOW_SHA256 --release /run/gpu-release.json --release-sha256 RELEASE_SHA256
python -m goodvision_harness.evaluation.queue summary --config /run/workflow.json --config-sha256 WORKFLOW_SHA256 --release /run/gpu-release.json --release-sha256 RELEASE_SHA256
```

The outer controller must release the exact owned GPU between capture and
postprocessing, including after shared failures. It must record provider
inventory and local container cleanup. These entry points do not allocate or
release cloud instances. A failed Docker connection does not remove the
controller's responsibility to release its remote instance.

`answers` makes no model calls. `summary` sends the public question and the same
saved observations to the configured final model. It rejects changed artifacts
and skips captures with any failed lane, missing EOF, missing drain state, or
`observation_manifest.complete=false`. It preserves semantic empty answers and
invalid model replies without retrying them for a better score.

## Frozen configuration

The workflow schema is `unified-evaluation-workflow-v1`. Required fields:

| Field | Meaning |
| --- | --- |
| `experiment_label`, `frozen_runtime_sha256` | Exact deployment and source fingerprint. |
| `public_questions`, `public_questions_sha256` | Public JSONL question file; no ground truth. |
| `output_directory` | This round's output directory. |
| `owner_lock_path` | One shared lock path across phases and recovery rounds. Its parent must exist. |
| `resources` | `identity_path`, `identity_sha256`, `instance_id`, and the exact owned `reset_endpoint`. |
| `arms` | Ordered entries containing a filesystem-safe `model` name, `plan`, and `plan_sha256`. |
| `model_ledger` | `seed_manifest` and `seed_manifest_sha256` for the cumulative request/response history. |
| `summary` | Explicit `settings`, `model`, `parameters`, and positive `max_input_bytes`. |

Paths resolve relative to the workflow file unless absolute. Model settings use
`ModelAPISettings` fields: HTTPS `endpoint`, `provider`, credential environment
variable name `api_key_env`, `timeout_seconds`, `max_retries=0`, and an empty
`inspection_endpoint`. Keep secrets outside configuration. Do not include
`model`, `messages`, or `stream` in `summary.parameters`. Setting
`defer_postprocessing=true` blocks final-model execution.

Each arm's plan uses `unified-capture-batch-v1`, with `method` and a `sources`
list. Each source contains `config` and `config_sha256`, with an optional `reuse`
descriptor. Source configurations retain the existing `unified-source-run-v1`
contract, frame hashes, timestamps, model settings, and independent observation
policy. `input_skips`, if supplied, must contain unique terminal `skipped`
sources disjoint from scheduled sources. Do not add unattempted sources there.

The resource receipt requires `owned=true`, one standard RTX 4090 with 24 GB,
`instance_id`, `experiment_label`, `source_manifest_sha256`, and
`sam_internal_patch=false`. The source fingerprint must match the frozen runtime.
The adapter only resets a model child after checking the expected label/model/
session and receiving confirmation that its process exited.

The postprocessing release receipt retains the controller's existing fields:

- `confirmed_released`, `local_confirmed`, and `remote_confirmed` are true.
- `gpu_label` matches the owner; `known_instance_ids` and
  `release_confirmed_instance_ids` contain the exact instance.
- `remaining` and `cleanup_errors` are empty; `network_containers` is empty.
- `creation_outcome_unresolved` is false; recorded containers are not running.

The workflow verifies the receipt's supplied file hash and fields. It does not
contact the provider or infer live release from a stale receipt. The outer
controller must produce fresh release evidence for this deployment.

## Resume without replaying completed work

A completed arm keeps its original files. Rerunning the same workflow does not
repeat that arm. A partially captured arm requires a new frozen recovery plan
and output directory; the queue refuses to replay its stateful frames.

Use `capture_recovery.prepare_reuse` with the old source directory, original
configuration, terminal outcome, original owner receipt, and their hashes.
Supply the confirmed source-recovery receipt for skipped sources. Put the
returned descriptor into the new plan's `reuse` field before freezing it.
The importer checks the original configuration hash in the source manifest,
runtime identity, hardware identity, terminal state, and source file hashes.
It copies terminal evidence with its original session and GPU provenance.

Only `stream_session_id` and `model_ledger` may differ between an original
source configuration and its reuse target. Other differences, including model
settings, runtime, source bytes, and window budgets, stop reuse. Do not bypass
this rule to attach old results to a new experiment method. An incomplete
terminal stays incomplete, retains failures and cleanup evidence, and receives
no paid final-model answer.

Carry cumulative ledger files into a hash-pinned
`unified-model-ledger-seed-v1` manifest for a new round. Successful HTTP responses,
including malformed answers, remain cached. Unknown outcomes and requests
without receipts stay blocked; a changed budget is not permission to repay
them. The existing ledger limits each full input to three sends and accounts
for each actual dispatch. Reusing a response does not create a new paid call.

An interrupted final-answer directory also requires independent recovery with
the cumulative ledger. Do not delete its request or overwrite an outcome.
Completed final answers with unchanged identities return their saved outcome.
Shared connection, I/O, ownership, or hash faults stop processing; they do not
turn the remaining question set into terminal failures.

## Local scoring

Keep private ground truth outside model and capture containers. Score the rule
and final-model answers separately, using the same question and truth hashes:

```sh
python -m goodvision_harness.evaluation.scoring --questions /run/public/questions.jsonl --questions-sha256 QUESTIONS_SHA256 --truth /private/ground_truth.jsonl --truth-sha256 TRUTH_SHA256 --cohort /run/output/yolo-answers/completion.json --cohort-sha256 COHORT_SHA256 --mode rule --output /run/reports/rule.json
python -m goodvision_harness.evaluation.scoring --questions /run/public/questions.jsonl --questions-sha256 QUESTIONS_SHA256 --truth /private/ground_truth.jsonl --truth-sha256 TRUTH_SHA256 --cohort /run/output/yolo-answers/completion.json --cohort-sha256 COHORT_SHA256 --mode model --summary-root /run/output/yolo-summary --output /run/reports/model.json
```

The adapter loads the repository's existing PR63 scorer only if its SHA-256 is
`956982bd59015fbd8591c9e8e5e5c58397189ab4d9e908428668094a84c53d43`.
It passes public clip metadata through the scorer's existing interface and
does not change tolerances or label matching. A relocated identical scorer can
be supplied with `--scorer`.

Results retain question IDs, original questions/options, ground truth, original
answers, source paths, and file hashes. Breakdowns cover capability, question
form, supplied source group, and event ground-truth label. Missing source groups
remain `unspecified`; the scorer does not guess their origin.

Terminal non-answer states contribute zero quality. Raw missing answers stay
null, and full-cohort MAE stays unavailable if numeric answers are missing.
Counting strict equality uses the full applicable denominator. Pending or shared
interrupted work keeps the full-cohort score unavailable. The combined PR63
score is not an accuracy percentage. Completion markers mean processing ended,
not that all answers were correct or successful.

Compare methods only on explicitly matched question IDs and disclose full-cohort
results beside paired subsets. The scorer does not generate historical baseline
comparisons, invent missing answers, or calculate an unselected latency metric.

No tests or model calls were run while integrating these entry points. Historical
run results do not validate this version. The original questions, answers,
frozen runs, caches, and ledgers remain unchanged.
