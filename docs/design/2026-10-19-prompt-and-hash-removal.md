# Prompt size and branch hash removal

Status: local prompt/ledger, runtime, artifact, worker and evaluation-tool migrations have completed bounded static reviews. Branch-wide removal is not complete. No push, deployment or new model calls. See [integration status](2026-10-19-hash-migration-integration-status.md) for repaired findings and remaining boundaries.

## Human decisions

The user requested timestamps instead of UUIDs in prompts, an audit of unnecessary input/output formatting, and removal of hash computation, propagation and checks across the team branch. Repeated requests for clarification must not narrow that last instruction back to prompt-only removal. Preserve historical observations, answers and paid-request records. Do not resume until the repair is complete and pushed.

## Failure and evidence

The final-answer route serialized the audit artifact as model evidence. Compression then preserved bookkeeping strings that the model did not need. That boundary was wrong. A dictionary made duplicate strings cheaper without removing their unnecessary definitions.

For saved question 0645, the capture contains 12,282 independent event records. JSON-serialized values of raw_reference occupy 2,235,324 characters; evidence_id occupies 739,800; raw_evidence_id 275,158; raw_observation_ids 338,608; window_id 208,794. These are character counts from the original capture, not token counts or additive savings after compression. The old prepared input still uses 420,299 tokens for a 260,096-token input budget.

The last completed short-answer run produced 13 replies with finish_reason=stop and 26–41 completion tokens. Its other 59 candidates were not dispatched because their prepared inputs exceeded capacity. That run does not measure the new projection.

## Model-visible field contract

| Input class | Treatment | Information retained |
| --- | --- | --- |
| UUIDs, request/message/session IDs | Remove from prompts | Source-video times; separate records at equal times |
| Hashes, local paths, weight URLs, deployment and billing metadata | Keep out of prompts | Method name, detector parameters and preprocessing settings needed to interpret observations |
| Original question and options | Preserve text and values | Public context, requested answer type and question semantics |
| QA/video/source bookkeeping identifiers | Keep local | Match the one response to the caller's question |
| Person observations | Retain records and coordinates | Source time, observed/predicted state, exact numeric precision |
| Subject references | Use short source-local aliases | Equal original subject IDs remain equal; different IDs remain different |
| Raw events and published judgments | Retain both, label their kind | Label, time, lane, tri-state detection, association and confirmation states |
| Evidence references | Render source-time intervals and lane | Reference order, multiplicity and unresolved references; timestamps do not prove unique identity |
| Native episodes | Retain records | Start/end, end reason, confirmation, subjects and temporal support |
| Coverage | Render counts, skipped times and window plans | Unknown time stays unknown; skips do not become negative observations |
| Provider errors | Keep local | Only affected lane and skipped time appear in the prompt |
| Repeated field names and values | Compare existing table/column encodings using token counts | Retain observations, explicit negatives, unknowns, source order and precision |
| Duplicate subject attributes | Omit only values exactly repeated at the subject's top level | Keep conflicting values and absent/null distinctions |
| Final response | `{"prediction":"answer"}` | Original option, number, timestamp, interval or requested short text |

The local archive remains the full record. The model projection is intentionally not a reversible copy of audit metadata. Lossless codecs apply to that projection, not to the discarded bookkeeping. Do not claim archival losslessness for the projection itself.

## Current bounded implementation

- `evaluation/summary_view.py` builds the timestamp observation view and records the alias map, counts and removed-field inventory locally.
- `evaluation/summary.py` applies the view only after looking up prior paid requests. It retains readers for old response contracts and maps the new single prediction to the existing scorer.
- `evaluation/summary_context.py` can optimize an input even when it already fits. Capacity alone is not an efficiency target.
- Existing capture and request history remain unchanged. No model, scoring, output-token budget or observation rerun changes belong to this patch.

These changes do not complete branch-wide hash removal. The ledger uses explicit operations and preserves historical dispatch state. Artifact readers use explicit source/configuration/question records. Current workers and evaluation tools have been migrated in groups. Independent recovery launchers are implemented locally; a fresh whole-change static review found two integration issues that received source fixes, without execution. Security-sensitive protocols and test-only legacy contracts remain separate unfinished boundaries; frozen historical copies are preserved.

## Branch-wide migration design

The source scan found hash-related logic in 32 core Python modules before expanding to workers, benchmark tools, customer-update tools and frozen experiment copies. A blind textual replacement would leave producers and consumers using different identities.

1. Replace evaluation artifact hashes with explicit run/source/question identities and saved file references. Update capture, recording, cohort, answer, scoring, batch, queue and recovery interfaces together. Keep schema checks, EOF/drain state, resource ownership and file readability checks; they are not hash checks.
2. Replace content-hash request keys with persisted logical operation keys. A final answer uses the benchmark revision, observation-run/source identity and QA ID. A window request adds lane, source-time range and its persisted dispatch sequence. Retries reuse the same logical operation, with a separate attempt count capped at three. Preserve successful and unknown outcomes across prompt changes.
3. Read legacy paid-request rows without recalculating their hashes. Import their explicit QA/source/lane context and dispatch states into a new index; retain the original files. If an old row cannot be mapped, mark that operation unresolved rather than resending it. Do not use a renamed hash or put complete request bodies into log IDs.
4. Remove hashing used for configuration caching, display colors and generated scene IDs. Use explicit immutable configuration values in memory, source-local counters and persisted identity maps. Do not serialize credentials from a cache key into logs or prompts.
5. Remove application-generated media, frame, weight, profile, provenance and error-body hashes and their checks. Update worker request/response contracts with their callers. Preserve file/version names, dimensions, parameters, source timestamps and exact resource handles.
6. Treat customer packaging and webhook signing as separate external-contract changes. They need the decision below before implementation. Do not change deployed production services or global account settings.

Git's own commit/object identifiers, third-party library hash tables, existing immutable experiment snapshots and historical result files are not application hash gates. Deleting historical source copies would destroy the baseline needed to explain previous results. Record any retained legacy fields as legacy data, not evidence that active hash removal is complete.

## External-contract decision

The user has now approved retiring signed webhook delivery and the customer
automatic updater, including its package builder. Their entry points reject use
before side effects. The descriptions below explain the retired protocols;
they are no longer unanswered choices. Official checkpoint loading and
maintainer-operated deployment are separate retained protocols, not new gates
for evaluation. The branch is not claimed free of all hash implementations.

`delivery.py` computes HMAC-SHA256 for `X-Harness-Signature-256`. Removing it means configured signed webhook delivery cannot continue under its current protocol. Choose a replacement authenticated transport or retire that signing feature and reject configurations that require it. Do not silently send unsigned messages where a signature was requested.

Customer updater/packager tools exchange content-checksum manifests. Removing both generation and verification changes compatibility with existing customer bundles. Choose whether to retire that update protocol or replace it with explicit release/file inventory checks while acknowledging the loss of content-integrity verification. File-size checks do not provide equivalent tamper detection.

`relay_auth.py` and `vlm_server.py` use `hmac.compare_digest` to compare supplied secrets. That call does not compute a hash; it is not a reason to remove authentication. The initial broad search matched its name, and source inspection corrected that classification.

## Evidence and acceptance

No tests, lint, syntax checks, builds or test-file edits have authorization for this change. Static review is not a passing test report. During the next authorized input-preparation stage, record per-question old/new bytes and tokens, selected encoding and actual capacity. Do not send model requests as a probe.

Before resuming, use the full 182-question inventory. The prior 72-question candidate union omitted eight duration-deferred questions and three older capacity-blocked questions. The completed 13 questions, historical successful/invalid requests and unknown requests remain excluded from paid redispatch.

The implementation must report prompt cleanup, local hash migration, worker compatibility and external-contract status separately. A pushed commit does not establish deployment, capacity coverage, answer correctness or reduced wall-clock latency.
