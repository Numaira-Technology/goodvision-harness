# Short final answers and lossless input preparation

The final model receives saved observations, not video or private ground truth.
Its output is the original question's option, number, timestamp, interval, or short text:

```json
{"predictions":[{"qa_id":"original-question-id","prediction":"original-answer-format"}]}
```

An empty prediction means insufficient evidence. The adapter records this as
`insufficient_evidence` and otherwise passes the unmodified prediction to the
existing scoring path. Audit provenance remains local; the model is not asked to
generate references, repeat observations, or explain its answer. Output budgets
are supplied by the frozen run configuration; this repair does not raise them.
`finish_reason=length` is recorded as `output_truncated`, without salvaging a
partial answer or sending another request.

The short envelope reuses the historical benchmark protocol, not the entire
historical observation method. The unified evidence stays question-independent.
Historical SAM-only methods, grounded-SAM Harness summaries, and the current
native-streaming method remain distinct. This incremental repair is not a claim
that the overall Harness matches the pre-task baseline
`6c28dddd26bed7a7be50762a4c7dd6b796b2dbe3`.

Input preparation extends the existing lossless window tables. Per-column
constants, exact dictionaries, stored defaults with explicit overrides, and
shared string affixes remove repetition. Decimal sequences are used only when
every original string matches exactly. Every record, field, type, ordering,
timestamp, negative, unknown, and identity relation is retained. The existing
round-trip guard remains part of preparation. Historical record truncation and
other lossy fallbacks are not imported. The local Qwen token budget and byte
limit are both applied; a smaller byte representation is not assumed to fit the
model context.

`execute_summary(..., prepare_only=True)` returns the complete prepared request,
input byte count, token preparation metadata, and readiness reason without
dispatching or creating a terminal answer. It is an evaluation-preparation
phase, not a model test. Independent recovery first inventories pending inputs
and only then dispatches those that fit and have never been sent.

The cumulative ledger is consulted before changing a prompt or encoding.
Previously dispatched requests retain their original messages and parameters;
successes, invalid responses, and unknown receipts must not be bypassed under a
new identity. Recovery uses a separate output directory and preserves historical
answers. Source EOF, drained queues, usable evidence, local-gap disclosure, and
duration-review policy are unchanged.

No independent tests, test edits, lint, compilation, builds, or health probes
were authorized for this repair. Read-only review and authorized evaluation
preparation must be reported separately from runtime success.
