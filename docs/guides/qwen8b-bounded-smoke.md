# Manual Qwen import diagnostic: report fields and limits

`tools.diag_qwen8b_vllm_import` remains a **manual-lifecycle diagnostic**, not a
bounded tool. It restores the pinned environment and checks historical/deep
imports and CUDA device visibility. It does not prefetch weights, start serving,
run official evaluation, or establish model/kernel inference readiness.

## Ownership checkpoint

The first report includes the run's exact unique `label` and `ok: false`.
Immediately before the unchanged shared create helper is invoked, the diagnostic
writes a report with `phase: creating` and a `create_intent` object containing
that exact label and `state: prepared`. A failed checkpoint prevents create
dispatch. The local `creation_attempted` flag is set only after the write returns.

Prepared intent means the call was planned, not that it was sent or that a rental
exists. The existing successful-create path records `instance_id`; the existing
exception path may reconcile an uncertain creation by its exact label. Neither a
missing ID nor a single empty list proves that no rental exists or that billing
has stopped. Shared offer deduplication, the three-machine maximum, explicit
unavailability fallback and stop-on-ambiguous-create behavior are unchanged.

The report remains `outputs/qwen8b-r2-vllm/diag-import.json`, with the existing
best-effort artifact mirror. Saving uses the existing synchronous writer. This
is **not** an atomic, fsynced, tamper-resistant or power-loss-safe ownership
journal. Later truncate-in-place writes, another run using the same path, or
external modification can remove earlier evidence. The checkpoint provides
ordinary pre-create visibility, not crash-proof recovery or bounded cleanup.

## Quote reporting

- `dph_cheapest_eligible` is the minimum eligible hourly offer quote, available
  after offer selection. A cheaper offer may be refused before another succeeds.
- `dph_selected` is JSON `null`: the selected price is **unknown**, never zero or
  free. The shared helper returns only a contract ID and does not establish an
  authoritative selected offer/contract price association. This diagnostic does
  not infer one from offer ordering, stdout or the cheapest quote.

The nullable selected-price field corrects a previous misleading numeric value.
External consumers that assume a number must handle unknown explicitly; consumers
that enumerate phases must also handle the additive `creating` phase. Existing
result, probe, release and exit-code fields retain their meanings.

## Remaining limits

No aggregate work/cleanup deadline, provider-enforced lifetime or all-in charge
cap is established here. The existing eligibility rates and individual operation
timeouts are unchanged; hourly price times planned duration is not a guaranteed
bill. Provider failure, delayed creation, controller loss or failed release can
leave charges active. Killing SSH does not stop a remote job or rental billing.

This reporting change does not alter provider retries, SSH/bootstrap/restore,
probe acceptance, cleanup, serving/evaluation behavior, frozen artifact identities,
scoring, thresholds or CI. Fake-provider tests establish only local reporting and
control flow. Real artifact contents, CUDA/GPU execution, serving/evaluation
acceptance and historical hosted timing gates remain unverified by these tests.
