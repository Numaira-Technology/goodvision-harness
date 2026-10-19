# Hash migration integration review

Review date: 2026-10-19. Incremental base: `33a70fa`; pre-task Harness baseline:
`6c28ddd`. The fresh reviewer read the uncommitted change set and new modules,
including the saved-answer runner/controller. No tests, probes, API calls or
resource mutations were performed.

Later scope decision: the human approved retiring signed webhooks and customer
automatic updates. A separate four-file source review found no Critical/Important
issue in the refusal paths. Two minor findings were addressed: published docs
now describe retirement, and the startup handler preserves the fixed public-safe
retirement message through a dedicated exception type. These are source findings
and edits, not runtime verification. The unresolved-choice statements below describe the earlier review,
not a current deployment gate. Checkpoint loading and maintainer deployment
remain outside that two-feature retirement; no repository-wide hash-free claim.

## Important findings and source fixes

1. The recovery catalog required every historical source-outcome path to be a
   file. The actual 0013 interruption points to a partial recording directory
   and a separate unknown-request reconciliation. The catalog now checks that
   directory's saved source/video manifest and the reconciliation file, retains
   both references and keeps the source pending with automatic replay disabled.
   It does not promote a partial recording to answerable evidence.
2. Removing the old session-reference helper from `continuous.py` left imports
   and a call in `cloud_pool.py`. Both import branches and the call were removed.
   Proxy timings retain transport/attempt IDs and camera/GPU routing. The
   capability token is not logged; the worker keeps its own non-secret session
   reference. This fixes the identified source dependency without restoring a
   hash calculation.

The author read the resulting changes against the actual historical manifest
and the remaining timing-helper definitions. These are source fixes, not
runtime-verified fixes. No second fresh review or prohibited test was run.

## Positive cross-boundary findings

- Legacy compressed requests retain their observation session/source/question
  scope. Unknown attempts take precedence over cached success. Unresolved old
  scope cannot become a fresh payable request.
- Prompt projection preserves repeated records, explicit negatives, nulls and
  ambiguous raw/published time references.
- The new recovery inventories the original 182 questions, reads the newest
  cumulative seed plus new ledger rows, and separates preparation, dispatch and
  saved-outcome subset scoring.
- Actual resume41 expired-association, invalid-response, timeout and explicit
  502 records have EOF/drained/accounted state and use the existing local-gap
  eligibility path. Static inspection did not find a restored blanket
  zero-failure-window or duration-deferral gate for those records.

## Deferred minor

The new association identity map in `runtime/scene.py` lasts until session close
and can grow in a long-lived stranger-observation session. It deliberately
preserves identity after the older result caches evict entries. A future bounded
lifecycle or reversible structural identity should preserve that property; an
arbitrary eviction that silently assigns a new identity is not a safe fix.
This item was not changed in this repair pass. It is not a claim of bounded
memory for indefinitely running production sessions.

## Review limits and rulings

- Tests and test edits remain prohibited by the user outside the specifically
  authorized evaluation and small wall-time trial. Static review does not
  establish runtime correctness, input fit, answer quality, latency or cost.
- The review read historical release records, not the current cloud inventory.
  The new controller must still reconcile the exact resources before launch.
- Webhook authentication and customer/checkpoint verification remain an
  unresolved human impact decision. Neither unsigned fallback nor unverified
  installation was implemented. The full branch is not ready to deploy while
  that decision and its corresponding changes remain open.
- Historical frozen source and test-only legacy contracts are preserved, not
  claimed migrated. The human's no-test-edit rule is not permission to report
  branch-wide hash removal as complete.
- Removing the local-GPU source checksum leaves required-file presence checks,
  not a guarantee that a prebuilt image matches current source. This limitation
  is explicit; the evaluation runner freezes/mounts committed source separately.

No commit, push, new input preparation, model dispatch, GPU creation or small
wall-time execution occurred during this review/fix pass.
