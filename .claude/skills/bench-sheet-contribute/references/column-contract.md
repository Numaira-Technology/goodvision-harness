# Historical pivoted Sheet contract: A:W

This records `results` in spreadsheet
`1D3-fwFW5Fv1ASDlbt3VUIFJJfAKKsrH2LXVNMkoPLjA` as checked on **2026-11-01**.
It is retained to interpret historical payloads and `scripts/validate_row.py`;
it is **not the current schema of results**. Read live headers before choosing a
mapping. Current field meanings, rather than fixed column letters, are recorded in
[results-column-contract.md](results-column-contract.md).

In this historical layout, headers occupy rows 1–2 and data begins at 3.
The old repository A:Q config does not describe even this A:W schema. Do not
rebuild or migrate formatting as a contribution. Rules below describe this
historical contract, not a replacement for the current unified scoring policy.

| Column | Row-2 header | Value |
|---|---|---|
| A | Model | nonempty text, exact model/protocol/variant identity |
| B | Backend | language / decision / vision |
| C | P50 @12 (s) | finite nonnegative number or blank |
| D | P95 @12 (s) | finite nonnegative number or blank; >= P50 |
| E | Overall % | 50-question block overall |
| F–I | Fight % / Smoking % / Fall % / Weapon % | 50 block subtasks |
| J | Overall % | 100-question base-4 block overall |
| K–N | Fight % / Smoking % / Fall % / Weapon % | 100 block subtasks |
| O | Overall % | full 120-question combined metric, only if agreed |
| P–S | Fight % / Smoking % / Fall % / Weapon % | 120 block base-4 subtasks |
| T | Hardware | nonempty text |
| U | Price | finite nonnegative number or blank; unit/source note required |
| V | Spec ver | nonempty protocol/variant text, e.g. v2 fixed |
| W | Run date | actual YYYY-MM-DD calendar date; latest filled block |

Percentages are JSON numbers on 0..100, rounded to one decimal, not fractions or
strings. Blanks are `""`/null in the offline payload. A new row needs at least one
filled overall. Every subtask percentage needs the block's overall. Missing
subtasks must be explained; don't treat blanks as zero. Numeric strings, boolean
numbers, NaN/infinity, objects/lists as text and invalid dates are rejected.

Backend `vision` is the Sheet label for margin models; do not rename it to threshold
without a coordinated schema change. Decision includes Luna and native verdicts;
the cell label alone does not establish a threshold or metric.

## Accuracy and note contract

For **v2 base-4**: `valid=TP+FN+TN+FP`, `correct=TP+TN`,
`attempted=valid+errors`, percentage = `round(100*correct/valid, 1)`.
No valid questions means blank accuracy. Report errors separately; they are never
negative predictions. Other runners' unknown/error policies need a separately
identified protocol, not silently substituted denominators.

The bundled validator supports binary confusion counts with errors excluded from
valid. If another runner counts unknown/abstained outputs inside the base-event
denominator, its result cannot pass this contract unchanged. Keep it in a separate
report until a reviewed adapter/metric contract exists; do not relabel unknown as
an excluded error or a false prediction merely to make the validator pass.

Each filled accuracy cell requires one unambiguous x/N plus `errN`, `attemptedN`
and `provisional=yes|no`. Subtask notes additionally carry TP/FN/TN/FP:

```
fight 11/12 · TP5 FN1 TN6 FP0 · err1 · attempted13 · provisional=no
```

Here N=12 is valid, attempted=13. When TN+FP=0 append
`no negatives (specificity n/a)`. The validator checks these counts against the
percentage. For 50/100 blocks with all subtasks present it also checks that their
counts add up to overall. The 120 overall is not the sum of just four subtasks;
it includes the additional task, whose accepted denominator and unknown policy
must be documented and audited before publication.

Overall note example (placeholders must be replaced by real provenance):

```
overall 45/50 · err0 · attempted50 · provisional=no · dataset=<manifest SHA256>
source=<result SHA256> · commit=<SHA> · protocol=v2 fixed · run_date=YYYY-MM-DD
```

Each block's attempted count must match its expected reviewed manifest; a failed
run still attempts the full cohort. The 50/100/120 band is a cohort, not a successful
answer count. Preserve original provenance notes when another block is added. The
validator checks syntax/arithmetic only; it cannot verify hashes, completeness,
review status or whether the selected cohort is the intended one.

## Timing and price notes

C/D must both be filled or both blank; @12 means 12, not an arbitrary N. Require a
complete measurement and matching frame/question/timing basis. Describe timestamp,
source artifact, attempt/success counts and request boundary in C/D notes. Other
runners' timings are separate report entries, not comparable @12 values.

U requires a note stating currency, unit, pricing date/source and basis. API input
and output token rates are separate; one cell cannot honestly represent both without
an explicitly defined weighting. GPU run dollars and dollars per million tokens
are not directly comparable. Leave U blank when no agreed single metric applies;
explain prices in the report. Never change global price formatting/rules here.

## Row identity and safe updates

Column A is the exact lookup key. Distinguish model, protocol, threshold artifact
and other variant parameters in that identity. Two exact matches or ambiguous
ownership block writing. Dataset blocks share a row only if shared metadata remains
true for all blocks; otherwise prepare a distinct variant. A changed dataset's
hash cannot silently replace another revision's results under the same block.

Offline rows contain 23 cells to make mapping reviewable; **updates are sparse**.
Blank unmeasured inputs never erase existing measurements. Read metadata, header
bands, cells, notes and formulas; preview the exact change; re-read before writing.
Use RAW/literal values, narrow masks and the append response's actual row range.
After uncertain writes, read back before retrying. See SKILL.md for the full flow.
