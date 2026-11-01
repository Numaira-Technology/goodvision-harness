---
name: bench-sheet-contribute
description: >-
  Prepare and, when authorized, contribute a model's benchmark results to the team
  Google Sheet. Use for upload my scores / contribute results / 把我的分数加到那个表.
  Validates the live schema, provenance and target row; previews changes before writing.
---

# Contribute benchmark results

Default to an **offline dry run**: prepare values, notes and a proposed change set.
A dry run performs no writes or uploads and needs no credentials. Reading the live
sheet to verify the target is a separate read-only step. Run or reuse models only
within the user's authorized scope; this skill does not authorize inference.

## Destination and contract

- Spreadsheet: `1D3-fwFW5Fv1ASDlbt3VUIFJJfAKKsrH2LXVNMkoPLjA`.
- URL: <https://docs.google.com/spreadsheets/d/1D3-fwFW5Fv1ASDlbt3VUIFJJfAKKsrH2LXVNMkoPLjA/edit>.
- Expected tab `results`, sheetId 0; headers rows 1–2, data starts row 3.
- Expected pivoted schema: **A:W**, detailed in `references/column-contract.md`.
  This was checked against live headers on 2026-11-01; re-read metadata and headers
  on every publication. **The repository's `sheet_config.json` describes a legacy
  A:Q layout and is incompatible.** Do not use its positions or absent
  `bench-sheet-push`/`build_rows.py` tooling. A header mismatch blocks writing;
  do not migrate the sheet, rename headers or modify the config as a side effect.

Model in column A is the lookup key, but identifies **model + protocol + variant**,
e.g. `luna / v2 / fixed 0.5` vs `luna / v2 / val-tuned <hash>`. Require explicit row
ownership/target agreement; sharing a model name does not authorize replacing
another contributor's result. Incompatible dataset versions, prompts, timing or
threshold artifacts require a distinct identity. Dataset size alone is not identity.

## Prepare and validate

1. Map saved measured artifacts to the exact block(s), with result/dataset hashes,
   source commit, protocol, label-review state, counts and dates. Use v2 rules only
   for v2 outputs. Preserve other runners' denominators in separate variants; do
   not make unlike metrics appear equivalent. Draft labels stay explicitly
   provisional; they cannot populate an official reviewed-v2 row.
2. Prepare `{"values": [23 cells A through W], "notes": {"E": "...", ...}}`.
   New-row blanks are `""`. Notes on every filled accuracy cell include correct/valid,
   `errN`, `attemptedN` and `provisional=yes|no`. Subtasks also include TP/FN/TN/FP
   and no-negative warnings. Overall notes carry the block's source artifact/hash,
   dataset/manifest hash, protocol/variant and run date. See the contract.
3. Run the offline validator (required before publishing):
   ```sh
   python3 -I .claude/skills/bench-sheet-contribute/scripts/validate_row.py my_row.json
   ```
   It checks values and arithmetic, not provenance truth, live schema, ownership or
   write authorization. Verify those separately. It never accesses the network.

## Read, preview, authorize, write, verify

1. Through the in-session OAuth connector, read metadata, bounded header ranges
   and the populated model-key range. Read candidate target cells **and notes and
   formulas**. Stop on schema mismatch, duplicate exact keys, ambiguous ownership,
   unexpected formulas, incompatible provenance or permission errors. Do not assume
   a row index or use case-folded/fuzzy matching to pick a row.
2. Build a **sparse change set**, only the measured dataset blocks and explicitly
   supplied metadata/notes. Preserve unmeasured speed/price, other dataset blocks,
   their dates/provenance notes and formulas. A blank in an offline row means
   unmeasured, not permission to erase an existing value. Deletions need explicit
   authorization. Shared metadata must remain accurate for every preserved block.
3. Preview the destination, exact identity, NEW/UPDATE, target cells, before/after
   values/notes, preserved cells, source hashes and provisional/incomplete flags.
   If this exact write is not already authorized in the current request, obtain
   explicit confirmation of the concrete preview. No automatic publish from a
   benchmark or tuning request. Declining or requesting dry run ends before writes.
4. **Immediately before writing**, re-read headers, all exact key matches and the
   target values/notes/formulas. Re-resolve row index. If anything differs from the
   approved snapshot, stop and rebuild the preview; never overwrite a stale row.
   Sheets has no compare-and-swap row upsert here. Coordinate a single writer for
   the target; if simultaneous edits cannot be excluded, stop instead of claiming
   atomic idempotency. Keep rows stable through value/notes verification.
5. **UPDATE:** write only approved cells with literal/RAW input; do not send a
   merged full row. Keep formulas and unmeasured cells unchanged. Use narrow
   `updateCells` masks for notes (or an atomic values+notes batch if supported).
   Never change formatting, merges, headers, freezes, dimensions or color rules.
   **NEW:** recheck the key is absent, append one A:W row with RAW values and obtain
   the server-returned updated range. Use that returned row for notes after checking
   the key. Never infer it from row count or reuse a previously guessed empty row.
6. Read back the written values and notes, confirm exact identity and uniqueness,
   and compare with the approved change set. On timeout/ambiguous outcome, read first
   to determine what executed; **never blindly retry append**. Resume only missing
   approved cells/notes, or stop on conflicts. If the same artifact and intended
   cells already match, report no-op. Record any partial result precisely.
7. Report the verified row, changed blocks/cells and source hashes with the sheet link.

## Boundaries

- @12 columns accept only a complete 12-concurrency measurement with the agreed
  request timing boundary. Other timing belongs in the report/notes, cells blank;
  do not change global headers to accommodate a contributor.
- The all-120 block requires the full reviewed cohort and an explicit versioned
  combined metric covering three-valued bag questions. A base-4 run has 100
  questions; leave all-120 blank. This skill supplies no new unknown-answer policy.
- Never invent a number or convert a missing/invalid model answer to false.
- Use **in-session OAuth only**. No service-account keys, stored tokens, `gspread`,
  credential creation or credential files. On connection/auth errors, stop and
  request reconnect through the app; do not switch authentication paths.
