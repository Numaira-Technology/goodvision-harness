---
name: bench-sheet-contribute
description: >-
  Add YOUR model's event-recognition scores to the shared goodvision-event-bench running Google Sheet
  (tab `results`) correctly — right columns, right ranking (color scales), without breaking the existing
  format or anyone else's row. Use this WHENEVER you (any contributor's agent) have benchmark numbers for
  a vision model on the fight/smoking/fall/weapon task and need to publish them to the team sheet:
  "把我的分数加到那个表 / upload my scores / contribute results / push my model to the bench sheet".
  This is the SELF-CONTAINED, shareable contributor path — it does NOT assume this repo's run_eval.py,
  build_rows.py, or any local file. It only needs your own numbers + the in-session Google OAuth connector.
  Writes are authorization-gated: preview → explicit confirm → write, and only your own model row is touched.
---

# bench-sheet-contribute

You have benchmark numbers for a vision model and want them on the shared running sheet, alongside everyone
else's, sortable by the built-in color scales. This skill gives you the sheet's exact contract and how to add
your row without breaking the format. It assumes nothing about how you produced the numbers: no repo scripts,
no fixed JSON schema. You bring values and map them to the columns below.

> If you are inside the `goodvision-harness` repo and produced results with its `run_eval.py`, use the
> `bench-sheet-push` skill instead; it builds the row for you. This skill is for everyone else.

## The sheet

- **Spreadsheet** `1D3-fwFW5Fv1ASDlbt3VUIFJJfAKKsrH2LXVNMkoPLjA`, tab **`results`** (`sheetId` 0).
- URL: <https://docs.google.com/spreadsheets/d/1D3-fwFW5Fv1ASDlbt3VUIFJJfAKKsrH2LXVNMkoPLjA/edit>
- **Row 1** = merged group bands, **row 2** = headers. Both frozen; column A also frozen. **Data starts at row 3.**
- The color scales (ranking) and group bands are **already configured on the sheet**. You never create or
  re-apply them — you only write cell values into your row. See `references/column-contract.md`.

## Layout: one row per model, dataset on top, no tracking columns

Dataset size (50 / 100 / all 120) is a **top band**, not a column. Each dataset block repeats the same five
accuracy cells; speed, cost and meta are single per-model columns. There are no Run/Validation columns. So
your model is **one row**, and running a new dataset fills that dataset's block in the same row. The upsert
key is **Model (column A) alone**.

## The column contract (A–W, in this exact order)

Full table, types, ranking direction, and the dataset-block map are in **`references/column-contract.md`** —
read it before writing. Summary:

| Col | Key | What you put | Blank if |
|-----|-----|--------------|----------|
| A | Model | your model name (**upsert key**) | never |
| B | Backend | `language` / `decision` / `vision` | never |
| C | P50 @12 (s) | median latency at N concurrency | not measured |
| D | P95 @12 (s) | p95 latency at N concurrency | not measured |
| E–I | DATASET · 50 | Overall + Fight/Smoking/Fall/Weapon % on the 50 set | 50 not run |
| J–N | DATASET · 100 | same five cells for the 100 set | 100 not run |
| O–S | DATASET · all 120 | same five cells for the full 120 set | 120 not run |
| T | Hardware | e.g. `OpenRouter API`, `4090`, `L4`, `A100` | never |
| U | Price | run cost: API → OpenRouter `$/Mtok`; self-hosted → GPU run `$` (note which) | unknown |
| V | Spec ver | spec you ran against, e.g. `v1` | never |
| W | Run date | `YYYY-MM-DD` (latest block's date) | never |

Each dataset block maps to its columns as **50 → E,F,G,H,I · 100 → J,K,L,M,N · all 120 → O,P,Q,R,S**
(Overall first, then Fight/Smoking/Fall/Weapon). Fill the block you ran; leave the others blank.

**The four subtask cells of each filled block carry a cell NOTE** with the exact counts:
`fight 11/12  ·  TP5 FN1 TN6 FP0  ·  err0` (or just `fight 28/32` if you only have x/N). The % is for the
color scale; the note is the ground truth.

### Column math (do not change it — this is how every other row was computed)

- Per class: `x = TP + TN`, `N = TP + FN + TN + FP` (errors excluded), `% = 100·x/N`.
- Overall: `% = 100·correct/total`.
- A class with no negatives (`TN+FP == 0`): put the % but add `no negatives (specificity n/a)` to the note —
  do not invent a false-positive rate.

## Flow

Work from anywhere; you only need the Google Sheets MCP connector (or equivalent in-session Google auth).

1. **Assemble your row** as 23 values in column order A→W, plus a note string for each non-blank subtask
   cell. Use `references/column-contract.md` for the exact type of each cell. Fill only the dataset block(s)
   you ran; send blanks (`""`) for every other cell so columns don't shift.

2. **Validate (recommended):**
   ```sh
   python3 -I .claude/skills/bench-sheet-contribute/scripts/validate_row.py my_row.json
   ```
   `my_row.json` = `{"values": [23 cells A→W], "notes": {"F": "...", "K": "...", ...}}`.
   It checks column count, types, % ranges, required cells, per-block consistency, and the subtask notes —
   **offline, no network.** Fix anything it flags.

3. **Read the sheet once** (so you upsert, not duplicate):
   `get_values(spreadsheetId, range="results!A3:W")`. Find the row whose **Model (column A)** matches yours.
   If none, your row is NEW (append). If it exists, you UPDATE it — **and you must merge**: keep the cells
   of dataset blocks you did not run exactly as they already are, changing only your block plus the
   per-model columns.

4. **Preview — ALWAYS, before writing.** Show the user:
   - target (sheet name, URL, tab `results`);
   - your row in column order, labelled **NEW** or **UPDATE**, and on an update which dataset block you fill
     and which existing blocks you preserve;
   - any blank cells, named (e.g. "100 block blank — not run");
   - a ⚠ note if any filled block's GT is still provisional.

5. **Wait for an explicit "yes".** Writing to a shared doc is an explicit-permission action. No auto-push.
   If the user declines, stop.

6. **Write — only your model's row:**
   - **UPDATE:** `update_values(range="results!A<row>:W<row>", values=[your 23 values])`, where the blocks
     you did not run hold the values you just read back (unchanged), not blanks.
   - **NEW:** `append_values(range="results!A3", values=[your 23 values])` (lands after the last data row).
   - **Notes:** one `update_spreadsheet` batch of `updateCells` with `fields="note"`, writing each subtask
     note to that cell of your row (only for the block(s) you filled).

7. **Report** what was written (row number, new vs updated, which block), and the sheet link.

## Guardrails (the whole point of this skill)

- **Never** run `update_spreadsheet` to change merges, freezes, headers (rows 1–2), number formats, or
  conditional-format (color-scale) rules. They are already set. Your contribution is **values + subtask
  notes only.**
- **Never** reorder, insert, or delete columns, and never add a column the contract doesn't list. 23 values,
  A→W, every time — even the blanks (send `""` so later columns don't shift left).
- **Never blank a dataset block you didn't run.** On an update, carry the other blocks' existing values
  through unchanged. The upsert key is Model alone, so a careless full-row write wipes other datasets.
- **Never** edit another model's row or any cell outside your model's row.
- **Never** fabricate a number. Unmeasured speed/price/class → blank, not a guess.
- The speed columns read `@N` where N is your concurrency. If your N differs from the header's and you're the
  first to post at that N, flag it in the preview and ask before changing the C/D header text — don't change
  it silently.

## Authorization & safety

- **Google auth = the in-session OAuth connector only.** No service-account keys, no `gspread`, no stored
  tokens. Nothing sensitive is written to disk.
- If the connector errors with an auth/permission problem, tell the user to reconnect it via `/mcp` and stop —
  do not fall back to any key-based path.
- Preview → explicit confirm → write, every single time.
