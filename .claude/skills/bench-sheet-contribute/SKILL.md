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
  Writes are authorization-gated: preview → explicit confirm → write, and only your own (Model, Dataset n)
  row is touched.
---

# bench-sheet-contribute

You have benchmark numbers for a vision model and want them on the **shared** running sheet so they sit
alongside everyone else's, sortable by the built-in color scales. This skill tells you the sheet's exact
contract and how to add one row **without breaking the format**. It assumes nothing about how you produced
the numbers — no repo scripts, no fixed JSON schema. You bring values; you map them to the columns below.

> If you are inside the `goodvision-harness` repo and produced results with its `run_eval.py`, use the
> `bench-sheet-push` skill instead — it builds the row for you. This skill is for **everyone else**.

## The sheet

- **Spreadsheet** `1D3-fwFW5Fv1ASDlbt3VUIFJJfAKKsrH2LXVNMkoPLjA`, tab **`results`** (`sheetId` 0).
- URL: <https://docs.google.com/spreadsheets/d/1D3-fwFW5Fv1ASDlbt3VUIFJJfAKKsrH2LXVNMkoPLjA/edit>
- **Row 1** = merged group bands, **row 2** = headers. Both frozen; column A also frozen. **Data starts at row 3.**
- The color scales (ranking) and group bands are **already configured on the sheet**. You never create or
  re-apply them — you only write cell values into your row. See `references/column-contract.md`.

## The column contract (A–Q, in this exact order)

Full table, types, and ranking direction are in **`references/column-contract.md`** — read it before writing.
One row = **one (Model, Dataset n)** result. Summary:

| Col | Key | What you put | Blank if |
|-----|-----|--------------|----------|
| A | Model | your model name (upsert key 1) | never |
| B | Backend | `language` / `decision` / `vision` | never |
| C | Single inf (s) | single-call latency, 3 dp | not measured |
| D | P50 @N (s) | median latency at N concurrency | not measured |
| E | P95 @N (s) | p95 latency at N concurrency | not measured |
| F | Overall acc % | `100·correct/total`, 1 dp | never |
| G–J | Fight/Smoking/Fall/Weapon % | per-class `100·(TP+TN)/(TP+FN+TN+FP)` | class not run |
| K | Hardware | e.g. `OpenRouter API`, `L4`, `A100` | never |
| L | Price ($/Mtok) | provider price **looked up now** | unknown |
| M | Dataset (n) | question count: 50 / 120 / … (upsert key 2) | never |
| N | Spec ver | spec you ran against, e.g. `v1` | never |
| O | Run date | `YYYY-MM-DD` | never |
| P | Provisional | `TRUE` if GT still under review, else `FALSE` | never |
| Q | Source | where your raw output lives (path/link) | never |
| R | Run | `yes` (eval ran) / `running` / `no` | never |
| S | Validation | `yes` if your model does threshold validation (decision / clip), else `no` (language models) | never |

**The four per-subtask cells (G–J) carry a cell NOTE** with the exact counts:
`fight 11/12  ·  TP5 FN1 TN6 FP0  ·  err0`. The % is for the color scale; the note is the ground truth.

### Column math (do not change it — this is how every other row was computed)

- Per class: `x = TP + TN`, `N = TP + FN + TN + FP` (errors excluded), `% = 100·x/N`.
- Overall: `% = 100·correct/total`.
- A class with no negatives (`TN+FP == 0`): put the % but add `no negatives (specificity n/a)` to the note —
  do not invent a false-positive rate.

## Flow

Work from anywhere; you only need the Google Sheets MCP connector (or equivalent in-session Google auth).

1. **Assemble your row** as 19 values in column order A→S, plus a note string for each of G/H/I/J.
   Use `references/column-contract.md` for the exact type of each cell. If you have a JSON blob of your
   numbers, run the validator (below) to catch mistakes before you touch the sheet.

2. **Validate (recommended):**
   ```sh
   python3 -I .claude/skills/bench-sheet-contribute/scripts/validate_row.py my_row.json
   ```
   `my_row.json` = `{"values": [19 cells A→S], "notes": {"G": "...", "H": "...", "I": "...", "J": "..."}}`.
   It checks column count, types, % ranges, required cells, and the dynamic `@N` speed label — **offline,
   no network.** Fix anything it flags.

3. **Read the sheet once** (so you upsert, not duplicate):
   `get_values(spreadsheetId, range="results!A3:S")`. Find any existing row whose **(A Model, M Dataset n)**
   matches yours. That tells you NEW (append) vs UPDATE (overwrite in place).

4. **Preview — ALWAYS, before writing.** Show the user:
   - target (sheet name, URL, tab `results`);
   - your row in column order, labelled **NEW** or **UPDATE** (and which cells change on an update);
   - a ⚠ PROVISIONAL flag if P = TRUE;
   - any blank cells, named (e.g. "speed blank — not measured").

5. **Wait for an explicit "yes".** Writing to a shared doc is an explicit-permission action. No auto-push.
   If the user declines, stop.

6. **Write — only your row:**
   - **UPDATE:** `update_values(range="results!A<row>:S<row>", values=[your 19 values])`.
   - **NEW:** `append_values(range="results!A3", values=[your 19 values])` (lands after the last data row).
   - **Notes:** one `update_spreadsheet` batch of `updateCells` with `fields="note"`, writing each G/H/I/J
     note to that cell of your row.

7. **Report** what was written (row number, new vs updated) and the sheet link.

## Do-not-break rules (the whole point of this skill)

- **Never** run `update_spreadsheet` to change merges, freezes, headers (rows 1–2), number formats, or
  conditional-format (color-scale) rules. They are already set. Your contribution is **values + G–J notes only.**
- **Never** reorder, insert, or delete columns, and never add a column the contract doesn't list. 19 values,
  A→S, every time — even the blanks (send `""` for a blank so later columns don't shift left).
- **Never** edit another model's row or any cell outside your (Model, Dataset n) row.
- **Never** fabricate a number. Unmeasured speed/price/class → blank, not a guess.
- A different dataset size is a **different row**, not an overwrite (50 and 120 coexist).
- The speed columns read `@N` where N is your concurrency. If your N differs from the header's and you're the
  first to post at that N, flag it in the preview and ask before changing the D/E header text — don't change
  it silently.

## Authorization & safety

- **Google auth = the in-session OAuth connector only.** No service-account keys, no `gspread`, no stored
  tokens. Nothing sensitive is written to disk.
- If the connector errors with an auth/permission problem, tell the user to reconnect it via `/mcp` and stop —
  do not fall back to any key-based path.
- Preview → explicit confirm → write, every single time.
