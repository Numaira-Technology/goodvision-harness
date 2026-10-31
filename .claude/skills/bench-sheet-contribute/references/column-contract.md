# Column contract — goodvision-event-bench · `results` tab

The sheet you are contributing to is **already built**: group bands, freezes, number formats, and the
color-scale (ranking) rules are all configured. Your job is to add exactly one data row of **values** (A→Q)
plus four cell **notes** (G–J), matching the contract below. Do not alter anything else.

Sheet: `1D3-fwFW5Fv1ASDlbt3VUIFJJfAKKsrH2LXVNMkoPLjA`, tab `results`, `sheetId` 0.
Row 1 = merged group bands, row 2 = headers (both frozen), column A frozen. **Data starts at row 3.**

## The 17 columns, in order

| Col | Header | Group band | Type / format | Ranking (color scale) | Rule |
|-----|--------|-----------|---------------|------------------------|------|
| A | Model | IDENTITY | text | — | **upsert key 1.** Your model's name. |
| B | Backend | IDENTITY | text | — | `language` (hard 0/1 output) · `decision` (continuous score + gate) · `vision`. |
| C | Single inf (s) | SPEED ↓ | num `0.000` | pale(min)→red(max) | Single-call latency. Blank if not measured. |
| D | P50 @N (s) | SPEED ↓ | num `0.000` | pale(min)→red(max) | Median latency at N concurrency. Blank if not measured. |
| E | P95 @N (s) | SPEED ↓ | num `0.000` | pale(min)→red(max) | p95 latency at N concurrency. Blank if not measured. |
| F | Overall acc % | ACCURACY | num `0.0` | red(min)→orange(max); **green only if =100** | `100·correct/total`. |
| G | Fight % | PER-SUBTASK % | num `0.0` | red→orange; green if =100 | `100·(TP+TN)/(TP+FN+TN+FP)`. **Note required.** |
| H | Smoking % | PER-SUBTASK % | num `0.0` | red→orange; green if =100 | same formula. **Note required.** |
| I | Fall % | PER-SUBTASK % | num `0.0` | red→orange; green if =100 | same formula. **Note required.** |
| J | Weapon % | PER-SUBTASK % | num `0.0` | red→orange; green if =100 | same formula. **Note required.** |
| K | Hardware | COST | text | — | e.g. `OpenRouter API`, `L4`, `A100`. |
| L | Price ($/Mtok) | COST | num | pale(min)→red(max) | Look up the provider price **at publish time** — prices drift. Blank if self-hosted/unknown. |
| M | Dataset (n) | META | num | — | **upsert key 2.** Question count: `50` / `120` / `250` … |
| N | Spec ver | META | text | — | Spec you ran against, e.g. `v1`. |
| O | Run date | META | text | — | `YYYY-MM-DD`. |
| P | Provisional | META | bool | — | `TRUE` if any GT is still under review, else `FALSE`. |
| Q | Source | META | text | — | Where your raw per-clip output lives (path or link). |
| R | Run | TRACKING | status | `yes`→green · `running`→yellow · `no`→red | Did the eval run? A posted result is normally `yes`. |
| S | Validation | TRACKING | status | `yes`→green · `no`→red | Did you run threshold/validation? `yes` for **decision + clip** models; `no` for **language** models (hard 0/1, no threshold to validate). |

**Upsert key = (A Model, M Dataset n).** One row per pair. Re-publishing the same pair overwrites that row;
a different dataset size is a new row.

### Status columns (R, S) — the two tracking cells

These are **text**, not numbers, and the sheet colors them by exact value (not a gradient):

- **R · Run:** `yes` (ran → green) · `running` (in progress → yellow) · `no` (not run → red).
- **S · Validation:** `yes` (ran threshold validation → green) · `no` (→ red).
  Only **decision** and **clip-style** models have a tunable threshold to validate, so they get `yes`;
  **language** models emit hard yes/no, so validation is meaningless for them → `no`.

Write them lowercase (`yes` / `no` / `running`); that is what the color rules match.

## Per-subtask note format (G, H, I, J)

Each of the four cells carries a note with the exact confusion counts behind the %:

```
fight 11/12  ·  TP5 FN1 TN6 FP0  ·  err0
```

- `x/N` where `x = TP+TN`, `N = TP+FN+TN+FP` (errors excluded from N).
- If the class has **no negatives** (`TN+FP == 0`), append `  ·  no negatives (specificity n/a)` and do not
  report a false-positive rate.

## Row math (frozen — identical for every row already on the sheet)

- Per class: `x = TP + TN`, `N = TP + FN + TN + FP`, `% = round(100·x/N, 1)`.
- Overall: `% = round(100·correct/total, 1)` where `total` excludes errored questions.
- `decision` backends choose yes/no by a gate on a continuous score (frozen default gate = 0.5);
  `language`/`vision` backends emit hard yes/no. The % math above is the same regardless.

## What is already on the sheet (do not recreate or modify)

- Merges on row 1 (group bands): `C1:E1`, `G1:J1`, `K1:L1`, `M1:Q1`, `R1:S1` (IDENTITY label sits in `A1`,
  unmerged — a merge can't cross the frozen A|B boundary).
- `frozenRowCount = 2`, `frozenColumnCount = 1`.
- **Muted warm palette** (green reserved for best-only): red `#E67C73`, orange `#F9CB9C`, pale `#FCE5CD`,
  green `#B6D7A8`, yellow `#FFE599`.
- Conditional-format rules (single-color/boolean rules come FIRST so they override the gradients):
  1. boolean on F:J: `NUMBER_EQ 100` → green (green shows **only** at a perfect 100%).
  2. boolean on R: `yes`→green, `running`→yellow, `no`→red.
  3. boolean on S: `yes`→green, `no`→red.
  4. gradient per accuracy column F,G,H,I,J: MIN red → MAX orange.
  5. gradient per speed column C,D,E and price L: MIN pale → MAX red.

If the sheet were ever lost, this block plus the table above is enough to rebuild it with one
`update_spreadsheet` batch — but as a contributor you should never need to.
