# Column contract — goodvision-event-bench · `results` tab

The sheet is **already built**: group bands, freezes, number formats, and the color-scale (ranking)
rules are all configured. Your job is to add or update **one row for your model**, filling the dataset
block(s) you ran. Do not alter anything else.

Sheet: `1D3-fwFW5Fv1ASDlbt3VUIFJJfAKKsrH2LXVNMkoPLjA`, tab `results`, `sheetId` 0.
Row 1 = merged group bands, row 2 = headers (both frozen), column A frozen. **Data starts at row 3.**

## Layout: one row per model, dataset on top

Dataset size is a **top band**, not a column. Each dataset block (50 / 100 / all 120) repeats the same
five accuracy cells. Speed, cost, tracking and meta are single per-model columns. So a model has **one
row**; running a new dataset fills that dataset's block in the same row.

**Upsert key = Model (column A) alone.** Re-publishing a model updates its row in place. You fill only
the block(s) you ran and leave the others as they are — never blank a block someone else populated.

## The 25 columns, in order (A → Y)

| Col | Header | Band | Type / format | Ranking | Rule |
|-----|--------|------|---------------|---------|------|
| A | Model | IDENTITY | text | — | **upsert key.** Your model's name. |
| B | Backend | IDENTITY | text | — | `language` (hard 0/1) · `decision` (continuous score + gate) · `vision`. |
| C | Run | TRACKING | status | `yes`→green · `running`→yellow · `no`→red | Did the eval run? A posted result is normally `yes`. |
| D | Val | TRACKING | status | `yes`→green · `no`→red | Threshold/validation run? `yes` for **decision + clip**; `no` for **language** (hard 0/1, nothing to tune). |
| E | P50 @12 (s) | SPEED ↓ | num `0.000` | pale(min)→red(max) | Median latency at N concurrency. Blank if not measured. |
| F | P95 @12 (s) | SPEED ↓ | num `0.000` | pale(min)→red(max) | p95 latency at N concurrency. Blank if not measured. |
| G | Overall % | DATASET · 50 | num `0.0` | red→orange; green if =100 | `100·correct/total` on the 50 set. |
| H | Fight % | DATASET · 50 | num `0.0` | red→orange; green if =100 | `100·(TP+TN)/(TP+FN+TN+FP)`. **Note required.** |
| I | Smoking % | DATASET · 50 | num `0.0` | red→orange; green if =100 | same formula. **Note required.** |
| J | Fall % | DATASET · 50 | num `0.0` | red→orange; green if =100 | same formula. **Note required.** |
| K | Weapon % | DATASET · 50 | num `0.0` | red→orange; green if =100 | same formula. **Note required.** |
| L–P | Overall / Fight / Smoking / Fall / Weapon % | DATASET · 100 | num `0.0` | red→orange; green if =100 | same five cells, for the 100 set. Subtask notes required. |
| Q–U | Overall / Fight / Smoking / Fall / Weapon % | DATASET · all 120 | num `0.0` | red→orange; green if =100 | same five cells, for the full 120 set. Subtask notes required. |
| V | Hardware | COST | text | — | e.g. `OpenRouter API`, `L4`, `A100`. |
| W | Price ($/Mtok) | COST | num `0.00` | pale(min)→red(max) | Provider price **looked up at publish time** — prices drift. Blank if self-hosted/unknown. |
| X | Spec ver | META | text | — | Spec you ran against, e.g. `v1`. |
| Y | Run date | META | text | — | `YYYY-MM-DD`. If the blocks were run on different days, use the latest. |

### The three dataset blocks

| Dataset | Overall | Fight | Smoking | Fall | Weapon |
|---------|---------|-------|---------|------|--------|
| 50      | G | H | I | J | K |
| 100     | L | M | N | O | P |
| all 120 | Q | R | S | T | U |

Fill the block that matches your run. A block you did not run stays blank. At least one block must carry
an Overall %.

### Status columns (C, D)

Text cells coloured by exact value (not a gradient). Write them lowercase — that is what the rules match.
- **C · Run:** `yes` (green) · `running` (yellow) · `no` (red).
- **D · Val:** `yes` (green) · `no` (red). Only decision/clip models have a tunable threshold to
  validate; language models emit hard yes/no → `no`.

## Per-subtask note format (Fight/Smoking/Fall/Weapon of each filled block)

Each non-blank subtask cell carries a note with the confusion counts behind the %:

```
fight 11/12  ·  TP5 FN1 TN6 FP0  ·  err0
```

- `x/N` where `x = TP+TN`, `N = TP+FN+TN+FP` (errors excluded from N).
- If the class has **no negatives** (`TN+FP == 0`), append `  ·  no negatives (specificity n/a)` and do
  not report a false-positive rate.

## Row math (frozen — identical for every row already on the sheet)

- Per class: `x = TP + TN`, `N = TP + FN + TN + FP`, `% = round(100·x/N, 1)`.
- Overall: `% = round(100·correct/total, 1)` where `total` excludes errored questions.
- `decision` backends choose yes/no by a gate on a continuous score (frozen default gate = 0.5);
  `language`/`vision` backends emit hard yes/no. The % math above is the same regardless.

## What is already on the sheet (do not recreate or modify)

- Row-1 band merges: `C1:D1` (TRACKING), `E1:F1` (SPEED @12), `G1:K1` (DATASET · 50),
  `L1:P1` (DATASET · 100), `Q1:U1` (DATASET · all 120), `V1:W1` (COST), `X1:Y1` (META). `IDENTITY`
  sits in `A1`, unmerged (a merge can't cross the frozen A|B boundary).
- `frozenRowCount = 2`, `frozenColumnCount = 1`. Thin left borders separate the three dataset blocks.
- **Muted warm palette** (green reserved for best-only): red `#E67C73`, orange `#F9CB9C`,
  pale `#FCE5CD`, green `#B6D7A8`, yellow `#FFE599`.
- Number formats: accuracy G:U `0.0`, speed E/F `0.000`, price W `0.00`.
- Conditional-format rules (single-color/boolean rules come FIRST so they override the gradients):
  1. boolean on G:U: `NUMBER_EQ 100` → green (green shows **only** at a perfect 100.0).
  2. boolean on C: `yes`→green, `running`→yellow, `no`→red.
  3. boolean on D: `yes`→green, `no`→red.
  4. gradient per accuracy column G..U (one rule each): MIN red → MAX orange.
  5. gradient per speed column E,F and price W (one rule each): MIN pale → MAX red.

  Rebuild gotcha: the Sheets MCP `addConditionalFormatRule` inserts at index 0 (it prepends), so the
  **last** rule you add ends up on top. Add the gradients first and the boolean rules (green=100,
  Run, Val) last — otherwise a 100 cell shows the gradient max (orange) instead of green.

If the sheet were ever lost, this block plus the table above is enough to rebuild it — but as a
contributor you should never need to.
