# Column contract — goodvision-event-bench · `results` tab

The sheet is **already built**: group bands, freezes, number formats, and the color-scale (ranking)
rules are all configured. Your job is to add or update **one row for your model**, filling the dataset
block(s) you ran. Do not alter anything else.

Sheet: `1D3-fwFW5Fv1ASDlbt3VUIFJJfAKKsrH2LXVNMkoPLjA`, tab `results`, `sheetId` 0.
Row 1 = merged group bands, row 2 = headers (both frozen), column A frozen. **Data starts at row 3.**

## Layout: one row per model, dataset on top, no tracking columns

Dataset size is a **top band**, not a column. Each dataset block (50 / 100 / all 120) repeats the same
five accuracy cells. Speed, cost and meta are single per-model columns. There are **no Run/Validation
tracking columns**. So a model has **one row**; running a new dataset fills that dataset's block in the
same row.

**Upsert key = Model (column A) alone.** Re-publishing a model updates its row in place. You fill only
the block(s) you ran and leave the others as they are — never blank a block someone else populated.

## The 23 columns, in order (A → W)

| Col | Header | Band | Type / format | Ranking | Rule |
|-----|--------|------|---------------|---------|------|
| A | Model | IDENTITY | text | — | **upsert key.** Your model's name. |
| B | Backend | IDENTITY | text | — | `language` (hard 0/1) · `decision` (continuous score + gate) · `vision` (CLIP/SigLIP-style). |
| C | P50 @12 (s) | SPEED ↓ | num `0.000` | pale(min)→red(max) | Median latency at N concurrency. Blank if not measured. |
| D | P95 @12 (s) | SPEED ↓ | num `0.000` | pale(min)→red(max) | p95 latency at N concurrency. Blank if not measured. |
| E | Overall % | DATASET · 50 | num `0.0` | red→orange; green if =100 | `100·correct/total` on the 50 set. |
| F | Fight % | DATASET · 50 | num `0.0` | red→orange; green if =100 | `100·(TP+TN)/(TP+FN+TN+FP)`. **Note required.** |
| G | Smoking % | DATASET · 50 | num `0.0` | red→orange; green if =100 | same formula. **Note required.** |
| H | Fall % | DATASET · 50 | num `0.0` | red→orange; green if =100 | same formula. **Note required.** |
| I | Weapon % | DATASET · 50 | num `0.0` | red→orange; green if =100 | same formula. **Note required.** |
| J–N | Overall / Fight / Smoking / Fall / Weapon % | DATASET · 100 | num `0.0` | red→orange; green if =100 | same five cells, for the 100 set. Subtask notes required. |
| O–S | Overall / Fight / Smoking / Fall / Weapon % | DATASET · all 120 | num `0.0` | red→orange; green if =100 | same five cells, for the full 120 set. Subtask notes required. |
| T | Hardware | COST | text | — | e.g. `OpenRouter API`, `4090`, `L4`, `A100`. |
| U | Price | COST | num | pale(min)→red(max) | Each run's cost. API: OpenRouter token price `$/Mtok` (look up at publish time). Self-hosted: computed GPU run cost `$` (rate × wall-hours). State which in the cell note. Blank if unknown. |
| V | Spec ver | META | text | — | Spec you ran against, e.g. `v1`. |
| W | Run date | META | text | — | `YYYY-MM-DD`. If the blocks were run on different days, use the latest. |

### The three dataset blocks

| Dataset | Overall | Fight | Smoking | Fall | Weapon |
|---------|---------|-------|---------|------|--------|
| 50      | E | F | G | H | I |
| 100     | J | K | L | M | N |
| all 120 | O | P | Q | R | S |

Fill the block that matches your run. A block you did not run stays blank. At least one block must carry
an Overall %. The full 120 set adds 20 three-valued bag-search questions; only score-based models answer
those, so language models fill 100 (base-4) and leave the 120 block blank.

## Per-subtask note format (Fight/Smoking/Fall/Weapon of each filled block)

Each non-blank subtask cell carries a note with the confusion counts behind the %:

```
fight 11/12  ·  TP5 FN1 TN6 FP0  ·  err0
```

- `x/N` where `x = TP+TN`, `N = TP+FN+TN+FP` (errors excluded from N).
- If the class has **no negatives** (`TN+FP == 0`), append `  ·  no negatives (specificity n/a)` and do
  not report a false-positive rate.
- If you only have `x/N` (no confusion breakdown), put just `fight 28/32` — that is enough.

## Row math (frozen — identical for every row already on the sheet)

- Per class: `x = TP + TN`, `N = TP + FN + TN + FP`, `% = round(100·x/N, 1)`.
- Overall: `% = round(100·correct/total, 1)` where `total` excludes errored questions.
- `decision`/`vision` backends choose yes/no by a gate on a continuous score (frozen default gate = 0.5);
  `language` backends emit hard yes/no. The % math above is the same regardless.

## What is already on the sheet (do not recreate or modify)

- Row-1 band merges: `C1:D1` (SPEED @12), `E1:I1` (DATASET · 50), `J1:N1` (DATASET · 100),
  `O1:S1` (DATASET · all 120), `T1:U1` (COST), `V1:W1` (META). `IDENTITY` sits in `A1`, unmerged
  (a merge can't cross the frozen A|B boundary).
- `frozenRowCount = 2`, `frozenColumnCount = 1`. Thin left borders separate the three dataset blocks.
- **Muted warm palette** (green reserved for best-only): red `#E67C73`, orange `#F9CB9C`,
  pale `#FCE5CD`, green `#B6D7A8`, yellow `#FFE599`.
- Number formats: accuracy E:S `0.0`, speed C/D `0.000`, price U `0.00`.
- Conditional-format rules (the boolean rule must outrank the gradients):
  1. boolean on E:S: `NUMBER_EQ 100` → green (green shows **only** at a perfect 100.0).
  2. gradient per accuracy column E..S (one rule each): MIN red → MAX orange.
  3. gradient per speed column C,D and price U (one rule each): MIN pale → MAX red.

  Rebuild gotcha: the Sheets MCP `addConditionalFormatRule` inserts at index 0 (it prepends), so the
  **last** rule you add ends up on top. Add the gradients first and the green=100 boolean last —
  otherwise a 100 cell shows the gradient max (orange) instead of green.

If the sheet were ever lost, this block plus the table above is enough to rebuild it — but as a
contributor you should never need to.
