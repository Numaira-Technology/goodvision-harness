# Publication font and data-dependent colors

Applies to authorized GoodVision sheet publications. Documentation edits and
previews do not themselves authorize changing the live workbook. Preserve values,
formulas, notes, numeric formats, borders, alignment, bold emphasis, header fills,
ordinary backgrounds, grouping, freezes and dimensions except for the explicitly
requested content and the font/metric colors below.

## Font

Set every cell whose content is written to **Calibri, 9 pt**, including a new
header if creating one was explicitly requested. Set only the font family and
size fields; do not replace the entire text format. Do not restyle untouched
historical rows or existing headers merely because one model is updated. A
requested whole-table font change may include those cells within that scope.

## Comparison scope

Before coloring, identify the current comparable result block from live model/run
identities. Recalculate against all eligible rows in that block after the proposed
value update, not just the changed model. Include any resulting recoloring of
unchanged peer cells in the preview. Exclude historical sections, blank separators,
totals, unrun cohorts and nonnumeric or unknown entries.

Treat each metric column separately: Overall202, Overall120, Overall100, each event, P50 and
P95 do not share a scale. Match workload, timing boundary, concurrency and units
before comparing speed; split incompatible groups or leave them neutral with an
explanation. Costs are comparable only for the same workload/scope with complete
cost evidence; exclude unknown costs and lower-bound amounts such as `>= $...`.
Do not treat blank or unknown as zero. Do not rank threshold values or hardware
capacity by color: larger values are not inherently better.

## Scale defaults

Use a continuous **red (#E67C73) -> pale yellow (#FFF2CC) -> green (#4EAD70)**
palette for worse -> middle -> better, with readable dark text. Compute anchors
from unrounded stored values, after resolving percentage storage (0..1 or 0..100).

- **Accuracy:** current minimum = red, median = yellow, maximum = green. Interpolate
  continuously between anchors, so values such as 80.8 and 89.2 are not placed in
  the same fixed bucket. This is relative ranking within the displayed cohort,
  not an absolute pass/fail standard. Always retain the numeric value.
- **Latency and comparable actual cost:** minimum = green, median = yellow,
  maximum = red. If the positive maximum/minimum ratio is at least 10, interpolate
  in log space so an outlier does not flatten the other cells to one color; record
  that choice and the raw numeric anchors. Never apply a logarithm to zero. If
  zero occurs, use the ordinary linear scale and identify the outlier in the note.
- **Completeness:** 100% coverage and zero final missing/incomplete counts are
  always green. For positive missing counts, map the smallest positive value to
  pale yellow and the largest to red; interpolate values in between. For coverage
  below 100%, color by its deficit (100 minus coverage) in the same way. A single
  distinct nonzero deficit uses pale yellow, not green. These colors communicate
  relative deficit severity; the count/percentage remains the factual measure.

With fewer than two distinct values for an accuracy/speed/cost group, use neutral
pale yellow rather than inventing differences. If the median equals an endpoint,
use a two-endpoint gradient. Equal values receive equal colors. Recompute scales
on the next update; a color can change when peers change even if its number does
not. Keep matching cohorts consistent across tabs when publishing both together.

## Apply and check

Save the comparison ranges, excluded cells/reasons, scale direction, numeric
anchors and any logarithmic transform in the local publication audit. Include a
short scale description in the corresponding metric-header note without erasing
existing provenance. Do not add a legend column or new tab automatically.

Read existing conditional-format rules before applying colors. Use a scoped
gradient rule where it expresses the chosen mapping, or computed background
colors for a logarithmic/deficit scale. Existing conditional rules can override
background fills: replace or narrow only overlapping metric-color rules while
preserving their effect outside the authorized current result block. Do not
delete unrelated rules. Read back effective colors, font family/size and values
after publication; verify distinct low/high values differ and preserved content
and formulas have not changed.
