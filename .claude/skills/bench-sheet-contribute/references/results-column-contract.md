# Results field semantics (resolve positions from live headers)

Target: `results`, sheetId 0, in the spreadsheet named by SKILL.md. Historical
sections and current rows may repeat model names; identify the requested run by
model/effort + protocol + dataset + prompt + date/source artifact. Prior rows25–34
and A:Z maps are history, not current positions.

Match Speed P50/P95, Dataset50, Dataset100, Dataset all120/all202, Hardware, Price and
metadata by both header levels. Dataset100, original120 and all202 must come from the same
run. Leave Dataset50 blank unless that specific cohort was actually evaluated;
never splice in a previous run. Preserve existing folded groups; metric colors
follow [publication formatting](publication-formatting.md).

Current202 headline is correct / 202, with all planned questions in the denominator.
Its six event denominators are32/22/62/66/10/10. Keep original120/robot82 summaries
and the inherited, pending-review status of robot82 explicit. Do not place202
results in an all120 block: use same-run original120 there, or obtain a scoped
header change before publishing202. No automatic new columns/tabs. Historical120
keeps denominator120; original Base100 uses100 and base counts32/22/26/20. Bag knife and bag bottle
contribute ten each to Overall and are separate from Weapon. Missing questions
contribute no correct answer and remain in the denominator.

Complete evidence is required before a question contributes a correct answer.
For a complete question: any yes -> yes; otherwise any no -> no; all unknown ->
unknown. Native/bag unknown is valid but incorrect against this binary reference.
Refusal, truncated or malformed output is missing. Exact string true/false may
be normalized only under the recorded boolean-string-v1 parser, never truthiness.
Keep the parser, selected effort and full run identity explicit.

Three attempts means initial + two retries, cumulative across resumptions.
Successful answers are immutable. Final missing is not a historical attempt count.
With 202 questions and no valid completions, headline accuracy is 0/202, not blank.
Only an empty cohort has no accuracy denominator. Show completion and accuracy
separately, and check formulas rather than merely replacing the displayed number.

If there are no dedicated Provider/precision/completeness columns, retain these
fields in model/backend/overall notes as appropriate; do not invent new columns.
Keep valid unknown, final missing units and incomplete questions separate.
Record complete-valid questions separately from correct questions. Unique send
coverage means units with durable transmission intent / all planned units; it does
not prove server receipt. Final valid coverage means unique units with all valid
target answers / planned units. Final missing means unique units lacking valid
target answers when the report is made (including unattempted units in a paused
run), and incomplete questions means questions missing any planned target evidence.
Distinguish exhausted retries from an earlier stop. Retain these definitions in notes where no
dedicated columns exist; retirement of another tab does not drop the metrics.
Hardware may combine capacity in one cell; Price is actual USD spend with scope
and receipt limitations, not old provider input/output quotes.

FPS is 1 and nominal concurrency is 12 slots for this protocol; record recovery
differences in notes. Record actual computation and score dtypes, or API disclosure
limits. Run date is the accuracy-run date; retained speed has separate provenance
when different. Last updated by is the actual editor, not a hardcoded teammate.

Inspect actual numeric storage and formatting before writing percentages. Do not
assume 0..1 versus 0..100; derive from live values/formulas/number formats. Keep the
same-run base100 block hidden/collapsed where already present. Preserve historical
rows, formulas and formatting except for the requested publication changes. The
legacy 23-cell validator and A:Q config cannot validate the current live layout.
