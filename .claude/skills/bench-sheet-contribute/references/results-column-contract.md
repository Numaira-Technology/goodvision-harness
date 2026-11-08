# Results field semantics (resolve positions from live headers)

Target: `results`, sheetId 0, in the spreadsheet named by SKILL.md. Historical
sections and current rows may repeat model names; identify the requested run by
model/effort + protocol + dataset + prompt + date/source artifact. Prior rows25–34
and A:Z maps are history, not current positions.

Match Speed P50/P95, Dataset50, Dataset100, Dataset all120, Hardware, Price and
metadata by both header levels. Dataset100 and all120 must come from the same
run. Leave Dataset50 blank unless that specific cohort was actually evaluated;
never splice in a previous run. Preserve existing folded groups and color rules.

Current120 headline is correct / 120, with all planned questions in the denominator.
Base100 uses 100; the four base classes use 32/22/26/20. Bag knife and bag bottle
contribute ten each to Overall and are separate from Weapon. Missing questions
contribute no correct answer and remain in the denominator. Apply the same current
semantics as [eval](eval-column-contract.md), including explicit parser and effort.

If there are no dedicated Provider/precision/completeness columns, retain these
fields in model/backend/overall notes as appropriate; do not invent new columns.
Keep valid unknown, final missing units and incomplete questions separate.
Hardware may combine capacity in one cell; Price is actual USD spend with scope
and receipt limitations, not old provider input/output quotes.

Updating eval does not update this tab automatically. Preserve historical rows,
formulas and formatting unless the user explicitly targets them. The legacy
23-cell validator and A:Q config cannot validate this sheet's current layout.
