# Eval field semantics (resolve positions from live headers)

Target: `eval-2026-11-04`, sheetId 1417937452, in the spreadsheet named by SKILL.md.
The layout has changed several times. Older A:AF mappings, separate Hardware/VRAM,
rate-based Price and success-only denominator formulas are historical snapshots,
not a current write contract. Never overwrite a cell based on those column numbers.

| Logical field | Required meaning |
|---|---|
| Model | Model and selected effort; full protocol/variant identity in notes |
| Backend / Provider | Backend class and provider observed in actual receipts |
| P50 / P95 | Seconds; record same-run source, concurrency, sample count and timing boundary |
| Overall 120 | Correct / 120 planned questions; missing stays in denominator |
| Base100 | Same run's base-four subset, correct / 100; preserve the collapsed group |
| Fight / Smoking / Fall / Weapon | Planned class denominators 32 / 22 / 26 / 20 |
| Bag | 10 knife + 10 alcohol-bottle questions contribute to Overall, never Weapon |
| Complete / correct counts | Complete-valid questions and correct questions remain distinct |
| Hardware | Combined hardware/capacity when present: RTX 4090 - 24G |
| Price | Actual spend USD and scope; not USD per million tokens |
| FPS / Concurrency | 1 FPS; nominal 12 slots; any recovery difference recorded in notes |
| Precision | Actual model computation dtype, score dtype and API disclosure limits |
| Run date | Accuracy-run date; speed has its own provenance/date if retained |
| Unique send coverage | Units with durable transmission intent / all planned units |
| Final valid coverage | Unique units with all valid target answers / planned units |
| Final missing | Unique units still lacking valid target responses after the cap |
| Incomplete questions | Questions missing any planned target evidence |
| Last updated by | Actual editor; do not assume a teammate is Haotian |

Complete evidence is required before a question contributes a correct answer.
For a complete question: any yes -> yes; otherwise any no -> no; all unknown ->
unknown. Native/bag unknown is valid but incorrect against this binary reference.
Refusal, truncated or malformed output is missing. Exact string true/false may
be normalized only under the recorded boolean-string-v1 parser, never truthiness.

Three attempts means initial + two retries, cumulative across resumptions.
Successful answers are immutable. Final missing is not a historical attempt count.
With 120 questions and no valid completions, headline accuracy is 0/120, not blank.
Only an empty cohort has no accuracy denominator. Show completion and accuracy
separately, and check formulas rather than merely replacing the displayed number.

Inspect actual numeric storage and formatting before writing percentages. Do not
assume 0..1 versus 0..100; derive from live values/formulas/number formats.
Keep a same-run base100 block hidden/collapsed where already present. Do not add
columns for bag or completeness when notes are the agreed destination.
