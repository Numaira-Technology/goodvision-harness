# Final-answer maintenance rules

- Final answers follow the original question: an option, number, timestamp, time interval, or short text. Do not require explanations, copied observations, or generated evidence-reference lists. Keep audit provenance locally. Fix an unnecessarily verbose response contract before increasing the output budget.
- Input compression and output truncation are separate problems. Reuse the existing lossless codecs; preserve every observation, explicit negative, unknown, timestamp, identity relation, and source ordering. Do not silently import historical lossy fallbacks.
- Preserve the cumulative request ledger and historical answers. A new prompt or encoding must not bypass a previously dispatched answer, including an invalid response or unknown receipt. Do not change scoring to salvage truncated replies without explicit authorization.
- Tests, test edits, lint, type checks, syntax checks, builds, and health probes require explicit task-specific human authorization. Read-only source and diff review is permitted. Do not claim tests passed when they were not run.
- Code, prompts, comments, published documentation, and commit messages are English. Preserve original questions, options, matching terms, observations, and answers verbatim.
