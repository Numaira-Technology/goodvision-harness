# Numeric final-answer representations

The human approved accepting JSON numbers for numeric questions and locally
reparsing the saved 0061 duration reply on 2026-10-19. That reply was
`{"prediction":43}` with a completed provider response, but the summary adapter
rejected it because it required a string. The unchanged PR63 scoring adapter
already accepts numeric final answers. This was a parser contract defect, not
an HTTP failure or an output-token limit.

## Bounded change

`parse_summary_answer` is shared by live summary execution and authorized local
reparsing. In `single_prediction_v2` and `short_predictions_v1`, it accepts a
finite JSON integer or float when the public `answer_type` is `numeric_count`
or `numeric_seconds`. It preserves the value and numeric type without rounding,
string extraction or correctness filtering. Zero remains an answer.

Strings retain their existing meaning, including empty or whitespace-only
strings as insufficient evidence. Option questions still require strings.
Booleans, null, arrays, objects, nonfinite numbers and extra envelope fields
remain invalid. The legacy status/answer/reference contract is unchanged.
Truncation is still rejected before parsing; this change does not salvage a
truncated reply. The model prompt and output budget do not change.

## Historical recovery boundary

Only the specifically approved saved 0061 response is reparsed in this action.
Read the original request, response and outcome; require the completed reply and
matching question; use the shared parser; publish a separate derived outcome
with the original receipt and source paths; then run the unchanged local scorer
into a new score file. Do not call the summary dispatcher or ledger completion
method, overwrite the original files, or replay other invalid replies.

The original answer remains 43 seconds; the original ground truth is 70 seconds.
Accepting its representation does not make it correct. The absolute error is
27 seconds and the existing scorer determines its quality score.

No unit tests, test edits, lint, syntax checks or builds are authorized. The
approved one-response local reparse and scoring are real result processing,
not a substitute claim that a test suite passed.
