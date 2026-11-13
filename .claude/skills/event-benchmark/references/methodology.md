For the current-frame 202-question variant, use `tools/unified_benchmark/README.md`.
The procedure below describes the retained legacy runner.

# Legacy v2/v2.1 benchmark methodology

Authority: `tools/event_benchmark/SPEC.md` and `run_eval.py`, currently v2.1.
`SPEC_V1.md`, `RESULTS.md` and checked-in result JSONs are historical v1 artifacts.
They cannot establish v2 accuracy or calibrate a v2 variant without fresh evidence.

## Inputs and provenance

Before any model request, require a `human_reviewed` manifest matching the exact samples
file and every scored media file. Pass `--manifest` explicitly. Missing media,
ambiguous filenames, changed questions/labels and hash mismatches are blockers.
`--checksums` is a compatibility flag: hash checking is always enforced.

Record dataset ID, samples/manifest/media hashes, source commit, runner/spec hashes,
model/backend/provider route, configuration, run timestamp, environment versions
and result-file hash. A count such as 50 or 100 is not a dataset identity. Record
question count and distinct-media count separately. Do not send labels or reference
answers to the model. Private media, keys and raw outputs stay outside Git.

Read the actual reviewed status and hash of each manifest. The 120 cohort has 100
base-4 questions and 20 bag-search questions. Review status must be resolved by
reviewers, not relabeled by an agent to enable a run. Bag questions allow unknown
and require a separate versioned task/metric; the base runner filters them out.

The legacy reviewed-scoring route still rejects the combined draft202 manifest.
The separate unified202 variant explicitly reports provisional reference agreement,
not a human-reviewed benchmark. Related views/cuts must stay in one split; a group
name alone does not prove reviewed source isolation. All 36 added fall views share
one recording. Original and robot weapon references retain different semantics;
a frozen current-image prompt does not establish relabeling equivalence. Do not
silently combine those references for calibration. The retained public historical
samples also do not match the exact hash pinned by the committed120 manifest.
Recover the matching frozen bytes rather than changing that hash.

## Frozen v2 procedure

- Same sample question for every candidate, stripping trailing `Answer yes or no.`.
- Language emits `{"answer": true/false}`; complete response and a real JSON boolean
  are required. Luna emits error-free `noul` with a finite numeric [0,1] value.
  Strings, booleans-as-probabilities, duplicate fields and malformed outputs fail.
- 1 fps: `n=max(1,round(duration/1.0))`, centered `t_i=duration*(i+0.5)/n`;
  JPEG width 720, height even. One independent call per frame.
- Temperature 0, chat max_tokens 30, timeout 90s, **three total attempts** per frame.
- All scheduled frames must decode and receive valid answers. Any failure makes
  the whole clip an error; partial positive evidence cannot rescue it.
- Clip max score `>= 0.5` is yes for both implemented backends. The uniform 0.5
  result remains the baseline; held-out tuning is a separate labeled variant.
- Default 12 workers is accuracy-run concurrency, not a measured latency result.

## Denominators and comparison

For each event: `valid=TP+FN+TN+FP`, `correct=TP+TN`,
`total=valid+errors`. Report `correct/total`, errors/total and coverage; use the
same all-question denominator overall. Errors and missing answers contribute no
correct answer and remain in the denominator. With questions but no valid answers,
accuracy is zero; only an empty cohort has unavailable accuracy. When negatives
are absent, specificity is unavailable. Error clips are not true negatives.

CLIP unknown handling and native VideoJev window scoring may differ. Preserve
those rules and label the cohort, unit, denominator and timing boundary. Do not
copy their OVERALL into a v2 base-event cell without a verified equivalent mapping.
Changing prompts, cadence, aggregation, parser or error policy changes the protocol;
changing samples or labels changes the dataset fingerprint. Both require new,
explicitly identified comparison evidence.

PR #61's explicit keep-unmerged acceptance hold remains separate from the current
unified workflow. It still requires its reviewed validation/media-hash manifest,
source-group/isolation evidence and fresh compatible lossless calibration/test
outputs. Legacy `--provisional` diagnostics record hashes without certifying review
or freeze status, and must not be described as eligible calibration evidence.
