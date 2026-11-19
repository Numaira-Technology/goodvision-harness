For current-frame202 and explicit legacy120, use `tools/unified_benchmark/README.md`.
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

The current unified Drive set has 202 questions and 192 videos: 182 base-4
questions and 20 bag-search questions. The original 120 questions remain reviewed;
the 82 added views inherit source labels and await separate-view review. Download
the entire media-only videos.json before filtering questions for a scoring task.
The historical 120 set contains 100 base-4 questions and 20 bag questions. Read
the selected manifest's status; do not relabel a draft to enable a run. Bag
questions allow unknown and retain their separate versioned task/metric; the
base runner filters them out. The current draft cannot pass the runner's input
check. Downloads and inherited labels do not make it
eligible for reviewed scoring or threshold calibration.

Keep both views and other cuts from the same original recording in one split.
The current `source_group` metadata is not a reviewed `source_video_group` mapping;
the combined manifest lacks group fields. The 82 views come from three recordings,
including one recording for all 36 added fall views. Weapon questions retain two
prompt protocols, which must not be silently combined or rewritten for tuning.
The retained historical samples also do not match the exact hash pinned by the
committed 120 manifest. Recover the frozen bytes rather than changing that hash.

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
