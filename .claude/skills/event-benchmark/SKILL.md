---
name: event-benchmark
description: >-
  Prepare, run and report the surveillance event benchmark (fight/smoking/fall/weapon).
  Use for benchmark/model comparisons, 跑一下测评, 测速, or preparing benchmark tables.
  Preserves reviewed inputs, protocol versions, fixed baselines and measured provenance.
---

# Event-recognition benchmark

Use `tools/event_benchmark/SPEC.md` as the authority. Current checkout: **v2.1**,
with strict answers, complete frame evidence and reviewed input hashes. Historical
v1 outputs are not v2 evidence. Inspect the current spec and runner before a run.
These instructions do not authorize paid inference, downloads, uploads or publishing;
use the user's authorized scope and budget.

## Workflow

1. **Freeze the inputs before scoring.** Identify the candidate, dataset, reviewed
   manifest, media directories and output path. Verify manifest review status,
   samples hash, question/label identities and media hashes. The runner does this
   before requests. Do not change a draft status to bypass review.
   Select the reviewed manifest from `tools/event_benchmark/datasets/` and obtain
   its exact matching samples file; not every samples file is committed there.
   Read review status from the selected manifest. The reviewed 120-question set
   includes 100 base-event questions plus 20 separate bag-search questions; the
   base runner scores only the former. Question count is not unique-video count.
   PR #73 records all 120 rows as `user_reviewed`, but these flags alone do not
   independently verify owner approval, source grouping or fresh run provenance;
   no `source_video_group` values are present in that manifest. The separate
   reviewed validation/media manifest and compatible lossless calibration/test
   evidence are still required by the threshold-study acceptance hold.
   Categories, questions, answers and source details come from the selected samples.
   Video source does not change the scoring workflow. Keep each cohort separately
   identified; supplementary data does not extend an existing result block.
   Preserve source groups across train/development, validation and test splits:
   cuts from one recording must not cross splits. Missing onset/person annotations
   do not establish reaction-time or track-ID accuracy. Never assume a colleague's
   Desktop path exists.
2. **Run accuracy from the repository root**, only after authorization for calls:
   ```sh
   python tools/event_benchmark/run_eval.py --candidate luna \
     --samples /path/to/samples.json --manifest /path/to/reviewed-manifest.json \
     --media /path/to/media1 /path/to/media2 \
     --interval 1.0 --workers 12 --out outputs/event-benchmark/luna-v2.json
   ```
   Repeat with qwen and gemini on identical inputs. Capture the source commit and
   result-file SHA-256 alongside the runner's samples/manifest/media/runner/spec
   hashes. Keep generated data in ignored `outputs/`, not tracked `results/`.
3. **Measure latency separately**, with the same input and question per candidate:
   ```sh
   python .claude/skills/event-benchmark/scripts/latency_bench.py \
     --candidate luna --video /path/to/representative-clip.mp4 \
     --concurrency 12 --rounds 6 --out outputs/event-benchmark/lat-luna.json
   ```
   This is single-frame, single-predicate request latency, not product latency or
   twelve-camera throughput. Inspect failures and completion status before reporting.
4. **Report** using `references/report-format.md`, including coverage and provenance.
   Use `bench-sheet-contribute` only when publishing is requested. Its default
   preparation is an offline preview, with no sheet write.

## Download media when needed

Use the selected samples/manifest pair for every source. The downloader handles
samples with `filename`, `size_bytes` and a `drive_file_id` (or a filename-keyed
download link). For other source schemas, obtain the exact media through the
source's authorized transfer method; do not invent metadata or change frozen files
to fit the downloader. On Vast, download directly to the GPU machine. Private
Drive files need `GOOGLE_DRIVE_ACCESS_TOKEN` or a private `--links` JSON map of
filenames/Drive IDs to temporary HTTPS download URLs; a Drive view link is not a
direct download URL. Keep credentials and temporary link maps outside Git.

```sh
python tools/event_benchmark/download_media.py \
  --samples /path/to/samples.json --manifest /path/to/reviewed-manifest.json \
  --output /path/to/media \
  --links /path/to/private-temporary-links.json
python tools/event_benchmark/download_media.py \
  --samples /path/to/samples.json --manifest /path/to/reviewed-manifest.json \
  --output /path/to/media --check-only
```

`--check-only` works with every evaluator-compatible samples/manifest pair and
uses the evaluator's input loader to verify reviewed labels and hashes of existing
local media. It reports scored video/question counts by category without downloading
or calling a model. A download's size check is not frozen-hash verification; run
`--check-only` after transfer, then use the accuracy command above with the same files.

## Frozen v2 baseline

- Same questions from `samples.json`, same media, 1 fps centered frame schedule,
  720px width, one independent request per frame; output envelope fixed by backend.
- Language: complete JSON boolean. Luna: finite numeric probability in [0,1].
  No truthiness coercion of strings/numbers, no invalid response treated as negative.
- Clip score = maximum of **all scheduled valid frames**; `score >= 0.5` is yes.
  Any missing frame or invalid answer after three total attempts makes the clip an
  error, even if another frame was positive.
- Accuracy is correct/all questions, per event and overall. Errors and missing
  answers stay in the denominator and are reported separately, never as negatives.
- `--dense` or changed cadence is a separately labeled sensitivity experiment.
  Tuning is a separate variant; see `bench-threshold-tune`.
- Dataset growth changes the dataset fingerprint and comparison cohort. It does not
  itself change the sampling spec; rerun every compared candidate on the same set.
- Provisional labels may be described as historical/exploratory evidence only, never
  as reviewed v2 results. No new official run until the manifest is reviewed.
  Explicitly authorized `--provisional` diagnostics use `base4-provisional-v2`
  (`tools/event_benchmark/PROVISIONAL.md`), support only binary base-4 questions,
  and record rather than verify input hashes. They cannot be used for v2 tuning.
  Bag yes/no/unknown questions require a separate reviewed scoring protocol;
  never collapse unknown into no or remove the third answer to fit this runner.
- No negatives means specificity unavailable, not zero false positives.

## Other model classes

`run_eval.py` currently supports language and Luna decision backends. Native
VideoJev and CLIP have different protocols. Read `references/model-classes.md`
before importing their results; shared formatting does not establish a controlled
comparison. Do not coerce margins or native verdicts into Luna probabilities.

## Files

- `references/methodology.md`: input, scoring and provenance checklist.
- `references/model-classes.md`: runner boundaries and cross-class reporting.
- `references/report-format.md`: columns and coverage notes.
- `scripts/latency_bench.py`: separate request latency measurement.
