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
For the locally authorized controlled rerun, also read
`tools/event_benchmark/CONTROLLED_RUN.md` (named routing/storage variant).
These instructions do not authorize paid inference, downloads, uploads or publishing;
use the user's authorized scope and budget.

## Choose the protocol before running

For this team's current 120-question rerun, use **unified-frame-120-v2** in
[the portable handoff](../../../tools/unified_benchmark/README.md) (path relative
from the repository root: `tools/unified_benchmark/README.md`). Its command-line
entrypoint and recorded model profiles are included in this branch. Download the
five frozen validation24/test120 JSON files from the restricted team Drive using
[the input acquisition and hash checks](../../../tools/unified_benchmark/README.md#get-the-frozen-inputs).
Use an authorized team account, keep the original bytes and filenames in one
private directory outside Git, and pass it as `--inputs`. Do not substitute the
repository's legacy manifest or change hashes to bypass a mismatch. The files,
videos, keys, weights and historical results are not included in the repository.
Start with offline preflight. For CLIP, SigLIP2, VideoJev and Decision3 use
[the native run guide](../../../tools/unified_benchmark/NATIVE.md): prepare and
verify the pinned deployment, then `native_run --phase validation|test|both`.
Without `--execute` the native CLI checks files only; approved execution runs the
original warmup/synthetic smoke before evaluation. Offline success is not GPU
acceptance. Preserve one/two-frame inputs, precision, prompts and three-attempt
budget; use explicit `--resume` for an interrupted run, never a fresh ledger to
reset that budget. The runner records deployment and threshold identities and
keeps historical results. Shared-sheet publication remains a separate request.

The current variant uses exact 1 FPS shared 720px JPEGs, current-image API prompts,
explicit effort settings when supported, and exact JSON string `"true"`/`"false"`
normalization (never general truthiness). Accuracy is correct/all planned questions
(120 for the whole test); missing questions remain in the denominator and are
reported separately. Every planned response must still be valid before a question
can contribute a correct answer. Native unknown is valid but wrong against the
binary references; a refusal or malformed answer is missing, not unknown.

CLIP/SigLIP2 retain single-frame margins. VideoJev/Decision3 retain adjacent
two-frame windows and native argmax; the proposed single-frame native experiment
was not run. Changing prompts, effort, windows, parser, thresholds or provider
creates a new recorded variant. Historical bundles need their retained source
snapshot; do not relabel their protocol or silently re-score old output files.

The legacy v2.1 API runner and robot41 workflow below remain separate protocols:
centered sampling and the original strict parser do not reproduce the current
120-question variant. Never mix their hashes or denominators in one result row.

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
   Categories, questions, answers and source details come from the selected samples.
   Video source does not change the scoring workflow. Keep each cohort separately
   identified; supplementary data does not extend an existing result block.
   Preserve source groups across train/development, validation and test splits:
   cuts from one recording must not cross splits. Missing onset/person annotations
   do not establish reaction-time or track-ID accuracy. Never assume a colleague's
   Desktop path exists.
   `--preflight` checks inputs offline without credentials or inference. Keep its
   pending-provider/hardware/authorization items distinct from input validity.
2. **Run accuracy from the repository root**, only after authorization for calls:
   ```sh
   python tools/event_benchmark/run_eval.py --candidate luna \
     --samples /path/to/samples.json --manifest /path/to/reviewed-manifest.json \
     --media /path/to/media1 /path/to/media2 \
     --provider-config /path/to/frozen-provider.json \
     --split-manifest /path/to/validation_split.json --split test \
     --interval 1.0 --workers 12 --out outputs/event-benchmark/luna-v2.json
   ```
   Repeat with qwen and gemini on identical inputs. Capture the source commit and
   result-file SHA-256 alongside the runner's samples/manifest/media/runner/spec
   hashes. Keep generated data in ignored `outputs/`, not tracked `results/`.
3. **Measure latency separately**, with the same input and question per candidate:
   ```sh
   python .claude/skills/event-benchmark/scripts/latency_bench.py \
     --candidate luna --video /path/to/representative-clip.mp4 \
     --provider-config /path/to/the-same-frozen-provider.json \
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

For the original default-provider v2.1 baseline without a validation24/test120
sidecar, add `--legacy-baseline` to the accuracy command and omit the controlled
provider/split options. It preserves centered sampling, the strict parser and
fixed gate, and records `v2.1-legacy-unpinned-lossless-v1`. Do not use that output
for validation threshold selection. The controlled workflow above still requires
the provider configuration and split sidecar. Keep supplementary cohorts separate
from existing result blocks and report their own full denominators.

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
