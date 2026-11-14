---
name: event-benchmark
description: >-
  Use when preparing or evaluating GoodVision event benchmarks, downloading the
  202-question dataset, or reporting model comparisons across surveillance events
  and bag-search tasks.
  Preserves reviewed inputs, protocol versions and measured provenance.
---

# Event-recognition benchmark

For the legacy runner, use `tools/event_benchmark/SPEC.md` as the authority: **v2.1**,
with strict answers, complete frame evidence and reviewed input hashes. Historical
v1 outputs are not v2 evidence. Inspect the current spec and runner before a run.
For the controlled provider-pinned, lossless accuracy and latency variant, read
`tools/event_benchmark/CONTROLLED_RUN.md`. The implemented current202 pipeline and
legacy unpinned baseline remain separately identified protocols.
These instructions do not authorize paid inference, downloads, uploads or publishing;
use the user's authorized scope and budget.

## Current dataset: 202 questions, 192 videos

Use the [unified Drive folder](https://drive.google.com/drive/folders/1kOyH59vxBp3uppyEoeyMjTMUa_CalTdh).
Its current `samples.json`, `videos.json` and `manifest_202.json` combine the
original 120 questions with 82 single-view questions. The old `samples_120.json`
and `videos_120.json` are historical snapshots, not the current download list.

| Existing event key | Questions |
|---|---:|
| fight | 32 |
| smoking | 22 |
| fall | 62 |
| weapon | 66 |
| bag_knife | 10 |
| bag_alcohol_bottle | 10 |

Read all 202 entries from `samples` and group by `event`. The base-4 runner
reads 182 questions; the 20 bag questions retain their separate task and metric.
Video filenames deduplicate to 192 because the bag clips have two questions each.
Source and `view` fields are metadata, not new categories or model branches.
Keep related views/cuts from the same original recording in one split. Existing
`source_group` metadata alone does not establish reviewed train/validation/test
isolation: the current samples lack `source_video_group`, and the combined
manifest has no group mapping. Do not invent or silently rewrite these fields.

The original 120 questions are reviewed. The 82 derived views inherit source
labels and remain `pending_view_review`; preserve the combined draft status.
Media preparation does not require changing these labels or starting inference.
The combined set is not ready for threshold calibration: weapon questions retain
two different prompt protocols, and the added 82 views come from only three
original recordings (all 36 added fall views share one). Human visibility review,
reviewed source isolation and compatible frozen prompts remain separate gates.

1. **Preflight:** run `tools.unified_benchmark.preflight` for the selected profiles.
   Verify all eight frozen inputs and video hashes, decode frames and save the new
   request/code snapshot. Keep pending hardware/provider/authorization checks
   separate from input validity. This step performs no model calls.
2. **Environment:** follow the execution route below. Verify pinned versions,
   credentials/access and hardware; do the authorized original warmup/smoke before
   evaluation. Offline checks do not prove a new GPU deployment can run.
3. **Validation:** use validation24 by default. For a requested CLIP/SigLIP2
   [bag validation supplement](../../../tools/unified_benchmark/NATIVE.md#optional-clipsiglip2-bag-validation),
   use the frozen 44-question plan on 34 videos. Preserve
   receipts, complete scores, missing units and run identity.
4. **Thresholds, when applicable:** use `bench-threshold-tune` and the route's own
   selector for Luna/CLIP/SigLIP2; require complete validation, then freeze the
   artifact before test. Boolean/argmax models do not gain a scalar threshold.
5. **Test:** run the frozen configuration on all 202 questions. Preserve successful
   answers and the cumulative three-attempt budget across interruptions. For the
   four-model native CLI, use explicit `--resume`; other routes follow their own
   documented continuation support, never a fresh ledger to reset the budget.
6. **Local report:** verify counts, coverage, unknowns, source identities, matching
   speed and actual costs using [report fields](references/report-format.md).
   Retain raw receipts, frozen inputs/configs and historical results.

The [202 dataset guide](../../../tools/unified_benchmark/DATASET202.md) defines
questions1–120 unchanged plus82 single views in test, with 24 base validation questions.
Keep inherited/unreviewed labels explicit; preflight is not human review. Report
original120 and robot82 separately alongside correct/202, and never count the old41
stereo sources again. Do not tune on the new test questions.

The unified runner defaults to the separate `unified-frame-202-v1` protocol.
Follow [its input and reporting guide](../../../tools/unified_benchmark/DATASET202.md):
preflight uses all eight frozen inputs, retains pending reference-review status,
and reports original120/robot82 separately. Any combined202 result is provisional
reference agreement, not a reviewed benchmark. Tune only on the separately frozen
validation24 or requested CLIP/SigLIP2 validation44 supplement.
The historical120 hashes, plans and denominators below remain unchanged; select
them explicitly with `--protocol unified-frame-120-v2`. Do not edit historical
hashes or relabel old results to fit the new protocol.

## Historical 120-question execution contract

For reproduction of the team's frozen 120-question runs, use **unified-frame-120-v2** in
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

This branch's self-hosted Qwen3-VL-8B HF path is profile `qwen8b_r2_host`
(`execution=self_hosted_r2`): restore the packed env and pinned weights from R2,
serve with `tools.qwen8b.serve`, and score with `tools.qwen8b.eval_local` through
`tools.start_qwen8b_r2`. Do not use a vLLM `qwen8b_selfhost` profile or the
OpenRouter `qwen8b` Parasail cell for that comparison.

An independently authorized serving engine is profile `qwen8b_r2_vllm` via
`tools.start_qwen8b_vllm` (vLLM OpenAI server, same R2 weights revision, bf16).
The source branch declares `envs/qwen8b-vllm/cu128-py312-vllm0310-ada.tar.gz`.
#98 records that object as missing `typing_extensions` in the prefix. The ada2
pack and its exam do not validate this pin.
This published identity is frozen; rebuilds require a new reviewed key and actual
artifact/runtime evidence. The launcher checks the active source profile and
sealed request bundle before rental and again before inference. A separate
synthetic 720px smoke is retained outside the scoring ledger. Preserve historical
bundles; changed active profiles require fresh bundles. This path does not relabel
the HF run, and CPU imports do not establish GPU acceptance. Frozen JPEGs, prompts,
parser, temperature, max_tokens=30, thresholds, retries and 12 queue slots remain
unchanged. Do not shrink inputs or reduce concurrency to fit a GPU.

This historical variant uses exact 1 FPS shared 720px JPEGs, current-image API prompts,
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
centered sampling and the original strict parser do not reproduce the frozen
120-question variant. Never mix their hashes or denominators in one result row.

## Workflow

1. **Freeze the inputs before scoring.** Identify the candidate, dataset, reviewed
   manifest, media directories and output path. Verify manifest review status,
   samples hash, question/label identities and media hashes. The runner does this
   before requests. Do not change a draft status to bypass review.
   For the current set, obtain `manifest_202.json` and its matching `samples.json`
   from the unified Drive entry below. Committed manifests in
   `tools/event_benchmark/datasets/` describe historical cohorts.
   Read review status from the selected manifest and its cohort metadata. The
   current set has 182 base-event questions and 20 bag-search questions; the
   historical 120 set has 100 and 20 respectively. Question count is not video count.
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
   Controlled latency does not turn the legacy accuracy output into a controlled run.
4. **Report** using `references/report-format.md`, including coverage and provenance.
   Use `bench-sheet-contribute` only when publishing is requested. Its default
   preparation is an offline preview, with no sheet write.

## Download media when needed

Download **all 192 videos** from the unified media-only `videos.json`; do not
filter this list by the four scored events or append the old stereo cohort.
On the machine storing media (the GPU machine when using Vast), use an existing
rclone Drive remote. Replace `TEAM_DRIVE` with its configured name:

```sh
mkdir -p outputs/event-benchmark/202
rclone backend copyid TEAM_DRIVE: 1cJhdVSiiVBOId6M6cybBJctNyJbS4e6u \
  outputs/event-benchmark/202/videos.json
python tools/event_benchmark/download_media.py \
  --videos outputs/event-benchmark/202/videos.json \
  --output outputs/event-benchmark/202/videos --drive-remote TEAM_DRIVE
```

On the offline scoring host, obtain the matching question and manifest files:

```sh
mkdir -p outputs/event-benchmark/202
rclone backend copyid TEAM_DRIVE: 1xzpCEgKtoQ26NbNOMc-85KxPiogjIzXr \
  outputs/event-benchmark/202/samples.json
rclone backend copyid TEAM_DRIVE: 1Hj8TjxeIeN9AH5bIZjww4BbjieojMz6S \
  outputs/event-benchmark/202/manifest_202.json
```

The downloader accepts `url`/`video_url` Drive references or `drive_file_id`, with
optional `size_bytes`. Existing `--samples`, `GOOGLE_DRIVE_ACCESS_TOKEN` and private
`--links` maps remain supported. It downloads unique filenames from every category,
retains existing nonempty files (also requiring the listed size when supplied),
and reports transfer failures without calling a model. These size checks do not
verify file identity or media hashes. Explicit temporary links take precedence,
then `--drive-remote`, then the environment token. Remote names refer only to an
existing configuration; the downloader does not configure authentication.
Keep credentials and temporary links outside Git; reference answers stay with the
offline scorer and never enter model requests.

After review and authorization for the existing input check, the scoring path is:

```sh
python tools/event_benchmark/download_media.py \
  --samples /path/to/samples.json --manifest /path/to/reviewed-manifest.json \
  --output /path/to/media --check-only
```

`--check-only` retains the evaluator's reviewed-manifest rules and checks only the
base-4 scoring cohort, not the complete 192-video download. The current draft
manifest is rejected before counts are reported. Once all input gates are met,
this set would contain 182 scored base questions; bag questions remain separate.
The retained `samples_120.json` is semantically historical but does not match the
exact samples hash pinned by the committed 120-question manifest. Reproduction
requires recovering those original bytes, not replacing the frozen hash. Downloading
media does not certify label review, scoring eligibility or model performance.

For the original default-provider v2.1 baseline without a validation24/test120
sidecar, add `--legacy-baseline` to the accuracy command and omit the controlled
provider/split options. It preserves centered sampling, the strict parser and
fixed gate, and records `v2.1-legacy-unpinned-lossless-v1`. Do not use that output
for validation threshold selection. The controlled workflow above still requires
the provider configuration and split sidecar. Keep supplementary cohorts separate
from existing result blocks and report their own full denominators.

```sh
python tools/event_benchmark/run_eval.py --legacy-baseline --candidate qwen \
  --samples /path/to/samples.json --manifest /path/to/reviewed-manifest.json \
  --media /path/to/media --interval 1.0 --workers 12 \
  --out outputs/event-benchmark/legacy-qwen.json
```

## Reviewed robot videos: 41-question supplement

Use `tools/event_benchmark/datasets/samples_robot41.json` with
`tools/event_benchmark/datasets/manifest_robot41.json`. These are the 41 clips
Pengyu reviewed on November 3, numbered 121–161; they do not replace or silently
extend the original 120-question cohort. Videos remain in the
[team Drive folder](https://drive.google.com/drive/folders/132wI2o9nLno7a_IGw7IG7zLJabBqTYxW).

| Category | Reference yes | Reference no | Questions |
|---|---:|---:|---:|
| fall | 9 | 9 | 18 |
| weapon | 11 | 12 | 23 |

Preserve the supplied questions and answers. Here `weapon` asks about visibly
holding or carrying a knife or weapon; it does not require an attack. The confirmed
beverage bottle and pen are `no`. Walking away, returning and crossing laterally
are separate routes; two near-camera knife displays remain separately identified.
Lateral weapon coverage still lacks two yes and one no clips for the requested
three-of-each quota. Clips share three source recordings: do not split cuts from
the same `source_group` between threshold tuning and evaluation. There are no
reviewed natural-onset times or person-ID labels; do not claim reaction-time or
track-ID accuracy from this supplement.

Prepare media on the machine that will run inference; on Vast, download there
directly rather than staging videos through a laptop. Use an existing Google
account with access to the folder. The downloader accepts a read-only Drive OAuth
token in `GOOGLE_DRIVE_ACCESS_TOKEN`, or `--links` pointing to a private JSON map
of filenames/Drive IDs to temporary HTTPS download URLs. A connected Drive fetch
can supply those URLs. Obtain them after environment preparation, refresh only
expired entries, and keep the map and credentials outside Git. Private Drive
view links are not anonymous media download URLs.

```sh
python tools/event_benchmark/download_media.py \
  --samples tools/event_benchmark/datasets/samples_robot41.json \
  --manifest tools/event_benchmark/datasets/manifest_robot41.json \
  --output outputs/robot41/media --links /path/to/private-temporary-links.json
python tools/event_benchmark/download_media.py \
  --samples tools/event_benchmark/datasets/samples_robot41.json \
  --manifest tools/event_benchmark/datasets/manifest_robot41.json \
  --output outputs/robot41/media --check-only
```

The second command uses the same input loader as the evaluator and should report
41 questions, 41 videos, `fall: 18`, `weapon: 23`, and zero model requests. It does
not start a model. For an authorized accuracy run, use the explicit cohort:

```sh
python tools/event_benchmark/run_eval.py --legacy-baseline --candidate qwen \
  --samples tools/event_benchmark/datasets/samples_robot41.json \
  --manifest tools/event_benchmark/datasets/manifest_robot41.json \
  --media outputs/robot41/media --interval 1.0 --workers 12 \
  --out outputs/event-benchmark/robot41-qwen.json
```

This explicitly unpinned baseline retains centered sampling, strict parsing and
the fixed gate. It records lossless received scores under
`v2.1-legacy-unpinned-lossless-v1`, not controlled provider/split evidence, and cannot
drive the controlled threshold selector. Historical rounded outputs remain
unchanged and ineligible. Use `download_media.py --check-only` above for offline
input verification, or the runner's `--preflight` without authorizing inference.

Report this as a separate robot41 cohort with full denominators 18, 23 and 41.
The existing Sheet's 50/100/120 blocks are not robot41 result cells. Availability
and reviewed labels do not establish model recognition accuracy; no model result
is supplied with these annotations. Other backends retain their existing runner
boundaries described below.

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
