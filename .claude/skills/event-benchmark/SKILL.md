---
name: event-benchmark
description: >-
  Use when preparing or evaluating GoodVision event benchmarks, downloading the
  202-question dataset, or reporting model comparisons across surveillance events
  and bag-search tasks.
---

# Event-recognition benchmark

Use `tools/event_benchmark/SPEC.md` as the authority. Current checkout: **v2.1**,
with strict answers, complete frame evidence and reviewed input hashes. Historical
v1 outputs are not v2 evidence. Inspect the current spec and runner before a run.
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
Keep related views/cuts in the same split using `source_group`.

The original 120 questions are reviewed. The 82 derived views inherit source
labels and remain `pending_view_review`; preserve the combined draft status.
Media preparation does not require changing these labels or starting inference.

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
keeps completed files and reports individual failures without calling a model.
Keep credentials and temporary links outside Git; reference answers stay with the
offline scorer and never enter model requests.

After review and authorization for the existing input check, the scoring path is:

```sh
python tools/event_benchmark/download_media.py \
  --samples /path/to/samples.json --manifest /path/to/reviewed-manifest.json \
  --output /path/to/media --check-only
```

`--check-only` retains the evaluator's reviewed-manifest rules and checks only the
base-4 scoring cohort, not the complete 192-video download. It reports 182 base
questions for this version; bag questions remain separate. Downloading media does
not certify label review, scoring eligibility or model performance.

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
