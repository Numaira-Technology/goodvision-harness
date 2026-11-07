---
name: event-benchmark
description: >-
  Prepare, run and report the surveillance event benchmark and reviewed robot-video
  supplement (fight/smoking/fall/weapon).
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
   `manifest_50.json` is reviewed; merged PR #73 records `manifest_120.json` as
   `human_reviewed` with all 120 rows `user_reviewed`. It includes 100 base-event
   questions plus 20 separate bag-search questions. These recorded flags do not
   independently verify owner approval, source grouping or fresh run provenance;
   no `source_video_group` values are present in that manifest.
   Question count is not unique-video count. Larger sets require their own reviewed
   manifest. Never assume a colleague's Desktop path exists.
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
python tools/event_benchmark/download_media.py --output outputs/robot41/media \
  --links /path/to/private-temporary-links.json
python tools/event_benchmark/download_media.py --output outputs/robot41/media --check-only
```

The second command uses the same input loader as the evaluator and should report
41 questions, 41 videos, `fall: 18`, `weapon: 23`, and zero model requests. It does
not start a model. For an authorized accuracy run, use the explicit cohort:

```sh
python tools/event_benchmark/run_eval.py --candidate qwen \
  --samples tools/event_benchmark/datasets/samples_robot41.json \
  --manifest tools/event_benchmark/datasets/manifest_robot41.json \
  --media outputs/robot41/media --interval 1.0 --workers 12 \
  --out outputs/event-benchmark/robot41-qwen.json
```

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
  Explicitly authorized `--provisional` diagnostics use `base4-provisional-v1`
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
