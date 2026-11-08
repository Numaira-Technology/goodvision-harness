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

This branch's self-hosted Qwen3-VL-8B HF path is profile `qwen8b_r2_host`
(`execution=self_hosted_r2`): restore the packed env and pinned weights from R2,
serve with `tools.qwen8b.serve`, and score with `tools.qwen8b.eval_local` through
`tools.start_qwen8b_r2`. Do not use a vLLM `qwen8b_selfhost` profile or the
OpenRouter `qwen8b` Parasail cell for that comparison.

An independently authorized serving engine is profile `qwen8b_r2_vllm` via
`tools.start_qwen8b_vllm` (vLLM OpenAI server, same R2 weights revision, bf16).
Pack that env on a CPU VM onto `envs/qwen8b-vllm/cu130-py312-vllm0310-ada.tar.gz`
(torch `+cu130`; keep the earlier `cu128-…vllm0310` object). The 4090 only
restores it. Do not overwrite the HF qwen8b pack. CPU import ≠ GPU serve. It does not
replace or relabel the HF run. Unified-frame-120-v2 exam rules stay
exactly as below: frozen JPEGs, prompts, parser, temperature, max_tokens=30,
thresholds, retries, and **12 queue slots**. Do not lower client concurrency,
shrink images, truncate context, or shorten outputs to fit a GPU.

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
   Read each manifest's actual status and fingerprint; `manifest_120.json`
   includes 100 base-event questions plus 20 separate bag-search questions.
   Question count is not unique-video count. Larger sets require their own reviewed
   manifest. Never assume a colleague's Desktop path exists.
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
python tools/event_benchmark/run_eval.py --legacy-baseline --candidate qwen \
  --samples tools/event_benchmark/datasets/samples_robot41.json \
  --manifest tools/event_benchmark/datasets/manifest_robot41.json \
  --media outputs/robot41/media --interval 1.0 --workers 12 \
  --out outputs/event-benchmark/robot41-qwen.json
```

`--legacy-baseline` explicitly uses the original default-provider v2.1 baseline
without the validation24/test120 sidecar. It preserves the centered sampling,
strict parser and fixed gate, and records `v2.1-legacy-unpinned-lossless-v1`.
Do not mix it with controlled provider/split options or use its output for
validation threshold selection. The controlled workflow above still requires
the provider configuration and split sidecar. Add `--preflight` for offline checks.

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
