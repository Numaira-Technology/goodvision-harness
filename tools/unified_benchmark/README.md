# Current-frame 120-question evaluation handoff

Protocol: **unified-frame-120-v2**. This packages the previously local workflow for
another checkout. It does not modify the original videos, questions or references,
re-run models, or reinterpret old bundles. Keep historical source snapshots with
historical artifacts. The original `tools/event_benchmark/run_eval.py` remains the
separate legacy v2.1 runner; it is not the current-frame 120-question entrypoint.

## What is included

- Mandatory byte hashes for the reviewed validation24/test120 inputs. The five
  frozen JSON files are private team inputs, not published in this public repository.
  Obtain the original files from the evaluation owner; do not regenerate their labels.
- Eleven recorded profiles: qwen8b, qwen235b, maverick, luna, gemini25_off,
  gemini25_low, gemini38_low, clip, siglip2, videojev, decision3.
- Offline input/media/frame/request preflight; lossless receipt/score retention;
  cumulative three-attempt queue; validation threshold selection and test reporting.
- API execution with an explicit selected credential file or per-model file mapping.
  No credentials, frozen question/answer files, model weights, video bytes or historical
  raw inference results are committed. Obtain media from the existing team dataset
  locations referenced by the private samples, and match every file hash before execution.

The API runner is reusable here. The four native profiles support offline planning
and response parsing; a **general native GPU execution CLI is not included**.
CLIP/SigLIP2's separate paced runner is included at `tools/clip_benchmark/run.py`,
with its own protocol in that directory's README. It is not the identical shared-JPEG
unified run. VideoJev/Decision3 additionally need their deployed service, original
warmup and identity/dtype checks. Do not advertise native GPU readiness from offline
preflight or substitute the paced runner while keeping the unified label.

## Fixed method

Test: 120 questions / 110 videos, fight32, smoking22, fall26, weapon20, bag knife10,
bag alcohol bottle10. Validation: 24 videos/questions, six per base event with
three yes and three no. There is no bag validation and no fabricated train split.
Source-group/hash separation is checked; unknown training/actor/scene overlap and
prior test reuse remain limitations. The robot41 supplement is a separate cohort.

- Sample t=0,1,2,... strictly before duration; common 720px JPEGs, frozen byte hashes.
- API profiles use the recorded **current-image** prompts, not whole-video wording.
  Each API request receives one image/question; reference answers never enter it.
- CLIP/SigLIP2 keep one-frame positive-minus-contrast margins and native precision.
  VideoJev/Decision3 keep adjacent two-frame windows, native prompts and argmax.
  Native single-frame validation was proposed but not run or adopted.
- Language output requires complete JSON. The current parser also maps exact strings
  `"true"`/`"false"` to booleans, without truthiness. Nulls, extra/duplicate fields,
  refusals, truncation and other strings remain invalid. Bag permits explicit unknown.
- Every planned response must be valid before a question contributes a correct answer.
  For a complete question: any yes -> yes; otherwise any no -> no; all unknown -> unknown.
  Unknown is valid but wrong against these binary references. Refusal is missing.
- Headline **accuracy = correct / all planned questions**, including missing questions
  (120 overall, 100 for base100, each event's fixed count for the breakdown).
  `complete_only_accuracy` is diagnostic, never the headline. All-missing is 0/120.
- Maximum three total attempts per unit (initial + two retries), cumulative across
  continuations. Retain the first valid answer; never retry a valid wrong answer.
  Wait 30/60 seconds or a longer Retry-After; do not skip work to maintain playback.
  Billing/auth/identity errors stop, preserving results. No automatic provider/key switch.
- Unique send coverage, final valid coverage, missing units, incomplete questions and
  valid unknowns are separate counters. Successful retry clears final missing; failed
  attempt history remains only in the ledger. A durable send intent cannot prove the
  remote server received an interrupted request.

Profiles record the earlier selected endpoints and quoted prices, **not a claim
that they are still available or fastest**. Before a new paid run, check access and
current pricing, estimate cost and obtain authorization. If fastest-provider selection
is needed, measure eligible providers first, then freeze one with fallback disabled.
Current provider/model must match actual receipts; API hardware/weights/dtype remain
unverified unless separately attested. A key must never be committed or put in a CLI argument.

Gemini 2.5 off uses `reasoning.enabled=false`; 2.5 low uses a 1024-token thinking
budget; 3.8 low requests `reasoning.effort=low`. All three recorded effort profiles
cap total output at 2048. Unspecified effort is recorded as unspecified, not off.
Verify current provider support instead of assuming all APIs support the same knob.

## Setup and offline preflight

Run from the repository root on Linux/macOS with Python3.12 and FFmpeg/ffprobe.
Use an isolated environment (`certifi`, `Pillow`; `pytest` for offline checks).
GPU dependencies are a separate pinned deployment. No CUDA imports/weights are
needed for this preflight. Place media under `/data/goodvision/validation/` and
`/data/goodvision/test/`, keeping frozen filenames. Put these five original files
in a separate private input directory: `manifest_validation_24.json`,
`validation-samples.json`, `manifest_120.json`, `test-samples.json`,
`validation_split.json`. The published legacy manifest is not a substitute for
these frozen inputs. Historical annotation paths inside them are provenance only.

```sh
python -m tools.unified_benchmark.preflight \
  --inputs /private/goodvision-inputs \
  --media /data/goodvision --models qwen8b luna gemini25_off \
  --output outputs/new-current-frame-bundle
```

Required `--inputs` locates the private frozen snapshots; optional `--profiles`
selects a reviewed configuration directory. The published hashes are mandatory;
changed datasets require a separately reviewed protocol, not relabelled hashes.
A new output directory is required. Preflight fully decodes videos, freezes frame
bytes and request hashes, and reports pending hardware/provider/cost authorization.
It sends zero network/model requests. Copy a bundle intact; do not mutate a frozen one.

## API execution after cost approval

Explicitly select models and phase. Start with validation; no past user's approval
is a standing authorization for someone else's new calls.

```sh
python -m tools.unified_benchmark.api_run \
  --bundle outputs/new-current-frame-bundle --models qwen8b luna \
  --phase validation --env-file /private/openrouter.env \
  --output outputs/new-validation --authorization-note 'Approved validation scope and cost' --execute
```

For different accounts, replace `--env-file` with `--credential-map /private/keys.json`:
that private JSON maps each selected alias to its dotenv file path, not to a key.
Models on the same provider execute sequentially; different provider groups may
run concurrently, each with twelve queue slots. The process stops on balance failure.
It never writes a Sheet or creates a PR.

Luna selects thresholds on complete validation24 and writes a frozen artifact.
Incomplete validation stops calibration; no implicit fixed-0.5 fallback. For test:

```sh
python -m tools.unified_benchmark.api_run \
  --bundle outputs/new-current-frame-bundle --models luna --phase test \
  --thresholds outputs/new-validation/luna/thresholds.json --thresholds-sha256 FROZEN_SHA256 \
  --env-file /private/openrouter.env --output outputs/new-test \
  --authorization-note 'Approved test scope and cost' --execute
```

Hard-answer profiles omit the threshold flags. Use a fresh output path per phase;
the public CLI refuses an existing run. The ledger supports bounded continuation,
but there is no automatic restart/recovery command here: inspect stopped records,
then use a reviewed wrapper retaining their contexts and remaining attempt budget.

## Thresholds, reports and publication

Luna: 0.01 grid, longest optimal contiguous run midpoint, `>=`; recheck midpoint
optimality and stop if it is worse. CLIP/SigLIP2: widest finite optimal margin
interval midpoint, strict `>`, with FP/FN tie rules. Selection requires complete
validation and cannot read test predictions. Freeze model/config/rules/bundle hashes.
Bag remains Luna0.5 or margin0, explicitly uncalibrated. Boolean/argmax models have
no scalar threshold. Past calibrated values are comparisons, not new-run defaults.

For normalized ledgers, offline commands are:

```sh
python -m tools.unified_benchmark.workflow select --bundle /bundle --model luna \
  --ledger /run/luna/validation/attempts.jsonl --output /new/thresholds.json
python -m tools.unified_benchmark.workflow report --bundle /bundle --model luna \
  --ledger /run/luna/test/attempts.jsonl --thresholds /thresholds.json \
  --thresholds-sha256 FROZEN_SHA256 --output /new/report.json
```

Keep known receipt USD spend and unknown-cost attempts separate. Specify whether
cost includes validation/test/retries. GPU actual cost needs a bill or verified
rate and duration. Test P50/P95 are successful complete transport attempt times,
including parsing/metadata lookup, excluding queue/backoff; report n and missing.
They are not independent-probe latency or end-to-end twelve-camera throughput.

Publish only when requested via bench-sheet-contribute, reading live headers and
preserving previous local artifacts. Model names include chosen effort, Hardware
may combine VRAM, Price is actual spend; thresholds and sources go in notes. Do not
add tabs/columns by default or replace old timing with a mismatched variant.
