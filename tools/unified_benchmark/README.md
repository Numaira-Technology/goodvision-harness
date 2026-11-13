# Current-frame 202-question evaluation handoff

Default protocol: **unified-frame-202-v1**; explicit **unified-frame-120-v2** remains
available for historical replay. Read [dataset membership and limits](DATASET202.md).
This packages the previously local workflow for
another checkout. It does not modify the original videos, questions or references,
re-run models, or reinterpret old bundles. Keep historical source snapshots with
historical artifacts. The original `tools/event_benchmark/run_eval.py` remains the
separate legacy v2.1 runner; it is not the current-frame entrypoint.

## What is included

- Mandatory byte hashes for validation24/test202 inputs. The eight
  frozen JSON files are private team inputs, not published in this public repository.
  Download the original files from the [restricted team Drive](#get-the-frozen-inputs)
  below; do not regenerate their labels. The added82 test references remain inherited
  and pending individual-view review; combined202 is provisional reference agreement.
- Thirteen recorded profiles: qwen8b, qwen235b, maverick, luna, gemini25_off,
  gemini25_low, gemini38_low, clip, siglip2, videojev, decision3, plus
  `qwen8b_r2_host` and `qwen8b_r2_vllm` (independent self-hosted R2 engines;
  not OpenRouter).
- Offline input/media/frame/request preflight; lossless receipt/score retention;
  cumulative three-attempt queue; validation threshold selection and test reporting.
- API execution with an explicit selected credential file or per-model file mapping.
  No credentials, frozen question/answer files, model weights, video bytes or historical
  raw inference results are committed. Obtain media from the existing team dataset
  locations referenced by the private samples, and match every file hash before execution.

The API runner and the four-model native execution CLI are included. Follow
[the native deployment/run guide](NATIVE.md) for pinned environments and weights,
original service warmup, identity/precision checks, validation selection, test and
bounded continuation. Its offline tests do not establish a new GPU result.
CLIP/SigLIP2's separate paced runner at `tools/clip_benchmark/run.py` retains its
own protocol; do not substitute it for the shared-JPEG unified run.

## Fixed method

Test: 202 questions / 192 videos, fight32, smoking22, fall62, weapon66, bag knife10,
bag alcohol bottle10. Validation: 24 videos/questions, six per base event with
three yes and three no. There is no fabricated train split. An optional
[CLIP/SigLIP2 bag supplement](NATIVE.md#optional-clipsiglip2-bag-validation)
adds ten validation videos/twenty questions, requiring complete validation44
before six-event calibration. Test202 remains separate and unchanged.
Source-group/hash separation is checked; unknown training/actor/scene overlap and
prior test reuse remain limitations. Original120 and robot82 stay separately
reported. The original41 stereo sources are not counted again, and the added82
test labels must not be used for calibration.

- Sample t=0,1,2,... strictly before duration and at/before the last decoded video
  frame timestamp relative to container start; common 720px JPEGs, frozen byte hashes.
  Current202 records excluded tail points in `media-audit.json`; legacy120 retains
  its original sampling boundary and matching source snapshot.
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
  (202 overall, 120/82 for the cohorts, 100 for original base100).
  `complete_only_accuracy` is diagnostic, never the headline. All-missing is 0/202.
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
`/data/goodvision/test/`, keeping frozen filenames. Obtain and verify the eight
private JSON inputs using the steps below, then run the offline preflight.

### Get the frozen inputs

Default202 requires the five original files below plus `manifest_202.json`,
`samples.json` and `videos.json` from [the current dataset guide](DATASET202.md#acquire-and-verify),
all under the same `--inputs` directory. For an explicitly selected legacy120 run,
only the five original files are required.

The original `unified-frame-120-v2` inputs are in the
[team Drive folder: unified-frame-120-v2-private-inputs-20261107](https://drive.google.com/drive/folders/1bCcadb0FDLxcmnnQI_tZ1ReR0x3avsoD).
Access is restricted to the existing benchmark team members; the link does not
grant public access. Sign in with your authorized team account. If access is
denied, ask Haotian Bi for access instead of changing the folder's link sharing.

| Original file | Download from Drive |
|---|---|
| `manifest_validation_24.json` | [File](https://drive.google.com/file/d/1OZrISz2JXSaUyvGI-wV8bN3B3oQhbsUR/view?usp=drivesdk) |
| `validation-samples.json` | [File](https://drive.google.com/file/d/1SxJYywf0HO7-aQegGCOt3gP4x6eO2Y1q/view?usp=drivesdk) |
| `manifest_120.json` | [File](https://drive.google.com/file/d/1uvCpUGnOB9PjcPGUsD2FfhjZSf6P5JGD/view?usp=drivesdk) |
| `test-samples.json` | [File](https://drive.google.com/file/d/1fK0kBHmqbbW59awOIAD17iyXq_72BbbD/view?usp=drivesdk) |
| `validation_split.json` | [File](https://drive.google.com/file/d/1ULNTVNbSzWH9wdynkSt-4DcPg6xMuf2q/view?usp=drivesdk) |

Use Drive's **Download** action for the folder or the five files. Extract any ZIP
and place all five JSON files directly in one private directory outside Git, for
example `/private/goodvision-inputs/`; pass that directory as `--inputs`, not the
ZIP or a parent directory. Preserve filenames and original bytes: do not convert
to Google Docs, copy text from a preview, or reformat/save the JSON. This folder
contains only the five original JSON files; add the three202 files for the default
protocol and obtain the videos separately from the existing
team dataset locations. No credentials or model weights are included.

From the repository root, use the prepared Python environment to check the files
against the runner's pinned SHA-256 values (replace the example input path).
For explicit legacy120 only, iterate `PINNED_INPUTS` without `PINNED_202_INPUTS`:

```sh
python - /private/goodvision-inputs <<'PY'
import hashlib
import sys
from pathlib import Path
from tools.unified_benchmark.preflight import PINNED_INPUTS
from tools.unified_benchmark.dataset202 import PINNED_202_INPUTS

inputs = Path(sys.argv[1])
for name, expected in (PINNED_INPUTS | PINNED_202_INPUTS).items():
    path = inputs / name
    if not path.is_file():
        raise SystemExit(f"Missing frozen input: {name}")
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise SystemExit(f"Frozen input hash mismatch: {name}; download the original again")
print("Verified all 8 frozen inputs.")
PY
```

The published legacy `tools/event_benchmark/datasets/manifest_120.json` is not a
substitute for the Drive snapshot. Historical annotation paths inside the inputs
are provenance only. Do not edit labels or pinned hashes to bypass a mismatch.
Upload readback on 2026-11-07 verified all five original SHA-256 values.

### Run the offline preflight

```sh
python -m tools.unified_benchmark.preflight \
  --inputs /private/goodvision-inputs \
  --media /data/goodvision --models qwen8b luna gemini25_off \
  --output outputs/new-current-frame-bundle
```

The command defaults to `--protocol unified-frame-202-v1`. For a deliberately
selected historical120 run, pass `--protocol unified-frame-120-v2` with its five
original inputs. Neither option changes the review status of its references.

Required `--inputs` locates the private frozen snapshots; optional `--profiles`
selects a reviewed configuration directory. The published hashes are mandatory;
changed datasets require a separately reviewed protocol, not relabelled hashes.
A new output directory is required. Preflight fully decodes videos, freezes frame
bytes and request hashes, and reports pending hardware/provider/cost authorization.
It sends zero network/model requests. Copy a bundle intact; do not mutate a frozen one.
Execution, offline selection and reporting also compare the current critical code
against the frozen source fingerprints. After changing code, build a new preflight
bundle; retain the original code with old artifacts for historical replay. Do not
rewrite old bundle hashes to pass the check.

## Native receipt continuation

Native `--resume` validates all requested phase identities/completion anchors,
then retained receipt hashes and reparsed answers before starting a backend. Every ledger result must have one earlier matching intent;
interrupted intents still consume the existing three-attempt budget. First valid
answers follow attempt order, even if an older orphan is recovered later.
A newly recovered fatal outcome is recorded and pauses this invocation before
startup. A subsequent explicit `--resume` can acknowledge that recorded failure,
as with a failure recorded during an ordinary run.

New native receipts preserve HTTP status, fatal classification, retry delay and
completion time. Incomplete orphan receipts missing these authoritative recovery
fields fail closed with the path and missing field names; do not replace them
with current time or rewrite historical evidence. Already-recorded legacy failures
may retain their existing anchored ledger outcome. Recovery keeps completion-based
cooldowns, including a final rate-limit response's shared delay.

Readers accept both historical `python-json-nonfinite-v1` global and
`json-text-nonfinite-v1` per-field raw encodings. New writes retain the per-field
format. Diagnostic decoding neither changes original bytes nor makes nonfinite
scores valid, and does not permit cross-source execution. Normalized CLIP evidence
contains only the three validated margin/cosine components; optional diagnostics
remain in the raw response. A differing historical normalized shape fails reparse
rather than being rewritten. Bundle checks require
both the complete physical inventory and recorded preflight config/plan/frame
bindings. The existing HF-to-vLLM converter's sealed lineage preserves the original
preflight bytes and binds only its derived serving profile/plans; it cannot repair
a changed source exam or drop an input from the recorded inventory.

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
Incomplete validation stops calibration; no implicit fixed-0.5 fallback. Failed
selection files and artifacts lacking successful calibration, full validation24
counts, matching plan/evidence identity or six finite thresholds are rejected before
any selected test model starts. Offline reporting applies the same check. Older
artifacts remain untouched; replay them with their retained code, or select a new
artifact offline from compatible complete validation evidence. For test:

```sh
python -m tools.unified_benchmark.api_run \
  --bundle outputs/new-current-frame-bundle --models luna --phase test \
  --thresholds outputs/new-validation/luna/thresholds.json --thresholds-sha256 FROZEN_SHA256 \
  --env-file /private/openrouter.env --output outputs/new-test \
  --authorization-note 'Approved test scope and cost' --execute
```

## Self-hosted Qwen8B from R2

### Prepare the control machine

The control machine runs the launcher, SSH tunnel and evaluator. Restoring an R2
environment on the GPU does not install this machine's dependencies. In the
isolated Python 3.12 preflight environment, install the additional controller
packages below. The Vast SDK pin matches this repository's controller Dockerfile;
keep GPU/model environments separate.

```sh
python -m pip install 'vastai==1.5.6' 'python-dotenv>=1,<2' 'PyYAML>=6,<7'
python -m pip check
```

Local tools required on PATH: Git, OpenSSH (`ssh`, `scp`, `ssh-keygen`), `tar`,
FFmpeg and ffprobe. Then check imports and tools **offline**, without loading
credentials, constructing a Vast client, contacting R2 or starting a launcher:

```sh
python - <<'PY'
import importlib
from importlib.metadata import version
import shutil
import sys

if sys.version_info[:2] != (3, 12):
    raise SystemExit("Use the Python 3.12 preflight environment")
for distribution, module in [("vastai", "vastai"), ("python-dotenv", "dotenv"),
                              ("PyYAML", "yaml"), ("Pillow", "PIL"), ("certifi", "certifi")]:
    loaded = importlib.import_module(module)
    installed = version(distribution)
    if distribution == "vastai" and (installed != "1.5.6" or not hasattr(loaded, "VastAI")):
        raise SystemExit("Install the documented vastai==1.5.6 controller SDK")
    print(distribution, installed)
missing = [name for name in ("git", "ssh", "scp", "ssh-keygen", "tar", "ffmpeg", "ffprobe")
           if shutil.which(name) is None]
if missing:
    raise SystemExit("Missing local tools: " + ", ".join(missing))
print("Controller imports/tools ready; credentials, R2 access and GPU remain unchecked.")
PY
```

Retain installed package versions with the run record. Missing imports/tools must
be resolved before using either launcher. This check is not rental authorization.

### Prepare the bundle and run the selected engine

Self-hosted Qwen3-VL-8B on a rented 4090 uses `qwen8b_r2_host` (HF generate +
FA2) or the independent engine profile `qwen8b_r2_vllm` (vLLM), not the
OpenRouter `qwen8b` profile and not a `qwen8b_selfhost` alias. Both keep the
same frozen JPEGs, prompts, parser, temperature, max_tokens=30, and **12 client
slots**. Do not relabel the completed HF run as vLLM.

These two launchers currently accept only the historical `unified-frame-120-v2`
protocol (validation24/test120). Explicitly select it below and supply its five
original frozen JSON inputs. The default 202-question bundle is rejected by these launchers
before rental. This limitation does not change the general preflight default or
make historical results evidence for the 202 protocol. Preserve existing bundles unchanged.
Create a fresh offline bundle containing the selected self-hosted aliases
(omit an alias if not needed):

```sh
python -m tools.unified_benchmark.preflight \
  --protocol unified-frame-120-v2 \
  --inputs /private/goodvision-inputs --media /data/goodvision \
  --models qwen8b_r2_host qwen8b_r2_vllm \
  --output outputs/new-qwen-r2-bundle
```

After preflight passes, use the corresponding launcher below only within the
approved rental/run scope. The launcher commands can create a billed Vast
instance; they are not dry runs and do not use the four-model native CLI's
`--execute` gate.

Transformers:

```sh
PYTHONPATH=. python -m tools.start_qwen8b_r2 \
  --eval-bundle outputs/new-qwen-r2-bundle \
  --eval-output outputs/qwen8b-r2-host \
  --eval-phase both \
  --authorization-note 'Approved self-hosted qwen8b R2 4090 run'
```

That launcher restores the packed env and pinned weights from R2, serves
`tools.qwen8b.serve` (transformers + flash_attention_2), and runs
`tools.qwen8b.eval_local` against the SSH tunnel. Requires gitignored `.env`
keys `VAST_API_KEY`, `SAM_SSH_KEY`, `R2_ENDPOINT`, `R2_BUCKET`,
`R2_ACCESS_KEY_ID`, `R2_SECRET_ACCESS_KEY`. Attach SSH immediately after create;
always release the labelled instance. Headline scoring remains correct/all.

The active HF and vLLM profiles bind typed `env_artifact` identities. Create fresh
bundles for updated active profiles and preserve historical bundles unchanged.
The captured preflight environment is authoritative through the final restore
acknowledgement; a changed registry or artifact stops before service/evaluation.

vLLM (independent engine, same exam):

```sh
PYTHONPATH=. python -m tools.start_qwen8b_vllm \
  --eval-bundle outputs/new-qwen-r2-bundle \
  --eval-output outputs/qwen8b-r2-vllm \
  --eval-phase both \
  --authorization-note 'Approved self-hosted qwen8b R2 vLLM 4090 run'
```

The Qwen evaluator currently requires a new output path and has no supported
`--resume` CLI. `--resume-instance-id` only reconnects to a rental; it does not
resume evaluation. Preserve interrupted ledgers and receipts. Do not restart the
same units with a fresh output to reset the three-attempt budget.

`eval_local` still uses 12 slots. vLLM may batch those 12 internally. Do not
change JPEG size, prompts, max_tokens, or client concurrency to fit VRAM.
The active declared vLLM artifact is
`envs/qwen8b-vllm/cu130-py312-vllm0310-ada2.tar.gz`, SHA-256
`59314d00783b6698ef83cbfbd2e028c2f094d76c9d3383f5a37c3fdf3687b95b`,
4,252,375,781 bytes. The profile records this identity and the launcher checks
it against the registry before rental. The cu130 pack requires host CUDA >=13.0
and NVIDIA driver >=580; the HF CUDA 12.8 image label is not a driver guarantee.
The old cu130 `ada` pack is incomplete and refused. All published Qwen pack keys,
including ada2, are frozen; CPU rebuilds require a new reviewed key. The 4090 only
restores the pinned object and existing weights. A CPU import is not a GPU serve.
See [pack evidence limitations](../qwen8b_vllm/BUILD.md); the declared identity
does not independently verify the claimed historical exam.

Bundle/profile/request hashes are checked before rental and rechecked before
inference. A separate synthetic 720px JPEG is used for smoke, with its request
and response saved to `vllm-smoke-<instance>.json` beside the evaluation output.
It is excluded from scored attempts and latency. Scored validation/test units
are sent only through the evaluation ledger. Create a new bundle for the updated
ada2 profile; never rewrite a historical bundle to pass identity checks.

Hard-answer profiles omit the threshold flags. Use a fresh output path per phase;
the public CLI refuses an existing run. The ledger supports bounded continuation,
but there is no automatic restart/recovery command here: inspect stopped records,
then use a reviewed wrapper retaining their contexts and remaining attempt budget.

## Thresholds, reports and publication

Luna: 0.01 grid, longest optimal contiguous run midpoint, `>=`; recheck midpoint
optimality and stop if it is worse. CLIP/SigLIP2: widest finite optimal margin
interval midpoint, strict `>`, with FP/FN tie rules. Selection requires complete
validation and cannot read test predictions. Freeze model/config/rules/bundle hashes.
Without the optional bag supplement, bag remains Luna0.5 or margin0, explicitly
uncalibrated. CLIP/SigLIP2 with the separately frozen validation44 supplement
select all six event margins; Luna still uses validation24 and bag0.5.
Boolean/argmax models have no scalar threshold. Past calibrated values are
comparisons, not new-run defaults.
Successful native CLIP/SigLIP2 validation reports use the selected thresholds and
cite their artifact hash, as does the subsequent test. Provisional202 test labels
remain outside validation selection.

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
