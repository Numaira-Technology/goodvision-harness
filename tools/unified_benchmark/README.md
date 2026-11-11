# Current-frame 202-question evaluation handoff

Default protocol: **unified-frame-202-v1**. See [dataset202 acquisition and limits](DATASET202.md).
This packages the previously local workflow for another checkout. It does not modify the original videos, questions or references,
re-run models, or reinterpret old bundles. Keep historical source snapshots with
historical artifacts. The historical `tools/event_benchmark/run_eval.py` included
here is not the current-frame entrypoint; archived v2.1 instructions
require their own matching implementation as described below.

Default sequence: **preflight → environment checks → validation → threshold
selection when applicable → test → local report**. Sheet publication follows only
the requested target/version/scope. Historical [v2.1](../../.claude/skills/event-benchmark/references/legacy-v2.1.md)
and [robot41](../../.claude/skills/event-benchmark/references/robot41.md) have separate
references requiring their matching retained code/data snapshots. Their archived
commands are not executable from this checkout and are not the current202 route.

## What is included

- Mandatory byte hashes for validation24/test202 inputs. The eight
  frozen JSON files are private team inputs, not published in this public repository.
  The additional82 labels remain inherited and pending review.
  Download the original files from the [restricted team Drive](#get-the-frozen-inputs)
  below; do not regenerate their labels.
- Thirteen recorded profiles: qwen8b, qwen235b, maverick, luna, gemini25_off,
  gemini25_low, gemini38_low, clip, siglip2, videojev, decision3, plus
  `qwen8b_r2_host` and `qwen8b_r2_vllm` (separate self-hosted R2 engines on one
  4090; not OpenRouter).
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

## Handoff checklist

These instructions extend the PR #98 code baseline
`d4fea6bfdab077cae37b207540922513b7a2cc82`; that revision contains the corrected
vLLM `ada2` launcher. Pass the complete checkout and its matching instructions to
the agent, not an isolated SKILL.md copied onto main. Record the actual
`git rev-parse HEAD`, working-tree changes and preflight source fingerprints for
the run. Preserve existing changes; do not reset a checkout to match this note.

| Required material | Ready when |
|---|---|
| Code and instructions | Selected profiles and execution entrypoints exist in the same checkout; revision/local changes are recorded. The agent reads `.claude/skills/event-benchmark/SKILL.md` and the selected route's guide explicitly. |
| Eight private JSON files | `manifest_validation_24.json`, `validation-samples.json`, `manifest_120.json`, `test-samples.json`, `validation_split.json` plus `manifest_202.json`, `samples.json`, `videos.json` are together under `--inputs`; all eight pass the [pinned-hash check](#get-the-frozen-inputs). |
| Original videos | 24 validation videos and 192 test videos (202 test questions), obtained from the private samples' team links, pass media fingerprints and full decoding. Do not substitute re-encoded copies. |
| Offline environment | Python 3.12, Pillow, certifi, FFmpeg/ffprobe; run the [offline preflight](#run-the-offline-preflight) for the selected aliases and retain the new bundle. |
| API route | Private key file or per-model key-file mapping, account access/balance and selected provider support are checked; model/provider identity must match actual receipts. No keys in Git or command arguments. |
| Four-model native route | Linux x86_64 RTX 4090 24GB, SSH access, enough disk, fixed model revisions and the Python/CUDA environments in [NATIVE.md](NATIVE.md#2-prepare-the-gpu-deployment-no-evaluation-inference). Deployment verification must pass. R2 is optional for this route. |
| Qwen R2 route | Complete the [control-machine check](#prepare-the-control-machine), then provide R2 read-only credentials, pinned weights/environment and private launcher credentials. Transformers uses its cu128 pack; vLLM uses the verified cu130 `ada2` pack and a compatible driver. A CUDA image label alone is not driver verification. |
| Authorization and costs | Record chosen models, validation/test scope, hardware use or rental, the cost estimate and applicable user approval. Offline preflight is not permission to rent or make paid requests. |
| Output and recovery locations | New local output/bundle paths and retained logs/receipts; retain old results. Resume only through the selected route's supported procedure with its original identities and remaining attempt budget. |

Passing offline preflight means the inputs and request plan are ready. Execution
readiness additionally requires the route's runtime identity checks and authorized
warmup/smoke. The native guide's synthetic tests do not establish a fresh GPU run.
PR #98 records a completed Qwen vLLM run with `ada2`; a new machine still needs its
own runtime checks. Stop at a failed prerequisite and report the missing item.

## Fixed method

Test: 202 questions / 192 videos, fight32, smoking22, fall62, weapon66, bag knife10,
bag alcohol bottle10. Validation: 24 videos/questions, six per base event with
three yes and three no. There is no bag validation and no fabricated train split.
Source-group/hash separation is checked; unknown training/actor/scene overlap and
prior test reuse remain limitations. The original41 stereo clips are not counted
again. New questions121–202 are all test; their inherited labels remain pending
single-view review. See [membership and limits](DATASET202.md#membership-and-reference-limits).

- Sample t=0,1,2,... strictly before duration and at/before the last decoded video
  frame timestamp (relative to container start). Current202 records the original
  grid, retained times and excluded tail points in `media-audit.json`; it never
  repeats the last frame to fill an empty tail or skips failed in-range frames.
  Use common 720px JPEGs with frozen byte hashes. Legacy120 retains its old boundary.
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
  (202 overall, 120/82 for the two cohorts, 100 for original base100, each event's fixed count for the breakdown).
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

For default202, obtain these original five inputs **plus the three202 files** in
[DATASET202.md](DATASET202.md#acquire-and-verify). Keep them together; the original
files validate the unchanged120 subset and validation split. For explicit legacy120,
only the five below are required.

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
contains only the original five JSON files; add the three202 files from the
current dataset folder, and obtain videos separately. No credentials or model weights are included.

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

Default `--protocol unified-frame-202-v1` uses202; for a deliberately selected
legacy120 run pass `--protocol unified-frame-120-v2` with the five original inputs.
Both protocols preserve the inference method; they never reuse or overwrite old bundles.
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

Hard-answer profiles omit the threshold flags. Use a fresh output path per phase;
the API CLI refuses an existing run. Its ledger supports bounded continuation,
but there is no automatic API restart/recovery command here: inspect stopped
records, then use a reviewed wrapper retaining contexts and remaining attempt
budget. The four-model native CLI has its own explicit `--resume` procedure.

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
slots**. Do not relabel the completed HF run as vLLM. First create an offline
bundle containing the selected self-hosted aliases (omit an alias if not needed):

```sh
python -m tools.unified_benchmark.preflight \
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

vLLM (independent engine, same exam):

```sh
PYTHONPATH=. python -m tools.start_qwen8b_vllm \
  --eval-bundle outputs/new-qwen-r2-bundle \
  --eval-output outputs/qwen8b-r2-vllm \
  --eval-phase both \
  --authorization-note 'Approved self-hosted qwen8b R2 vLLM 4090 run'
```

`eval_local` still uses 12 slots. vLLM may batch those 12 internally. Do not
change JPEG size, prompts, max_tokens, or client concurrency to fit VRAM.
Restore the existing package; ordinary reruns do not build/upload an environment.
The authoritative `tools/models.yaml` entry and launcher pin:

- R2 object: `envs/qwen8b-vllm/cu130-py312-vllm0310-ada2.tar.gz`
- Manifest: `envs/qwen8b-vllm/cu130-py312-vllm0310-ada2.json`
- SHA-256: `59314d00783b6698ef83cbfbd2e028c2f094d76c9d3383f5a37c3fdf3687b95b`
- Size: `4252375781` bytes; Python 3.12, torch `2.13.0+cu130`, vLLM `0.31.0`, Ada.

The R2 restore path verifies the downloaded archive before unpacking. See the
[package acquisition and verification guide](../qwen8b_vllm/BUILD.md#restore-the-current-package).
Both previous vLLM `…ada.tar.gz` objects are historical and refused as launch
fallbacks; preserve them and the independent HF pack. `pack_provenance.json`
describes the old incomplete cu130 object, not `ada2`. A CPU import check alone
does not prove GPU serving. Preserve old bundles after this metadata correction;
create a new preflight bundle for a new run rather than editing old fingerprints.

## Thresholds, reports and publication

Luna: 0.01 grid, longest optimal contiguous run midpoint, `>=`; recheck midpoint
optimality and stop if it is worse. CLIP/SigLIP2: widest finite optimal margin
interval midpoint, strict `>`, with FP/FN tie rules. Selection requires complete
validation and cannot read test predictions. Freeze model/config/rules/bundle hashes.
Bag remains Luna0.5 or margin0, explicitly uncalibrated. Boolean/argmax models have
no scalar threshold. Past calibrated values are comparisons, not new-run defaults.
After successful native CLIP/SigLIP2 calibration, the validation report uses the
selected thresholds and cites their artifact hash, as does the subsequent test.

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
`run_date_utc` comes from the first timestamped request intent in the saved ledger,
including resumed runs; it is null if no such evidence exists. Offline re-scoring
records its own `report_generated_at_utc` and does not change the inference date.
For invalid JSON or oversized HTTP responses, receipts retain up to 2 MiB of raw
body bytes as `body_base64`, the HTTP status, and truncation/redaction flags. Such
attempts remain invalid and consume the existing three-attempt budget; missing
cost evidence stays unknown. API key bytes are redacted if echoed in that body.

An agent can prepare a local publication preview and carry out an authorized
update using [bench-sheet-contribute](../../.claude/skills/bench-sheet-contribute/SKILL.md).
The user chooses the result version, target tabs and overwrite/append scope; an
evaluation request alone is not publication authorization. The publication flow
reads live headers, matches exact model/variant rows, saves before-values and a
sparse proposed diff, writes only the agreed cells, and verifies the readback.
Keep a requested review step pending until the user approves; an existing explicit
write authorization does not need to be requested again. Preserve previous local
artifacts and styles. Model names include chosen effort, Hardware
may combine VRAM, Price is actual spend; thresholds and sources go in notes. Do not
add tabs/columns by default or replace old timing with a mismatched variant.

## Offline regression checks

In the prepared lightweight environment with pytest, run:

```sh
PYTHONDONTWRITEBYTECODE=1 python -m pytest \
  tests/test_unified_benchmark.py tests/test_unified_dataset202.py tests/test_unified_luna_adapter.py \
  tests/test_native_workflow.py tests/test_start_qwen8b_r2.py tests/test_qwen8b_vllm.py \
  -o addopts='' -q -p no:cacheprovider
```

The Luna checks use synthetic receipts through the real parser, transport and
queue into validation threshold selection, including invalid-answer retries and
missing counts. No model calls are made. Native source-export tests require the
public pinned commits described in [NATIVE.md](NATIVE.md#offline-regression-checks)
to be present locally. Test success does not verify credentials, live R2 access
or a new GPU deployment.
