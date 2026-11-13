# Four self-hosted models: reproducible execution

This is the native execution path for **unified-frame-202-v1**, alongside the API
runner in [README.md](README.md). Run commands from the repository root. The
installer and runner are opt-in; importing them or running offline preflight
loads no models. The portable flow is covered by synthetic offline tests; a new
GPU deployment still requires its own runtime checks and smoke. No new GPU
benchmark result is implied by these tests. Explicit `unified-frame-120-v2`
remains available with its original inputs and source fingerprints.

## Fixed method

| Profile | Input / decision | Precision | Validation |
|---|---|---|---|
| `clip` | One common JPEG; positive-minus-contrast cosine margin | FP32, TF32 off | Four base events on validation24 by default, six events with the optional bag supplement |
| `siglip2` | Same; original lowercase text preprocessing | FP32, TF32 off | Same selection, independently for this model |
| `videojev` | Two adjacent distinct 1 FPS frames; native yes/no/unknown argmax | BF16 backbone, original FP32 scoring | Evaluate native decisions; no scalar threshold |
| `decision3` | Two adjacent distinct 1 FPS frames; native yes/no/unknown argmax | BF16 backbone, FP32 readout | Same; no scalar threshold |

The runner uses the frozen profiles, model revisions, prompts, JPEGs, question
mapping and scorer. Twelve video workers process ordered units with at most one
in-flight request per source. CLIP/SigLIP2 GPU calls remain serialized under the
original lock. No work is dropped to keep playback moving. Each unit has at most
three total attempts, including interrupted sends and continuations; retain its
first valid target answers. Retry invalid/missing responses after the original
30/60-second cooldown. Valid unknown/wrong answers are not retried.

A question needs every planned target response. For complete evidence: any yes
means yes, otherwise any no means no, otherwise unknown. Headline accuracy is
correct / **202**, with incomplete questions still in the denominator. Report
missing units, incomplete questions and unknowns separately. A native window is
two frames, so missing windows and missing single-frame calls are different units.
This does not establish twelve-camera real-time throughput.
The added82 test references remain inherited and pending individual-view review;
keep [the dataset limitations](DATASET202.md) and original120/robot82 breakdowns.
Validation24 remains the default. The optional bag supplement below extends it
to validation44; added test labels are never calibration data.

## 1. Inputs, media and offline preflight

Obtain the eight private JSON inputs and original videos as described in
[the input guide](README.md#get-the-frozen-inputs). Use Python 3.12, Pillow,
certifi, FFmpeg and ffprobe for preflight. Keep inputs, generated JPEGs and raw
receipts private. Model requests do not contain reference answers; the frozen
bundle does contain answers for local scoring, so only copy it to an authorized
evaluation machine.

```sh
python -m tools.unified_benchmark.preflight \
  --inputs /private/goodvision-inputs \
  --media /data/goodvision \
  --models clip siglip2 videojev decision3 \
  --output /private/native-bundle
```

The output must be new. Preflight verifies the frozen inputs and all media,
decodes frames, builds request fingerprints and snapshots execution code. It
does not download weights, contact an API or import CUDA. Rebuild a new bundle
after changing runner code; never edit an old bundle's hashes. Copy the complete
bundle to the GPU machine without changing its contents, using the same checkout.
For a deliberately selected legacy120 run, add `--protocol unified-frame-120-v2`
and use its five original JSON inputs. Preserve historical bundles unchanged.

### Optional CLIP/SigLIP2 bag validation

For current202, select `--models clip siglip2` (or either alias) and add the
following preflight arguments to use a private, versioned validation supplement.

```sh
  --bag-validation /private/goodvision-inputs/bag-validation.json \
  --bag-validation-sha256 FROZEN_MANIFEST_SHA256
```

The manifest has `schema: "bag-validation10-v1"` and a `videos` array of ten
original clips under the existing media `validation/` directory. Each entry
contains `filename`, media `sha256`, `source_video_group` and two `questions`.
Each question contains its `event` (`bag_knife` or `bag_alcohol_bottle`),
`proposed_answer` (`yes` or `no`) and `human_review` describing the actual
reference provenance, including any absence of individual human review. Record
the manifest digest before preflight. Preflight preserves its bytes and checks
both labels occur per event, unique video hashes and split separation by source
group. Recording cuts can share a group within validation.

Original validation questions 1–24 retain their identifiers. The supplement
adds 25–44 in manifest video order, knife then bottle. The two questions per
clip share the same 1 FPS frames. Use the existing `native_run --phase both`
command with an authorization note covering 34 validation videos / 44 questions
and test202. Each model selects all six midpoints on complete validation44 and
freezes them before its first test request. The top level of each completed
phase report gives calibrated results. Its `zero_margin_baseline` gives the
same scores evaluated at zero, including per-event and test-cohort counts.

## 2. Prepare the GPU deployment (no evaluation inference)

Prepare a Linux **x86_64** machine with **one RTX 4090 24GB**, a driver supporting
CUDA 12.8, `git`, `patch`, FFmpeg/ffprobe and Python with venv support. Supply
Python **3.11** for CLIP/SigLIP2 and **3.12** for VideoJev/Decision3; these preserve
the recorded runtime families. Use an empty deployment directory and enough
disk for three isolated environments, all four checkpoints, media and outputs.
Check free space before renting/running; the script does not rent or resize disks.
Public GitHub, PyTorch wheel and Hugging Face access are required during setup.

Inspect the plan first (does not install or download):

```sh
python -m tools.unified_benchmark.native_deploy prepare \
  --root /workspace/goodvision-native \
  --clip-python python3.11 --native-python python3.12
```

After the machine/setup scope is authorized, append `--execute` to that command.
It creates isolated environments, pins the recorded compute dependencies, checks
the native wheel ABI, runs `pip check`, downloads exact model revisions and verifies
every selected file against official LFS SHA-256 or Git blob checksums. It writes
per-file SHA-256 inventories, installed package versions and `deployment.json`.
Keep the printed `deployment_sha256` for every subsequent command.

CLIP/SigLIP2 share an environment and cache. VideoJev and Decision3 have separate
environments and their exact prior service source. `native_assets/sources.json`
pins public commits `dcea0063…` and `a9ee8a2…`, the already-used 1 FPS patches,
and every exported source-file hash. Setup fetches those public commits if the
checkout is shallow; it does not merge their branches or switch the current one.
It preserves the original warmup code and does not substitute the current main
runtime. The installer itself does **not** load model weights for inference.

The default downloader uses verified official HTTP (`HF_HUB_DISABLE_XET=1`).
R2 is optional and not a prerequisite; this command does not need R2 credentials.
Do not substitute another revision or bypass the checksum gate. Setup errors
leave `failure.json`, logs and partial files intact. Stop and inspect the failed
step; setup has no automatic retry/reinstall mode and refuses a populated root.

Verify a completed deployment without loading a model:

```sh
python -m tools.unified_benchmark.native_deploy verify \
  --root /workspace/goodvision-native --model clip \
  --deployment-sha256 DEPLOYMENT_SHA256
```

Repeat verification for the other model aliases. This hashes checkpoint bytes,
not just filenames. Runtime startup additionally compares installed package
versions against the recorded environment. Do not install extra packages into
these environments after freezing them.

## 3. Validate, freeze thresholds, then test

First check the execution inputs offline. The command below intentionally omits
`--execute`, so it neither starts a service nor creates the output directory:

```sh
/workspace/goodvision-native/clip/env/bin/python -m tools.unified_benchmark.native_run \
  --model clip --bundle /private/native-bundle \
  --deployment /workspace/goodvision-native --deployment-sha256 DEPLOYMENT_SHA256 \
  --phase both --output /workspace/results/clip-current202
```

Once the **validation and test** run is authorized, append:

```sh
  --authorization-note 'Approved native validation24 and test202 scope' --execute
```

Use the same command for each model, with a distinct output directory and these
interpreters. Run the models sequentially on one GPU; a deployment lock rejects
overlapping runner processes. Stop unrelated services using the same GPU first.

| Alias | Python executable | Suggested output |
|---|---|---|
| `clip` | `/workspace/goodvision-native/clip/env/bin/python` | `/workspace/results/clip-current202` |
| `siglip2` | `/workspace/goodvision-native/clip/env/bin/python` | `/workspace/results/siglip2-current202` |
| `videojev` | `/workspace/goodvision-native/videojev/env/bin/python` | `/workspace/results/videojev-current202` |
| `decision3` | `/workspace/goodvision-native/decision3/env/bin/python` | `/workspace/results/decision3-current202` |

The native runner starts its own loopback service; no externally exposed inference
port is needed. `--port` defaults to 8093 and must be free. Startup checks GPU,
model/revision, dependencies, service batch/queue configuration and precision,
runs the original service warmup plus a synthetic smoke, then starts evaluation.
A failed startup consumes **no** evaluation attempt budget. Runtime and smoke
records live under `invocations/<id>/`. Cleanup stops only its own service process.

`--phase both` runs validation before test. With default validation24, each
margin model requires all 24 answers, chooses each base threshold using the
original widest finite optimal interval midpoint and strict `>` comparator,
and writes `thresholds.json` **before** test starts. Both bag thresholds stay
at zero for validation24. Missing validation evidence blocks
selection and test; there is no silent zero-threshold fallback. VideoJev/Decision3
have no threshold selection; their validation coverage remains separately reported.

To review validation before authorizing test, use `--phase validation` first.
For a later CLIP/SigLIP2 `--phase test` in a **new** output directory, add:

```sh
  --thresholds /workspace/results/clip-validation/thresholds.json \
  --thresholds-sha256 THRESHOLD_FILE_SHA256
```

Compute the exact file hash with `sha256sum`. The test entrypoint requires complete
validation evidence and matching bundle, profile, code and deployment hashes.
Use native-runner-generated threshold artifacts: the generic offline selector
does not attest the native deployment. Native argmax models omit both threshold flags.

## 4. Interrupted runs, outputs and reporting

On interruption, retain the whole output directory. Diagnose a failure before
resuming. Repeat the **same** authorized command with `--resume`; keep model,
phase, bundle, deployment and any initial threshold artifact unchanged. The
runner validates run/phase identities, existing receipts and cumulative attempts,
recovers a valid receipt saved just before an interruption, skips completed units
and phases, and preserves frozen thresholds. An interrupted intent without a
recoverable receipt consumes one attempt. A unit already at three attempts cannot
be sent a fourth time. Do not use a new output directory to evade that limit.
Every result must match one preceding intent for the same unit, request and attempt;
duplicate or unmatched results stop recovery before backend startup. First valid
answers are selected in original attempt order, even if an older orphan receipt is
recovered after a later result. Historical ledger rows and receipt files are not rewritten.

A fatal receipt discovered during recovery is recorded and pauses that invocation
before backend startup. Diagnose it before a subsequent explicit `--resume`, which
may retry an already recorded fatal response using only the remaining attempt budget.
An incomplete legacy receipt, including a fatal orphan, must retain its
authoritative recovery fields and a finite `completed_at_unix`. Missing or invalid
timing stops recovery without inventing a completion time or resetting attempts.
Complete legacy receipts need no retry time.

Each continuation retains its own startup, telemetry and failure records.
An incomplete final result is still incomplete; `completed` means the bounded
queue finished, not that every question succeeded. There is no automatic infinite
retry loop. Inspect `status.json` and the current invocation's logs before resuming.

Receipts remain strict JSON. Readers accept both historical
`python-json-nonfinite-v1` global and `json-text-nonfinite-v1` per-field raw
encodings through the strict receipt decoder. New writes retain the per-field
format. Raw nonfinite values remain lossless diagnostics, never valid scores;
optional invalid cost metadata remains unknown. Preserve original receipt bytes.

For each phase, keep `run.json`, `attempts.jsonl`, `receipts/`, `report.json` and
`completion.json`; a failed stage leaves a partial report in its invocation.
CLIP/SigLIP2 also keep `thresholds.json`. Keep runtime/smoke, hardware, GPU telemetry,
peak allocation records and `result-files-sha256.json`. Verify that inventory after
copying results locally, before releasing a GPU with unbacked-up files. The inventory
excludes itself; earlier inventory snapshots remain in continuation records.

The test report includes combined202, original120/robot82 and per-event results, missing units,
incomplete questions and successful-attempt P50/P95. Latency excludes worker queue
and retry backoff; CLIP's preprocessing/GPU-lock wait/readback are retained separately
inside each receipt. Restarts can change cache warmth and are recorded. These are
test timings, not an independent 12-camera throughput measurement. Actual GPU cost
requires the rental bill or verified rate and duration; unknown cost is not zero.

Results stay local. Publishing to a sheet is a separately requested operation,
with live-header matching and existing styles preserved. Historical results and
the separate paced CLIP runner are not overwritten or relabelled by this flow.

## Offline regression checks

In the lightweight preflight environment (Pillow and pytest; no torch required):

```sh
PYTHONDONTWRITEBYTECODE=1 python -m pytest \
  tests/test_native_workflow.py tests/test_unified_benchmark.py \
  tests/test_eval_review_regressions.py -o addopts='' -q -p no:cacheprovider
```

Source-export tests need the two public Git commits locally. A shallow checkout
can fetch each pinned commit first; the tests themselves never fetch, install,
download weights, start services or call a model. Synthetic results only verify
the flow; they are never evaluation results for the shared table.
