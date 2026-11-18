---
name: event-benchmark
description: >-
  Prepare, run and report the surveillance event benchmark and reviewed robot-video
  supplement (fight/smoking/fall/weapon).
  Use for benchmark/model comparisons, 跑一下测评, 测速, or preparing benchmark tables.
  Preserves reviewed inputs, protocol versions, fixed baselines and measured provenance.
---

# Event-recognition benchmark

Default for this team's current evaluation: **unified-frame-202-v1**.
The current dataset is **222 test questions on 202 videos** when the requested
test supplement below is included; the retained 202-question baseline stays available.
Follow **preflight → environment checks → validation → threshold selection when
applicable → test → local report**. Publishing is a separately scoped action.
Use [the unified run guide](../../../tools/unified_benchmark/README.md), not the
legacy `tools/event_benchmark/run_eval.py` examples, for this default protocol.
These instructions use the user's existing scope and budget; they do not by
themselves authorize paid inference, downloads, uploads or publication.

## Before starting

Read the [handoff checklist](../../../tools/unified_benchmark/README.md#handoff-checklist).
It covers the code revision, eight private frozen JSON files (five original plus three202 files), original videos,
credentials or pinned weights, hardware, cost authorization and output locations.
Keep the original input bytes and filenames in a private `--inputs` directory.
Do not substitute the repository's legacy manifest or alter hashes to pass a check.
Another agent must read this file and its selected execution guide; discovering
`.claude/skills` automatically is not a prerequisite.

## Current202 workflow

1. **Preflight:** run `tools.unified_benchmark.preflight` for the selected profiles.
   For the current222 test, add `--test-supplement /private/path/test-supplement.json`
   from [the unified dataset folder](https://drive.google.com/drive/folders/1kOyH59vxBp3uppyEoeyMjTMUa_CalTdh):
   [supplement](https://drive.google.com/file/d/1SgKsoMVYQm9w5a1fE84bR7dcxPuscEbV/view),
   [all222 questions](https://drive.google.com/file/d/1qvcl8vEWNcfxlrw_DhU79Tfxe1q2VBQg/view),
   [all202 videos](https://drive.google.com/file/d/1uYN1PdiF13Aa1L6yj88zDACaO-0i5kqd/view).
   Use the existing downloader with the latter list. Keep the original eight
   frozen inputs unchanged; do not replace their `samples.json` with `samples_222.json`.
   Haotian's ten bag recordings add questions203-222 in the existing bag categories.
   Their labels are assistant annotations, not human-confirmed. Ashley's ten bag
   validation recordings remain separate; do not tune thresholds on the new test.
   Verify all eight frozen inputs and video hashes, decode frames and save the new
   request/code snapshot. Keep pending hardware/provider/authorization checks
   separate from input validity. This step performs no model calls.
2. **Environment:** follow the execution route below. Verify pinned versions,
   credentials/access and hardware; do the authorized original warmup/smoke before
   evaluation. Offline checks do not prove a new GPU deployment can run.
3. **Validation:** for the current repeated-evaluation task, use validation44 on
   34 videos for every selected model. Include the existing private
   [bag validation supplement](../../../tools/unified_benchmark/NATIVE.md#bag-validation-for-every-model)
   in preflight; missing inputs block execution, with no fallback to validation24. Preserve
   receipts, complete scores, missing units and run identity.
4. **Thresholds, when applicable:** use `bench-threshold-tune` and the route's own
   selector for Luna/CLIP/SigLIP2; require complete validation, then freeze the
   artifact before test. Boolean/argmax models do not gain a scalar threshold.
5. **Test:** run the frozen configuration on all planned questions (222 with the
   supplement, otherwise202). Preserve successful
   answers and the cumulative three-attempt budget across interruptions. For the
   native, API and local Qwen CLIs, use explicit `--resume` for compatible
   outputs; never use a fresh ledger to reset the budget.
6. **Local report:** verify counts, coverage, unknowns, source identities, matching
   speed and actual costs using [report fields](references/report-format.md).
   Retain raw receipts, frozen inputs/configs and historical results.

The [202 dataset guide](../../../tools/unified_benchmark/DATASET202.md) defines
questions1–120 unchanged plus82 single views in test, with 24 base validation questions.
Keep inherited/unreviewed labels explicit; preflight is not human review. Report
original120, robot82 and any test supplement separately alongside correct/all planned,
and never count the old41
stereo sources again. Do not tune on the new test questions.

## Repeated runs

For the current ten-round task, follow [BATCH.md](../../../tools/unified_benchmark/BATCH.md).
Run `batch check` first, then the authorized `batch run`. Use `--detach` only when
the host's network and credentials remain available after the attached command
ends; otherwise keep an attached command session. Detached mode supervises bounded
controller restarts, but cannot preserve a session-scoped network proxy.
Validate once, freeze applicable thresholds,
and execute ten independent test rounds. Resume the original matrix and deadlines.
Read `state.json`, logs and final coverage; a PID or exit zero is not proof of complete
evidence. Preserve incomplete/failed rounds; do not add a replacement eleventh run.

The batch supports pre-existing GPU deployments/services. It does not rent or
release instances, restore R2 packs, or prove a live deployment from offline tests.
Verify ownership or independent lease cleanup, actual hardware identity, pinned
runtime, current provider access and bounded costs before paid execution. Missing
materials or an unverified cleanup arrangement remain concrete blockers.

## Execution routes

| Selected models | Guide and entrypoint |
|---|---|
| OpenRouter API profiles | [API guide](../../../tools/unified_benchmark/README.md#api-execution-after-cost-approval); `tools.unified_benchmark.api_run` |
| CLIP, SigLIP2, VideoJev, Decision3 | [Native guide](../../../tools/unified_benchmark/NATIVE.md); pinned deployment then `native_run --phase validation\|test\|both` |
| Qwen8B R2 / Transformers | [Self-hosted Qwen guide](../../../tools/unified_benchmark/README.md#self-hosted-qwen8b-from-r2); `qwen8b_r2_host` via `tools.start_qwen8b_r2` |
| Qwen8B R2 / vLLM | Same guide; `qwen8b_r2_vllm` via `tools.start_qwen8b_vllm` |

The four-model native CLI without `--execute` checks files only. Qwen R2 launchers
can rent a GPU and run the selected phases; they are not offline preflight commands.
Those launchers also write a hardware run log with no extra flag. The same log
is written for `python -m tools.unified_benchmark.native_run --execute` when
`--model` is `clip`, `siglip2`, `videojev`, or `decision3` (the d3-nano
weights). Other models are not armed. Optional
`GOODVISION_OPERATOR` sets the name on the index; otherwise the log uses git
`user.name` and `$USER` (Windows: `%USERNAME%`). The series index is
`outputs/run-logs/<alias>/logs/manifest.json`. Each run folder there holds
`vast-offer.json` (redacted offer, written when the instance is created) and
`entry.json` (offer fields, nvidia-smi probe, timings, score). Rental logs are
also copied under a fresh `<eval-output>/<alias>/logs/`. Native logs stay outside
the sealed result directory and read `<output>/<split>/report.json` only after
a successful invocation with a fresh output. Failed or resumed native runs keep
the score `unknown`; the log records the CLI outcome without changing it.
A logging failure is printed and does not change the rental, the exam, or the
exit code. Details: [Hardware run log](../../../tools/unified_benchmark/README.md#hardware-run-log).
Keep the HF, vLLM and OpenRouter Qwen variants separate. The active vLLM environment
is `envs/qwen8b-vllm/cu130-py312-vllm0310-ada2.tar.gz`; use the exact registry digest
and restore instructions in [the package guide](../../../tools/qwen8b_vllm/BUILD.md#restore-the-current-package).
The old cu128 and incomplete cu130 `…ada.tar.gz` objects remain historical, not
fallbacks. Preserve the HF pack. Do not rebuild an environment during an ordinary
rerun or lower concurrency, shrink images or shorten outputs to fit the GPU.

The self-hosted Qwen3-VL-8B HF path is profile `qwen8b_r2_host`
(`execution=self_hosted_r2`): restore the packed env and pinned weights from R2,
serve with `tools.qwen8b.serve`, and score with `tools.qwen8b.eval_local` through
`tools.start_qwen8b_r2`. Do not use a vLLM `qwen8b_selfhost` profile or the
OpenRouter `qwen8b` Parasail cell for that comparison.

An independently authorized serving engine is profile `qwen8b_r2_vllm` via
`tools.start_qwen8b_vllm` (vLLM OpenAI server, same R2 weights revision, bf16).
The active declared pack is `envs/qwen8b-vllm/cu130-py312-vllm0310-ada2.tar.gz`
(torch `+cu130`; requires host CUDA >=13.0 and NVIDIA driver >=580). Its exact
digest/size are in the profile and registry. This published key is frozen;
any CPU rebuild needs a new reviewed key. The older cu130 `ada` pack is
incomplete and refused. Keep historical objects and records. The 4090 only
restores the pinned artifact. Do not overwrite the HF qwen8b pack. CPU import ≠ GPU serve. It does not
replace or relabel the HF run. Unified-frame-120-v2 exam rules stay
exactly as below: frozen JPEGs, prompts, parser, temperature, max_tokens=30,
thresholds, retries, and **12 queue slots**. Do not lower client concurrency,
shrink images, truncate context, or shorten outputs to fit a GPU.
The launcher verifies the frozen bundle and active pack identity before rental
and again before inference. Its separate 720px synthetic smoke is retained as a
diagnostic; it must never resend a scored validation/test unit outside the ledger.
Updated profiles require a fresh bundle. Preserve historical bundles unchanged.

## Fixed interpretation

The current variant uses exact 1 FPS shared 720px JPEGs and current-image API prompts.
For current202, the last decoded video timestamp bounds the sampling grid;
`media-audit.json` records any excluded tail points without duplicating frames.
Retained frames must all decode successfully. The variant also uses
explicit effort settings when supported, and exact JSON string `"true"`/`"false"`
normalization (never general truthiness). It uses 12 queue slots; that is not proof
of twelve-camera real-time throughput. Each unit has at most three total attempts
(initial plus two retries); keep the first valid answer.

Every planned response must be valid before a question can contribute a correct
answer. For complete evidence: any yes → yes; otherwise any no → no; otherwise
unknown. Headline accuracy is correct / all planned questions (222 with the current
supplement; 202 for the retained baseline). Missing questions stay in the denominator
and are reported separately.
Native/bag unknown is a valid response; a final unknown is wrong against the binary
references. Refusal or malformed output is missing, not unknown.

CLIP/SigLIP2 retain single-frame margins. VideoJev/Decision3 retain adjacent
two-frame windows and native argmax; the proposed single-frame experiment was
not run. Changing prompts, effort, windows, parser, thresholds or provider creates
a new recorded variant. Historical bundles need their retained source snapshot;
do not relabel their protocol or silently re-score old output files.

## Results and shared-sheet publication

An agent can prepare the local update preview and perform an authorized sheet
update; manual cell-by-cell transcription is not required. Route publication to
[bench-sheet-contribute](../bench-sheet-contribute/SKILL.md), which reads live
headers, matches the run/variant, preserves styling and verifies saved cells.
The user chooses the result version, target tabs and overwrite/append scope.
Show the proposed cell/note changes first when review was requested. An evaluation
request alone does not authorize overwriting the shared results, and publication
does not authorize new model calls. No new columns or tabs by default.

## Other protocols (read only when selected)

- `unified-frame-120-v2`: explicit `preflight --protocol unified-frame-120-v2`;
  five original frozen files,120 test questions. Old bundles keep their original code.
- [Legacy v2.1](references/legacy-v2.1.md): historical centered sampling and strict
  parser. This checkout includes its controlled/legacy CLI; keep the matching
  retained code and inputs with historical artifacts.
- [Robot41 supplement](references/robot41.md): historical 41-question fall/weapon
  cohort. Its downloader and reviewed JSON files are included; use their matching
  fingerprints for an explicitly selected run. Never mix it into the 120 denominator.
- [Model classes](references/model-classes.md): historical runner boundaries.

Current202, current120 and legacy protocols have different fingerprints and interpretation.
Shared table formatting does not make their results a controlled comparison.

For an explicitly requested robot41 legacy run after its own input verification,
the retained entrypoint is:

```sh
python tools/event_benchmark/run_eval.py --legacy-baseline --candidate qwen \
  --samples tools/event_benchmark/datasets/samples_robot41.json \
  --manifest tools/event_benchmark/datasets/manifest_robot41.json \
  --media outputs/robot41/media --interval 1.0 --workers 12 \
  --out outputs/event-benchmark/robot41-qwen.json
```
