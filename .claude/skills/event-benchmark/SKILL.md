---
name: event-benchmark
description: >-
  Prepare, run and report the surveillance event benchmark and reviewed robot-video
  supplement (fight/smoking/fall/weapon).
  Use for benchmark/model comparisons, 跑一下测评, 测速, or preparing benchmark tables.
  Preserves reviewed inputs, protocol versions, fixed baselines and measured provenance.
---

# Event-recognition benchmark

Default for this team's current evaluation: **unified-frame-120-v2**.
Follow **preflight → environment checks → validation → threshold selection when
applicable → test → local report**. Publishing is a separately scoped action.
Use [the unified run guide](../../../tools/unified_benchmark/README.md), not the
legacy `tools/event_benchmark/run_eval.py` examples, for this default protocol.
These instructions use the user's existing scope and budget; they do not by
themselves authorize paid inference, downloads, uploads or publication.

## Before starting

Read the [handoff checklist](../../../tools/unified_benchmark/README.md#handoff-checklist).
It covers the code revision, five private frozen JSON files, original videos,
credentials or pinned weights, hardware, cost authorization and output locations.
Keep the original input bytes and filenames in a private `--inputs` directory.
Do not substitute the repository's legacy manifest or alter hashes to pass a check.
Another agent must read this file and its selected execution guide; discovering
`.claude/skills` automatically is not a prerequisite.

## Current120 workflow

1. **Preflight:** run `tools.unified_benchmark.preflight` for the selected profiles.
   Verify all five frozen inputs and video hashes, decode frames and save the new
   request/code snapshot. Keep pending hardware/provider/authorization checks
   separate from input validity. This step performs no model calls.
2. **Environment:** follow the execution route below. Verify pinned versions,
   credentials/access and hardware; do the authorized original warmup/smoke before
   evaluation. Offline checks do not prove a new GPU deployment can run.
3. **Validation:** run the same frozen configuration on validation24. Preserve
   receipts, complete scores, missing units and run identity.
4. **Thresholds, when applicable:** use `bench-threshold-tune` and the route's own
   selector for Luna/CLIP/SigLIP2; require complete validation, then freeze the
   artifact before test. Boolean/argmax models do not gain a scalar threshold.
5. **Test:** run the frozen configuration on all 120 questions. Preserve successful
   answers and the cumulative three-attempt budget across interruptions. For the
   four-model native CLI, use explicit `--resume`; other routes follow their own
   documented continuation support, never a fresh ledger to reset the budget.
6. **Local report:** verify counts, coverage, unknowns, source identities, matching
   speed and actual costs using [report fields](references/report-format.md).
   Retain raw receipts, frozen inputs/configs and historical results.

## Execution routes

| Selected models | Guide and entrypoint |
|---|---|
| OpenRouter API profiles | [API guide](../../../tools/unified_benchmark/README.md#api-execution-after-cost-approval); `tools.unified_benchmark.api_run` |
| CLIP, SigLIP2, VideoJev, Decision3 | [Native guide](../../../tools/unified_benchmark/NATIVE.md); pinned deployment then `native_run --phase validation\|test\|both` |
| Qwen8B R2 / Transformers | [Self-hosted Qwen guide](../../../tools/unified_benchmark/README.md#self-hosted-qwen8b-from-r2); `qwen8b_r2_host` via `tools.start_qwen8b_r2` |
| Qwen8B R2 / vLLM | Same guide; `qwen8b_r2_vllm` via `tools.start_qwen8b_vllm` |

The four-model native CLI without `--execute` checks files only. Qwen R2 launchers
can rent a GPU and run the selected phases; they are not offline preflight commands.
Keep the HF, vLLM and OpenRouter Qwen variants separate. The active vLLM environment
is `envs/qwen8b-vllm/cu130-py312-vllm0310-ada2.tar.gz`; use the exact registry digest
and restore instructions in [the package guide](../../../tools/qwen8b_vllm/BUILD.md#restore-the-current-package).
The old cu128 and incomplete cu130 `…ada.tar.gz` objects remain historical, not
fallbacks. Preserve the HF pack. Do not rebuild an environment during an ordinary
rerun or lower concurrency, shrink images or shorten outputs to fit the GPU.

## Fixed interpretation

The current variant uses exact 1 FPS shared 720px JPEGs, current-image API prompts,
explicit effort settings when supported, and exact JSON string `"true"`/`"false"`
normalization (never general truthiness). It uses 12 queue slots; that is not proof
of twelve-camera real-time throughput. Each unit has at most three total attempts
(initial plus two retries); keep the first valid answer.

Every planned response must be valid before a question can contribute a correct
answer. For complete evidence: any yes → yes; otherwise any no → no; otherwise
unknown. Headline accuracy is correct / all planned questions (120 for the full
test). Missing questions stay in the denominator and are reported separately.
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

- [Legacy v2.1](references/legacy-v2.1.md): historical centered sampling and strict
  parser. Requires its matching retained code snapshot; this checkout does not
  implement the documented controlled/legacy CLI flags.
- [Robot41 supplement](references/robot41.md): historical 41-question fall/weapon
  cohort. Its downloader and frozen JSON files are absent here; obtain the matching
  snapshot before considering a run. Never mix it into the 120 denominator.
- [Model classes](references/model-classes.md): historical runner boundaries.

Current120 and legacy protocols have different fingerprints and interpretation.
Shared table formatting does not make their results a controlled comparison.
