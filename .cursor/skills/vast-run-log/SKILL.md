---
name: vast-run-log
description: >-
  Any Vast GPU rental must call tools.vast_run_log.attach() before create.
  Covers every launcher, ad-hoc script, notebook, agent, VideoJev or future
  model, and the vastai SDK, CLI, or REST. Records the offer, probe, 2s GPU
  samples, timings, and score.
---

# Vast run log

Any Vast GPU rental must call `tools.vast_run_log.attach()` before the instance
is created. That includes every launcher, an ad-hoc or one-off script, a
notebook, another agent, VideoJev, any future model, and a rental made with
the vastai SDK, the `vastai` CLI, or the REST API. There is no opt-out.

`tools/vast_run_log.py` is the only recorder. Do not copy the probe, the
sampler, offer redaction, or the manifest writer.

`tools/start_qwen8b_r2.py` (HF Qwen) is already covered: `tools/__init__.py`
calls `arm_if_launcher` for that file only, with sampling off. Do not edit
that file. Every other rental calls `attach()` itself.

## Call

Put this at the start of `main`, before `create_instance`, `create_from_offers`,
`search_offers`, `vastai` CLI, or REST create:

```python
def main() -> int:
    try:
        from tools.vast_run_log import attach
    except ImportError:
        from vast_run_log import attach
    attach()
```

`attach()` takes no arguments for a supported launcher. It reads the running
filename and patches that launcher's shared rental/SSH/readiness helpers.
It does not intercept arbitrary `VastAI.create_instance`, CLI, or REST calls.
A new script or notebook needs an explicit `__file__` in the passed namespace,
a `LAUNCHERS` registration, and the same helper integration, with an offline
creation/readiness/release test before use. Merely calling `attach(globals())`
does not provide coverage for an unregistered caller or another transport.

Register that filename in `LAUNCHERS` inside `tools/vast_run_log.py` as
`(module, alias, engine, sample)` before the run. `sample` is `True`.
Without that row, `attach()` does not record the rental.

### Required inputs

- **alias.** The `LAUNCHERS` alias for this filename. If the process has
  `--model`, that flag overrides the table. Example: `qwen8b_r2_vllm`.
- **eval-output.** `--eval-output` on `sys.argv` when the run writes a report.
  The stable copy is always `outputs/run-logs/<alias>/logs/`. When
  `<eval-output>/<alias>/` exists, the same folder is copied to
  `<eval-output>/<alias>/logs/`.
- **authorization note.** `--authorization-note` on `sys.argv`: the same
  approved-run sentence the launcher already requires. `attach()` does not
  grant permission and does not read a new flag.

## Operator

Who is recorded in this order: `GOODVISION_OPERATOR` if set, otherwise
`git user.name`, otherwise `$USER` (`%USERNAME%` on Windows). Leave
`GOODVISION_OPERATOR` unset so the git name is the operator.

## What is recorded

- Offer fields from the accepted Vast offer, with secret-looking values removed.
- One nvidia-smi probe: PCIe gen and width (current and max), power limit,
  SM clocks, utilization, driver, plus CPU model, cores, RAM, cgroup limits,
  and CUDA once.
- GPU samples every 2 seconds from server-ready until evaluation ends or the
  instance is released (diagnostics start after the first successful SSH command):
  GPU index, PCIe gen and width, power draw and limit, SM clock, utilization,
  memory used, temperature, and `t_unix`. Multi-GPU rentals retain each GPU's
  rows and separate per-GPU phase statistics.
- Timings: rent, ready, eval start, eval end, release.
- Score read from that run's `summary.json` (unknown when there is no exam).

Missing numbers are `unknown`, not 0. Idle PCIe gen 1, a low SM clock such as
210 MHz, and 0% utilization are power-saving, not throttling. The sampler uses
only `nvidia-smi`, `/proc`, and `sh` on the image. No pip on the GPU. The log
is not uploaded to R2. Offer search, prompts, scoring, thresholds, frames,
retry, and client concurrency stay unchanged. A recorder failure is printed
and does not change the rental or the eval exit code.

## Output

```text
outputs/run-logs/<alias>/logs/manifest.json
outputs/run-logs/<alias>/logs/<run-id>/vast-offer.json
outputs/run-logs/<alias>/logs/<run-id>/entry.json
outputs/run-logs/<alias>/logs/<run-id>/gpu-samples.jsonl
<eval-output>/<alias>/logs/
```

`manifest.json` has one row per run (time, who, machine, GPU, price, location,
score) and `samples` (a path, or `unknown`). `entry.json` `hardware.samples`
has the path, the one-time driver / CUDA / CPU record, and the per-phase
min, median, and p95. A replacement rental after startup failure gets its own
run ID and manifest row, so offers and SSH targets cannot mix between hosts.
Phase labels use only a report created by this invocation and a known clock
offset; otherwise statistics describe the sampled run without phase attribution.
Unavailable PCIe generations are excluded from the busy-time fraction.
A launcher that can retain an instance must call `finish()` before removing its
temporary SSH credentials; `start_sam.py` does this even if analysis backup fails.

## Verify

After the run, stderr contains a `run log:` line with the `entry.json` path.
That file exists, `vast-offer.json` has the machine and no secrets, and
`outputs/run-logs/<alias>/logs/manifest.json` has a new row whose `who.display`
is `GOODVISION_OPERATOR` or `git user.name`. When sampling ran,
`gpu-samples.jsonl` exists and `hardware.samples.path` points at it.

Colleague command (same launcher as before; the recorder is already inside it):

```sh
PYTHONPATH=. python -m tools.start_qwen8b_vllm \
  --eval-bundle outputs/new-qwen-r2-bundle \
  --eval-output outputs/qwen8b-r2-vllm \
  --eval-phase both \
  --authorization-note 'Approved self-hosted qwen8b R2 vLLM 4090 run'
```

## Checklist

- The rental calls `attach()` before create, whatever the client (SDK, CLI, REST, notebook, agent).
- The filename is in `LAUNCHERS` with an alias and `sample` True.
- `--eval-output` and `--authorization-note` are set when that launcher has them.
- `GOODVISION_OPERATOR` is unset unless it should override `git user.name`.
- `tools/start_qwen8b_r2.py` was not edited.
- No copied probe, no opt-out flag, no R2 upload, no offer or scoring change.
- `tests/test_vast_run_log.py::test_every_vast_rental_calls_the_recorder` passes.
