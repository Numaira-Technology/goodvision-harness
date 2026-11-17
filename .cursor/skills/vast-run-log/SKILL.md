---
name: vast-run-log
description: >-
  Repo Vast GPU rentals must call tools.vast_run_log.attach() and then rent,
  SSH, and release only through the tools.start_sam helpers. Direct
  create_instance, the vastai CLI, and REST are not recorded.
---

# Vast run log

Every Vast GPU rental in this repo must call `tools.vast_run_log.attach()` at
the start of `main`, then rent, SSH, and release only through the
`tools.start_sam` helpers `create_from_offers`, `run`, and `release_instance`.
`attach()` records those helper calls. A direct `vast.create_instance`, the
`vastai` CLI, or the REST API is not recorded and is not allowed for repo
work. There is no opt-out.

`tools/vast_run_log.py` is the only recorder. Do not copy the probe, the
sampler, offer redaction, or the manifest writer.

`tools/start_qwen8b_r2.py` (HF Qwen) is already covered: `tools/__init__.py`
calls `arm_if_launcher` for that rental file, with sampling off. Do not edit
that file. Every other rental calls `attach()` itself.

The package hook also logs `tools.unified_benchmark.native_run --execute` for
`clip`, `siglip2`, `videojev`, and `decision3`, without editing the frozen runner.
Native runs use a bounded local probe without login shell startup scripts and
no sampler. The hook records the CLI outcome and wall time before probing.
Only successful invocations with a fresh `--output` can claim their
`<output>/<split>/report.json` score; failed or resumed invocations keep the score
`unknown`. Native logs remain in the stable series directory and are never
mirrored into the sealed native result inventory. No Vast offer is fabricated.

## Call

```python
def main() -> int:
    try:
        from tools.vast_run_log import attach
    except ImportError:
        from vast_run_log import attach
    from tools import start_sam

    attach()
    instance_id = start_sam.create_from_offers(
        vast, offers, disk_gb=disk_gb, label=label
    )
    try:
        start_sam.run([*ssh, "nvidia-smi"])
    finally:
        start_sam.release_instance(vast, instance_id)
```

`attach()` reads the running filename from the caller's globals. A notebook or
other caller passes `attach(globals())` and must first set `__file__` to its
registered script filename. Without `__file__`, no recorder is attached.

Call helpers through the module as above, or import them at module scope. Do
not bind helper functions to local variables before `attach()`: it patches
module attributes and launcher globals, but cannot replace local references.

Register that filename in `LAUNCHERS` inside `tools/vast_run_log.py` as
`(module, alias, engine, sample)` before the run. `sample` is `True` except
for the HF launcher. Without that row, `attach()` does not record the rental.

`attach()` patches `create_from_offers`, `run`, and `release_instance` in the
launcher namespace and on `tools.start_sam`. The offer is saved when
`create_from_offers` accepts a contract. The probe runs on the first `run()`
SSH command. `release_instance` stops sampling and marks the rental released.
Calling `vast.create_instance` yourself skips the offer, the probe, and the
samples.

### Required inputs

- **alias.** The `LAUNCHERS` alias for this filename. If the process has
  `--model`, that flag overrides the table. Example: `qwen8b_r2_vllm`.
- **eval-output.** `--eval-output` on `sys.argv` when the run writes a report.
  The stable copy is always `outputs/run-logs/<alias>/logs/`. The copy under
  `<eval-output>/<alias>/logs/` and the score are written only when that
  eval-output directory did not exist at eval start. A resume or a reused
  directory is left alone, and its score stays `unknown`, so an older report
  is not claimed. Use a fresh `--eval-output` directory for each run.

`--authorization-note` is the launcher's own approved-run sentence when that
launcher requires one. `attach()` does not read it and does not grant
permission.

## Operator

Who is recorded in this order: `GOODVISION_OPERATOR` if set, otherwise
`git user.name`, otherwise `$USER` (`%USERNAME%` on Windows). Leave
`GOODVISION_OPERATOR` unset so the git name is the operator.

## Where the files go

`GOODVISION_RUN_LOG_ROOT` overrides the log root. Unset, the root is
`outputs/run-logs/<alias>/logs` in the repo. Set, it is
`<GOODVISION_RUN_LOG_ROOT>/<alias>/logs`. If that path would sit inside
`--eval-output`, the recorder moves it beside the eval output so the eval
mkdir is not the log directory. Leave the variable unset unless the logs
should leave `outputs/run-logs`.

## When sampling starts

Sampling is on for every launcher except `start_qwen8b_r2.py`. It starts when
the worker is ready: at the return of `_wait_ready` or `wait_for_worker` when
the launcher defines one of those. A launcher with neither starts sampling
after the first SSH `run()`, which is also when the one-time probe runs.
Samples stop when evaluation finishes, or inside `release_instance` before
the instance is destroyed. A caller retaining an instance must call
`tools.vast_run_log.finish()` before removing its temporary SSH credentials.
The interval is 2 seconds.

## What is recorded

- Offer fields from the accepted Vast offer, with secret-looking values removed.
- One nvidia-smi probe: PCIe gen and width (current and max), power limit,
  SM clocks, utilization, driver, plus CPU model, cores, RAM, cgroup limits,
  and CUDA once. The stored probe is stdout. The Vast SSH welcome banner on
  stderr is not kept.
- GPU samples every 2 seconds from ready until eval completion or release:
  PCIe gen and width, power draw and limit, SM clock, utilization, memory
  used, temperature, and `t_unix`.
- Timings: rent, ready, eval start, eval end, release.
- Score read from that run's `summary.json` when this run created the
  eval-output directory. Otherwise the score is `unknown`.
- Manifest `status`: `rented` after a contract or a resumed instance id,
  `not_rented` when `main` exits before any rental (bundle check, budget
  precheck, or no offer). A `not_rented` row is an attempt, not a machine.

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

`<eval-output>/<alias>/logs/` is written only when that eval-output directory
did not exist before the eval. `manifest.json` has one row per attempt (time,
who, status, machine, GPU, price, location, score) and `samples` (a path, or
`unknown`). `entry.json` `hardware.samples` has the path, the one-time driver
/ CUDA / CPU record, and the per-phase min, median, and p95.

## Verify

After the run, stderr contains a `run log:` line with the `entry.json` path.
That file exists. On a rental, `vast-offer.json` has the machine and no
secrets. `outputs/run-logs/<alias>/logs/manifest.json` has a new row whose
`who.display` is `GOODVISION_OPERATOR` or `git user.name`. A run that exits
before create has `status` `not_rented`. When sampling ran,
`gpu-samples.jsonl` exists and `hardware.samples.path` points at it.

Colleague command (same launcher as before; the recorder is already inside it):

```sh
PYTHONPATH=. python -m tools.start_qwen8b_vllm \
  --eval-bundle outputs/new-qwen-r2-bundle \
  --eval-output outputs/qwen8b-r2-vllm \
  --eval-phase both \
  --authorization-note 'Approved self-hosted qwen8b R2 vLLM 4090 run'
```

Use a fresh `--eval-output` directory. The authorization note is the
launcher's check, not an `attach()` argument.

## Checklist

- `attach()` runs at the start of `main`, and the filename is in `LAUNCHERS`.
- Rent, SSH, and release go through `create_from_offers`, `run`, and `release_instance`.
- No direct `vast.create_instance`, `vastai` CLI, or REST create.
- `--eval-output` is a fresh directory when the run writes a report.
- `--authorization-note` is set when that launcher requires it. `attach()` does not read it.
- `GOODVISION_OPERATOR` is unset unless it should override `git user.name`.
- `GOODVISION_RUN_LOG_ROOT` is unset unless logs should leave `outputs/run-logs`.
- `tools/start_qwen8b_r2.py` was not edited.
- No copied probe, no opt-out flag, no R2 upload, no offer or scoring change.
- `tests/test_vast_run_log.py::test_every_vast_rental_calls_the_recorder` passes.
