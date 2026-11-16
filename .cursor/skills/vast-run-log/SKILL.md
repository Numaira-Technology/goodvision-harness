---
name: vast-run-log
description: >-
  Record every Vast GPU rental with tools/vast_run_log.py: offer, nvidia-smi
  probe, 2s GPU samples, timings, score, and manifest. Use when adding or
  changing a script that rents a Vast instance, or when reading run logs.
---

# Vast run log

One recorder, `tools/vast_run_log.py`, owns the hardware probe, GPU samples,
offer fields, timings, score, and manifest. Launchers call it. They do not
copy that logic.

VideoJev (`tools/videojev_probe`, `tools/videojev_benchmark`) and
`tools/vast_vlm` do not rent a Vast instance. Do not attach the recorder there.

## Call

At the start of `main`, before the instance is created:

```python
def main() -> int:
    try:
        from tools.vast_run_log import attach
    except ImportError:
        from vast_run_log import attach
    attach()
```

Add the script's filename to `LAUNCHERS` in `tools/vast_run_log.py`:
`(module, alias, engine, sample)`. `sample` is `True` except the HF Qwen
launcher below. `attach()` reads the alias from that table and reads
`--eval-output` / `--model` from `sys.argv`.

`tools/start_qwen8b_r2.py` is the exception. Do not edit it. `tools/__init__.py`
calls `arm_if_launcher` for that file only, with sampling off. A new renter
must call `attach()` itself.

Logging is always on. Do not add a flag that skips it. A recorder failure is
printed and does not change the rental, the eval exit code, offer selection,
scoring, prompts, frames, retry, or client concurrency.

## Colleague command

Leave `GOODVISION_OPERATOR` unset to record `git user.name`. Set it only when
the name should differ from git. Then run the existing launcher, for example:

```sh
PYTHONPATH=. python -m tools.start_qwen8b_vllm \
  --eval-bundle outputs/new-qwen-r2-bundle \
  --eval-output outputs/qwen8b-r2-vllm \
  --eval-phase both \
  --authorization-note 'Approved self-hosted qwen8b R2 vLLM 4090 run'
```

The same pattern covers:

```sh
PYTHONPATH=. python -m tools.start_qwen8b_r2 --eval-bundle ... --eval-output ...
python tools/start_sam.py
python tools/start_detector.py
PYTHONPATH=. python -m tools.diag_qwen8b_vllm_import
```

## Output

```text
outputs/run-logs/<alias>/logs/manifest.json
outputs/run-logs/<alias>/logs/<run-id>/vast-offer.json
outputs/run-logs/<alias>/logs/<run-id>/entry.json
outputs/run-logs/<alias>/logs/<run-id>/gpu-samples.jsonl
<eval-output>/<alias>/logs/
```

`<eval-output>/<alias>/logs/` is a copy written after that report directory
exists. `manifest.json` is one row per run (time, who, machine, GPU, price,
location, score) plus `samples`, a path to `gpu-samples.jsonl` or `unknown`.
`entry.json` `hardware.samples` holds that path, the one-time driver / CUDA /
CPU record, and a per-phase summary (min, median, p95 of utilization, SM clock,
and power; fraction of busy time at a PCIe gen below the max).

Samples are one JSON object per line, with `t_unix` aligned to the controller
clock so they can sit next to `summary.json` `started_at_unix` /
`finished_at_unix`. The loop uses `nvidia-smi`, `/proc`, and `sh` already on
the image, every 2 seconds, from server-ready until just before release. It
does not pip-install on the GPU and is not uploaded to R2.

Missing measurements are `unknown`, not 0. Idle PCIe gen 1, a low SM clock
such as 210 MHz, and 0% utilization are power-saving, not throttling.

## Checklist

- The script contains `from vastai import VastAI` or it does not rent.
- Its filename is a key of `LAUNCHERS`, with the right alias and `sample`.
- `main` calls `attach()` with the import above, and nowhere else copies the probe.
- `tools/start_qwen8b_r2.py` was not edited.
- No opt-out flag, no R2 upload, no offer-search or scoring change.
- The sampler string has no `pip`.
- `tests/test_vast_run_log.py::test_every_vast_rental_calls_the_recorder` passes.
