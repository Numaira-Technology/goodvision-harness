# Qwen3-VL-8B latency measurement

This directory prepares a hardware latency comparison. It does not rent a GPU,
does not upload to R2, and does not change the formal 202-question benchmark.
Questions, prompts, frames, parser, decoding, `max_tokens`, scoring, retries,
the 12 client slots, and offer selection stay on their existing path.

The replay engines are pinned standard tools:

| Role | Tool | Version |
|---|---|---|
| Primary, closed loop | `vllm bench serve` custom image dataset | vLLM **0.31.0** (the restored exam environment) |
| Secondary, timestamp trace | AIPerf `profile` single-turn `--fixed-schedule` | AIPerf **0.13.0** |

This is a standard-method variant: production replay of a fixed recorded set at
fixed concurrency 12. It is not a new client and it is not the formal
benchmark latency.

## Clocks

Closed loop (`vllm bench serve --request-rate inf --max-concurrency 12`):

In vLLM 0.31.0 `vllm/benchmarks/serve.py`, the concurrency semaphore is taken
before `RequestFuncOutput.start_time`. Reported E2EL (`latencies` in the
`--save-detailed` JSON) starts at **actual HTTP send**. Time spent waiting for
a free client slot is `client_queue_time` and is not added to E2EL when the
request rate is infinite. The per-bucket table uses those `latencies` values.

Open loop (AIPerf 0.13.0 `--fixed-schedule`):

The exported file carries each recorded inter-arrival as `timestamp` in
milliseconds. AIPerf releases requests on that schedule. Its stored
`request_latency` starts at dispatch (`request_start_ns`), not at the
scheduled timestamp. The results table adds the least-late schedule lag back:

`lag = (request_start_ns - scheduled_ns) - min(request_start_ns - scheduled_ns)`

Primary P50/P95 use `request_latency + lag`, so client delay after the
scheduled send counts. `p50_actual_send_seconds` and
`p95_actual_send_seconds` keep the dispatch clock. The anchor is the
least-late request because AIPerf does not record schedule zero. The notes
column says `clock=scheduled_send`.

Neither command passes `ignore_eos`. `max_tokens` is the recorded cap. vLLM
bench serve sends that cap as `max_completion_tokens` and the model stops on
its own. `--save-detailed` keeps output token counts and generated text so
they can be compared with the baseline text stored in the manifest.

Warmup requests are taken from the complement of the timed set. Restart vLLM
with the existing `tools.qwen8b.serve_vllm` command, run the warmup file, then
start each timed round. The open-loop replay restarts and repeats the same
disjoint warmup so it does not inherit the closed-loop timed-set caches. Warmup latencies are not included in P50/P95. If the timed set uses the whole
population, there is no disjoint warmup and the plan omits that step. Weight
download and env restore are recorded on the hardware run log as
`timings.setup` and are not included either.

## What the run log already had

`tools/vast_run_log.py` already records the offer, CPU model, host cores, host
RAM, CUDA, cgroup limits, and per-GPU PCIe, clocks, power, temperature,
memory, and utilization. Those fields are not copied.

The same probe now adds one `EXTRA` line: CPU governor, turbo, NUMA node
count, each GPU's NUMA node, and compute-app process names. Per-GPU
utilization during the run remains the existing sample file. Client TCP RTT
and whether the HTTP port is an SSH `-L` tunnel or a direct port are stored
as `hardware.client_access` on the vLLM launcher.

## vLLM `/metrics`

Before and after the official eval, and in the replay command plan, scrape
`/metrics` and diff queue time, time to first token, prefill, decode, and
preemptions. The diff is a counter delta and a mean. It is not a P95. A reset
or a missing or changed series is `unknown`, not 0. Each replay has its own
before/after files (`closed-loop-metrics-*.prom`, `open-loop-metrics-*.prom`).
Warmup completes before the first timed snapshot; server-wide metrics can
include unrelated traffic, so run replays on an otherwise idle server.

## Per-request split timing

`GOODVISION_SPLIT_TIMING=1` writes `transport-splits.jsonl` next to the native
or Qwen eval output: upload done, first response-body byte, and read done.
The phases match `scene_model`'s HTTP trace. The client does not stream, so
one `read` of a short body can stamp first byte and read done together.
Leave the variable unset for an official run: the request bytes, the receipt,
and `elapsed_seconds` / `transport_seconds` stay on the existing path. The
formal P50/P95 definition is unchanged. Sidecar I/O is best effort and cannot
turn a successful request into a failure or replace its original HTTP error.

Changing `native_run.py`, `eval_local.py`, or `start_qwen8b_vllm.py` changes
the source hash a frozen bundle checks. The latency package, including the HTTP split dependency, is also frozen.
Create a new preflight bundle before
the next authorized GPU run. Do not edit an old bundle in place.

## Export

```sh
python -m tools.latency_investigation export records.json /tmp/replay-set
```

`records.json` is a list of recorded chat requests (`request_id`,
`prompt_tokens`, `body` or `text` plus image bytes, `max_tokens`,
`temperature`, optional `arrival_offset_seconds` and baseline output text).
The writer keeps the source share of 500–999 token requests, clamps a large
population to 200–300 requests, and writes:

- `vllm-custom-image.jsonl` and `aiperf-single-turn.jsonl`
- JPEG files for every image
- `warmup-vllm-custom-image.jsonl` (ids disjoint from the timed set)
- `manifest.json` and `manifest.sha256`
- `commands.json` (not executed; `do_not_run_until_authorized` is true)

Run the commands from that directory so the relative image paths resolve.
Do not run them until a GPU run is authorized. This package never calls Vast.

## Results table

```sh
python -m tools.latency_investigation report \
  --manifest /tmp/replay-set/manifest.json \
  --vllm-result /tmp/replay-set/results/closed-loop.json \
  --host host.json
```

Columns: run date, location, GPU, CPU model, vLLM version, concurrency,
input-size bucket, P50, P95, P95/P50, then P50/P95/P95/P50 from actual send,
then notes. For a closed loop the two clocks match. For an open loop the
first trio is scheduled send and the second trio is actual send. P50/P95 use
the same ceil-rank as the formal benchmark (`ceil(fraction * n) - 1`). There
is no P99 column. vLLM reports require complete success metadata for every
manifest request and reject failed or inconsistent runs rather than counting
failed zero-latency requests. AIPerf reports reject runs with no usable successful
latency samples, and require exact successful coverage of the manifest.

`python -m tools.latency_investigation summarize <eval-dir>` prints that table
from a saved `attempts.jsonl` when the per-request log is on disk. It reuses the
formal request-latency summary: `metadata.transport_seconds` takes precedence,
legacy whole-attempt fallback is counted, and failures, timeouts and missing
outcomes are reported rather than silently dropped.
