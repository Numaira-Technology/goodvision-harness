# Repeated current202 evaluation

## Validation-only effort comparison

For the Astra/Hydrogen low/medium comparison, use `preflight --phase validation
--require-validation44` with the original bag manifest and its independently
recorded SHA-256. This freezes only validation frames/plans; test videos are not
decoded and no test requests are prepared. The eight original JSON inputs are
still required to verify unchanged references and prevent split overlap.

Use the same batch commands below, with `schema: "event-benchmark-validation-v1"`
and `repeats: 0`. This schedules exactly one validation run for each selected
alias (`astra_low`, `astra_medium`, `hydrogen_low`, `hydrogen_medium`) and cannot
start test rounds. Reports are `summary.json` and `validation.csv`.

For this API-only scope, `budget_mode: "rolling_reservation"` allows a bounded
partial run instead of reserving the maximum cost of every possible retry before
starting. Each request atomically reserves its entire supplied per-attempt upper
bound before its intent is written. Known persisted charges release the unused
part; unknown or interrupted charges retain their full reservation. `initial_spent_usd`
includes technical smoke already billed. The same total cap and reservations
survive restarts; a budget stop blocks further requests. The cap is conditional
on the supplied input/output price bounds being valid. A provider charge above
its bound stops subsequent requests. Keep the budget state with the run.
Creating a file named `STOP` in the batch output prevents new API request intents
in this mode, while allowing already sent requests to retain their outcomes.
Optional `budget_wait_seconds` (0 by default, at most 120 and less than the stall
timeout) waits for this worker's in-flight receipts to release unused reservations.
Waiting does not hold the admission lock or count as model request latency.
Interrupted calls and completed calls with unknown cost keep their full reservation;
they are not treated as in-flight work that can unblock admission. STOP and queue
cancellation interrupt waiting before another request intent is written.
Do not use detached execution when the host's network proxy exists only for the
lifetime of an attached command; retain the active command session instead.

The default ten-round schema retains its full upfront reservation. Validation-only
runs make no claim of cross-round stability and do not automatically select or
launch a formal test configuration.

## Ten-round evaluation

This batch runs **validation44 / 34 videos once per selected model**, then
**ten test202 / 192-video rounds**. The existing twenty bag validation questions
are required for every selected profile. Test questions, labels, media, prompts,
precision, frame sampling, queue slots and three-attempt rules are unchanged.

The batch controller, API resume and local Qwen resume are implemented here.
They have synthetic offline tests; those tests are not model measurements or
proof of a live GPU/API deployment. The controller currently runs on POSIX.
Single-run legacy validation24 remains readable by its matching runner, but the
batch rejects it and never falls back to it.

The selected November 14 run scope is recorded in the
[execution plan](../../docs/architecture/event-benchmark-implementation-20261114.md):
Astra and Hydrogen each run once first, followed by nine selected configurations
with ten test rounds each. Llama4 Maverick is excluded from this batch. This
selection does not remove reusable profiles or historical results.

## Prepare inputs and freeze code

1. Use the eight original private snapshots from [README.md](README.md) and the
   existing private `bag-validation10-v1` manifest. Do not manufacture labels or
   reconstruct the manifest from old predictions.
2. Put all 34 validation videos under `media/validation` and all 192 test videos
   under `media/test`. Verify the manifest's externally recorded SHA-256.
3. Finish code changes, record the commit, then create a **new** bundle:

```sh
python -m tools.unified_benchmark.preflight \
  --inputs /private/inputs --media /private/media --output /private/bundle44 \
  --models qwen8b luna clip siglip2 videojev decision3 \
  --bag-validation /private/inputs/bag-validation.json \
  --bag-validation-sha256 FROZEN_MANIFEST_SHA256
```

The example model list is illustrative; use the already selected experiment
scope. All thirteen recorded profiles accept the supplement. Requests remain
single-frame for API/CLIP/SigLIP2 and adjacent-two-frame for VideoJev/Decision3.
Both bag questions on a clip use the same sampled frame bytes.

Every model must finish all planned validation responses before test.
CLIP/SigLIP2 select six margin thresholds; Luna selects six probability thresholds.
Boolean/argmax models save validation results without a numeric threshold.
Old validation24 threshold artifacts cannot be reused for this batch.

## Configuration

Copy [batch.example.json](batch.example.json) to a private location and replace
every placeholder. It deliberately contains no runnable budget, price or key.
Use absolute paths. Set `python` to the intended environment's interpreter.

Required batch fields:

| Field | Meaning |
|---|---|
| `schema` | `event-benchmark-batch-v1` |
| `repeats` | Exactly 10; validation is separate |
| `models` | Explicit ordered list; models and rounds execute sequentially |
| `budget_usd` | Total authorized upper budget |
| `phase_timeout_seconds` | Finite limit for one worker invocation, including startup |
| `stall_timeout_seconds` | Finite interval without evidence/progress |
| `batch_timeout_seconds` | Wall deadline starting at first batch launch; resume does not reset it |
| `cleanup_timeout_seconds` | Finite graceful worker cleanup, then forced termination |
| `max_process_restarts` | Integer 0–10 per validation/test run; does not reset request attempts |
| `max_controller_restarts` | Integer 0–10; detached watcher restarts a signal-killed controller using the same batch |

Each model specifies `alias`, `route` (`api`, `qwen`, `native`),
`bundle`, `bundle_sha256` (hash of `bundle-files-sha256.json`), and `python`.
Different bundles must have identical validation/test membership, references and
shared frame hashes. All runners and bundles must use the same frozen checkout.

API entries additionally specify:

- `env_file`: selected dotenv file; its contents are never put in configuration.
- `credential_file_sha256`: freeze that file's identity.
- `attempt_cost_upper_usd` and `cost_upper_bound_basis`: verified upper cost of
  one request at this profile/input/output size, including any billable overhead.
  Historical catalog prices are not sufficient evidence of a current upper bound.

Before execution the controller reserves the upper cost of **all three possible
attempts for every validation/test unit in all ten rounds**. This conservative
reservation includes concurrent in-flight requests and interrupted intents.
Unknown actual charges are not reported as zero. Bounds are conditional on the
supplied billing evidence; provider overcharging is not a strict billing guarantee.

GPU entries additionally specify:

- `resource_management: "external"` and `resource_lifetime_note`: an already
  prepared host owned by the operator, or an independently enforced lease/cleanup
  arrangement covering the batch. This controller does not rent or release leases.
- `hourly_cost_upper_usd`: reserve the full batch deadline plus cleanup duration.
  For an owned machine this may be zero; this does not measure energy cost.
- `hardware_probe`: an argv array for a **read-only** command on the actual GPU
  host, returning a stable JSON identity. No shell expansion or API keys in argv.
- `hardware_sha256`: canonical JSON digest of the expected probe result.
  The probe runs before every worker invocation. A different host/GPU fails closed.

For a local native host, a probe command can be:
`["/absolute/python", "-m", "tools.unified_benchmark.hardware"]`.
It records GPU UUID/model/memory/driver/power limit, host, CPU and RAM.
For remote Qwen, run the corresponding probe on the **server**, not the controller;
record the SSH target and independently verified runtime/weight/environment
provenance. GPU identity alone is not proof of server model weights or cache state.

Native entries also require `deployment` and `deployment_sha256`, verified by
the existing [native deployment workflow](NATIVE.md). CLIP/SigLIP2 workers must use
the prepared CLIP interpreter. Qwen entries require `endpoint` for an already
verified running service. Keep that same service/host for the batch. Do not invoke
the rent-and-destroy launcher separately for each round. The batch does not
restore R2 environments or incorporate the entire unmerged environment PR stack.

## Check, execute, resume

```sh
python -m tools.unified_benchmark.batch check --config /private/batch.json
python -m tools.unified_benchmark.batch run \
  --config /private/batch.json --output /private/results/batch-001 \
  --execute --authorization-note "Previously agreed model, hardware and budget scope" \
  --detach
```

`check` verifies local files/configuration and worst-case reservation, with no
inference or renting. Runtime hardware probes happen at execution. Offline checks
do not confirm a live provider's credits/access, a GPU model load, or lease cleanup.
`--detach` returns the watcher PID and controller log path, **not a completion claim**.
The output directory and `state.json` record actual progress. The sibling
`<output>-supervisor.json` retains controller invocation counts and the original
watcher deadline. A signal-killed controller resumes automatically within that
limit, after its previous worker has released the execution lock. A normal
failed/blocked result does not trigger a replacement run.

If the watcher or whole host stops, resume with the same command and `--resume`.
Restarting the host/watcher itself requires an external service manager. The
same config, source, credentials, data, deadline, run IDs and attempt budgets are
required. Do not start a new batch to conceal a failed round. A worker process
can be restarted within the frozen limit; an authentication, identity, corrupted
evidence or calibration failure blocks its model rather than being auto-acknowledged.
There is no extra eleventh replacement round.

Individual API and local Qwen runners also accept `--resume` for outputs created
by this version. Their receipt/outcome checks reject incompatible old outputs.
A response saved before a crash is re-parsed before another request is sent.
An intent without a recoverable response consumes an attempt. If response timing
did not survive, its valid answer is kept but latency is unknown, not zero.

The supervisor detects loss of the controller through a pipe and terminates its
owned worker process group. Native services join that group in supervised mode.
This bounds worker cleanup; it does **not** cancel provider-side accepted requests
or replace independent cleanup of a remote rented GPU. Output is written to the
chosen persistent location; remote file replication is an operator responsibility.

## Results

`summary.json` contains per-model, per-round status, accuracy, valid coverage,
missing units, latency sample count, P50/P95, restarts and cost evidence.
JSON and CSV also retain the latency source, failed and timeout attempts,
unresolved outcomes, missing successful timings, legacy timing fallback count,
and failure-penalized P95 with its sample count. Missing historical qualifications
remain null in JSON and empty in CSV; they are not inferred as zero failures.
See the [shared latency definition](README.md#latency-definition-p50p95).
`rounds.csv` is the compact comparison table. Each run also retains its raw
receipts, attempt ledger, immutable outcome records, calibration and full report.
Hardware probe records and worker logs live beside each run.

Accuracy is correct/202, including incomplete questions. Unstarted runs have no
invented score. Mean, sample standard deviation, min/max and actual contributing
round count are reported separately for accuracy and per-round latency quantiles.
Null latency is never replaced by zero. Statistics of ten P95 values are not a
pooled P95, and twelve queue slots do not prove twelve-camera real-time throughput.
Calibration cost is retained with the validation run; GPU reservation is not a bill.

Offline report refresh after the controller has stopped:

```sh
python -m tools.unified_benchmark.batch report --output /private/results/batch-001
```

The report command takes the batch lock and does not send model requests. No
shared spreadsheet, PR or external message is published by these commands.

## Selected upstream fixes

The ledger's append-time validation and strict historical receipt codec are
adapted from PR #97 at `5b96f20f3f52f03d7a14ae7a7c6f374c643fd537`.
Other PR #94/#97/#98 environment, deployment and launcher changes are not
implicitly merged. This implementation uses main
`31e675f0334a630bec3886217671fcc273d7afb3` as its base.
