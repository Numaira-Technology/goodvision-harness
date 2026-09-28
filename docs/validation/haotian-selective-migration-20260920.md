# Haotian selective migration — 2026-09-20 UTC

Baseline: main `2f7bdaa`, incorporating the entity/event engine. This is a
selective migration, not acceptance of the old branch's legacy inference path.

## Retained and adapted

- Exact final temporal request JPEGs, dimensions and source sequence/timestamps.
- Separate model-response validity, runtime evidence acceptance and confirmed
  event state; missing/null labels remain partial while valid positives survive.
- Failure categories and bounded per-label source-time gaps. Memory is bounded,
  late responses cannot revive expired evidence, and collectors stay per run.
- Paced replay through the production scene runtime, worker and temporal model
  adapter, using recorded tracks or explicitly synthetic offline tracks.
- Slow/auth/timeout/parse fault injection, compressed source timelines and one
  atomic request budget shared across clips, speeds and clients.
- Service-aligned sensitivity metadata; legacy single-frame evaluation is
  explicitly distinguished from the active temporal entrypoint.

Current main's image settings, concurrency, scheduler, event lifecycle and
prompt/schema implementations are unchanged. The fixed 360×202 canvas, default
concurrency increase, old branch scheduler, local tracking experiment and
parallel live-output/event-log implementation were not migrated.

## Checks and scope

Full regression on the identical source/test tree in an isolated worktree:
**4166 passed, 10 skipped, 2 warnings** in 91.09 seconds, using
`PYTHONPATH=src python -m pytest --disable-warnings --tb=short -rs`.
Skips cover unavailable PyAV, a Linux-root-only check and PowerShell launchers;
those platform/runtime paths are not validated here.

Environment: local macOS, Python 3.13, Node.js 22, FFmpeg 7.1. The focused set
below passed all **62 tests** in the original working directory:

```bash
PYTHONPATH=src python -m pytest \
  tests/test_diagnostics.py tests/test_replay_stress.py \
  tests/test_low_resolution_model_eval.py tests/test_fleet_outputs.py \
  tests/test_dashboard_classification.py
```

Coverage includes payload-byte equality, all-frame dimensions, missing/null
labels, retained positives, image eviction, old/new run isolation, late expired
responses, accepted evidence and event confirmation, camera-scoped APIs,
out-of-order completions, source-time compression, missing tracking records,
shared request caps, preflight rejection and refusal to overwrite reports.
Follow-up review coverage also checks that incomplete drains fail, changed input
files are rejected and failed runs leave no empty or partial report.
The out-of-order test explicitly gates the first response; it does not assume
identical source endpoints between independently scheduled unpaced runs.

Replay fixtures are generated constant-rate clips and synthetic HTTP responses.
They establish wiring and failure-handling behavior, **not real model accuracy
or production latency**. No paid model calls, remote GPU/SAM/Face inference,
customer-video acceptance, or interactive browser acceptance was performed.

New/modified focused Python files pass Ruff; repository hygiene and
`git diff --check` pass. Root/package default configs are identical to main.

## Local environment caveats

FFmpeg was initially absent from PATH. The existing local FFmpeg binary was
added to the test process PATH; no business-code workaround was needed.

Three existing tests fail when the user's local `.env` is loaded: default
provider/model assertions in `test_harness` and `test_reference_memory_inference`,
and the rendered cloud configuration check in `test_modular_deploy`. All three
failures were reproduced against unchanged main with the same working-directory
environment. The user's `.env` and model configuration were not modified.
Use an isolated worktree without local service credentials for full regression.

## Recovery

The old branch head is retained at
`codex/haotian-before-pr-review-20260920` (`c811259`). All pre-existing tracked
and untracked edits are saved in stash `76690af`, named
`haotian before selective PR migration 2026-09-20`. Restore those changes on a
separate recovery branch based on the old head, not on the migrated tree.

See [usage and interpretation](../guides/diagnostics-replay.md).


## Independent-process follow-up — 2026-09-28

Connected the live independent VLM process to the existing diagnostic HTTP and
JPEG endpoints. The same real model adapter captures the final encoded inputs;
a dedicated bounded queue and reader transport them separately from reports.
Full, compact, events and parallel modes are covered, including distinct lane
records for one shared source window. The camera diagnostic page now scopes its
external script's metadata and JPEG fetches to the selected camera.

Re-review checked process/run isolation, parallel lane identity, bounded image
retention, diagnostic backpressure, failed requests and the distinction between
model observations and confirmed scene events. No remaining blocking finding was
identified in this scope. Independent diagnostics deliberately report completed
model requests only; downstream evidence acceptance, event confirmation and
coverage gaps are not inferred. Saturation/shutdown may drop diagnostics; normal
report delivery remains independent. See the guide for bounds and field meanings.

Validation on Windows with Python 3.12.14, using an isolated checkout and a new
workspace-local pytest temporary directory:

- 267 related Python tests passed: diagnostics, independent process/service,
  parallel lanes, reorder capacity, scene model/worker, fleet endpoints, replay,
  low-resolution evaluation and dashboard classification.
- An additional oversized-diagnostic-image test passed (metadata survives while
  JPEG retention is refused).
- Five Node diagnostic-panel tests passed, including scoped fetch/image URLs and
  unobserved event-confirmation display.
- New Python files pass Ruff; repository hygiene and diff whitespace checks pass.

The spawned-process integration tests exercise the production message builder,
model adapter, IPC, cache and HTTP/JPEG routes with synthetic model responses.
They assert exact equality between retained JPEGs and the multimodal payload at
the child process's request boundary. They make no paid inference calls and do
not establish model accuracy, real GPU performance or production latency. The
full repository suite was not rerun for this follow-up; CI remains the full-suite
gate. Two installed-library deprecation warnings occurred in focused tests.
