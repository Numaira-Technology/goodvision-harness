# Regression tests

The default suite uses synthetic frames, reviewed fixtures, fake model backends
and intercepted HTTP requests. It does not require model weights, GPUs, live
cameras or paid API credentials. Some integration tests start a loopback HTTP
server, encode local video with FFmpeg, or run Node/PowerShell subprocesses.

## Run the suite

Install the development environment from [CONTRIBUTING.md](../CONTRIBUTING.md),
including Node and FFmpeg, then run from the repository root:

```bash
python -m pytest
```

This also runs every dashboard `.cjs` suite through the shared Python runner.
PowerShell launcher cases skip when PowerShell is unavailable. Missing Node
skips dashboard checks; FFmpeg is required for the full evidence suite. CI
installs the dependencies on Linux and Windows. Treat skipped tests as missing
coverage for that local run, not evidence that the feature works.

For focused iteration, select the modules that own the behavior:

| Area | Representative tests |
| --- | --- |
| Configuration and service contracts | `test_harness.py`, `test_deployment_settings.py`, `test_service_state.py` |
| Independent scheduling and failures | `test_independent_branch_pipeline.py`, `test_event_pipeline_latency.py`, `test_stream_failures.py` |
| Event lifecycle and evidence | `test_event_archive*.py`, `test_event_overlay_continuity.py`, `test_event_service.py` |
| People, identity and overlays | `test_face_*.py`, `test_person_*.py`, `test_tracking_observations.py` |
| Worker and video behavior | `test_paddle_*.py`, `test_video_*.py`, `test_display_*.py` |
| Model protocols and privacy | `test_model_api*.py`, `test_vlm_*.py`, `test_privacy.py`, `test_multi_event.py` |
| Deployment and delivery | `test_delivery.py`, `test_start_delivery.py`, `test_runtime_gateway.py`, `test_windows_hybrid_deployment.py` |
| Dashboard behavior | Dashboard, workflow canvas, task observation and summary suites; shared DOM harness in `helpers/dashboard.cjs` |

```bash
python -m pytest tests/test_event_service.py tests/test_video_evidence.py
python -m pytest tests/test_deployment_settings.py tests/test_rich_config.py
node --test tests/*.cjs
```

The Node command is an alternative dashboard-only run, not an extra step after
the full Python suite. Run the full suite before finishing a cross-cutting change.

## Keep coverage useful

- Preserve concrete regression scenarios: late callbacks, stale sessions,
  identity crossings, bounded deadlines, exact source/evidence correlation,
  partial results and privacy failures. Similar setup does not make these
  different behaviors duplicates.
- Prefer observable results, state changes and dependency interactions over
  searching Python/JavaScript source text. Keep static build/Compose checks
  where the offline suite cannot execute the target deployment environment.
- Parameterize identical setups with different meaningful inputs. Remove a case
  only when another retained test protects the same behavior and failure mode.
  Retiring an obsolete standalone tool also removes its exclusive tests; keep
  the independent production regressions for related behavior.
- Root and packaged `config.yaml` byte equality is owned by
  `test_deployment_settings.py`. Detailed prompt and negative-response matrices
  exercise the distinct source and delivery profiles; packaged-config loading
  remains covered in `test_single_event_envelope.py`.
- Keep helper modules outside `test_*.py` collection. Use the dashboard harness
  for common DOM mechanics, while keeping each suite's clock, network and
  rendering assumptions explicit.
- Keep fixtures small and reviewed, with attribution where needed. Generate
  temporary outputs under `tmp_path`; private captures and model weights stay
  outside Git.

## Real-model and deployment validation

Tests named `gpu`, `model_eval` or `benchmark` can still be offline tests
of those tools using fake backends. Keep these regressions in the normal suite.
They do not establish real model accuracy, GPU execution or production latency.

Real model runs, rented GPU experiments and Docker image checks are separate
entrypoints in [tools/README.md](../tools/README.md). Run them explicitly with
the required environment and authorized inputs. Store generated results in an
ignored output directory and keep reviewed summaries in
[docs/validation](../docs/README.md#validation-records).

## CI partition and timeout evidence

The two Linux regression shards deterministically assign entire test modules using
SHA-256 of the collected module path. They retain each module's original test
order and fixtures. Neither shard skips a slow test or excludes camera tests.
Every shard records the full collection and its selected node IDs; the existing
`Tests (ubuntu-latest, Python 3.11)` aggregate check fails unless both shards pass
and the manifests plus completed setup/call/teardown records prove every collected
test was executed exactly once. Legitimate recorded setup skips remain skips.
Windows also uses two shards with the same partition and evidence plugin. Its
existing `Tests (windows-latest, Python 3.12)` check independently verifies all
Windows tests were covered exactly once. Each Windows shard retains the 25-minute
job limit and bounds the test step to 20 minutes. Its parent process stops the
test process tree after 16 minutes; stdout/stderr go to files so console output
cannot block pytest shutdown. This leaves time to upload collection, incremental
stage records, logs and JUnit. A timed-out step or missing
completion record fails the aggregate gate; it never becomes a successful skip.

Linux shards use a 16-minute test-process deadline inside the unchanged 20-minute
job limit, leaving time for `always()` artifact upload. The `regression-linux-*`
artifacts contain raw output, incremental test/stage progress, collection coverage,
process snapshots and JUnit when pytest completes. Test stdout/stderr go directly
to artifact files rather than synchronously mirroring to the job console: a stalled
console must not block capture, deadline cleanup or interpreter shutdown. Consult
`stdout.log`, `stderr.log` and `progress.jsonl` for test-by-test details.
On deadline, the runner requests
an all-thread stack dump, terminates its owned test descendants (including detached
services), and fails rather than
turning the interrupted tests into skips. A hard runner loss or external cancellation
can still prevent upload; absence of evidence is never treated as success.

Periodic and deadline SIGUSR1 stack requests use a Python signal handler so
all-thread dumps run with the GIL, avoiding the asynchronous handler crash in
CPython issue 116008. The bootstrap disables pytest's separate asynchronous C
watchdog; the existing 60-second requests provide periodic slow-test stacks instead
of per-test timer dumps. A fully deadlocked interpreter or blocking native code may
not service this handler. In that case, incremental progress and process snapshots
remain available and the unchanged parent deadline still fails and cleans up the
run; a missing stack is never evidence of success.
