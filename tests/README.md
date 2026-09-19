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
| Dashboard behavior | Five `dashboard_*.cjs` suites; shared DOM harness in `helpers/dashboard.cjs` |

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
