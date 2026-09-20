# Tools

Run command-line tools from the repository root in an activated development
environment. Most support `python tools/<name>.py --help`; runtime helpers are
used through their documented callers. See [CONTRIBUTING.md](../CONTRIBUTING.md)
for installation and offline checks. Generated outputs belong in `output/`,
`outputs/` or `dist/`.

## Startup and delivery

| Tool | Purpose |
| --- | --- |
| `start_delivery.py` | Start the recorded `2026.09.10-full.4` customer package. |
| `start_windows_hybrid_demo.ps1` | Start the native Windows / Docker worker demo. |
| `export_full_delivery.py` | Export the recorded full.4 image pair, metadata and checksums. |
| `run_full_delivery_smoke.ps1` | Run isolated image protocol checks; accepts candidate image tags. |
| `delivery_smoke.py` | Exercise a disposable main container using synthetic frames. |
| `paddle_delivery_smoke.py` | Check worker HTTP behavior with synthetic models. |
| `face_gpu_smoke.py` | Check runtime packages or run real accelerated face warmup. |
| `customer_check.py` | Installed check dispatcher; mode comes from its installed filename. |
| `runtime_gateway.py` | Runtime process supervisor and customer diagnostic filtering. |

See [release development](../docs/guides/release.md) and the
[customer deployment guide](../DEPLOY.md). Recorded image tags and release
checks describe that delivery, not subsequent source changes.

## Evaluation

| Tool | Purpose |
| --- | --- |
| `face_provider_benchmark.py` | Compare ONNX Runtime face inference providers. |
| `face_llm_interop_eval.py` | Check real CPU face inference alongside synthetic remote blocking. |
| `low_resolution_face_eval.py` | Evaluate local media/models with a disposable face gallery. |
| `low_resolution_model_eval.py` | Replay listed samples through a real model service. |
| `replay_stress.py` | Paced active-scene replay with recorded tracks, temporal HTTP fault injection, compressed timelines and a shared request budget. [Scope and usage](../docs/guides/diagnostics-replay.md). |
| `low_resolution_tracking_eval.py` | Replay saved observations through person tracking. |
| `low_resolution_evidence_eval.py` | Exercise synthetic event continuity and H.264 evidence offline. |
| `benchmark_display_cadence.py` | Compare local synthetic display cadence, lag and frame drops. |
| `rich_label_benchmark.py` | Benchmark the JPEG `/v1/labels` worker on local video. |
| `vlm_speed_matrix.py` | Measure the native rich client with an explicit local `--cases` JSON file. |

Model/GPU/service runs need the corresponding environment and authorized inputs.
Protocol success is not model accuracy. See the
[validation reports](../docs/README.md#validation-records) for reviewed measurements.

## Repository hygiene

`check_repo_hygiene.py` checks tracked paths against repository ignore rules;
run it without arguments locally or in CI.

Superseded deployment recipes, the old fixed-bundle auditor, and redundant VLM
probes/fixed comparison cases have been deleted. Their source is available in the
pre-cleanup main revision `a7e5ec8`; there is no second archive
copy to maintain in the working tree. Dated measurement reports remain in
`docs/validation/`.

`audit_customer_bundle.py` checks customer bundles or the repository (`--repository`)
for the anonymization patterns recorded in [the validation report](../docs/validation/anonymization-audit-20260918.md).
