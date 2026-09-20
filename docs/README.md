# Documentation

[Repository overview](../README.md) · [Development setup](../CONTRIBUTING.md) · [Tests](../tests/README.md) · [Tools](../tools/README.md)

Use the guide for the task below. Commands and unlinked paths are relative to
the repository root. Dated measurements describe their recorded versions and
environments, not a fresh validation of the current checkout.

## Delivery and integration

These files keep their root paths for customer delivery and launch workflows.

| Document | Purpose |
| --- | --- |
| [DEPLOY.md](../DEPLOY.md) | Customer package installation and target-host checks |
| [API_DELIVERY.md](../API_DELIVERY.md) | Public APIs, result/frame correlation, SSE and Webhooks |
| [RELEASE.md](../RELEASE.md) | Recorded image versions and completed release checks |
| [WINDOWS_HYBRID.md](../WINDOWS_HYBRID.md) | Native Windows DirectML setup |
| [AnalysisResult v2 schema](../schemas/analysis-result-v2.schema.json) | Machine-readable result contract |

## Guides

| Guide | Contents |
| --- | --- |
| [Runtime](guides/runtime.md) | Scheduling, rich labels, GPU face recognition and VLM configuration |
| [Source deployment](guides/deployment.md) | LAN topology, Compose, operations and dry-run mode |
| [Configuration](guides/configuration.md) | Settings precedence, inference routes and reference memory |
| [Release builds](guides/release.md) | Image targets, customer boundaries, staging and verification |

Local setup and gallery enrollment live in [CONTRIBUTING.md](../CONTRIBUTING.md#local-development);
test selection and coverage rules live in [tests/README.md](../tests/README.md).

## Architecture and implementation

| Document | Scope |
| --- | --- |
| [Events and video evidence](architecture/event-evidence.md) | Identity continuity, persistent events, evidence revisions and migration |

## Validation records

Each report preserves distinct measurements, failures and unverified boundaries.
Use the guides above for current operating instructions.

| Dates | Record |
| --- | --- |
| September 6–10 | [VLM evaluation](validation/vlm-evaluation.md) |
| September 10 | [Event validation](validation/event-validation.md) |
| September 10, later source updates | [30 FPS display and independent overlays](validation/display-cadence-20260910.md) |
| September 10 | [Low-resolution validation](validation/low-resolution-20260910.md) |
| September 10 | [Stranger / LLM interoperability](validation/stranger-llm-interop-20260910.md) |

All dates above are in 2026. The low-resolution study includes a reviewed
[source manifest](validation/low-resolution-sources.json) and
[aggregate results](validation/low-resolution-results.json). Raw media, traces
and benchmark outputs remain in ignored local directories.

## Internal reference

[Route catalog](internal/route-catalog.md) contains internal route mappings and
historical provider comparisons; it is separate from customer documentation.

## Continuous scenes and multi-camera services

- [Continuous scene guide](guides/continuous-scene.md) — modes, evidence, runtime limits and verification records.
- [Temporal diagnostics and replay](guides/diagnostics-replay.md) — actual request images, partial results, failures, source gaps and budgeted replay.
- [Entity/event engine](architecture/entity-event-engine.md) — state model and active operation.
- [Local GPU deployment](local-gpu.md) and [modular services](modular-services.md).
- [Multi-camera API](multi-camera-api.md).
- [SAM scene worker](../workers/sam_scene/README.md) and [standalone Contour node](../workers/contour/README.md).
