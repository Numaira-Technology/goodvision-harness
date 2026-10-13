# Building and verifying customer releases

[Documentation index](../README.md) · [Repository overview](../../README.md)

Commands and unlinked file paths in this guide are relative to the repository root.

## Customer release builds

The two GPU Dockerfiles each define `runtime`, `customer`, and `development`
targets. The final `development` target is still the default, retaining the
existing developer entrypoints; CPU Dockerfiles and the local CPU/hybrid demo
path are unchanged. `compose.gpu.yaml` now builds its worker from the repository
root so the explicit customer target can use the shared process gateway.

Customer targets build the current source and dependencies from the public base
image. They do not require a product image from an earlier local delivery. From
the repository root, use distinct candidate tags while validating changes:

```bash
docker build --platform linux/amd64 -f Dockerfile.face-gpu --target customer -t goodvision-harness:consolidation-test .
docker build --platform linux/amd64 -f workers/paddle_rich_label/Dockerfile.gpu --target customer -t goodvision-rich-label-worker:consolidation-test .
```

The customer main image fixes `HARNESS_INFERENCE_PROFILE=customer-standard`.
That internal preset uses OpenRouter and `qwen/qwen3.5-35b-a3b:nitro`.
All A-H selections also use their base model's `:nitro` variant, allowing
throughput-based provider selection and fallbacks while excluding Novita and
CoreWeave. Existing IDs and saved selections retain their base models;
the eight slots contain six distinct models now that providers are not pinned.
See the [Nitro documentation](https://openrouter.ai/docs/guides/routing/model-variants/nitro).
Delivery Compose does not forward
provider, model or endpoint overrides; the development target keeps configurable
routing. Customer configuration, API profiles and UI continue to use neutral
feature names. Saved credentials are scoped to their service: do not silently
reuse a previous service's key. Supply the preset's credential through the
existing console or neutral `INFERENCE_API_KEY` when upgrading.

The corresponding developer checks use `--target development` with different
test tags. A clean dependency build needs network access and substantial free
disk space. Recipe/static checks and tests using previously installed dependency
layers do not prove that a fresh full GPU build has completed; record which
validation was actually performed. Dependency version ranges also mean a clean
build is not a promise of byte-for-byte reproducibility.

The current delivery uses `20260923-full1` for both product images. See
[RELEASE.md](../../RELEASE.md) for its exact artifacts and validation record.
Both images were rebuilt with the customer targets above; the old
`Dockerfile.release` remains a historical recipe whose local runtime bases may
not be installed. Use a new product tag and archive name for every later release,
and update the launcher, Compose files, deployment guide, exporter, smoke
defaults and tests together.

### Customer-facing boundary

Neutral names apply to the supported UI, API, event/webhook output, deployment
materials, default service logs and supplied check entrypoints. Known engine,
model and provider markers are handled by the shared `privacy.py` helpers; errors
must not expose upstream exception text or echo private input. The customer
process gateway also handles native stdout/stderr while preserving exit status
and shutdown behavior. A single `tools/customer_check.py` selects its check mode
from the installed filename and enters that gateway.

Business identifiers (for example person, source and message IDs) remain exact,
even when customer-supplied text resembles a brand. They are correlation/CRUD
keys, not implementation metadata; redacting them would break lookups or merge
distinct records.

This is not removal of working dependencies or a promise against image
inspection. Preserve upstream licenses, copyright notices and metadata. Runtime
internals, required dependency files and historical image layers may retain
implementation names; directly invoking internal tools bypasses the supported
customer-output boundary. Model authorization remains separate from code
packaging, and no real keys, model weights, photographs or galleries belong in a
release staging directory.

### Release staging and verification

Keep one deployment source, `DEPLOY.md`, in the repository. Use a new staging
directory and copy these reviewed sources:

| Repository source | Customer bundle destination |
| --- | --- |
| `DEPLOY.md` | Same name |
| `API_DELIVERY.md` | Same name |
| `docs/architecture/event-evidence.md` | Same path (linked from the delivery guides) |
| `compose.full-gpu.delivery.yaml` | `compose.yaml` |
| `compose.full-gpu.accelerated.yaml` | `compose.gpu.yaml` |
| `.env.full-gpu.delivery.example` | `.env.example` |
| `config.delivery.yaml` | Same name |
| `tools/start_delivery.py` | `start_delivery.py` |
| `schemas/analysis-result-v2.schema.json` | Same path |
| `schemas/analysis-result-v1.schema.json` (historical contract) | Same path |

Add a reviewed `RELEASE.md` recording the exact candidate versions, completed
checks and unverified boundaries, plus `SOURCE_HASHES.json` for the application
files verified inside both images. The existing exporter targets the recorded
`20260923-full1` tags and archive name:

```bash
python tools/export_full_delivery.py dist/goodvision-full-20260923-full1
```

The exporter streams the two recorded image tags into a binary-safe compressed tar
archive, verifies the complete stream and manifest, and writes `IMAGES.json`
and `SHA256SUMS.txt`. Load the archive directly with `docker load -i`.
Before a later release, update the exporter's tags and archive name along with
the release files. Archive/tag verification does not establish that image contents
match the checkout; retain the source-hash and protocol checks for every new candidate.

Before publishing, run the full unit suite and isolated image protocol check
with the candidate image tags:

```powershell
pwsh -NoProfile -File tools/run_full_delivery_smoke.ps1 -HarnessImage goodvision-harness:consolidation-test -WorkerImage goodvision-rich-label-worker:consolidation-test
```

The smoke uses disposable resources, not production sessions or data volumes.
Reload the new archive and repeat the checks against its selected tags. Synthetic
protocol tests and `--runtime` dependency checks do not establish actual GPU or
model correctness. Real face/detection/attribute inference, authorized-video
results, evidence photos and performance still require the target-host checks
in the delivery guide. Do not restart the local demo merely to validate a release.

### Minimal customer ZIP (2026-09-23 packaging revision)

The rich staging materials above are internal release records. Do not send them
to customers. Run `tools/package_customer_delivery.py` against the full staging
ZIP to retain only installation files. Its configuration is already in the image.

Customer updates now use Docker images, not an exposed application payload.
Prepare a build context using `tools/harness_update/package_image_update.py
--prepare <context>`, build the configured update image, then use `--output <zip>`
to stream it into the customer ZIP. The image contains the application and update
helper. The outer ZIP contains only the Docker archive, installation entrypoints
and a short usage note. Do not distribute source-update ZIPs made by the historical
`package_update.py` / `package_customer_update.py` workflows.

Customer instructions contain only required files and installation steps. Keep
internal evidence outside the delivery folder. Per the requested workflow, do
not run post-packaging hash checks.
