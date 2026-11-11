For current-frame 120-question runs use `tools/unified_benchmark/README.md`.
This document covers the older base-four controlled runner and v2/v2.1 replay.

# Controlled rerun: v2-provider-pinned-lossless-v1

## Implementation status

This checkout includes the controlled **evidence verifier**, threshold-selection
consumer and separate controlled latency CLI. It does **not** include a controlled
accuracy collector or `prepare_provider.py` catalog selector. `run_eval.py` is the
legacy default-routing collector: it rounds clip scores and has no provider/split
sidecar, per-attempt receipt export, `--preflight` or `--legacy-baseline` switch.
Its output is ineligible for the controlled calibration contract below. Do not
relabel it as lossless/provider-pinned or add markers to make verification pass.

Use `tools/unified_benchmark/README.md` for the implemented current120 pipeline.
For this older controlled protocol, collect new accuracy evidence only after a
separately reviewed producer is available, or use eligible retained records with
their original source snapshot. The remaining sections describe the verifier's
acceptance contract; they are not claims that the legacy CLI produces it.

This is a named variant of the existing v2 sampling/scoring protocol. SPEC.md
remains the historical fixed baseline. Questions, labels, centered 1 FPS sampling,
720px JPEGs, independent per-frame requests, clip-max aggregation, gate 0.5,
three total accuracy attempts, 90-second timeout and error denominators are unchanged.
Native CLIP/SigLIP2 and VideoJev are separate protocols. The authorized native
CLIP/SigLIP2 integration is documented in `../clip_benchmark/README.md`.

## Provider selection and evidence

The latency CLI accepts a frozen `--provider-config`; it does not choose an
endpoint from a live catalog. A reviewed configuration pins one endpoint and
its requested model, with fallbacks disabled. A pinned provider alone does not
prove fastest latency, server weight revision or actual precision.

Eligible controlled accuracy records must retain each attempt's returned
generation ID, actual provider/model, available usage/cost, elapsed time and
lossless score or sanitized error. The verifier rejects missing or inconsistent
identity evidence. The latency CLI can resolve missing response metadata through
GET /api/v1/generation without repeating inference. Its request latency includes
answer parsing but excludes the later metadata lookup. This differs from the
current120 API runner's complete transport timing. This is not an accuracy collection path.

Controlled evidence requires the returned model ID to exactly match the frozen
configuration's model ID. Case, whitespace, provider prefixes and version suffixes
are not normalized or treated as aliases. A differing receipt stays incomplete
until its identity is resolved; no alias policy is inferred from a similar name.

Request endpoint and catalog precision are recorded; actual endpoint/precision,
server weights, hardware and VRAM remain unknown unless
separately attested. A fixed provider does not prove fixed server weights.

## Scores and splits

`score` and per-frame scores retain the received numeric value as Python float,
without additional rounding. `score_display` is rounded for display only. The
lossless marker describes local storage, not the server's numeric precision.
Historical rounded scores cannot be upgraded by changing a marker or adding zeros.

Eligible records must identify the frozen split sidecar and validation/test split. The original
sample and manifest files are not rewritten. The sidecar is joined on question ID,
filename and SHA-256; complete membership, inline-group consistency and cross-split
filename/hash/source-group isolation are checked. Its original review scope,
unknown train/development membership and near-duplicate/person/scene limits remain
in the output. This is not a new leakage-free or untouched-test claim.

## Offline preflight and staged evaluation

The implemented latency CLI has offline `--preflight`, including JPEG preparation;
it does not authorize inference or validate accuracy records. The legacy accuracy
CLI has no equivalent preflight. Use the media downloader's `--check-only` for
reviewed media/manifest checks; it does not establish provider or split evidence.

The commands below consume existing eligible controlled records; they cannot
upgrade legacy rounded outputs or collect missing evidence.
Select using validation only; freeze the artifact before reading test predictions:

```sh
python -I tools/event_benchmark/tune_threshold.py select \
  --val /path/validation-run.json --val-manifest /path/validation-manifest.json \
  --split-manifest /path/validation_split.json \
  --reference-thresholds /path/historical-luna-thresholds.json \
  --out /new/path/thresholds.json
python -I tools/event_benchmark/tune_threshold.py evaluate \
  --thresholds /new/path/thresholds.json --thresholds-sha256 FROZEN_HASH \
  --test /path/test-run.json --test-manifest /path/test-manifest.json \
  --split-manifest /path/validation_split.json --out /new/path/comparison.json
```

Luna retains the 0.01 grid, >= comparator and longest-best-run midpoint rule.
If its midpoint is not optimal, this variant stops for review instead of silently
falling back to a grid point. Historical reference thresholds are frozen inside
the selection artifact and used only for a third test comparison alongside newly
selected thresholds and fixed 0.5. No additional model calls are needed for these
three calculations. Test never selects which threshold to publish as the winner.

Calibration checks returned provider/model, fixed routing configuration, helper/
runner/spec hashes, prompt identity, sampling, workers and sidecar hash across runs.
All outputs remain local. Twelve workers describe evaluation concurrency, not
twelve-camera end-to-end throughput; latency timing still excludes image preparation
and the product pipeline. GPU rental, inference and publication require their
separately authorized steps.
