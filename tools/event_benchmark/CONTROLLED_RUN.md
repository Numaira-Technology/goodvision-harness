For current-frame 120-question runs use `tools/unified_benchmark/README.md`.
This document covers the older base-four controlled runner and v2/v2.1 replay.

# Controlled rerun: v2-provider-pinned-lossless-v1

The default CLI requires the provider configuration and validation/test sidecar.
For the separate original baseline (including robot41), explicitly pass
`--legacy-baseline`; it uses default routing without this sidecar and cannot be
mixed with controlled options or accepted by this threshold-tuning protocol.

This is a named variant of the existing v2 sampling/scoring protocol. SPEC.md
remains the historical fixed baseline. Questions, labels, centered 1 FPS sampling,
720px JPEGs, independent per-frame requests, clip-max aggregation, gate 0.5,
three total accuracy attempts, 90-second timeout and error denominators are unchanged.
Native CLIP/SigLIP2 and VideoJev are separate protocols. The authorized native
CLIP/SigLIP2 integration is documented in `../clip_benchmark/README.md`.

## Provider selection and evidence

`prepare_provider.py` performs a public catalog GET only. It saves the original
catalog bytes and a separate routing configuration. The default chooses the lowest
published latency (p50 when provided), with endpoint slug as the deterministic tie
break. If multiple eligible endpoints have missing latency, it stops rather than
claiming one is fastest. A sole eligible endpoint is explicitly not a speed comparison.
An explicit `--provider-slug` is recorded as an override, not a fastest-provider claim.

Accuracy and latency require the same `--provider-config`. It pins exactly one
catalog endpoint with fallbacks disabled. Both chat and Decisions request shapes
support ProviderPreferences in OpenRouter's official Python SDK:
https://github.com/OpenRouterTeam/python-sdk/blob/main/src/openrouter/decisions.py
https://github.com/OpenRouterTeam/python-sdk/blob/main/src/openrouter/components/providerpreferences.py

Each attempt records the returned generation ID, actual provider/model, available
usage/cost, elapsed time, score or sanitized error. Missing provider/model metadata
may be resolved through GET /api/v1/generation after inference, outside timing.
Metadata failure never causes another inference request or silently turns a score
into a negative. Missing/inconsistent actual receipts block calibration even when
the prediction itself is valid. Request endpoint and catalog precision are recorded;
actual endpoint/precision, server weights, hardware and VRAM remain unknown unless
separately attested. A fixed provider does not prove fixed server weights.

## Scores and splits

`score` and per-frame scores retain the received numeric value as Python float,
without additional rounding. `score_display` is rounded for display only. The
lossless marker describes local storage, not the server's numeric precision.
Historical rounded scores cannot be upgraded by changing a marker or adding zeros.

Supply the frozen `--split-manifest` and `--split validation|test`. The original
sample and manifest files are not rewritten. The sidecar is joined on question ID,
filename and SHA-256; complete membership, inline-group consistency and cross-split
filename/hash/source-group isolation are checked. Its original review scope,
unknown train/development membership and near-duplicate/person/scene limits remain
in the output. This is not a new leakage-free or untouched-test claim.

## Offline preflight and staged evaluation

Add `--preflight` to run_eval.py to validate media and the supplied frozen provider
and split configuration. It never reads credentials or sends HTTP. Missing provider
configuration is listed as pending; paid inference will refuse to start without it.
The latency script has its own offline `--preflight`, including JPEG preparation.
Both refuse an existing output path. A preflight pass does not authorize inference.

After separate permission for paid calls, collect complete validation evidence.
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
