# Temporal diagnostics and replay

This change selectively migrates the diagnostics and replay work from `haotian`
onto main `2f7bdaa`. It does not migrate the legacy branch scheduler, fixed
360×202 canvas, higher default concurrency, local tracking experiment, prompt
changes, or a second live-output/event-log implementation. Current scene image
size, parallelism, sensitivity defaults and event lifecycle remain unchanged.

## Live diagnostics

Expand **连续场景诊断** in the detailed console during an active scene session.
The panel polls only while open and visible, with one request in flight.

- `GET /v1/scene/diagnostics`: bounded request metadata and aggregate counters.
- `GET /v1/scene/diagnostics/{request_id}/{frame_sequence}.jpg`: exact JPEG bytes
  from the final temporal multimodal payload, after its normal resizing/encoding.
- Both paths also exist under `/v1/fleet/cameras/{camera_id}/scene/diagnostics`.
  Camera-scoped paths are discoverable in the fleet output catalog.

All two to four request frames retain their source sequences, source timestamps
and actual encoded dimensions. These are distinct from `/v1/scene/evidence`,
which exposes the retained original source images. Images live only in memory:
at most eight requests and 16 MiB per run; evicted JPEGs return 404. Metadata is
bounded to 256 requests. No provider response text, prompts, credentials or
camera addresses are recorded. Persistent model clients use the diagnostic
collector attached to each immutable job, so old responses stay with their run.

Interpret the stages separately:

| Field | Meaning |
| --- | --- |
| `frames`, `request_attempted` | Payload prepared / adapter attempted; not proof of successful upload. Transport status is supplied when observed. |
| `response_outcome` | Pending, validated model response, or failure. |
| `result` | Positive, negative, partial, pending or unavailable, for returned labels only. |
| `evidence_outcome`, `accepted_verdicts` | Runtime acceptance; late/expired results cannot become fresh evidence. |
| `confirmed_event_labels` | Accepted labels whose scene events are confirmed at completion, not a claim every positive is a new alarm. |

Missing labels and `detected=null` display **部分完成 / 无法判断**. Other valid
positives are retained. An empty response array is not an all-negative result.
This addresses the review's partial-result display concern without importing
the old letterbox/classification path or changing temporal output schema.

`unanalyzed_gaps` measures source-time gaps between **accepted, non-null label
observations** and includes the current trailing gap. It does not establish
coverage of every person. When the observation history rolls over, scope is
`recent`, not whole-run. `source_arrival_to_completion` starts at arrival in
SceneRuntime, not sensor exposure; `claim_to_response` includes pool/adapter
waiting and is not pure model compute time. Existing HTTP and worker timing logs
remain the detailed stage breakdown.

## Replay inputs and scope

`tools/replay_stress.py` now uses the production `SceneRuntime`,
`SceneInspectionWorker`, temporal message builder, `TemporalVLMInspector`,
response validator and event lifecycle. It no longer uses the old
`build_selected_features_grounding_client` entrypoint.

Tracking inputs are either recorded, source-aligned `RichLabelSnapshot` JSONL
observations or explicit synthetic tracks for offline plumbing tests. The
recorder's frame sequence and timestamp must match the video's decode sequence
and constant-rate timeline. Missing track records remain missing; they are not
converted into observations of an empty room. Use constant-frame-rate sources;
variable-rate media must first be normalized with matching track timestamps.

The tool **does not run SAM or Face, render a browser, or encode final evidence**.
Its report explicitly says so. Recorded tracks test the downstream scene/VLM
path and cannot establish end-to-end camera/SAM latency or tracking accuracy.
For full-system acceptance replay the same inputs through the deployed service
and measure those stages separately. Oracle HTTP simulation tests wiring and
failure handling, never model accuracy.

Manifest example (paths resolve beside the manifest):

```json
{
  "clips": [
    {
      "id": "hall-01",
      "path": "hall-01.mp4",
      "tracks": "hall-01.tracks.jsonl",
      "events": [{"label": "fall", "start": 12.0, "end": 16.0}],
      "ignore": [{"start": 30.0, "end": 31.0}]
    }
  ]
}
```

Every clip requires an `events` field; an empty list explicitly declares a
negative clip. Annotate uncertain
intervals as `ignore`; do not treat unreviewed video as known negative material.

Run fault injection against recorded tracks (no model API calls):

```bash
python tools/replay_stress.py --manifest /path/to/manifest.json \
  --speeds 1,2,3 --latency-ms 800 --slow-rate 0.1 --slow-ms 4000 \
  --timeout-rate 0.05 --parse-error-rate 0.03 --max-requests 300 \
  --output output/hall-faults.json
```

For a manifest without `tracks`, add `--synthetic-tracks` for explicit plumbing
tests. `--no-realtime` is permitted only for simulation and suppresses reported
first-alert/confirmation latency. Real inference is opt-in:

```bash
python tools/replay_stress.py --manifest /path/to/manifest.json --real \
  --speeds 1 --max-requests 300 --event-sensitivity smoking=standard \
  --output output/hall-real.json
```

The request cap is shared across all clips, speeds and parallel clients, with
a preflight estimate and an atomic check before dispatch. Exhaustion is a
classified failure, not a negative. Reports include the fault profile and seed,
source/track/config/tool hashes and effective settings. Real mode rejects synthetic tracks,
unpaced playback and injected faults. Reports never overwrite an existing file.

`--speeds 2` halves source timestamps, event intervals and ambiguous intervals;
sampling stays fixed in compressed time, so brief actions can fall between
samples. Under overload the replay applies explicit backpressure and reports
arrival lag; it does not silently discard an admitted task. This is not a
simulation of upstream live-camera dropping. Simulated positives only use new
source evidence, never old context images as a second independent vote.

Use the same sensitivity map for comparisons. Defaults come from the service
(`smoking=high`); override explicitly when desired. The legacy
`low_resolution_model_eval.py` now accepts the same override and records the map,
but marks itself `legacy_single_frame`. Do not aggregate its metrics with active
temporal replay as if they used the same pipeline or prompts.

For latency tuning freeze source/track hashes, config, model route, sampling and
sensitivity. Compare latency alongside missing labels, failures, source gaps,
first alert versus confirmation, and false alarm instances. Keep original
event onset as the latency origin. Repeated snapshots of one alarm are not new
false alarms. Any accuracy/latency conclusion still requires reviewed real
clips and the relevant real components.

See the [selective migration validation record](../validation/haotian-selective-migration-20260920.md)
for the tested scope and local-environment caveats.
