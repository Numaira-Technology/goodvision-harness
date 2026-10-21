# Saved person queries

Record current counts, sparse observed peak counts, and tracked boxes from the
existing Harness person route. This entry uses `VideoFileStream`,
`RichLabelCoordinator`, `RichLabelHTTPClient`, `ResultBus`, and the existing stream
recorder. It does not start or probe a detector service, load VideoJev, enable
attributes or faces, or read annotations. It runs video sources sequentially.

```bash
PYTHONPATH=src python -m tools.person_query_benchmark.run --execute \
  --video mot17:09=/private/MOT17-09.mp4 \
  --queries /private/mot17-09/queries.jsonl \
  --endpoint http://127.0.0.1:8091/v1/tracks \
  --detector-model yolo26-s --sample-fps 5 \
  --output /private/new-person-query-run
```

Use an already configured, task-owned detector endpoint. The command sends real
video frames only; it does not send a warm-up or readiness request. Detector
weights, device, confidence, input-size settings, and any server-side startup
behavior belong to that service and must be recorded with the actual run. This
entry does not establish that a requested model header was honored.

The output directory must be new. Repeat `--video VIDEO_ID=PATH` and `--queries`
for additional sources. Optional `--start-seconds` and `--end-seconds` select a
source range. Original EOF, range end, request drain, and failures are recorded
separately. Public query files contain only `query_id`, `video_id`, `task`,
`query_time_seconds`, `query_frame_index`, and `question`. Do not pass private
reference files or annotation subject IDs to this command.

## Timing and answers

Each decoded source frame receives an actual input clock before candidate
sampling and JPEG encoding. The default 5 FPS selects candidates; the existing
coordinator limits request starts and replaces older waiting work when busy.
Actual processed and replaced counts remain in the completion record. A raw
source-frame index is not the candidate delivery index.

Reports retain the original person snapshot. Counts include only currently
observed `person` or `pedestrian` objects with matching observation metadata;
predicted tracks do not count as new observations. Unclear metadata produces an
unknown count, not zero. `sparse_observed_peak_estimate` is computed from valid
observed counts up to that report's source frame and published with it. It does
not recover skipped frames or count distinct people over time.

Queries are read when their original frame enters the program. The chosen
report must already be readable then and contain no future source frame. Among
eligible reports, the newest source frame wins, not the latest callback. An
older readable result is retained when a newer publication straddles the query
clock; that uncertainty is recorded. A stale result is still an answer and is
scored against the current reference, with its age reported separately.

Each source directory contains `manifest.json`, `inputs.jsonl`, `reports.jsonl`,
`publication.jsonl`, `queries.jsonl`, `failures.jsonl`, and `completion.json`.
Keep these runtime files private. Publication rows link to physical report
lines such as `reports.jsonl:12`; the recorded readable-time bounds, not the
file modification time, define availability. Individual HTTP or format failures
are saved and later frames continue. An actual 429 waits at least 60 seconds,
or longer numeric `Retry-After`, without resending that frame. Saving failures
and workers that have not exited are not hidden by starting the next source.
Explicit HTTP 401/403 responses stop further requests to that service, including
queued work and later video sources; ordinary window failures do not.

## Offline comparison and limits

Reference derivation and scoring live in the separate
[benchmark repository](https://github.com/Numaira-Technology/goodvision/tree/codex/walltime-queries/evaluation/lmms_eval/tasks/walltime_queries).
Its exact-frame table measures saved observation quality; its online table
scores the actual query selection recorded here. Neither changes original QA
or scores.

Positions need an observation-side subject choice and one fixed private mapping
to the annotation subject. No mapping is inferred from each query's GT overlap.
This entry records all tracks but does not interpret a subject description or
claim that a runtime track is the requested annotation subject. The current
person route has no walking-state coverage output, so walking duration remains
unsupported rather than being inferred from movement or filled with zero.
Source annotations and media frames/PTS still need alignment for each source.

This is an unexecuted experimental interface, not a measured result or a claim
that the three task types are complete. No independent software tests were run.
