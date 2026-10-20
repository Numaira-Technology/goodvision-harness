# Native VideoJev event route

## Scope and status

This integration reuses Haotian's native VideoJev implementation from
[`8825a9b`](https://github.com/Numaira-Technology/goodvision-harness/commit/8825a9b90893b16b6b13d5df738563db3a7c43a8).
It connects that model to the current Harness independent event branch rather
than importing the separate `tools/benchmark_runtime` copy. This branch,
`codex/videojev-walltime-main`, starts directly from `main` at
`9dbd5103a5cc1c4a7886d7c80492f70d8abe1e63`. It does not merge the historical
`codex/research-testing` branch. The VideoJev work previously published as
`49ab100` and `672af15` is ported here with only the input/publication timing,
optional report callbacks, and event-policy interfaces needed by this route.
Historical QA, detector controllers, paid-request storage, paired scheduling,
and lean grounding implementations are not imported.

The code includes a real-file 1–12 stream entrypoint. It has not been deployed or
measured in this task. It does not demonstrate that one GPU can sustain twelve
streams. Completed Qwen-based 50-video results remain separate and unchanged.
No software tests, imports for validation, builds, or health probes were run.

## What changes in the running path

- `videojev_model.VideoJevInspector` implements the existing `inspect(job)` event
  interface for the explicit local `ZhuYuhan/VideoJev-4B` route. It sends original
  selected JPEGs and independently scores three choices for every requested event:
  present, assessed absent, and insufficient evidence. It does not generate a
  JSON answer or use a generated-token budget.
- `videojev_contract` and `videojev_ppe` contain the CPU-only question and choice
  conversion shared by the packaged client and repository tools. The model uses
  the existing event policy, sensitivities, or explicit fixed definitions. Ground
  truth and benchmark questions are not supplied. Source timestamps and optional
  window-local person aliases are the only evidence metadata in question text;
  transport IDs, paths, revisions, and machine details are not model context.
- `tools/videojev_probe/runtime.py` retains the team's pinned model revision
  `1c1a253bf90dfa66176124d25a48ffb791d517d5`, BF16 model loading, native feature
  scoring, and attention configuration. `prepare()` does CPU image preparation;
  `predict_prepared()` performs visual encoding and choice scoring on the sole GPU
  thread. All event questions in one window share that visual encoding. There is
  no cross-window feature cache or multi-video GPU batch.
- `tools/videojev_benchmark/server.py` keeps a resident model on one GPU. CPU
  preparation can overlap GPU work. FIFO remains the default; the explicit
  `latest-per-stream` policy retains one latest waiting candidate per stream and
  serves waiting streams in order. Replacing a pending candidate does not reset
  that stream's waiting position. Work already preparing or running is not
  cancelled to make room for a newer frame.

This is scene-level event marking only. It does not provide person bounding boxes,
subject confirmation, OFF transitions, a natural action-onset frame, or frontend
alarm latency. Native VideoJev temporal grounding is not the Harness person
grounding interface, and is not enabled by this route.

## Effective window parameters

The independent VLM branch now accepts these optional `VLMProcessConfig` fields:

| Field | Meaning | Default |
| --- | --- | --- |
| `candidate_fps` | Candidate source frames retained per second | Existing `max(2, 2 * observation_fps)` |
| `observation_window_frames` | Maximum frames, or exact count when delta is explicit | 4 |
| `observation_frame_delta_seconds` | Target spacing between selected original frames | Existing selection |
| `observation_fps` | Scheduling opportunities per second, not completed requests per second | 1.25 |

Explicit spacing selects distinct past frames ending at the latest available
frame. Insufficient history or missing target frames does not duplicate images or
create future evidence. The original default path still uses the original
four-frame maximum, minimum 0.75-second span, and three-second history. An explicit
two-frame, 0.5-second window no longer inherits that old span requirement. Longer
configured windows retain enough history, within the existing 16 MiB JPEG limit.
The shared inspection/IPC boundary accepts 2–64 frames; defaults do not change.

The model's native positional encoding still assumes uniform FPS. With explicit
spacing its nominal FPS is `1 / delta`; otherwise it uses the candidate FPS.
Actual irregular source times are retained separately, not silently retimed.
An odd frame count may receive the official processor's internal temporal padding;
the external source frame count and padding count are recorded separately.

The server's `--max-pixels` and `--context` control native processing. Additional
frames can force lower spatial resolution under the context budget. Source size,
processed size, visual tokens, and the effective pixel cap must therefore be read
alongside frame count, rather than attributing every change to FPS alone.

## Service use

Run the service from this checkout with the Harness package available. Keep its
Python 3.12 CUDA environment separate from the ordinary application environment.
The team's dependency versions are retained in
`tools/videojev_probe/requirements.txt`; PyTorch and native attention extensions
require the CUDA installation steps described there. Model files must already be
available in the supplied cache. This integration does not download weights or
start a GPU automatically.

Use `python -m tools.videojev_probe download --cache-dir /path/to/cache` only when
explicitly preparing the model. It reuses/downloads the pinned official snapshot
without loading it. The existing `tools.vlm_benchmark.run_remote.bundle()` now
includes both VideoJev tool directories as well as the packaged source. Reuse
that bundling and SSH transport, not its cloud-API bootstrap or old requirements.

The repository entrypoint is `python -m tools.videojev_benchmark.server`. Required
arguments are `--cache-dir` and `--result-log`. It binds only to `127.0.0.1`, port
8093 by default; forward that port for a remote GPU rather than exposing this
unauthenticated endpoint publicly. The adapter endpoint is the full `/predict`
URL, configured through `ModelAPISettings(provider="local", endpoint=...)` with
the exact model name and `output_mode="events"`.

The service defaults to FIFO, one CPU worker, two preparing/ready windows, a
20-second HTTP limit, and a 19.5-second queue age. These are retained baseline
settings, not claimed latency targets. Set `--queue-policy latest-per-stream`
explicitly to evaluate fair latest-window processing. More than one CPU worker
is opt-in; shared processor concurrency has not been measured.

No synthetic warmup or health request is sent by the adapter. Its first request
uses actual arrived frames. The local adapter does not retry, follow redirects,
use proxy environment settings, or fall back to another model. Client and server
deadlines must be chosen together; an elapsed client deadline does not mean an
already-started CUDA operation has stopped. Recent completed outcomes and active
requests are reused by transport ID, without repeating GPU work.

## Failures and timing

A malformed native choice remains a format failure, not a negative observation.
Ordinary window exceptions, queue expiry, replacement, HTTP failures, and CUDA
out-of-memory remain distinct in the service results. Explicit device loss or an
unrecoverable CUDA context, an actual worker exit, or failure to save results
stops acceptance. One ordinary `RuntimeError` does not permanently disable the
service. No failed window triggers an automatic model restart.

The client forwards named numeric service measurements into existing Harness
reports: CPU and GPU waiting, CPU preprocessing, visual encoding, choice scoring,
native total computation, frame/padding counts, dimensions, visual tokens, and
peak memory. Raw native scores and terminal window status are saved by the
service. Scores are not claimed to be calibrated confidence.

Real event response time still starts when the original event frame enters the
Harness before sampling, and ends when the corresponding report can be read.
HTTP includes server waiting and computation: do not add those internal
measurements to HTTP a second time. Server durations use the server clock; its
absolute clock is not subtracted from the local input clock.

## Remaining evaluation work

`python -m tools.videojev_benchmark.run --config CONFIG.json` prints the selected
settings without opening videos, creating workers, or contacting the model.
Add `--execute --output NEW_DIRECTORY` to run the listed real files once. The
config accepts only observation settings, not ground truth. A minimal example is:

```json
{
  "streams": [{"id": "camera-1", "video": "/path/to/video.mp4"}],
  "endpoint": "http://127.0.0.1:8093/predict",
  "labels": ["fight", "smoking", "fall", "weapon", "ppe_violation", "ground_litter"],
  "candidate_fps": 2.5,
  "window_frames": 4,
  "frame_delta_seconds": null,
  "observation_fps": 1.25,
  "max_in_flight": 1,
  "http_timeout_seconds": 20,
  "drain_timeout_seconds": 30
}
```

List up to twelve distinct short stream IDs. A stream may additionally specify
`start_seconds`, exclusive `end_seconds`, and `offset_seconds` for staggered
arrival. Relative video paths resolve against the config directory. Each stream
decodes independently at original speed; a repeated file remains multiple
independent streams, not proof of twelve different cameras or content diversity.
The default event set is all six existing safety events. A selected-label tuning
run must not replace the full-label capacity result.

All workers prepare before the common release time. Raw-frame input timestamps
are recorded before candidate selection and JPEG encoding. Pre-encoding
selection uses the same actual source-time phase as Harness admission; an
independently phased second sampler is not applied. Files use the existing
VideoFileStream pacing and IndependentVLMBranch scheduling and publication.

Per stream, buffered `inputs.jsonl` retains raw frame number, source time, actual
arrival, selection, and lateness. `reports.jsonl` saves original reports;
`publication.jsonl` links them to ResultBus readable-time bounds. `completion.json`
separates requested-range completion, original-media EOF, event-queue drain,
cleanup, and errors. The run-level completion includes every stream, not just
successful ones. Input files are buffered rather than opening/fsyncing one file
per raw frame; report callbacks flush their rows. Final completion follows a
flush/fsync of all stream files.

A source-local preparation/decoder failure leaves that stream incomplete while
other sources continue. Ordinary failed windows do not stop playback. Explicit
native shared-device or result-storage failures propagate as closed error codes
and stop new source input; missing/failed windows never become negative labels.
The runner does not rent, release, or restart a GPU. The external resource owner
must stop/release the exact owned GPU after the last requested run and retain
server results, including calculations that outlive a client deadline.

GPU execution and offline quality scoring remain next steps. Parameter tuning
must report quality, dropped/expired windows, per-stream
completion and update intervals, failures, and event response delay together.
Dropping most requests can reduce the latency of survivors without meeting
twelve-stream capacity. Small tuning clips are not a held-out final benchmark.
Counting, location, duration, OFF evaluation, and offline GT derivation are not
implemented by this event-model integration.
