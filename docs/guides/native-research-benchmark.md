# Native research evaluations

Run the current scene and identity engines against source-timed files, then
extract answers deterministically. The checkout entrypoint is:

```bash
.venv/bin/python benchmark/run_native.py --help
```

Install the ordinary Harness `video` extra in the client environment. A real
run also needs an already configured SAM or detector worker. GPU worker
requirements remain separate from the client environment.

Two explicit runtime profiles are available:

| Profile | Scene/identity/event engine | Provenance |
| --- | --- | --- |
| `current-main` | The current checkout's organized runtime and built-in events | Current commit, working patch hash, effective model/sampling settings, media/input/extractor hashes |
| `frozen51` | Exact Ashley source in `experiments/ashley-detector182-20261012/frozen_runtime/`, with the archived extended vocabulary | Original publication commit and verified source/publication hashes, plus the new runner's independent provenance |

Both profiles use the new sequential native evaluation runner. Neither sends
question text or answer options into observation windows, nor calls a final
answer LLM. Public event labels select the native detectors for each question.
Counting/location/count-temporal questions skip the event VLM. Event questions
use the selected runtime's normal temporal inspector and confirmation lifecycle.
Each question receives its own source session and canonical identities.

This is a source-time quality experiment. Sequential backpressure preserves
sampled frames and ordered observations; it does not measure production
realtime queue drops, browser delay or end-to-end streaming throughput. The
segmentation and VLM pipeline suites measure those separately.

## Inputs and plan

Supply public question JSONL and a media-path mapping. Media paths must be
relative to `--media-root`:

```json
{"video_id":"example","path":"example.mp4","sha256":"optional verified source SHA256"}
```

The published `bench_v1_core`, `bench_v1_sample54` and `bench_v1_sample80`
question/media manifests and samplers are retained. If their media manifests
lack local `path` values, supply a separate local mapping rather than editing
public question files. Private answers are only used by the separate grader.

Validate the complete input plan, profile labels and archive hashes without
contacting a worker or creating output:

```bash
.venv/bin/python benchmark/run_native.py \
  --questions /data/questions.jsonl --media /data/media-paths.jsonl \
  --media-root /data/videos --output-dir outputs/research/native-example \
  --runtime-profile current-main --backend detector \
  --detector-model yolo26-s --tracking-endpoint http://127.0.0.1:7902 \
  --max-model-calls 100 --plan-only
```

Remove `--plan-only` to run. For a geometry-only cohort, add `--geometry-only
--max-model-calls 0`; an event question in that cohort is rejected before
worker requests. The VLM call budget spans all questions and counts actual
requests, including failures. Model calls and worker inference occur only in
a real run. API credentials come from the selected Harness configuration.

Use `--backend sam` with a `sam31_stream` configuration and its configured
session/perception endpoint to retain native SAM tracking. Authenticated Contour
endpoints require `--tracking-key-env NAME`, where `NAME` is an environment
variable holding the worker token. The token is passed to the tracking client
without entering manifests, records or command-line arguments. Detector mode
supports `yolo26-s` and `rfdetr-l`; it verifies returned model, weight SHA256,
CUDA parameter device and source/session identity for every frame. The exact
historical detector worker and its dependency pins remain under
`experiments/ashley-detector182-20261012/gpu/source/workers/detector/` for use in
a separate worker environment.

For extended Ashley event labels, choose `--runtime-profile frozen51` in a
fresh process. The wrapper verifies and selects the archived root source
without replacing the live application's built-ins. `--config` can select an
explicit run configuration; resolved sampling and model settings are recorded.
The frozen profile is a historical engine comparison under the new runner,
not reproduction of the original 182-question cohort, budget, infrastructure
or published results. Original publication scripts and their bytes remain
unchanged; do not invoke their old `prepare.py`/`start.py` entrypoints.

## Evidence and completion

Every new output directory contains `manifest.json`, `predictions.jsonl`,
`details.jsonl` and per-question `records.jsonl`/`capture.json`. Existing runs
are never overwritten. The audit retains original source timestamps, decoded
and sampled frame counts, natural EOF, empty observation frames, canonical
identities, native provenance, event revisions, complete model responses and
observation completion. Image payloads in model audit records are replaced by
hashes; credentials and request headers are not written.

The decoder uses presentation timestamps relative to the first decoded frame;
missing PTS can use decoder DTS, which is explicitly recorded. AVI preserves
the historical frame-index/file-FPS clock. Original source dimensions remain
separate from resized inference images for location extraction. Natural EOF,
complete worker emission and drained temporal work are required before a
complete capture is written.

Audit, decode, tracking, model or completion failure stops subsequent calls.
Failed and unattempted questions retain blank prediction rows and explicit
statuses. Geometry and temporal extractors retain their original scoring semantics
and documented limitations. Native event answers distinguish missing or
unconfirmed evidence from explicit accepted negatives; combinations and
single-action choices preserve exact public options and aliases. Ambiguous
answers remain blank. The legacy extractor files remain unchanged. Scoring and historical results stay separate from execution.

Run the CPU checks with:

```bash
.venv/bin/python -m pytest benchmark/tests -q
```

The tests use real scene/identity state and real video decoding with a local
mock detector, including both runtime profiles and a confirmed extended
frozen51 event. They do not run a GPU, contact a model provider or measure
accuracy/performance on the real cohort.
