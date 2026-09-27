# Benchmark V1 runner

## Full Harness evaluation from video files

Use `--offline` to process each video once through continuous SAM Live tracking,
rolling-window event analysis, identity updates, and evidence finalization.
You supply video files without a real-time playback wait. GPU and model processing
can still take longer than the video duration. Report answer quality separately
from live latency.

Each observation window receives that video's original questions, response
options, and public context. Each session detects the events its own video is
asked about (see "Everyday events" below); videos with only counting and
location questions stay on the built-in safety events, as earlier runs did.
The four existing safety verdicts are retained;
question-relevant observations are additional output. In this mode only, sampled
frames with no detected people can also reach the model. Empty detections do not
establish that no people are visible.

Observation requests include a 1-based window index and source-video start/end
times. The prompt distinguishes a window boundary from the end of the video or
an action; it does not label a window as final. The final text-answer request is
sent after video processing finishes and is explicitly identified as that stage.

After tracking and model work drain and identity/archive updates finish, the
same configured model route receives the collected **text and structured records
only**, with no images or video, to answer every question. All four capabilities
use this final answer step; no per-question rule extractors are used. Private
answers and annotations are not sent to the Harness.

### Requirements and media paths

- Run this branch's Harness with scene mode `active`, person tracking enabled,
  and the **SAM Live** processing method (`sam31_stream` backend). Configure the
  existing SAM worker and event model route as usual. An idle, single-camera
  deployment is sufficient; do not use an active demo session for evaluation.
- The runner uses Python's standard library and HTTP. Run it from `benchmark/`.
- Videos must be readable **on the Harness server/container**, not necessarily
  on the machine running the command. Prepare files through the existing media
  workflow; this runner does not download media or rent a GPU.
- Supply public questions and a media JSONL mapping. Each media row needs
  `video_id` and a `path` relative to the server-side `--media-root`:

  ```json
  {"video_id": "example_video", "path": "example.mp4"}
  ```

  For Docker, place videos in the existing `/data` volume, for example
  `/data/benchmark/media/example.mp4`, and pass `/data/benchmark/media`.
  Supply a separate path mapping if the original media manifest has no `path`;
  do not alter the public question files. Directories of frames must first be
  assembled into video by the existing preparation workflow.

### Run

With the configured Harness already running, from `benchmark/`:

```bash
python -m bench.run_bench \
  --offline \
  --base-url http://localhost:8080 \
  --questions /path/to/questions.jsonl \
  --media /path/to/media-paths.jsonl \
  --media-root /data/benchmark/media \
  --output-dir /path/to/results
```

This makes model calls. Videos run sequentially, and each video's questions share
one recording. Sampling, window sizes, image settings, and routing come from the
Harness configuration. The runner does not increase FPS or reduce resolution.
Timestamps follow the existing file reader and its source FPS; incorrect media
timing metadata must be fixed during media preparation.

Optional `--observation-max-tokens` and `--answer-max-tokens` control the two
output budgets separately. If omitted, both use `scene_max_tokens`. A too-small
budget or oversized evidence context is recorded as a failure, not silently
truncated or followed by an extra summarization call.

### Everyday events

Questions about everyday activity — texting on a phone, opening a door, loading
a vehicle — need events the harness does not ship with. `events/vocabulary.yaml`
holds their criteria, written by hand so a score can be traced back to what was
actually asked of the model.

Declaring them is a one-time server-side step, because definitions are read from
the config when the harness starts. Print the block and paste it into the
harness config, then restart:

```bash
python -m bench.vocabulary >> /path/to/harness/config-events.yaml
```

Declaring an event only makes it selectable; nothing changes about what a live
camera detects. The runner then ticks the right ones per video, reading each
question's event label, so a video asked about texting runs a session detecting
texting and nothing else.

Before the first session the runner checks the harness knows every label the run
needs, and names the missing ones rather than failing once per video. Pass
`--no-auto-features` to skip both the check and the per-video selection and
leave every session on the harness default.

Labels whose subject is a vehicle are not supported: the tracker admits people
only. Those questions cannot be answered by this path yet.

### Results and reuse

- `predictions.jsonl`: every selected `qa_id`, including an empty prediction for
  failed or unanswered questions. Pass this to the existing **goodvision** grader;
  this runner does not replace its scoring or omit MAE.
- `details.jsonl`: per-question capability, answer/error, and artifact location.
- `videos/video-<encoded-video-id>/result.json`: latest result and actual model,
  route, sampling, image, and token-budget settings.
- Session/attempt subdirectories retain previous `result.json` and `records.jsonl`
  files. Records contain accepted person frames (including empty frames), event
  revisions, identity merges, observations, raw model responses, and errors.
  In benchmark mode, a rejected safety-verdict batch is recorded as a
  `stage: verdicts` error without discarding valid question observations from
  the same response. Rejected verdicts never update live safety-event state.
- The server retains its session JSONL under its configured events directory,
  in `benchmark/`. Keep these records private: they contain questions, model
  responses, person locations, and configured model/route identifiers.

Rerunning overwrites the top-level predictions/details and latest-result index,
but retains previous attempt directories. Use a new output directory for a
separate experiment.

Add `--reuse-results` to reuse matching complete local results with downloaded
records and run only missing/incomplete videos. Reused videos make no model calls.
This explicitly retains their recorded configuration, even if the current
Harness settings differ.

To request **only a new final text answer** from saved sessions, use the same
command and output directory with `--reanswer` instead of `--reuse-results`.
The original Harness process must still be running; its frozen model configuration
is reused. No video is replayed. New token-budget options do not change an existing
session. This action makes another model call. A processing failure before evidence
finalization cannot be repaired with `--reanswer`.

After the command prints a session ID, Ctrl+C stops that video session.
During `--reanswer`, Ctrl+C only stops waiting; the submitted answer request may
finish on the server. A disconnected runner does not automatically stop the server.
Its `session.json` identifies the run; retrieve its status and records with:

```text
GET /v1/benchmark/<stream_session_id>
GET /v1/benchmark/<stream_session_id>/records
```

These routes use the service's existing coordinator routing guard, not a new
authentication system. Use a trusted/local service connection.

## Legacy capture and rule extraction

Feed the Benchmark V1 sample videos into the streaming harness and emit one
answer per question, in the exact `{"qa_id", "prediction"}` shape
`bench_v1_sample/how_to_feed.md` asks for. The private answers and scoring live
on the eval side.

The legacy `--captures` mode reads answers from recorded Harness output using
fixed rules, without sending question text to a VLM:

| Capability | qa_id suffix | Video | Harness path → what we read |
| --- | --- | --- | --- |
| event_recognition | `…_event_recognition` | A (fight .mpeg) | scene event `fight` confirmed? → `yes`/`no` |
| temporal_questions | `…_temporal_questions` | A (fight .mpeg) | first `fight` event `ended_since` → seconds |
| counting | `…_counting_…` | B (MOT17) | SAM tracks → peak simultaneous people (centers in frame) |
| location_tracking | `…_location_tracking_…` | B (MOT17) | SAM tracks in 9–13 s → busiest horizontal third |

`physical_conflict` (benchmark) = `fight` (harness). See `bench/schema.py`
`EVENT_TARGET_TO_HARNESS_LABEL`.

## Layout

```
benchmark/
  bench/schema.py        capture format + question/media loaders
  bench/extract.py       PURE answer extractors (unit-tested, no GPU)
  bench/run_bench.py     join questions × captures → predictions.jsonl
  bench/capture_live.py  GPU-side: poll a running harness → <video_id>.json capture
  replay/                ffmpeg → RTSP → MediaMTX(WHEP) replay + MOT17 assembly
  tests/                 extractor unit tests
```

The design splits into **capture** (feeds the harness) and **extract** (pure
functions). The answer logic is fully tested offline; only capture touches a
running harness.

## Two ingestion schemes

These legacy schemes publish files through ffmpeg, using `rtsp://`, `rtmp://`, or
`webrtc+http(s)://` (WHEP). They feed the same capture → rule-extraction core.
The direct-file `--offline` mode above does not use either scheme.

| | Scheme A — Live (WHEP) | Scheme B — Offline (file replay) |
| --- | --- | --- |
| Path | ffmpeg `-re` loop → RTSP → **MediaMTX → WHEP** → service | ffmpeg single-pass → **RTSP → service directly** (no MediaMTX WHEP) |
| Pacing | real time, looped | as fast as the pipeline consumes; one pass |
| Proves | production WHEP + concurrency | simplest offline run |
| Covers | all 4 questions (scene engine, GPU) | all 4 questions (scene engine, GPU) |
| Capture | `capture_live.py` (poll `/v1/scene`) | `capture_live.py` (poll `/v1/scene`) |

**Bonus — video A without a GPU.** Video A's two questions (event_recognition,
temporal) are the safety pipeline's own output, so they can run through the
headless CLI with no streaming infra and no GPU:

```bash
goodvision-harness-cli --video fight.mpeg --no-realtime --output rawA.jsonl
python -m bench.from_cli --cli-output rawA.jsonl \
  --video-id sample_91ffb5b98379cc5a --out caps/sample_91ffb5b98379cc5a.json
```

`from_cli` folds the per-window `fight` judgments into capture events; its
temporal "end" is the first contiguous positive run's end (an approximation of
the scene engine's confirmed `ended_since`). Video B (counting/location) still
needs the scene engine (Scheme A or B), because the safety pipeline has no person
tracking.

## Prerequisites (GPU box)

- The harness on `codex/entity-event-engine`, installed, with SAM weights and a
  CUDA GPU (SAM worker `/health/tracking` reporting `ready` + `gpu_ready`).
- A configured event model route + API key (for the temporal VLM; not a GPU cost).
- `ffmpeg` and the `mediamtx` binary (github.com/bluenviron/mediamtx).
- The two media items from `bench_v1_sample/how_to_feed.md`:
  - Video A `sample_91ffb5b98379cc5a`: the `.mpeg` file (verify `media.sha256`).
  - Video B `crowd_mot17_mot17_02`: the MOT17-02 `img1/` frame folder.

## Step 1 — prepare the videos

```bash
# Video B: assemble MOT17 frames into a 30 fps clip
benchmark/replay/assemble_mot17.sh /path/to/MOT17-02-FRCNN/img1/ mot17_02.mp4
# Video A is already a file; use it as-is.
```

## Step 2 — publish the videos to the harness

Start MediaMTX once (it is the RTSP hub for both schemes):

```bash
mediamtx benchmark/replay/mediamtx.yml &
```

**Scheme A — Live (WHEP), production parity.** ffmpeg can't publish WHEP, so we
publish RTSP with real-time pacing + loop and MediaMTX serves WHEP:

```bash
benchmark/replay/publish_rtsp.sh fight.mpeg   benchA &   # -re + loop
benchmark/replay/publish_rtsp.sh mot17_02.mp4 benchB &
# harness reads: webrtc+http://localhost:8889/benchA/whep  and  /benchB/whep
```

**Scheme B — Offline (file replay), simplest.** Single pass, no `-re`; the
service reads RTSP directly (MediaMTX WHEP layer unused):

```bash
benchmark/replay/publish_rtsp_offline.sh fight.mpeg   benchA &
benchmark/replay/publish_rtsp_offline.sh mot17_02.mp4 benchB &
# harness reads: rtsp://localhost:8554/benchA  and  rtsp://localhost:8554/benchB
```

### Pacing: two independent knobs

`-re` and `-stream_loop` do different things — don't conflate them:

- `-re` paces frames at wall-clock speed → **simulates a live camera** (Scheme A).
- `-stream_loop -1` replays the file endlessly → **keeps the stream alive** (a
  camera that never stops), for a live-monitoring / concurrency demo.

For **benchmark scoring, do NOT loop**: a question like "when does the first
conflict end" needs a single, well-defined timeline. Use Scheme B single-pass, or
the CLI with `--no-realtime` (faster than real time). Looping is only for the
live-monitoring demo.

### Local backup media

If the online media (private R2 / Drive) is unavailable, point the scripts at
local copies in the repo folder — e.g. the fight `.mpeg` file and the MOT17
`img1/` frame directory (assemble the latter with `assemble_mot17.sh` first).
Verify each against the `media.jsonl` `sha256` before trusting a result. Media
files are gitignored and never committed.

## Step 3 — run harness sessions + capture (concurrent)

Start one harness session per video in `active` scene mode, then capture each.
Running both at once is the concurrency proof.

```bash
export SCENE_ENGINE_MODE=active PERSON_TRACKING_ENABLED=true
# Scheme A: --stream-url webrtc+http://localhost:8889/benchA/whep
# Scheme B: --stream-url rtsp://localhost:8554/benchA
# session A on :7900, session B on :7901 (separate instance/port = concurrency)

python -m bench.capture_live --base-url http://127.0.0.1:7900 \
  --video-id sample_91ffb5b98379cc5a --out caps/sample_91ffb5b98379cc5a.json &
python -m bench.capture_live --base-url http://127.0.0.1:7901 \
  --video-id crowd_mot17_mot17_02   --out caps/crowd_mot17_mot17_02.json &
wait
```

Run `bench/*` from the `benchmark/` directory (so `-m bench.…` resolves).

## Step 4 — produce predictions

```bash
python -m bench.run_bench \
  --questions ../bench_v1_sample/questions.jsonl \
  --media     ../bench_v1_sample/media.jsonl \
  --captures  caps \
  --output    predictions.jsonl
```

`predictions.jsonl` (one line per question):

```json
{"qa_id": "benchmark800_qa_0001_event_recognition", "prediction": "yes"}
{"qa_id": "benchmark800_qa_0051_location_tracking_07dacfd8...", "prediction": "右侧"}
{"qa_id": "benchmark800_qa_0051_counting_9fe11d0d...", "prediction": "3"}
{"qa_id": "benchmark800_qa_0001_temporal_questions", "prediction": "18.0"}
```

## Tests

```bash
python -m pytest benchmark/tests -q
```

## Scoring (eval side, for reference — from how_to_feed.md)

- choice → case-folded exact match; counting/duration → VSI-Bench MRA (+MAE),
  denominator floored at 0.5; time points → abs error within
  `max(0.5 s, 2% × duration)`; invalid/negative/out-of-range numbers score 0.

## Assumptions & caveats (tell the eval side)

- **event_recognition** answers `yes` only when the harness *confirms* the event
  (its two-source-frame bar); suspected-only reads as `no`.
- **temporal end** uses the confirmed episode's `ended_since`; if it never ends
  in-capture, falls back to the last observed time.
- **capture cadence**: `capture_live.py` polls `/v1/scene` (~scene snapshot rate),
  which is coarse for 30 fps counting/location. For frame-exact counts use the
  high-fidelity in-process hook below.

### High-fidelity capture (optional)

Instead of polling HTTP, record every canonical scene snapshot in-process from
`HarnessRuntime._on_tracking_snapshot` (service.py) into the same schema
(`entities[].{subject_id,bbox,frame_sequence,last_observed_timestamp_seconds}` →
`frames[]`, scene `events[]` → `events[]`). This captures all 30 fps frames, so
`counting`/`location_tracking` see the true per-frame track set. Event answers
(Q1/Q4) are unaffected by cadence.
