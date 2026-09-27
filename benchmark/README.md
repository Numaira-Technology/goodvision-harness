# Benchmark V1 runner (streaming harness)

Feed the Benchmark V1 sample videos into the streaming harness and emit one
answer per question, in the exact `{"qa_id", "prediction"}` shape
`bench_v1_sample/how_to_feed.md` asks for. The private answers and scoring live
on the eval side.

We do **not** send the question text to a generic VLM. We run the harness's real
capabilities and **read the answers from its structured output**:

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

The service ingests only `rtsp://`, `rtmp://`, or `webrtc+http(s)://` (WHEP) — not
a bare file — so a file is always published through ffmpeg. Both schemes feed the
same capture → extract core.

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

**Scheme B — Offline single pass, realtime-paced (use this for scoring).** The
harness timestamps RTSP frames by wall-clock, so the stream MUST be paced at the
source rate (`-re`) or the capture timeline is compressed and every time-scored
answer is wrong. Use the realtime single-pass publisher:

```bash
benchmark/replay/publish_rtsp_realtime_once.sh fight.mpeg   benchA &
benchmark/replay/publish_rtsp_realtime_once.sh mot17_02.mp4 benchB &
# harness reads: rtsp://localhost:8554/benchA  and  rtsp://localhost:8554/benchB
```

`publish_rtsp_offline.sh` (no `-re`, pushes faster than realtime) is **only** safe
for fps-independent answers (counting peak, event yes/no) — never for location
windows, temporal, or count-duration.

### Pacing: two independent knobs

`-re` and `-stream_loop` do different things — don't conflate them:

- `-re` paces frames at wall-clock speed → source-time alignment (required for
  scoring) and a live-camera cadence (Scheme A).
- `-stream_loop -1` replays the file endlessly → **keeps the stream alive** (a
  camera that never stops), for a live-monitoring / concurrency demo.

For **benchmark scoring, use `-re` single pass, do NOT loop**: a question like
"when does the first conflict end" needs one well-defined, source-aligned
timeline. The CLI `--no-realtime` path is fine only for video A's fps-independent
event/temporal-via-episode answers.

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

## The recommended set — `bench_v1_core/` (116, optimized)

For scoring the harness, use **`bench_v1_core/`** (regenerate with
`tools/sample_bench_v1_core.py`), not the sets below. It is sampled **sub-form
aware** (location is three different tasks — busiest-third, relative-horizontal,
relative-vertical — each needing its own extractor), balanced for a per-capability
score (counting 30 / event 30 / location 30 / temporal 26; ~30 each, temporal
capped by video scarcity), with choice-answer priors balanced against a majority
baseline (event is exactly 15 yes / 15 no). See `bench_v1_core/SELECTION.md`. The
80/54 sets below are kept for provenance; the 54 silently mis-answered 11
relative-position location questions, which `bench_v1_core` fixes via the
`relative_displacement` extractor.

## Batch run — the 80-question diversity set

The 4-question sample above is the wiring proof. The real evaluation set is an
80-question, 74-video stratified sample (`bench_v1_sample80/`, regenerate with
`tools/sample_bench_v1_80.py`) that covers all 9 source datasets, all 4
capabilities, all 4 answer formats, and all 18 feasible (dataset × capability)
cells. Running it is the same capture → extract core, looped over 74 videos.

**1. What to fetch, and expected coverage.** Generate the media fetch list (with
per-sequence assembly fps) and the answerability report:

```bash
cd benchmark
python -m bench.manifest \
  --questions ../bench_v1_sample80/questions.jsonl \
  --media     ../bench_v1_sample80/media.jsonl \
  --fetch-out fetch_manifest.tsv --report-out answerability.tsv
```

`fetch_manifest.tsv` lists each media item's `drive_file_id` / path + `sha256` to
download and verify. **fps varies per sequence** (MOT17-02=30, MOT17-05=14,
MOT20=25) — always assemble at the manifest's `assemble_fps`, or every
location/temporal timestamp shifts.

**Answerability (54/80).** This harness answers from its own detectors/tracks, so
coverage is bounded by what it perceives, not by the runner. The 54 it can answer
are packaged as `bench_v1_sample54/` (regenerate with
`tools/filter_answerable_54.py`); point the run at that directory for a set where
every question is answerable. The 26 it cannot are documented in
[`CAPABILITY_GAP.md`](CAPABILITY_GAP.md) — meva/caviar semantic events outside the
harness's fight/smoking/fall/weapon taxonomy.

| bucket | n | how |
| --- | --- | --- |
| `cli_fight` | 15 | fight event_recognition + fight-end temporal — no GPU, CLI path |
| `gpu_tracks` | 39 | counting, location, and MOT count-over-time temporal — SAM capture |
| `no_detector` | 26 | meva/caviar **semantic events** (talks-to-person, opens-door, walking, …) — the scene engine has no such detector, so these are **structurally unanswerable** here and are left unpredicted rather than guessed |

The 6 MOT "count-over-time" temporal questions (e.g. *第一次恰好出现 N 名行人是第几秒*)
are computed from the SAM per-frame track set by `extract.parse_count_temporal`
+ `count_exact_time` / `count_max_duration` / `count_first_reaches_max` — unit
tested, no GPU.

**2. Generate the run plan** (one bash script for all 74 videos, in source order,
reusing one harness session):

```bash
python -m bench.run_batch \
  --media      ../bench_v1_sample80/media.jsonl \
  --questions  ../bench_v1_sample80/questions.jsonl \
  --media-root /data/benchmark_v1 \
  --base-url   http://127.0.0.1:7900 \
  --caps caps --out run_batch.sh
```

Every line it emits is a command this README already defines (`replay/*.sh`,
`bench.capture_live`). Start MediaMTX + one harness session
(`SCENE_ENGINE_MODE=active PERSON_TRACKING_ENABLED=true`, ingesting
`rtsp://localhost:8554/bench`), review the script, then `bash run_batch.sh` on the
GPU host. It writes `caps/<video_id>.json` per video.

**3. Predictions** — join captures to questions:

```bash
python -m bench.run_bench \
  --questions ../bench_v1_sample80/questions.jsonl \
  --media     ../bench_v1_sample80/media.jsonl \
  --captures  caps --output predictions.jsonl
```

`run_bench` skips (does not fabricate) any question whose capture lacks the needed
evidence, so the 26 `no_detector` items simply produce no line — tell the eval
side to score them as unanswered.

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
