# Model interfaces and protocol identity

For new current-120 runs use `tools/unified_benchmark/README.md`. All adapters share
1 FPS JPEG input, but APIs/CLIP/SigLIP2 consume one frame while VideoJev/Decision3
consume adjacent two-frame windows. Native precision is retained and recorded.
The older interfaces below are retained for historical interpretation; they do
not override the named current protocol or establish identical model inputs.

# Model classes and runner boundaries

A shared table does not make unlike experiments controlled comparisons. Record the
runner commit, dataset/reference hashes, question set, sampling, aggregation,
threshold comparator, eligibility policy and timing boundary for every row.
Only claim a model-only comparison when those inputs and rules match (apart from
the specified per-backend output envelope). Otherwise label a system comparison.

## Language and Luna: current v2 runner

`tools/event_benchmark/run_eval.py` supports:

| Candidate | Model ID | Backend | Score | Fixed baseline |
|---|---|---|---|---|
| qwen | qwen/qwen3-vl-8b-instruct | language | JSON boolean -> 0/1 | max >= 0.5 |
| gemini | google/gemini-2.5-flash | language | JSON boolean -> 0/1 | max >= 0.5 |
| luna | openai/gpt-6-luna-decisions | decision | finite noul probability [0,1] | max >= 0.5 |

Same reviewed inputs, 1 fps centered schedule, width 720, independent frame calls.
Every scheduled frame must succeed. Output includes attempted/valid/error counts
and frozen hashes. qwen is the reference row. Language has no continuous threshold
to tune. A Luna score's [0,1] range does not establish empirical calibration.

## VideoJev: separate native experiment

Read `docs/guides/videojev-candidate.md`, `tools/videojev_probe/` and the specific
`tools/videojev_benchmark/` experiment being used. The native probe consumes 2–4
original JPEGs per window, with six independent present/absent/insufficient-evidence
questions and a pinned model revision. Its option scores are uncalibrated. Preserve
uncertain verdicts; missing/invalid answers fail the window. Do not apply a scalar
0.5 or claim it uses the event runner's frame protocol.

The probe guide explicitly distinguishes its sampled windows from a twelve-camera
throughput test. The benchmark suite has additional person/scene questions and its
own deadlines. Neither is automatically the base-4 exam. Check actual timestamps,
question mapping and frozen references; do not assume a universal 0.8s cadence or
that a historical/private run is reproducible from these public tools.

## CLIP / SigLIP2: authorized native integration

`tools/clip_benchmark/` was selectively imported from draft PR #62 commit
`99d515a3f561460561d21e50cff994ad85f0fd99`, then adapted with user authorization;
the PR was not merged. Read its README and staged runner before use. No CLIP
backend exists in `run_eval.py`. Offline preflight does not verify CUDA/model loading.

A cosine positive-minus-negative margin is not a probability. Record its text
pairs, model revision, comparator and threshold. An untuned zero-margin baseline
and validation-tuned thresholds belong to distinct labeled variants. Only reuse
a selector after verifying its validation-only inputs, completeness checks and
deterministic tie rule against `bench-threshold-tune/references/protocol.md`.
The controlled variant samples at 1 FPS from source time zero, using original
decoded pixels and the checkpoint processor, FP32/TF32-off, twelve paced file
slots and native error denominators. API centered JPEG sampling is different.
These are explicitly labeled system comparisons, not a model-only comparison.

## Importing results

1. Obtain the exact run artifact and hashes; do not infer data from source code.
2. Verify cohort, reviewed labels, source-video disjointness for tuning and full
   expected coverage. Historical v1, provisional and incomplete outputs remain
   explicitly labeled; they cannot become v2 results by renaming a file.
3. Preserve each runner's denominator and unknown/error counts. Only map equivalent
   base-4 questions into base-4 cells. All-120 includes a separate three-valued task
   and needs an agreed combined metric; a 100-question run is not all-120.
4. Keep probe timing, last-frame-to-ready latency and API round-trip latency distinct.
   Leave @12 cells blank unless measured at 12 with the agreed boundary.
5. Use `bench-sheet-contribute` for a requested publication. In the pivoted layout,
   Model identifies a **model/protocol/variant**, and dataset blocks share that row
   only when their provenance is compatible. Fixed and tuned results never overwrite
   each other. Record per-block run provenance in cell notes.
