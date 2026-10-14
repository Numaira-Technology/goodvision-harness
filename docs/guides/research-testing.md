# Unified research testing

`codex/research-testing` combines current main, GPU-host analysis, VLM research
experiments and Ashley detector evaluation. Use a separate worktree and output
directory for experiments. Historical runs and the preserved Supervision worktree
remain separate.

## Integrated sources

| Source | Revision | Role |
| --- | --- | --- |
| Main | `6c28dddd26bed7a7be50762a4c7dd6b796b2dbe3` | Current application and organized event runtime |
| GPU-host analysis | `f93b91e` | Authenticated local gateway, camera isolation, migration and release lifecycle |
| YOLO/VLM research | `139711ef2162cf8d892d12b3ed78faff5cb1331c` | VLM mode/quality/concurrency studies and detector preview |
| Ashley native evaluation | `d167188a4dcca7b2e7fc8fe9c4de121ef7e87b25` | Benchmark cohorts, native detector evaluation and frozen protocol reference |

Original source branches remain available. The branch uses current runtime
implementations for normal execution. Historical Ashley publications retain
their own source fingerprints and experiment environment. New runs record new
provenance.

## VLM modes and scheduling

Simulation is the default. Inspect a plan before selecting a real backend:

```bash
python -m tools.vlm_benchmark.modes --plan-only
python -m tools.vlm_benchmark.modes \
  --streams 1 2 --duration 3 --repeats 1 \
  --event-policy main-baseline --output outputs/research/vlm-baseline
python -m tools.vlm_benchmark.modes \
  --streams 1 2 --duration 3 --repeats 1 \
  --event-policy research-v1 --parallel-scheduling paired \
  --grounding-format lean-v1 --output outputs/research/vlm-paired-lean
```

`main-baseline` retains main's event prompts; normal runtime configuration also
retains main's result retirement behavior.
`research-v1` selects the shared research criteria. Paired scheduling and lean
grounding are explicit alternatives. Research runners retain completed responses
for delivery accounting and record that setting; normal runtime defaults retain
main's behavior. Compare only runs with matching media, annotation and policy
fingerprints.

The mode manifest hashes nested runtime implementations, including `runtime/`,
`events/`, `actions/` and `evaluation/`. It also records effective admission,
scheduling, model, provider and output settings.

Related tools: [behavior comparison](vlm-behavior-compare.md),
[grounding concurrency](vlm-grounding-concurrency.md),
[parallel concurrency](vlm-parallel-concurrency.md) and
[VLM mode benchmark](vlm-mode-benchmark.md).

## Detector monitoring

Set `PREVIEW_DETECTOR_URL` and `PREVIEW_DETECTOR_MODEL` when using the VLM monitor
with an existing detector worker. Detector boxes are presentation evidence; they
do not enter VLM observations. The detector evidence view pairs the actual
captured image with its boxes and source time, including slow responses. Moving
video overlays require a fresh matching frame. Loop changes, stage changes and
stop clear stale evidence.

## Ashley native evaluation

`python benchmark/run_native.py --help` exposes the integrated deterministic
runner. It records source-time tracking and native event transitions, then answers
the benchmark questions from those records. Questions are never inserted into
observation windows, and there is no final answer-model call.

Use `--runtime-profile current-main` for the integrated application or
`--runtime-profile frozen51` for the isolated archived Ashley engine. The latter
verifies the original source hashes before import. Its new orchestration and
results are identified separately from the original publication.

Example preflight with local authorized media and an existing detector endpoint:

```bash
python benchmark/run_native.py \
  --questions /path/to/research/geometry-questions.jsonl \
  --media /path/to/research/media-paths.jsonl --media-root /path/to/media \
  --tracking-endpoint http://127.0.0.1:8091 \
  --backend detector --detector-model yolo26-s \
  --runtime-profile current-main --geometry-only \
  --output-dir outputs/research/ashley-geometry --plan-only
```

Supply a geometry-only question selection and a local media-path mapping for
that example. The retained public cohort manifests do not contain private local
media paths. Full input and authentication instructions are in
[Native research evaluations](native-research-benchmark.md).

Use question selections supported by the chosen runtime profile. Geometry-only
execution requires a tracking worker and performs no event-model calls. Event
evaluation records each observation request and enforces `--max-model-calls`;
choose that budget before removing `--plan-only`. The detector worker must report
the requested model, weights and CUDA provenance. Frozen worker requirements live
with the archive and should use their own environment.

Native replay is sequential and backpressured in source time. Its timing is
offline evaluation timing; use the VLM or gateway suites for realtime throughput
and deployment latency measurements. Raw predictions and audit records preserve
empty frames, source/session identity and observed versus predicted tracks.

## Experiment environments and output

Follow [CONTRIBUTING](../../CONTRIBUTING.md) to create a worktree-local environment.
Use separate pinned environments for archived detector/model stacks. Keep model
weights and credentials external. Put generated databases, videos, JSONL and
reports under a new `outputs/research/<run-id>/` directory. Do not reuse a
preserved event database or historical results directory.

## Checks and maintenance

```bash
python tools/check_repo_hygiene.py
python -m pytest
git diff --check
```

The normal suite includes `benchmark/tests`. CI runs WHEP receiver stress in its
own process and separately checks two authenticated gateway relays inside the
deployment container. Local synthetic regressions establish protocol and runtime
behavior; real model accuracy, GPU throughput and private-media scores require
separate explicitly configured experiments.

Merge new main changes into this shared branch and rerun the baseline regressions
before publishing comparisons. Preserve historical experiment revisions and
record the exact current revisions with each new run.
