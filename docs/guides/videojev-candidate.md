# VideoJev-4B small candidate validation

> 2026-10-20：本文实验工具使用独立的 `tools/benchmark_runtime/`；产品 grounding 已恢复 main。历史结果不等于恢复后的产品性能，见[代码与测试边界](pr48-49-person-grounding-integration.md)。

This replaces the **Qwen judgment candidate**, not YOLO. It does not modify any
production route, six-class criteria, existing formal benchmark, or recording UI.
It is **not** a twelve-camera throughput test. The first probe selects 12 original
windows across `output.mp4` without looking at labels, and scores six independent
present/absent/insufficient-evidence questions per window. Multiple classes can
be positive simultaneously. Smoking remains HIGH; the other five remain STANDARD.

## Source and input contract

- Model: `ZhuYuhan/VideoJev-4B`, revision
  `1c1a253bf90dfa66176124d25a48ffb791d517d5` (weights and custom code pinned).
- BF16 backbone, with official `prepare_for_inference()` promoting vision/head
  to FP32. Record actual peak memory; do not infer it from weight size.
- Official Transformers/FlashAttention API, **not the existing vLLM server**.
- Exact original 2–4 JPEGs, no mosaics, crop, or frame selection. Native video
  preprocessing may resize/temporally pad; record its grid and padding. Source
  times stay in the prompt; native positional times use nominal 2.5 fps.
- One encode per window; six questions grouped with the official API. Context
  32768; upstream may split larger person groups to fit. No cross-window cache
  or incremental-streaming claim.
- Reference is an existing independent Astra packet. Its verdicts never enter
  model input. Require exact video/policy/image fingerprints. No new Astra calls.
- Option scores are uncalibrated; preserve the selected uncertain option, with
  no extra confidence threshold. Missing/invalid answers fail the entire window.

Upstream: <https://github.com/MCG-NJU/VideoJev> and
<https://huggingface.co/ZhuYuhan/VideoJev-4B/blob/1c1a253bf90dfa66176124d25a48ffb791d517d5/USAGE.md>.

## Offline preparation

Run from this checkout with its `src` on PYTHONPATH (avoids an unrelated editable
checkout). Planning/preparation never downloads weights or starts GPU work.

```bash
PYTHONPATH=src python -m tools.videojev_probe plan
PYTHONPATH=src python -m tools.videojev_probe prepare \
  --source-packet /path/to/astra-unified-v2-20261009/packet \
  --output outputs/videojev-candidate/packet
```

The output includes original JPEGs, unchanged reference subset, frozen policy,
query preview and selection provenance. Use a new output directory each time.

## CUDA environment (only on an available, authorized GPU host)

Do not install these dependencies into vLLM's Python environment. Create a new
Python 3.12 environment on the retained volume; first verify driver compatibility
and sufficient free storage/GPU memory. Do not kill an existing model process to
free memory. The scripts do not rent, start or destroy Vast instances.

```bash
python3.12 -m venv /workspace/videojev-env
/workspace/videojev-env/bin/pip install --upgrade pip setuptools wheel packaging ninja
/workspace/videojev-env/bin/pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
/workspace/videojev-env/bin/pip install -r tools/videojev_probe/requirements.txt
/workspace/videojev-env/bin/pip install flash-attn==2.8.3.post2 causal-conv1d==1.7.0 --no-build-isolation
```

These native dependency versions passed the 2026-10-19 RTX 4090 24GB scene
probe with Python 3.12.15 and CUDA 12.8. Save `pip freeze` with each deployment.
Transformers 5.14.1 requires safetensors >=0.8.0; the requirements file pins 0.8.0.
This validates the small sequential scene probe, not twelve-stream capacity.
No CPU fallback. Keep the existing YOLO worker environment separate.

For Vast, the verified base image is
`pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel`, in SSH launch mode with no custom
on-start command. That image contains Python 3.11: install Python 3.12 separately
before creating this environment. On the tested host the persistent mount is
`/data`, so use `/data/videojev/env` and `/data/videojev/models/huggingface`
instead of the example `/workspace` paths. Verify the mount before installation.
The unchanged YOLO worker's OpenCV build also needed the system packages
`libxcb1 libgl1 libglib2.0-0` on this Ubuntu image.

Downloading is explicit, on the persistent volume (several GB):

```bash
PYTHONPATH=src /workspace/videojev-env/bin/python -m tools.videojev_probe download \
  --cache-dir /workspace/models/huggingface
PYTHONPATH=src /workspace/videojev-env/bin/python -m tools.videojev_probe run \
  --packet outputs/videojev-candidate/packet \
  --cache-dir /workspace/models/huggingface \
  --output outputs/videojev-candidate/scene-run
```

`run` only loads cached weights; it does not download implicitly. Check model load
time, first-window timing, preprocessing, visual encode, decision time and peak
allocated/reserved GPU memory separately. No warmup is silently discarded.
Application timer excludes initial packet validation/model loading but includes
YOLO (when enabled), model preprocessing/inference and answer validation. It is
not source-to-publication timing under the formal twelve-stream scheduler.

## Optional person-ID compatibility probe

Start/warm the **existing** YOLO26-S `/v1/tracks` service, then use a new output:

```bash
PYTHONPATH=src /workspace/videojev-env/bin/python -m tools.videojev_probe run \
  --packet outputs/videojev-candidate/packet \
  --cache-dir /workspace/models/huggingface \
  --scope person --detector-url http://127.0.0.1:8092 \
  --output outputs/videojev-candidate/person-run
```

The broker is reused unchanged. Per-frame normalized positions identify people
in original full-scene images; no boxes/IDs are drawn on those images. Each sparse
window starts a new session, avoiding false continuity across omitted frames.
All six scene questions remain; each last-frame ID adds five person questions
(litter remains scene-only). More than four people fails a window rather than
silently dropping people; this is a **probe budget**, not a production limit.

This evaluates whether VideoJev understands location-referenced IDs. It does not
claim reliable identity binding or replace the main grounding contract yet.
The retained Astra labels measure scene judgments only; independent person-level
review is still required, especially ambiguous/overlapping people and small objects.

## Reading results and next gate

`manifest.json` records code hashes, model revision, input fingerprints, runtime
versions and status. Per-window files retain request/response, detector snapshots
and timing. `summary.json` uses all planned windows as denominator, reports missing
attempts, invalid answers, uncertainty, comparable count and per-class errors.
Per-class reference coverage explicitly exposes classes with no positive examples.
This clip cannot establish sensitivity for unseen positive classes.

No conventional output-token TPS is reported (direct scoring). Judge complete
windows/sec and latency only after real runs; do not infer capacity from mocked
tests. Successful small probes gate a later twelve-stream test; they do not prove
the 3-second/95% SLA or equivalence with Qwen. Do not change the formal routes until
quality, ID binding and runtime compatibility have been reviewed.

## First GPU probe: 2026-10-19

RTX 4090 24GB, both scene and person probes completed 12/12 windows with no OOM.
Scene: <=3s 11/12, P50 0.694s, P95 26.708s; first window 58.485s, remaining
windows 0.670–0.708s. Person: <=3s 5/12, P50 4.918s, P95 17.516s. No window
was excluded from these headline statistics. Both produced 45/64 (70.3125%)
agreement with the same certain scene reference judgments; eight uncertain
smoking references were excluded from the binary denominator. All 12 PPE
reference positives were missed, with five litter false positives. This is
not sufficient evidence to replace Qwen, despite the successful runtime check.

Person windows had one to three last-frame IDs (11/16/21 total questions).
Their measured detector stage includes broker creation/destruction per sparse
window (about 0.501s), including the HTTP server shutdown polling interval;
it is NOT pure YOLO inference time. Snapshot queue/HTTP timings are retained.
The scene run preceded the person run; cache states differ, and no profiler
separated JIT/autotuning time. Do not attribute every long response to compilation
or report these figures as steady-state twelve-stream latency. Peak PyTorch
reserved memory was 10.30 GiB (scene) / 17.13 GiB (person), excluding the separate
YOLO process and driver overhead. Person-level accuracy remains unmeasured.

Evidence: `outputs/videojev-candidate-20261019/remote-audit/results/scene-02`
and `person-01`; runtime freezes and deployment logs are in the sibling `logs/`.
The earlier `scene-01` failed before inference because a model download was
incomplete; retain it as setup evidence, not as a valid inference benchmark.

## PPE adaptation controls (prepared offline; not model-tested)

The default remains `--ppe-variant baseline --ppe-option-order canonical`.
Its actual requests are byte-for-byte equivalent as JSON values to the first
probe requests. Never overwrite the old results or present previews as results.

Three explicit variants are available:

| Variant | PPE question | Model questions per scene / additional person |
| --- | --- | --- |
| `baseline` | Original six-policy prompt and abstract options | 6 / 5 |
| `focused` | Exact original PPE rule, relevant common rules, concrete options | 6 / 5 |
| `equipment` | Separate helmet/head and vest/torso observations | 7 / 6 |

The other five event query texts and options, original evidence frames, policy
fingerprint, sensitivities and reference labels remain unchanged. PPE is still
one of SIX output events. The additional observations are retained separately.
The focused/equipment variants explicitly apply the existing equipment rule
regardless of office/factory setting; no industrial-scene condition is introduced.
Equipment observations are missing / assessed compliant / unassessable:
either missing => PPE positive; both compliant => negative; otherwise uncertain.
Never treat cropped/occluded areas, a lost ID or a missing response as compliant.
Scene observations cover all visible people, not just YOLO's last-frame IDs.

PPE options alone support `canonical`, `rotate`, and `reverse` ordering. The
letters are relabelled A/B/C and decoding uses the option's meaning, not its
index. Actual option-to-value mappings are recorded. Changing PPE questions
changes shared context and workload; inspect all six classes for regressions,
and do not call this a fixed-load performance comparison.

Inspect actual requests without a GPU, download, model call or network:

```bash
PYTHONPATH=src python -m tools.videojev_probe preview \
  --packet outputs/videojev-candidate-20261019/packet \
  --output outputs/ppe-focused-preview \
  --ppe-variant focused --ppe-option-order canonical
```

Add `--person-run /path/to/original/person-01` to preview using the exact saved
YOLO IDs/locations. Replay requires a finished person run, matching input/policy/
reference fingerprints, complete windows and valid frame-aligned snapshots.
No predictions or Astra reasons enter the generated questions. The preview
writes requests, semantic mappings and input/request hashes, never predictions.

For a FUTURE explicitly authorized GPU run, the existing `run` command accepts
the same PPE flags. In person scope, choose exactly one of a live
`--detector-url` or `--person-run`. Saved-ID replay isolates prompt effects but
EXCLUDES live YOLO latency; its manifest and summary are labelled accordingly.
It is not comparable to complete pipeline timing. Live detector timing retains
the original broker lifecycle overhead; this patch does not silently change
timing definitions. All windows remain in the completion denominator.

Suggested later order: focused canonical vs baseline; PPE option-order control;
then equipment observations if needed. Do not automatically execute all variants.
The current 12-window packet has no PPE negatives: also obtain independently
annotated compliant/occluded examples before accepting a quality improvement.
Synthetic unit-test fixtures only verify logic and are not model-quality evidence.

## Twelve-stream focused-PPE experiment (2026-10-19)

The user authorized an exploratory twelve-stream run after the three-way PPE
probe. `tools.videojev_benchmark` adapts the native choice API to the existing
tracked suite, without replacing production default routes. The profile retains
12 streams, 2.5 FPS evidence, 1.25 observations/s/camera, original 55.798021s clip,
48 global / 4 per-camera admission slots, paired parallel scheduling, six-class
policy, real YOLO26-S IDs, and IDs-only rendering. Each mode has two rounds;
preflights and ten-second warmups are separately marked. Calls expire at 20s;
three seconds remains the scoring deadline. `parallel` makes separate events
and grounding calls and reports same-window joint completion.

The local scoring server uses **one serialized GPU worker**, a bounded 48-entry
FIFO, and drops queued work that has expired. It does not implement continuous
batching, cross-window visual reuse, vLLM, or synthetic duplicate-camera caching.
Consequently the result measures this native serving implementation, not an
upper bound on VideoJev capability. Preserve queue/inference timing and failed
requests in the audit. The client sends no cloud credentials and has no fallback.

PPE uses `focused` with canonical option order. Events retains the small-probe
six scene questions. Grounding adds five questions per last-frame person and
narrows the scene questions (except litter) to **unassigned** visible occurrences,
as required by the shared tracked-grounding aggregation contract. This target
clarification is recorded; it is not a change to the six-class criteria. No
people are silently truncated: the shared limit is 64 IDs, with invalid or
oversized groups failing closed. The earlier four-person limit remains only
in the standalone small probe.

The original independent 69-window Astra packet is matched by video, image and
policy fingerprints. It supplies scene-level agreement only, not person-ID
accuracy. Twelve identical source clips create the historical load experiment;
results must not be described as twelve different physical camera scenes.

Offline regression covers all three modes with twelve spawned camera workers,
HTTP contract, uncertainty, native response identity and real tracked result
assembly. Run results and browser recordings are retained separately from the
small PPE study under `outputs/videojev-twelve-20261019` after collection.
