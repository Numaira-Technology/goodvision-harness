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

The code includes a real-file 1–12 stream entrypoint. The initial sixteen real
parameter/load runs used `fe95fa9` and the repeated question layout. Separate
seven-run layout and seven-run prefix-cache comparisons have also completed.
They do not establish usable twelve-stream capacity or justify making the
experimental prefix cache the default. A later seven-run comparison has also
deployed the fixed-template optimization below. All sent native requests
completed, but input coverage, CPU preprocessing and quality still prevent a
twelve-stream usability claim. A subsequent same-host startup-thread comparison
completed eight real-file runs. All twelve streams received regular updates on
that host, but changing the CPU thread limit produced no material throughput
gain or quality improvement. Event false positives, missed intervals and unscored
categories still prevent a full usability claim. Completed Qwen-based 50-video
results remain separate and unchanged.
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

CPU preparation workers and Torch intra-op threads are different settings.
For a startup-thread ablation, set `OMP_NUM_THREADS` and `MKL_NUM_THREADS` before
starting the service; MKL's setting takes precedence. Unset both for the default
baseline. Loaded-model metadata records `torch_cpu_threads` and
`torch_interop_threads`; these describe the model-loading thread, not an
independent measurement of every worker. The request protocol, pictures and event
rules do not change. Compare configurations on the same GPU and installed
environment, restarting only the service between drained configuration groups.
Record cold-start and steady real-file results separately. This is an optional
performance experiment, not a changed production default or a proven fix for
input lateness. See the [PyTorch 2.8 environment-variable documentation](https://docs.pytorch.org/docs/2.8/threading_environment_variables.html).

The completed comparison retained the default thread setting: limiting it to
one, four or eight did not improve throughput materially, and the default had
fewer negative clips with positive reports in this development sample. Later
service restarts reused the same environment and persistent compilation cache;
their shorter first requests are not an isolated thread-count speedup or an
independent cold-start measurement. GPU scoring and service queueing, rather
than CPU image preparation, dominated the steady twelve-stream runs on that host.

### Optional single-sequence scoring

`--single-sequence` is off by default. For a fixed non-prefix template it sends
each complete window to the language model as one unpadded sequence, without
`cu_seq_lens_q/k`, `max_length_q/k` or all-zero `seq_idx`. Visual encoding, full
native mRoPE, causal attention, BF16 language weights, FP32 choice head, task
questions and formatting are preserved. Each context group starts from no cached
state. Actual two-window packed scoring still supplies its distinct boundaries;
prefix layouts and the generic non-template interface retain their existing paths.

The author's single-window wrapper always supplies packed metadata. In the pinned
[FLA index preparation](https://github.com/fla-org/flash-linear-attention/blob/v0.4.2/fla/ops/utils/index.py#L121),
this can copy sequence lengths to the CPU; same-window layer calls are already
cached, so the extra copy must not be counted once per layer. Without those
arguments, [FlashAttention](https://github.com/huggingface/transformers/blob/v5.14.1/src/transformers/modeling_flash_attention_utils.py#L757)
uses its ordinary unpadded call. The unchanged native three-plane mRoPE produces
no text packing positions in [Qwen3.5](https://github.com/huggingface/transformers/blob/v5.14.1/src/transformers/models/qwen3_5/modeling_qwen3_5.py#L1176).
FLA retains its zero-initialized chunk kernel with one equal-length sequence,
and causal-conv1d retains its CUDA implementation without a segment-index tensor.

The existing scoring helper also bypasses the wrapper's repeated GPU boolean
checks for an already constructed fixed template. Any combined timing change
must not be attributed only to omitting packed metadata. Native per-window mRoPE
and formatter synchronizations remain. New kernel specializations can add first
use costs or alter near-boundary numeric results.

The optional path was deployed at `56c5007` in six real-file trials on one RTX
4090. Each configuration ran the original cold clip, three-stream case and
twelve-stream input. Both used the same installation, weights, six event rules,
shared two-frame layout, 0.5-second frame spacing and 0.8-second scheduling
opportunities. The controller restarted only the drained service to change the
startup option. It did not lease another GPU or redeploy between groups.

| Twelve-stream result | Native single window | Single-sequence option |
| --- | ---: | ---: |
| Completed / 1,800 opportunities | 922 | 924 |
| Busy opportunities not sent / waiting expiry | 878 / 0 | 876 / 0 |
| Valid reports during input | 899 | 901 |
| Valid updates/s, common input span | 7.482 | 7.499 |
| Longest valid-update gap | 2.434 s | 2.461 s |
| Source intervals reported | 6/8 | 7/8 |
| Source intervals reported within 3 s / all intervals | 4/8 | 5/8 |
| Negative clips with a positive | 6/8 | 4/8 |
| Positive negative-clip windows | 46/615 | 42/616 |
| Median service elapsed time | 1.575 s | 1.569 s |
| Median service queue | 1.429 s | 1.423 s |
| Median GPU-work interval | 132.258 ms | 131.860 ms |
| Median CUDA scoring interval | 103.850 ms | 103.064 ms |

The option ran for all 924 candidate windows; all 922 baseline windows used the
native path. Both retained one GPU window at a time. The update-rate difference
was about +0.22%, too small to support a material speedup or a default switch.
All twelve streams received valid updates with no gap above three seconds on
this short load, but roughly 1.58-second per-stream intervals still missed about
half the fixed 0.8-second opportunities. Timing intervals overlap and must not be
added. GPU memory peaks were almost unchanged.

The six commonly reported source intervals were four slower and two faster with
the option; the median paired difference was +0.179 s. The candidate's extra
report used different selected frames. It does not establish improved model
recognition. For the 132 matching original two-frame/time combinations, all 792
event choices and their saved probabilities matched. That observation covers
only those saved windows, not all inputs or mathematical equivalence. Only fight
has reference labels; the other five categories remain unscored. Eight negative
clips come from four original videos, and source ON is not a separately accepted
natural action onset or subject match.

All 2,051 sent requests completed across the six trials; fight semantic unknowns,
invalid judgments and request failures were zero. The native first full HTTP
request took 60.437 s, completing only one of sixty cold-clip opportunities, with
58 busy opportunities unsent and one waiting expiry. After the same-host service
restart, the option's first request took 45.875 s and four of sixty completed.
First-use kernel work and reused environment/cache order differ, so the 14.6 s
gap is not a steady-state optimization result. Both later three-stream trials
completed all 100 requests. No extra warm-up requests were sent.

The controller collected all six results and released the GPU; a separate exact
provider query confirmed the instance was absent. The default remains native
single-window scoring; this option remains an explicit research setting.
No independent software tests were run.

### Optional ready-window packed scoring

`--gpu-batch-size 1|2` defaults to one, preserving the original single-window
route. Two-window mode takes only an immediately ready second choice window
from the existing pipeline. It does not wait to fill a batch, reorder the queue,
combine videos into one timeline, remove event questions, or change model weights.
Each window retains its own request, source times, future, result and failure.
Prefix layouts and windows requiring multiple native context groups retain the
single-window path without repeating a model call.

The author's wrapper accepts one video. The optional runtime adapter still
encodes each video separately and computes its own complete mRoPE positions.
It packs the resulting unpadded language sequences using explicit cumulative
query/key lengths and the longest individual sequence length. Distinct `seq_idx`
segments reset causal-convolution boundaries; FLA recurrent states start from
zero for each segment. The trained FP32 choice head is unchanged, with candidate
and decision indices offset within their own sequences. Output tensors are copied
once to the CPU and split by window before the original formatter and parser.
The loaded linear-attention layers must actually use the FLA and causal-conv1d
implementations: their plain Torch fallbacks ignore packed boundaries. If those
implementations are unavailable, the existing single-window route remains usable.

This relies on the pinned [Transformers FlashAttention interface](https://github.com/huggingface/transformers/blob/v5.14.1/src/transformers/modeling_flash_attention_utils.py#L794),
[Qwen3.5 DeltaNet implementation](https://github.com/huggingface/transformers/blob/v5.14.1/src/transformers/models/qwen3_5/modeling_qwen3_5.py#L441),
[FLA sequence boundaries](https://github.com/fla-org/flash-linear-attention/blob/v0.4.2/fla/ops/common/chunk_delta_h.py#L62),
and [causal-conv1d segment handling](https://github.com/Dao-AILab/causal-conv1d/blob/v1.7.0/csrc/causal_conv1d_fwd.cu#L304).
No past-video cache is retained. A shared forward failure affects only its actual
participating windows; formatting or parsing errors remain per-window. There is
no automatic resend or whole-video restart.

Source review found that an early fallback could reverse window order or delay
classification of a fatal CUDA error. Eligibility is now decided before model
calls, and indexed outcomes return to the service immediately. A fallback second
window gets its actual start time and original expiry check; an individual visual
encoding error does not fail its unattempted neighbor. The GPU owner is restored
before resuming remaining work. These are source-reviewed changes, not executed
error-path validation.

Per-window timings record the actual `gpu_batch_size`. Shared scoring time is
the real wait experienced by both windows, not divided by two. Consequently
overlapping batch intervals must not be added as if they were independent GPU
work. Extra memory and the first window waiting for the second visual encoding
are trade-offs to measure, not an assumed speedup. The default is unchanged.

The optional implementation was deployed at `00b035d` and measured in six real-file
trials on one RTX 4090, using the same installation and weights. Each configuration
ran the original cold clip, three-stream case and twelve-stream input. Changing
the startup option restarted only the drained scoring process, not the machine
or environment. All 2,158 requests completed; fight semantic unknowns, invalid
judgments and request failures were zero. No independent software tests were run.

| Twelve-stream result | Single window | Up to two ready windows |
| --- | ---: | ---: |
| Completed / 1,800 opportunities | 960 | 953 |
| Valid reports during input | 937 | 930 |
| Valid updates/s, common input span | 7.803 | 7.745 |
| Longest valid-update gap | 2.276 s | 2.375 s |
| Source intervals reported | 6/8 | 7/8 |
| Negative clips with a positive | 6/8 | 6/8 |
| Positive negative-clip windows | 47/640 | 55/635 |
| Median service elapsed time | 1.497 s | 1.508 s |

The packed case actually used two-window scoring for 952/953 requests, so its
roughly 0.75% lower update rate was not caused by widespread single-window fallback.
The shared GPU-work interval increased from a median 124.8 to 252.5 ms; dividing
that shared interval by two would misrepresent either window's actual latency.
The six commonly reported source intervals were three faster and three slower,
with a median paired difference of +0.193 s. The changed reported-event set and
real-time frame selection prevent attributing the extra report to better model
recognition. Negative coverage and false positives must accompany latency.

The first single-window service request took 37.84 s. A later restarted service
reused the environment and compilation cache; its first 1.03 s single-window
request is not a packed cold-start improvement. The first actual packed forward
took 8.87 s in the three-stream trial, which missed 29/100 opportunities. Those
startup costs remain in the results rather than being hidden by warm-up calls.

In the saved twelve-stream outputs, 193 matching original-frame/time pairs shared
1,156/1,158 six-event choices. Two near-boundary fall/PPE choices changed, so neither
numerical nor output equivalence is established. Those classes lack ground truth
in this comparison. The eight negative clips come from four original videos;
source intervals are not accepted natural-onset or subject-level annotations.
Packed scoring remains an opt-in experiment, not the recommended configuration.
The GPU was released after all results were retrieved. Twelve-stream quality and
the full P0 goal remain unresolved.

### Optional question-layout ablation

`--question-layout repeated|shared|prefix|prefix-cache` selects the service default;
`repeated` remains the default. A `/predict` request may specify `question_layout`.
Omitting it uses the fixed service default. The selected value belongs to that
request and does not mutate the service or restart its loaded model. Ready output
records `question_layout_default`.
The real-file runner accepts `videojev_question_layout` in its observation JSON
and passes it through the native event worker as `question_layout`; the default
is `repeated`. Its run and stream manifests retain that selected configuration.

The original repeated-layout group boundaries are retained. The fixed-template
path computes them from compiled message lengths, including this window's actual
source-time tokens; the legacy full-question path still calls the pinned HF
revision's `prepare_query_groups`. Shared mode examines the exact prefix before the last
newline followed by `For ` in every query in that group. Only when all prefixes
are present, nonempty, and identical does it keep that complete prefix in the
first query and keep the remaining query suffixes in their original order.
Otherwise the entire group uses the repeated layout. Single-question groups also
remain repeated. Each original group retains its own full prefix; reducing text
does not merge groups across the original context boundary. If rebuilding a shared
group cannot retain that boundary, it falls back to the original repeated batch.

Event definitions, source times, all requested events, question IDs, options, and
three-value meanings remain available. The original rows still drive the native
result formatter and unchanged choice parser. This is a prompt-layout ablation:
removing repeated token occurrences changes positions and the context seen by
later questions, so outputs are not claimed to be equivalent. It is not a
multi-video batch. Its speed and quality must be distinguished from the separate
prefix-cache layout below.

### Fixed streaming questions

The native event client now prepares its fixed rules, scene questions and options
once when its inspector is created. On the first real request it sends a
`question_template` through the existing `/predict` endpoint. Each row has the
existing choice ID, task, options and two `query_parts` surrounding the actual
source-time list. Later windows on that session send pictures, `frame_times` and
the existing transport fields, not the rules again. No template UUID, content
fingerprint or extra registration request is introduced. Explicit person/PPE
probe callers may still supply full `questions` using the legacy interface.

The service binds that template to the existing session. Windows already queued
hold the template they arrived with, so a later configuration cannot change their
questions. Identical templates from different streams share compiled text. The
service retains at most 64 session bindings; a missing binding affects only that
window. The client includes its template on the next new window rather than
resending a failed window or restarting the service.

`tools/videojev_probe/question_templates.py` compiles the fixed native chat text,
options and repeated/shared/prefix variants. A small eight-entry cache handles
native frame-count/FPS/spatial-grid shapes. Only the short source-time span is
tokenized per window; adjacent pre-tokenizer units stay with that span to avoid
cutting through a BPE unit. Original greedy group boundaries are derived from the
current lengths rather than repeatedly formatting and tokenizing every candidate
question group. Final tensors and option/decision indices still describe this
window's actual sequence. The dynamic visual sequence and positions are not
reused across windows.

This replaces two costs found in the completed cache comparison: the client
rebuilt all fixed questions on every request, and the runtime first built native
groups only to rebuild them in another layout. It does not change event criteria,
the three choices, model weights, the trained head, or the formatter. Source
review is not an executed token/output equivalence comparison. The real-file
comparison confirmed sub-millisecond median question preparation; slower CPU
image preparation and missed input opportunities still dominated the complete
pipeline on that host. This is not an end-to-end speedup or quality guarantee.

### Experimental static-policy prefix cache

`prefix` and `prefix-cache` are opt-in layouts for a separate real-file comparison.
The native layout puts video before question text, so question states already
depend on that window's pictures. Reusing those states for different pictures
would be incorrect. Both new layouts instead move only the identical policy
before `Evidence metadata:` into a complete text-only user message before the
original video message. Source times, person metadata, judgment instructions,
questions and options remain in the dynamic suffix. The same original group
boundaries, original rows, context budget and result formatter are retained.
Unsupported prefix/group shapes use the original repeated layout and report it.

Both experimental modes construct the same full token sequence. `prefix` scores
that sequence without state reuse. `prefix-cache` prefills the text-only prefix
once. The immutable policy object owns its token IDs, GPU token IDs, embeddings
and pure-text base state. A cache hit reads that object directly: it does not
copy prefix IDs or positions back to the CPU, compare their contents or retokenize
the policy. Every window deep-copies the whole Transformers hybrid cache,
including full-attention KV, convolution and DeltaNet recurrent state. The
video-updated copy is discarded; it never becomes the base for another window.
Changing the fixed configuration creates a corresponding policy object. No hash,
UUID, old picture or past event answer is used as a cache key.

The previous implementation accepted complete questions on each window and
reconstructed a key from GPU prefix tokens and positions. That guarded a generic
changing-query interface at the wrong frequency for a fixed streaming policy and
forced device-to-host synchronization on cache hits. Binding the state to the
configuration removes that comparison rather than replacing it with a per-window
CPU fingerprint. The legacy full-question path still extracts its policy from the
supplied queries; it is not the fixed-template fast path.

The opt-in scorer calls the existing language backbone, injects this window's
fresh visual features and recomputes positions from its complete new sequence.
It slices suffix positions and candidate/decision indices only after computing
them. Transformers constructs unpadded continuation masks without native packed
q/k lengths, which no longer match after prefix caching. The original trained
decision head, FP32 scoring, mask, softmax and weights are unchanged. Native
`repeated`/`shared` forward calls remain unchanged.

`prefix_cache_groups` records prefix tokens, hit/miss, real prefill/copy time and
the tokens forwarded in that call. Initial prefill and layout first-call costs
remain part of the actual trial. This is genuine model-state reuse, not only
tokenization caching. It does not promise bitwise equality: prefix placement
changes context relative to native layouts, and splitting a hybrid forward can
change floating-point rounding. Compare native shared, prefix without cache and
prefix with cache on quality, raw scores, wall-time and stream coverage before
selecting a deployment setting. No performance result is implied by this code.

The runtime retains the pinned model module's original `prepare_query_groups`
for legacy full-question requests and its
`format_predictions(output, rows, record, decoder)` for both interfaces. Native
`repeated`/`shared` use the same model-forward call as `predict_features`;
the two prefix modes use the explicit language-backbone/head path above.
No downloaded model source or weights are edited. Default-mode equivalence is a
read-only source assessment, not an executed comparison. The interfaces were read
from the pinned HF [model source](https://huggingface.co/ZhuYuhan/VideoJev-4B/raw/1c1a253bf90dfa66176124d25a48ffb791d517d5/modeling_videojev.py)
and [processing source](https://huggingface.co/ZhuYuhan/VideoJev-4B/raw/1c1a253bf90dfa66176124d25a48ffb791d517d5/processing_videojev.py).

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

Additional service timings record `native_group_count`, total
`native_input_tokens`, and `native_group_input_tokens`. Token counts include the
native visual slots and text in every actual group; they are not generated-token
counts. `question_group_prepare_seconds` measures text/group preparation. For a
template, this includes its initial compilation when needed and the current
dynamic source-time fill; legacy requests include grouping and reconstruction.
`gpu_scoring_seconds` sums CUDA-event intervals around the group model-forward
calls. It excludes visual encoding, text/group preparation, and native CPU result
formatting, but is not a sum of profiled kernel times. Existing `decision_seconds`
still covers the whole post-encoding decision stage, including CPU work and
formatting; all prior timing fields retain their boundaries. These internal
measurements must not be added to full HTTP time a second time.
The four numeric measurements also pass through the Harness report transport.
The extra event/timing instrumentation can itself add overhead. Compare repeated
and shared requests on the same updated service; do not assume a new repeated
measurement is performance-identical to the historical `fe95fa9` run.

Timings also retain `question_layout_requested`, actual `question_layout`
(`repeated`, `shared`, or `mixed`), and `question_layout_groups`. These are service
metadata and are not appended to model text. A shared request can legitimately
report repeated or mixed layouts when prefixes differ or a group falls back.

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
arrival, selection, and lateness. `reports.jsonl` saves original reports.
Decoder-reported frame position and milliseconds are retained beside the nominal
frame grid, including after a seek. Missing or disagreeing decoder positions must
be reported during offline time matching, not assumed to establish alignment.
These values are recording fields, not additional model context.
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
