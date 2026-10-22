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

The code includes a real-file 1–12 stream entrypoint. This document describes
implementation, configuration, timing boundaries and remaining scope. Experiment
records, raw responses, measurements, quality tables, comparisons and run status
belong in the project's Google Drive materials, not this repository. Keep them
separate from implementation documentation.

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

## Optional scene-prompt ablation

The real-file input configuration accepts `videojev_prompt_style`: `full`
(default), `scene`, or `brief`. `full` preserves the existing template. `scene`
omits position/ID association instructions from the scene-only template, which
has no person snapshots. It also removes duplicate last-frame/evidence guidance
only for `research-v1` without fixed definitions, where the policy already states
those rules. Main-baseline and fixed-definition templates retain that guidance;
the person-snapshot question interface is unchanged.

`brief` adds one separate change: its choices are `A. Present.`, `B. Assessed
absent.`, and `C. Insufficient evidence.` The six event definitions, sensitivities,
scene aggregation, source times, native task markers and three semantic values
remain. The parser checks the returned answer against the actual options, keeps
the original probabilities, and maps the shortened canonical options to the same
three values. Existing PPE option semantics retain their path.

The client constructs the chosen template once, then reuses the existing session
registration and compiled-text path. It adds no per-window text rewriting or GPU
comparison. These are optional prompt factors, not a claim of lossless model
equivalence.

## Service use

### Isolated vLLM pooling adapter

`--execution-backend vllm` selects the experimental adapter in
`tools/videojev_vllm`. The default remains `native`. Use a separate environment
with the adapter's pinned dependencies; do not install them into the native
PyTorch 2.8 environment. Its Linux x86_64 requirements select the official
vLLM 0.30.0 cu129 wheel, Torch 2.13 / torchvision 0.28 from the official cu129
index, and HF 5.14.1. Use a CUDA 12.9 development image and a compatible GPU
driver. Pin the official CPU TorchCodec wheel: the default Linux package can
require CUDA 13 libraries during vLLM imports. This prepared-input path uses
PIL/PyAV and the original HF processor, not TorchCodec CUDA media decoding;
visual and language computation still executes on the GPU. Fatal worker
errors retain their chained traceback in the existing server log.
FA2 and causal-conv1d require builds for this installation; their old
Torch 2.8 binaries cannot be reused. The same new environment also provides the
native comparison phase. The adapter is not a stock VideoJev endpoint provided
by vLLM, and implementation alone does not establish runtime compatibility,
classification equivalence, or a throughput improvement.

The adapter uses the vLLM 0.30 bare Qwen3.5 language executor, not a generative
language-model head. Its configuration callback leaves the engine's initial
type-discovery config unchanged, allowing the actual VideoJev config to load
before selecting its backbone and original decision-head settings. This is a
CPU configuration step, not a synthetic model invocation.
Original language weights are mapped into its fused
projections and hybrid attention/DeltaNet implementation. The final layer keeps
attention and its residual, without its removed MLP or post-attention norm.
The model's final norm adds that residual once. This changes language execution
kernels, not the trained decision task; numerical differences remain possible.

The original HF 5.14.1 visual module keeps FP32 parameters and BF16 autocast,
temporal chunks, and video-grid inputs. It is constructed with real CPU storage
before moving to the GPU, preserving non-persistent rotary buffers. The original
native load rounds visual parameters to BF16 before promoting them to FP32; the
adapter preserves that parameter path without rounding rotary buffers. The head
keeps its original FP32 loading exception. The original
processor, tokens, source-time questions, repeated/shared layouts and fresh-window
3D positions are reused. The FP32 original cosine head scores the original
candidate/question indices inside the GPU pooler. Only logits, probabilities and
the score mask return to the original formatter; the complete hidden sequence
does not pass through CPU memory.

The existing GPU worker owns an in-process `EngineCore` with `UniProcExecutor`.
Already-prepared tokens and video fields enter directly, without rendering or
tokenizing the prompt again. The static-budget processor declares that it does
not require a tokenizer; the original HF processor still prepares real inputs,
and the adapter's public preprocessing entry remains unsupported.
The first version allows up to two independent
requests and a 32768-token context, allocates 4 GiB for request-local attention
and recurrent state, and explicitly keeps recurrent state FP32. Prefix sharing,
multimodal processor caching, CUDA graphs and compilation are disabled. This
allocation is a configuration choice, not a measured memory-capacity guarantee.
The adapter does not reuse a previous window's video or hidden state. Short
internal request identifiers use timestamps and are not supplied as model text.

A small worker override skips stock full-model profiling and synthetic warmup;
an explicit multimodal size bound avoids synthetic video sizing. The first real
request therefore retains first-use kernel costs. Existing server scheduling,
input/publication clocks, parser and window-failure handling remain in use.
Service shutdown releases the engine on the GPU thread. Native-only optimization
switches and prefix layouts are not supported by this separate backend.

Compare native and adapter phases on the same GPU and new installation, with
identical original frame windows, labels, rules, resolution and scheduling.
Record the planned service switch, first-use costs, every processing opportunity,
unknown/failed windows, task quality and real event-report latency. A change from
the older Torch 2.8 installation is not an isolated engine improvement. For an
optional two-window engine batch, CUDA segment totals are marked `engine-batch`
and must not be interpreted as independent per-window kernel durations.

### Optional single-window operator profile

`--profile-request N` selects one real `predict_prepared()` call for PyTorch CPU
and CUDA profiling; the default zero uses the original runtime without a call
counter or profiler wrapper. Use single-window GPU batches for this capture.
`--profile-output DIR` selects its output directory, otherwise `kernel-profile`
beside the service result log is used. The directory receives `trace.json` and
an operator table grouped by input shape. No model text, prediction-request
fields, weights, choices, or classification rules change.

The selected call includes profiler overhead and trace export. Keep all of its
CPU, CUDA, service, and HTTP durations separate from normal inference and
event-response latency. Earlier real
windows can establish the existing runtime's warm state; retain their first-use
cost rather than adding dummy warmup calls. Subsequent windows run without
profiling in the same model process. Profiler start, stop, and export failures
are reported separately; they do not discard a successful prediction or resend
it. If profiling cannot start, the selected window still runs once through the
original inference path. Model exceptions retain the existing handling. This
trace is a
performance artifact, not proof of classification equivalence or task quality.

### Optional language MLP compilation

`--compile-language-mlp` is a disabled-by-default startup factor. After the
official inference preparation, it calls PyTorch's in-place `Module.compile()`
on the existing language-layer MLPs, using Inductor, dynamic lengths, and no CUDA
Graphs. The attention-only final layer has no MLP and is left unchanged. The
original modules remain in place; compilation itself does not quantize weights.
Visual encoding, attention,
DeltaNet, positions, hybrid state, the FP32 decision head, prompts, and parsing
remain on their existing paths. No new prediction-request fields or model text
are required. The existing loaded-model metadata includes the startup option.

Registration does not mean compilation has finished. The first real input and
new shapes can incur compilation cost; retain those costs separately from
steady-state measurements, without dummy warmup requests. Compiler exceptions
use the existing window failure handling, without suppressing errors or
resending the window through an eager fallback. Normal PyTorch graph-break or
recompilation behavior can still leave regions eager; this option alone does
not prove that every operation ran as compiled code or improved performance.

Compare an eager phase and a compiled phase on the same GPU, installation, and
weights. Drain the eager phase before changing this startup factor. Such a
comparison restarts the scoring service once and must not be described as the
same model process. Keep input layouts and cache settings fixed, compare saved
classifications and probabilities on the same original frames, then retain
full quality and completion denominators in the real-stream comparison.

### Optional language MLP GEMM autotuning

`--language-mlp-gemm-autotune` is a disabled-by-default startup factor that
also enables the existing regional MLP compilation path. It adds
`max_autotune_gemm=True` and `max_autotune_gemm_backends="ATEN,TRITON"` to
the MLP-only compile options, retaining dynamic lengths and disabled CUDA
Graphs. It does not enable global Inductor autotuning or exhaustive search.
The loaded-model metadata reports the effective MLP compilation startup option
and this factor, not proof that compilation has finished or covered every layer.
It adds no prediction-request fields, model text, or dependencies.

Inductor times candidate matrix kernels during first-use compilation. Its
internal timing uses compiler-generated buffers, not extra video or full-model
requests. Keep this first-use cost and any later shape compilation separate
from steady-state latency. Dynamic lengths do not guarantee that every shape
shares a compiled variant or receives its own optimal kernel. ATEN remains a
candidate, but this does not guarantee that compilation cannot fail. Existing
window failure handling remains unchanged, with no suppressed errors or
automatic eager resend.

For a BF16 comparison, enable `--compile-language-mlp` in both phases and change
only the GEMM-autotune startup factor. Keep FP8, prefix state caching, and
profiling disabled. Reuse the GPU, installation, weights, and fixed frame
inputs, draining before the one planned service restart. Compare complete
probability vectors and decisions, then real-stream quality and processing
coverage. Kernel selection can change floating-point evaluation, so neither
bitwise equivalence nor a speed improvement follows from enabling the flag.
The original visual, attention, DeltaNet, position, decision-head, prompt, and
parsing paths remain unchanged.

The implementation uses the pinned [PyTorch 2.8 matrix selection path](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/_inductor/kernel/mm.py#L663)
and [dynamic compilation behavior](https://docs.pytorch.org/docs/2.8/generated/torch.compile.html).

### Optional ordinary language normalization and visual block compilation

`--compile-language-norm` and `--compile-vision-blocks` are independent,
disabled-by-default startup factors. Both use in-place `Module.compile()` with
Inductor, dynamic lengths, and disabled CUDA Graphs after the author's inference
preparation. They do not enable language MLP compilation, GEMM autotuning, FP8, or prefix
state caching. No prediction-request fields, model text, or dependencies change.

The normalization factor selects only `Qwen3_5RMSNorm` instances in the language
model, including its input, post-attention, final, and attention q/k RMSNorms.
Their original FP32 normalization and `(1 + weight)` expression remain the source
computation. This is distinct from DeltaNet's already fused gated normalization,
which stays untouched. Attention projections and kernels, recurrent and
convolution state, language MLPs, visual encoding, positions, and the FP32 decision
head retain their paths.

The visual factor compiles each existing `backbone.visual.blocks` module in place.
It retains the author's visual parameter dtypes, original autocast, FlashAttention
backend, position inputs, residuals, and MLP definitions. It does not compile or
replace the entire language model, reuse prior images, lower image resolution, or
change sampling. Dynamic attention lengths and supported graph breaks can leave
parts eager; the flag is not evidence that every visual operation was compiled.

Compare an unchanged eager phase, ordinary-normalization-only phase, and
visual-block-only phase before combining factors. Keep all other startup options,
rules, frames, and offline references fixed. Reuse one GPU, installation, and
weight source, draining each phase before its planned scoring-service restart.
The phases therefore do not share a single model process. Preserve real first-use
and shape-compilation cost without synthetic warmup or health requests.

Compilation can change floating-point evaluation and may fail. Existing window
failure handling remains unchanged, without suppressed compiler exceptions or
automatic eager resends. Compare probabilities and decisions on fixed original
frames and retain full event, negative, unknown, and processing denominators in
the streaming comparison. Neither numerical equivalence nor a speed improvement
follows from these startup options. Loaded-model metadata records the effective
options, not completed compilation or all-layer execution coverage.

The regions follow the pinned [Qwen3.5 implementation](https://github.com/huggingface/transformers/blob/v5.14.1/src/transformers/models/qwen3_5/modeling_qwen3_5.py)
and [PyTorch Module compilation interface](https://docs.pytorch.org/docs/2.8/generated/torch.nn.Module.html#torch.nn.Module.compile).

### Optional merged language gate/up projections

`--fuse-language-gate-up` is a disabled-by-default startup factor for the pinned
bias-free Qwen3.5 language MLPs. After the author's inference preparation, it
replaces each remaining language MLP with `FusedGateUpMLP`. The final decoder
layer, whose MLP was removed by the author, is left alone.

The original computation is `down(act_fn(gate(x)) * up(x))`. The replacement
concatenates the original BF16 gate and up weights along their output dimension
once during loading. It performs one linear projection, splits the output in
the original gate/up order, then uses the original activation and down projection.
The combined parameter keeps the original device and dtype and disables gradients.
A meta-device Linear avoids random initialization of another CUDA matrix. Replacing
the original MLP drops its gate/up module references rather than keeping an extra
permanent copy. The weights on disk and the model's trained values are unchanged;
runtime parameter names differ, so this inference wrapper is not a checkpoint
export format.

This changes matrix output width, not the input token count or total arithmetic.
Different matrix kernels and split views may affect floating-point results and
performance. Neither exact numerical equivalence nor a speed improvement is
promised. Nothing is concatenated or copied back to the CPU on each window.
Vision, attention, recurrent state, positions, the FP32 decision head, prompts,
sampling, scheduling, and existing failure handling retain their paths.

The flag does not enable compilation, FP8, or prefix caching. Compare it alone
against an unchanged phase on one GPU and installation, using the same fixed
original frames and streaming inputs. Drain before the planned service switch,
retain first-use costs, and score quality, unknowns, failures, coverage, and
wall-time separately. Loaded-model metadata records the selected startup factor;
no prediction-request fields or dependencies are added.

The expression follows the pinned [Qwen3.5 MLP implementation](https://github.com/huggingface/transformers/blob/v5.14.1/src/transformers/models/qwen3_5/modeling_qwen3_5.py).

### Optional FP8 language MLPs

`--language-mlp-fp8` is disabled by default and requires the optional
`torchao==0.13.0+cu128` wheel for the pinned PyTorch 2.8 CUDA 12.8 environment:

```sh
python -m pip install --no-deps --index-url https://download.pytorch.org/whl/cu128 \
  torchao==0.13.0+cu128
```

After official inference preparation, only the Linear modules inside existing
language-layer MLPs use TorchAO's version-2 dynamic FP8 activation/FP8 weight
configuration. It explicitly selects per-row scaling, the PyTorch scaled-matrix
backend, and `use_fast_accum=False`. The pinned model has 31 MLPs with three Linear
modules each; the attention-only final layer remains unchanged. Weight storage
and matrix inputs in those Linear operations are quantized, with BF16 outputs.
This is a lossy precision experiment, unlike exact prefix-state reuse.

`set_inductor_config=False` prevents TorchAO's recommended-settings helper from
changing global FP32 matrix precision or unrelated Inductor settings. Vision's
existing parameter/autocast behavior, attention, DeltaNet state, positions, FP32
decision head, prompts, choices and parser retain their original paths. The
optional import and weight conversion run only at startup; no new prediction
fields, per-window comparisons or alternate-answer retries are introduced.

The flag is independent of `--compile-language-mlp`. For a compiled FP8 comparison,
enable compilation in both the BF16 and FP8 phases; otherwise the comparison
mixes precision with compilation. Conversion precedes regional MLP compilation.
Keep the GPU, installation, weights source, media and scheduling fixed, drain
each phase before restarting the model service, and retain first-use costs.
The saved startup metadata identifies the option and TorchAO version, but does
not prove every operation used an FP8 kernel or that quality was preserved.

References: [TorchAO 0.13 configuration](https://github.com/pytorch/ao/blob/v0.13.0/torchao/quantization/quant_api.py#L1602),
[FP8 Linear dispatch](https://github.com/pytorch/ao/blob/v0.13.0/torchao/quantization/quantize_/workflows/float8/float8_tensor.py#L232),
and [PyTorch 2.8 SM89 rowwise dispatch](https://github.com/pytorch/pytorch/blob/v2.8.0/aten/src/ATen/native/cuda/RowwiseScaledMM.cu#L945).

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

The default remains native single-window scoring; this option is an explicit
research setting. Compare steady behavior separately from first-use kernel work,
service restarts and reused compilation caches.

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

Packed scoring remains opt-in. Compare the same source frames and timestamps
when assessing changed choices or probabilities; changed real-time window
selection does not by itself establish improved recognition. Report quality and
negative coverage alongside latency and throughput.

### Optional question-layout ablation

`--question-layout repeated|shared|prefix|prefix-cache|prefix-cache-aligned` selects the service default;
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

This avoids rebuilding fixed questions in the client on every request and
building native groups only to reconstruct them in another layout. It does not
change event criteria, the three choices, model weights, the trained head, or
the formatter. Source review does not establish token/output equivalence or
end-to-end performance. CPU image preparation and source coverage remain
separate parts of the pipeline to measure.

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
remain part of the actual trial. This reuses model state as well as tokenization.
It does not promise bitwise equality: prefix placement changes context relative
to native layouts, and splitting a hybrid forward can change floating-point
rounding. Compare native shared, prefix without cache and
prefix with cache on quality, raw scores, wall-time and stream coverage before
selecting a deployment setting. No performance result is implied by this code.

`prefix-cache-aligned` is a separate experimental factor. The pinned
[FLA 0.4.2 DeltaNet forward](https://github.com/fla-org/flash-linear-attention/blob/v0.4.2/fla/ops/gated_delta_rule/chunk.py#L35)
uses 64-token chunks. Splitting a full sequence at an arbitrary text-prefix end
changes those chunk boundaries. The aligned mode caches only the largest multiple
of 64 tokens within that pure-text prefix. Its remaining policy tokens are
computed with the current window suffix; no padding, deletion, reordering or new
model context is introduced. A prefix shorter than one chunk uses the full
uncached text-first forward rather than an empty prefill. Full-prefix and aligned
base states are stored separately on the fixed policy object, so switching this
request factor can reuse the same loaded model without mixing their states.

Both modes still compute positions from the complete current sequence, preserve
its candidate indices and copy the entire hybrid state before suffix evaluation.
The existing `prefix_tokens` count describes the actually cached span when
caching is enabled; first prefill and remaining forward tokens keep their actual
costs. This factor investigates a concrete difference in evaluation order, not a
confirmed defect or a promise of equal probabilities. Attention/projection shapes
and the explicit recurrent-state boundary still differ from a full forward.
Compare the same original frames, raw probabilities, three-valued decisions and
paced-stream results before selecting it. The original `prefix-cache` mode and
all defaults remain available and unchanged.

The runtime retains the pinned model module's original `prepare_query_groups`
for legacy full-question requests and its
`format_predictions(output, rows, record, decoder)` for both interfaces. Native
`repeated`/`shared` use the same model-forward call as `predict_features`;
the two prefix modes use the explicit language-backbone/head path above.
No downloaded model source or weights are edited. Default-mode equivalence is a
read-only source assessment, not an executed comparison. The interfaces were read
from the pinned HF [model source](https://huggingface.co/ZhuYuhan/VideoJev-4B/raw/1c1a253bf90dfa66176124d25a48ffb791d517d5/modeling_videojev.py)
and [processing source](https://huggingface.co/ZhuYuhan/VideoJev-4B/raw/1c1a253bf90dfa66176124d25a48ffb791d517d5/processing_videojev.py).

### Grouped result transfer

Single-window scoring now transfers each complete logits, probabilities and
score-mask tensor to the CPU before calling the original native formatter. The
existing packed-window path already uses this arrangement. The formatter's
per-question mask counts and list conversions therefore run on CPU tensors,
rather than triggering repeated small GPU readbacks for each event class.

The native formatter, FP32 conversion, valid-option slicing, choice ordering and
parser remain unchanged. Model execution, prompts, cached state and video
selection do not change. This is a result-transfer optimization, not a fix for
prefix-cache score differences or a claim of numerical equivalence between
cached and uncached model forwards. No additional request option, timing field,
comparison pass or synchronization is introduced. Existing decision-stage and
total timings include the transfer and formatting work. Performance and output
behavior still require the authorized real-input evaluation; source review alone
does not establish a speedup. No independent software tests were run.

No synthetic warmup or health request is sent by the adapter. Its first request
uses actual arrived frames. The local adapter does not retry, follow redirects,
use proxy environment settings, or fall back to another model. Client and server
deadlines must be chosen together; an elapsed client deadline does not mean an
already-started CUDA operation has stopped. Recent completed outcomes and active
requests are reused by transport ID, without repeating GPU work.

## Fixed original-frame classification

`tools.videojev_benchmark.points` uses the existing `VideoJevInspector`, event
question template, JPEG preparation, service and choice parser. It does not use
the paced source scheduler. This separates a change in the model's judgment on
identical frames from a faster configuration selecting different streaming
windows. The regular `run` entry point and application defaults are unchanged.

Provide a regular observation config with `streams`, labels and layout, plus a
separate observation-only window list. Each window contains only `id`,
`stream_id`, ordered zero-based `frame_indices`, and their actual
`timestamps_seconds`. The runner decodes sequentially to those original frames;
it does not infer frame numbers from nominal FPS or annotation seconds. Source
timestamps and the configured native positional FPS remain separate.

```bash
python -m tools.videojev_benchmark.points \
  --config /path/to/observations.json --windows /path/to/windows.json \
  --execute --output /path/to/new-results
```

The caller supplies references only to a later offline analysis, never to this
runner. One stable client session registers the existing template once, and each
window is sent once. A window's input, HTTP or parsing failure is retained while
later windows continue; the existing explicit shared-service failures and
authorization denial stop further calls. A 429 cools down for at least 60 seconds
or a longer finite numeric Retry-After, without replaying the window.

The output separates processed windows, completed requests, failed windows and
three-valued judgments. The unchanged service log retains native choices and
probabilities. HTTP duration excludes local decoding/JPEG preparation; the
separate client duration includes it. Neither duration is an event's real-stream
reaction latency. Point annotations do not establish natural ON/OFF or negatives
at nearby times, and scene judgments do not identify an annotated actor.

Compare layouts or requested class sets on identical original frames before
attributing streaming differences to model classification. Fixed-window
classification and paced-stream quality/latency have separate denominators.
Labels without references remain unscored. Event sensitivities stay explicit
in the observation configuration; smoking uses HIGH and fight, fall and weapon
use STANDARD under the existing policy.

### Optional category and window factors

Observation configurations can vary the policy profile, requested labels and
window geometry without changing application code. Compare research-v1 with
main-baseline, combined labels with single labels, and alternative window
settings as distinct factors. Keep answer options, sensitivities and offline
references fixed when they are not the factor being examined. Reuse one GPU,
environment and resident model for compatible configurations.

A policy-profile comparison evaluates existing profiles as a whole.
Main-baseline omits the research-wide guidance but retains some confidence
wording; it is neither a pure deletion ablation nor the official model's built-in
policy. Single-label groups remove other definitions and preceding questions as well as reducing
question count. Their requested shared layout uses the existing repeated
fallback for one question. They do not establish combined multi-class capacity.

For a window-geometry comparison, keep the selected final frame. For example, a
four-frame candidate can add earlier frames over about one second with nominal
FPS 3. It changes frame count, temporal span and positional FPS together; actual
source times remain separate. Added earlier frames may cross a shot cut. Offline
scoring uses only requested labels, preserving each label's reference denominator;
unrequested classes are not format failures. Default application behavior, event definitions,
weights and original labels do not change. These are configuration choices,
not performance or quality claims.

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
and shared requests on the same updated service; instrumentation changes can
prevent direct attribution to the layout alone.

Timings also retain `question_layout_requested`, actual `question_layout`
(`repeated`, `shared`, or `mixed`), and `question_layout_groups`. These are service
metadata and are not appended to model text. A shared request can legitimately
report repeated or mixed layouts when prefixes differ or a group falls back.

Real event response time still starts when the original event frame enters the
Harness before sampling, and ends when the corresponding report can be read.
HTTP includes server waiting and computation: do not add those internal
measurements to HTTP a second time. Server durations use the server clock; its
absolute clock is not subtracted from the local input clock.

Source-label ON times do not by themselves establish natural action onset or
actor attribution. Keep that annotation boundary distinct from the actual
input-to-readable-report measurement. For streaming summaries, the existing
analyzer uses nearest-rank percentiles. Compute input-period update rates over
the common input span, excluding drain-period reports; distinguish all reports
from definite-label judgments that exclude insufficient evidence. Retain all
scheduling opportunities when reporting completion, including those not sent.

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

Further quality and sustained-load evaluation remain. Parameter tuning must
report quality, dropped/expired windows, per-stream
completion and update intervals, failures, and event response delay together.
Dropping most requests can reduce the latency of survivors without meeting
twelve-stream capacity. Small tuning clips are not a held-out final benchmark.
Counting, location, duration, OFF evaluation, and offline GT derivation are not
implemented by this event-model integration.
