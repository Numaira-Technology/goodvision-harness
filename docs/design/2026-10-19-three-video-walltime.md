# Three-video native streaming timing run

## Scope and authority

The user authorized a small wall-time trial, with physical conflict first if
needed. The existing local plan selects clips 0018, 0028 and 0002, in that order.
The partial-observation answer recovery is a different run and has finished.
This trial makes new observations only for these three videos. It does not replay
old unknown requests, replace original answers, or send ground truth to a model.

## Operational adaptation

`tools/streaming_walltime_trial.py` prepares the current published application,
extends the latest cumulative ledger, and adapts the previously used resource
controls to a separate experiment. It uses timestamp-based session names and
explicit configurations, without adding content hashes or hash gates. Public
capture scope contains only video identities, not questions or reference labels.

The capture entry remains `evaluation.answer.run` without a release/answer
phase. Native scheduling, full timed playback, source/drain limits, stock YOLO26-S
1280 / confidence 0.10 / max_det 300, Qwen3-VL-8B, events 1600, compact 8192,
75/180-second request limits and face-disabled behavior remain in place. No
additional health probe, dummy request, unit test, lint or build is performed.

The outer owner retains both existing task locks. It confirms the dedicated
network and exact historical task GPUs are absent before one launch. Creation
has one attempt per experiment; an unknown create outcome never permits another
card. Existing provider selection and exact release logic are reused. Container
inspection is restricted to ownership and state fields. The release path runs
after capture or failure and independently handles remote release if Docker is
unavailable. Exited owned containers are removed only after logs and release
evidence are saved. Results, media, weights, caches and ledgers are preserved.

The frozen remote worker/deployment source comes from resume41 and retains its
existing official checkpoint protocol; this is not a claim that every historical
resource tool is hash-free. Only the current capture app and new wrapper are
published here; experiment data and private resource records remain local.

## Measurement and limitations

Measure each actual request, frame-to-report publication, EOF/drain, and setup
separately. Preserve retries, explicit 429 cooldown, failures and skipped frames.
Analyze event matches offline using the source-backed event table. The three
positive source intervals are two in 0018 and one in 0028; 0002 is a video-level
negative. Source annotations include boundary conventions and are not claimed to
be independently confirmed natural contact onsets. Missed events remain in the
denominator and do not receive zero latency. A broad window spanning two events
is ambiguous, not two independent discoveries. Never subtract video seconds
directly from host timestamps or generalize this small sample to all 182 questions.

## Execution ledger

- Base: published application `a6df607`; original pre-task Harness baseline is
  `6c28ddd`, not the previous recovery round. This adapter changes orchestration,
  source selection and reporting scope only, not the detector weights or native
  processing method. The earlier cumulative application changes remain separate.
- User testing restrictions override default development test workflows. Only
  the authorized three-video trial will execute; no independent tests ran.
- Preparation, deployment and successful timing results are distinct states;
  the existence of this wrapper alone proves none of them.
- Read-only review found the reused launcher requires its existing creation
  budget identity. Preparation now carries that identity and prior attempt count
  forward; it does not replace the budget to bypass an unknown outcome. This
  compatibility repair was inspected in source, not independently tested.
