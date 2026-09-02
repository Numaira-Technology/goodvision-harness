# VLM rich annotation audit — 2026-09-03

The complete, correct rich result has **not** met the 2.5-second target. This
audit used saved responses and offline tests; it made no new model requests.

## Frame and box ownership

Paddle and VLM feed the same `RichLabelSnapshot` interface and
`RichLabelCoordinator`. Display frames and annotations retain their source
session, frame sequence, and timestamp. The shared compositor associates event
alerts with rich tracks, then draws the combined result. A matched event does
not draw an additional event rectangle. Unmatched or ambiguous event evidence
keeps its own box so that an uncertain association cannot suppress an alert.

JPEG resizing and the experimental person crops prepare API inputs. They do
not generate a second video timeline. The exact analysis-image output and the
continuous delayed output are existing views, not separate event/Rich videos.

Two confirmed defects were fixed:

- A fusion exception previously retried Rich drawing and then independent event
  drawing. Both exact and delayed output now fall back to event rendering on
  the original frame; failed event rendering retains the original frame.
- Crop merging previously discarded detections based on which half owned their
  center. A right-only detection at full-frame center 0.4975 was lost. Merging
  now keeps unmatched detections and removes only one-to-one opposite-crop
  duplicates with IoU at least 0.5. This does not recover people never detected
  by either request, nor prove perfect identity association in crowded scenes.

## Comparable measurements and their limits

All rows below used the same authorized office image. The OpenRouter rows used
`qwen/qwen3.5-35b-a3b`, DeepInfra only, no provider fallback, reasoning disabled.

| Run | Measured time | Result and timing boundary |
| --- | --- | --- |
| Actual normal four-event path | 3.129 / 2.088 / 2.156 s; mean 2.458 s | Three valid calls, all events negative and no boxes. Includes input encoding, response and event parsing. Local face recognition is separate. |
| Full Rich repeat, objects protocol | 10.949 s | Valid 4 people + 8 objects. First content 2.124 s; another 8.824 s to completion. |
| Original OpenRouter Rich smoke | 111.433 s | Valid 4 people + 8 objects. No phase timing was saved, so the exceptional delay cannot be assigned to a specific phase. |
| Parallel compact runtime 06 | 2.788 s | Failed parsing; not a successful full-frame latency. |
| Parallel structured diagnostic 09 | 2.478 s request group; 2.858 s with initial client setup | Only one merged person before the crop fix. Domain parsing/merging was outside the group timer. Not a quality or latency acceptance pass. |
| Parallel structured runtime 08 | 9.998 s | Whole-frame timeout with one branch incomplete. |

Local evidence is in Git-ignored `output/normal-event-baseline-10/`,
`output/vlm-latency-baseline-01/`, and the numbered Rich experiment directories.
The earlier matrix's event prompt had an extra 233 tokens and used a streaming
probe instead of the normal production client. Use baseline-10 for the actual
normal-event comparison.

## What can explain slower Rich results

Normal events returned 227–228 output tokens. Full Rich returned about 1,100.
The 8.824 seconds after first content in the full Rich repeat cannot be explained
by uploading its input image. It includes response generation and delivery;
the exceptional 111-second run remains unexplained without phase timing.

The normal input was 320×180, JPEG quality 75: 10,300 base64 image bytes and a
14,253-byte complete JSON request. Rich used 720×405, quality 85: 41,340 base64
image bytes. Three Rich branches uploaded 92,456 image bytes in total. Parallel
requests can overlap server work but increase aggregate image traffic. Image
bytes alone do not establish an upload bottleneck. Reducing max side to 480
does not shrink the 432×405 person crops; max side 320 would. Attribute accuracy
at smaller sizes still needs validation.

There are separate runtime delays:

- Rich has one in-flight frame and one replaceable pending frame. Normal events
  permit six concurrent requests. Thus call latency and label refresh rate are
  different measurements; a 2.5-second single-worker call yields at most about
  0.4 rich updates per second. Continuous frame replacement prevents unbounded
  queue growth, but reset does not cancel an old in-flight request.
- The experiment still configures a 25-second display buffer, versus 2.5 seconds
  in normal config. This directly delays displayed video, independently of API
  timings. The old objects/single configuration and buffer have not silently
  been replaced by the unvalidated fast experiment.
- VLM spatial IoU tracking uses previous successful detections. It is weaker
  through occlusion/crossing than Paddle MOT. Slow updates often hold old boxes
  beyond the interpolation interval; shared rendering does not solve that.

## Instrumentation and next comparison

Coordinator status now separates local queue, inference, and total completion
time. Replacing a pending frame updates its timing origin, and old-session
completions cannot overwrite new-session measurements. These exclude display
buffering. Each timing identifies its attempt frame and success/failure, so a
fast failed response cannot be mistaken for the last successful frame's time.
The streaming diagnostic now records HTTP/1 and HTTP/2 request-body
write duration and time to first content after that write. Client write
completion is not proof of server receipt, and the remaining interval is not a
direct measure of server queue time.

Optional strict compact JSON schema prevents some malformed response shapes;
it does not guarantee all people or requested attributes are present. It is
disabled by default. Existing experiments also varied sampling and transport,
so their differences cannot all be attributed to resolution or parallelism.

The next controlled comparison should hold model, route, sampling, output
contract and timing boundary fixed; compare complete Rich coverage before
accepting a speed gain. Start with one full-frame compact request, then change
only input size. Shorter output addresses the measured response tail; compare
parallel execution only with correct merging and all branches completed. Both
API latency and the displayed result's age must pass separately. Do not count
partial, malformed, or missing-person responses as 2.5-second successes.

Full offline regression: **1,049 passed, 7 skipped**. A subsequent focused
coordinator regression covers explicit timing-to-attempt association. These
tests verify software behavior, not live recognition quality or provider latency.
