# RF-DETR 1288 count comparison

Run the existing 36 counting questions with RF-DETR Large 1.3.0 at resolution 1288, confidence 0.10 and num_select 300. The user selected 1288 because the old RF model does not support 1280.

The reference YOLO26-S arm uses imgsz 1280, confidence 0.10 and max_det 300. The dimensions differ by eight pixels. RF selects up to 300 candidates across classes before the person filter; YOLO limits person detections to 300. These are not identical capacity settings.

Use the original RF checkpoint (SHA-256 `86879a31a42ea4c091ee8d758654bf5c8c465f773f08e95a9143653f0f529cba`, 1571684963 bytes), JPEG bytes, PTS, tracker and count parser from `detector-count36-4090-stock-resume3-20261014`. This comparison changes RF resolution and confidence together. It cannot establish the effect of either setting alone.

## Execution contract

`detectors.py` is a separate adapter derived from the frozen count worker. It does not replace a production worker. `run_count.py` consumes the existing read-only `/run/public`, `/run/frames`, `/run/input-manifest.json` and `/run/extraction` mounts and writes new results under `/run/results`. Set `DETECTOR_EXPERIMENT_LABEL` to the exact frozen round label. The gateway must enforce the same label for reset requests.

Start this arm with fresh preservation maps. Do not import old RF baseline answers under the new parameter identity. Within a later recovery of this same configuration, reuse only hash-verified outcomes. Never resend an in-flight stateful frame. The parent controller must freeze code and input identities, enforce one standard 24 GB RTX 4090, and release its exact resources in a finally block.

An OOM or server-side operation timeout becomes a skipped question only after explicit model-process cleanup. Shared IO, transport, ownership, parameter or extraction failures interrupt the round; unattempted questions remain pending.

The gateway retains its 74-second frame-inference deadline. The HTTP client allows 115 seconds to receive the result, including the gateway's separate process termination (up to 30 seconds). Model loading uses the existing 295-second gateway deadline and a 335-second response wait. These waits do not extend the inference deadline or authorize a repeated frame.

Score all 36 questions with the unchanged PR63 rule. Terminal failures receive zero MRA and strict-match credit while retaining null answers. Report full-set MAE as unavailable if any numeric answer is absent. Keep full-set and paired completed-subset results separate. Preserve original answers and historical scores.

This is a person-counting arm, not the three-lane Harness. It makes no event, compact or final-answer model calls.

Tests were not run because the user did not authorize them.

