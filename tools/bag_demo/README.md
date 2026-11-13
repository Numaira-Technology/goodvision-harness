# Bag-inspection demo (work in progress)

This standalone demo is separate from the frozen benchmark and production console.
It is **not a completed combined knife + hosted-alcohol detector**. The page calls
VideoJev for knife/weapon candidates; `alcohol_vlm.py` is a standalone helper and is
not called by `server.py`. `alcohol_status: "not_integrated"` is explicit in the API.

The original PR reports a ten-clip smoke sample with only two knife positives and
one alcohol positive. Those author-reported counts, and the demo's existing 0.95
knife/weapon option-score gate fitted on that sample, are not independent validation,
calibrated confidence, or benchmark acceptance evidence. No benchmark labels,
protocols or thresholds are changed here. A combined real-model run is still missing.

## Run locally

Install the repository runtime dependencies plus `ffmpeg` and `ffprobe`. Put only
media you are authorized to process in the ignored `demo-media/bag/` directory
(or set `BAG_DEMO_MEDIA`). The supported clip extension is `.mov`, case insensitive.
Symlinked clips/sidecars and paths outside that directory are rejected.

```sh
VIDEOJEV_ENDPOINT=http://127.0.0.1:8093 PYTHONPATH=src python -m tools.bag_demo.server
```

The page binds to `127.0.0.1:8099`. `BAG_DEMO_PORT` changes the port. This is an
unauthenticated local demo: only loopback hosts are supported, cross-site browser
requests are rejected, and responses are not cached. Do not expose it through a
public proxy. Media frames are sent to the configured VideoJev scorer when playing.

Optional YOLO overlays require **already installed local weights** at
`workers/detector_rich_label/yolo26s.pt` and the matching Ultralytics dependencies:

```sh
PYTHONPATH=src python -m tools.bag_demo.yolo_boxes --video /absolute/path/clip.MOV
```

The helper writes `clip.MOV.boxes.json` beside the clip. Missing weights fail before
Ultralytics can download them. Bottle/wine-glass boxes are object candidates, not
alcohol or contraband verdicts. Corrupt sidecars produce an explicit overlay error.

## Verdicts and failures

- The headline reports a knife/weapon **candidate**, uncertainty, or that no
  knife/weapon was detected in sampled windows. It never declares the bag clear.
- A positive below the unchanged demo gate, or an insufficient native weapon/knife
  answer, stays uncertain. The VideoJev alcohol bar is experimental, not hosted VLM.
- A positive candidate is retained within a continuous playback segment. Clip changes
  and seeks reset it. Late responses from a previous segment are discarded. The
  displayed source timestamp is the sampled window, not the current playback position.
- Clip changes and seeks cancel the page's previous scoring wait; leaving the page
  also cancels overlay loading. Scoring waits expire after 60 seconds and overlays
  after 10 seconds, including reading the response body. A timeout stays unknown;
  subsequent playback windows can retry. Client cancellation does not guarantee
  cancellation of work already running in the server/scorer. The server retains
  its single scoring slot until that work finishes or hits its own bounded timeout,
  and returns unavailable while busy.
- Before 0.4 seconds there are insufficient past frames. No future frame is sampled
  to fabricate a full window. Invalid/out-of-range timestamps are rejected; extraction
  is bounded, empty ffmpeg output fails, and only one scoring request runs at a time.
- Failures remain unavailable/unknown. The optional truth display is a **whole-clip
  annotation**, not a correctness badge for a partially viewed window.

The standalone hosted helper reads exported `OPENROUTER_API_KEY` or
`DASHSCOPE_API_KEY` (it does not load `.env`). It sends whole frames to a paid hosted
provider when explicitly invoked; choose approved media/provider/model beforehand.
It requires HTTPS and a strict boolean + finite 0..1 confidence reply. Failed or
low-confidence frames remain unknown; a valid positive can coexist with partial
errors. These scores are model self-reports, not calibrated correctness probabilities.

## GPU launcher and recovery

`python -m tools.start_bag_demo` **creates a paid Vast instance**. Run it only with
separate rental and data-processing authorization. It requires `VAST_API_KEY`,
`SAM_SSH_KEY`, and local `ssh`, `ssh-keygen`, `git` and `rsync`. R2 credentials are
optional; `HF_TOKEN` is optional for the fallback. It loads the local ignored `.env`.
Only tracked `src/` and `tools/` files are uploaded. Prefetch credentials go through
SSH stdin, never command arguments. The tunnel binds only to loopback.

Cleanup is attempted on interruption and startup failures. It is not guaranteed
across kill -9, host loss or provider failures. `--status-file PATH` writes JSON
states (`starting`, `ready`, `stopped`, `cleanup_required`) with the instance ID,
unique label and cleanup confirmation. `stopped` confirms cleanup, not successful
inference. A lost create reply is reconciled only when the unique label identifies
one instance. If creation or release is unconfirmed, inspect the reported ID/label
in Vast, destroy the instance if present, and confirm billing stopped before retrying.
`--resume-instance-id ID` takes cleanup responsibility only for a bag-demo-labelled
instance explicitly selected by the operator.

## Offline review validation

```sh
PYTHONPATH=src python -m pytest tests/test_bag_demo*.py tests/test_start_bag_demo.py
```

The tests use synthetic media and mocked Vast, HTTP and model boundaries. The page's
actual JavaScript is exercised with delayed responses using Node. They do not rent a
GPU, download weights, call a paid model or upload customer clips, and cannot replace
an authorized combined real-model run. The author must confirm whether this PR is to
land as this partial demo or wait for the missing combined integration and evidence.
