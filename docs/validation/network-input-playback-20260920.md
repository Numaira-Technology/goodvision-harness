# Network input and bounded playback follow-up

Branch: `codex/sam-realtime-queue`. These changes are independent of the opt-in
SAM overload policy. Keep playback at **4 seconds**, with analysis running during
the buffer. No single-frame model reset, person-confirmation change, or inference
resolution reduction is introduced.

## Changes

- Network sampling uses media PTS when available. The fallback advances a fixed
  sampling grid instead of restarting the interval at each arrival. Bursty
  delivery therefore does not discard otherwise valid 15 FPS source frames.
  Reconnects and backward source-clock jumps rebase to a monotonic local timeline.
- `HARNESS_STREAM_DECODER=ffmpeg` enables a separate RTSP/RTMP receive process.
  `opencv` remains the default for a controlled rollout. Restart the connection
  after changing the variable. A missing dependency fails explicitly; this mode
  never silently substitutes a color-incorrect HDR path.
- FFmpeg probes source color metadata. PQ/HLG uses linear-light float conversion,
  Hable tone mapping and BT.709 output. This needs system FFmpeg with `zscale` and
  `tonemap`, plus `ffprobe` and the application's existing `video` dependency.
  Official container images include the tools; do not assume PyAV's embedded
  FFmpeg has `zscale` (the tested PyAV 17.1 wheel does not).
- Source-time selection occurs before expensive color conversion. BGR frames
  travel over a local NUT pipe that preserves PTS. Decoder/filter threads are
  bounded; shutdown terminates the process to interrupt reads, and the capture
  thread alone closes its PyAV decoder. RTSP uses TCP and corrupt-packet discard;
  we do not arbitrarily delete compressed P/B-frame data.
- Source format, normalization, decoder errors and timestamp discontinuities
  appear in bounded receive diagnostics. The FFmpeg path's received-frame count
  is **after** its source-time selection. It is not total source decoder output.
- Browser playback discards presentation copies more than 0.5 seconds behind
  the configured delayed playhead. Frames inside the four-second buffer survive.
  Stale overlay requests/results are retired with those copies. Analysis inputs,
  evidence, identities, confirmation rules and SAM state remain independent.
  `display_catchup_expired` makes this loss explicit.

## Targeted verification

- 25 Python input/connection checks, including bursty 15 FPS, 30→15 sampling,
  missing PTS, PTS restart, reconnect and explicit unsupported-decoder behavior.
- 96 existing/new browser display checks, including four-second catch-up without
  early presentation, stale-overlay rejection and six-second buffer preservation.
- One short real-codec check in the existing Linux image: 60 HDR and 60 SDR
  1280×730 frames, valid pixels and 15 FPS source timestamps; approximately 3.16
  and 2.00 seconds including process setup/cleanup. This is not a live throughput
  claim or a visual color-quality measurement.
- Full pipeline: delegated to “测试双路视频分析性能”, using its existing rental
  authorization. Reuse its proper-SDR baseline and one five-minute run with
  4-second buffering, 2 FPS SAM, face and VLM. No fresh GPU is required for
  receive-only checks. Record exact deployed commits and settings.

## Live receive verification

The first RTSP check exposed an actual compatibility problem: this deployed
FFmpeg RTSP demuxer rejects `-rw_timeout`. RTSP now uses `-timeout 5000000`, while
other protocols retain `-rw_timeout`. Safe stderr categories distinguish missing
options/filters from a downstream NUT pipe error without logging camera URLs.

After that correction, a single ten-second live HDR dual-stream check completed:

| Metric | Lane 1 | Lane 2 |
|---|---:|---:|
| Delivered frames in measured window | 151 | 155 |
| Arrival FPS | 14.99 | 15.39 |
| PTS-derived FPS | 15.00 | 15.00 |
| Maximum arrival gap | 0.187 s | 0.160 s |
| Gaps over 200 ms | 0 | 0 |
| Adapter sampling drops | 0 | 0 |
| Decoder error chunks | 0 | 0 |
| Initial probe/connection to first frame | 6.64 s | 6.64 s |

The 15.39 arrival FPS includes slight initial catch-up, not a source-rate increase.
Both sources were BT.2020/PQ H.264 1280×730 and logged `normalized_to_sdr=true`.
This receiver check uses no model/GPU and does not establish end-to-end contour
quality or long-run stability. Initial connection/probing latency is a remaining
tradeoff. Raw summary: `network-input-live-hdr-20260920.json`.

Separately, the peer task verified commit `0748e76` in its actual main program:
proper-SDR input now has decoded FPS equal to emitted FPS at approximately 15,
with no active rate-limit drops (previously approximately 9.6 emitted FPS).

## Limits and next decision

The new FFmpeg receiver initially applies to RTSP/RTMP; WebRTC keeps its existing
decoder/converter and receives the source-clock sampling fix. Normalize HDR at
the source gateway for that route until a validated WebRTC color path is added.
Source color changes during a connection require a reconnect/reprobe. Native
resolution BGR pipes add local memory bandwidth; assess this before a many-camera
default rollout. Metadata is trusted to describe the source; unknown color
information cannot be treated as proven SDR color accuracy.

Do not claim that result publication or disk I/O is fixed by these changes. Use
the peer task's post-SDR lock/stage measurements to select the next small change.
Do not remove synchronous journal guarantees based only on CPU-starved timings.

References: [FFmpeg tone mapping](https://ffmpeg.org/ffmpeg-filters.html#tonemap-1),
[PyAV containers](https://pyav.basswood.io/docs/stable/api/container.html).


## Full-pipeline follow-up outcome

The peer task's second H100 reached readiness, but its optional CPU monitoring
launcher attempted `docker cp` into a read-only container and failed before the
five-minute analysis began. Cleanup released that instance. No new five-minute
result exists, and the failed startup is not counted as a performance test.
No third instance was created: no budget-compatible Taiwan H100 was available,
so paid recovery was stopped. Original containers/settings are being restored by
the peer task. Keep the successful input checks and first H100 baseline separate
from this failed attempt; no result-path speedup has been measured or claimed.

The display catch-up counter is included in the server's strict numeric telemetry
schema, so reporting remains valid after deploying the browser change. Eight
telemetry validation checks cover accepted counters and invalid values. Deploy
`display_telemetry.py` together with `camera_display.js`.
