#!/usr/bin/env python3
"""Roll a harness run's structured log into a per-stream, per-stage pressure table.

Answers: *when you add one more video stream, where does the marginal cost land —
CPU, GPU, or bandwidth, and at which pipeline stage?*

Each harness process serves one stream (multi-stream = sibling processes), so one
log file = one stream. Pass several logs (one per stream, or one run per stream
count) and this prints, per stream, the CPU/mem/thread resource curve, per-stage
latency p50/p95, and the three bandwidth legs — then a side-by-side summary that
flags which columns grow with stream count (linear cost) vs. saturate (ceiling).

It reads events emitted through `latency_trace.log_timing` (lines of the form
`<event> <json>`):
  network_receive_timing          - receive/decode window: fps, loss, loop-lag,
                                     qdrop, ingest bytes (stage 1)
  resource_timing                 - this process: cpu%, rss, threads, per-stage
                                     thread CPU (the CPU attribution)
  sam_client_http_timing          - per remote-SAM request: latency + upload bytes
                                     (stage 4 round-trip, stage 3 transport size)
  sam_coordinator_timing          - per request: append/commit/source_to_result
                                     (stages 4 & 6)
  sam_display_timing              - per published frame: render decode/draw/encode,
                                     publish (stage 7)
  network_display_delivery_timing - output delivery window: bytes (stage 7 egress)

Optionally merge a remote-GPU CSV from diag/gpu_sampler.sh (--gpu-csv) for the
GPU leg. Stdlib only.

Usage:
    python parse_pipeline_log.py stream1.log
    python parse_pipeline_log.py s1.log s2.log s4.log        # compare 1/2/4 streams
    python parse_pipeline_log.py run.log --gpu-csv gpu.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from statistics import median

# event name -> handler key
RECEIVE = "network_receive_timing"
RESOURCE = "resource_timing"
SAM_HTTP = "sam_client_http_timing"
SAM_COORD = "sam_coordinator_timing"
DISPLAY = "sam_display_timing"
DELIVERY = "network_display_delivery_timing"

_EVENTS = (RECEIVE, RESOURCE, SAM_HTTP, SAM_COORD, DISPLAY, DELIVERY)


def _num(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _percentile(values, pct):
    """Nearest-rank percentile on a copy; 0.0 for an empty series."""
    data = sorted(v for v in values if v is not None)
    if not data:
        return 0.0
    if pct <= 0:
        return data[0]
    if pct >= 100:
        return data[-1]
    rank = max(0, min(len(data) - 1, round(pct / 100.0 * len(data) + 0.5) - 1))
    return data[rank]


def _find(obj, key):
    """Depth-first search for the first value under `key` anywhere in obj."""
    if isinstance(obj, dict):
        if key in obj:
            return obj[key]
        for v in obj.values():
            found = _find(v, key)
            if found is not None:
                return found
    elif isinstance(obj, list):
        for v in obj:
            found = _find(v, key)
            if found is not None:
                return found
    return None


def _iter_events(path):
    """Yield (event_name, record) for every recognised log line."""
    for line in Path(path).read_text(errors="replace").splitlines():
        for event in _EVENTS:
            token = event + " "
            i = line.find(token)
            if i == -1:
                continue
            brace = line.find("{", i)
            if brace == -1:
                break
            try:
                yield event, json.loads(line[brace:])
            except json.JSONDecodeError:
                pass
            break


class StreamSummary:
    """Aggregates one log file (= one stream)."""

    def __init__(self, path):
        self.path = path
        self.camera_index = None
        # resource
        self.cpu_percent = []
        self.rss_bytes = []
        self.num_threads = []
        self.system_cpu_max = []
        self.stage_cpu_seconds = {}  # stage -> [seconds per interval]
        self.resource_interval = None
        # latency series (seconds)
        self.lat = {
            "receive_decode_fps": [],
            "loop_lag_p95": [],
            "sam_http_total": [],
            "sam_append": [],
            "sam_commit": [],
            "source_to_result": [],
            "render": [],
            "render_decode": [],
            "render_draw": [],
            "render_encode": [],
            "publish": [],
        }
        # WebRTC transport bytes_received and the `totals` counters are CUMULATIVE
        # per connection. Convert them to per-window deltas (prev starts at 0; a
        # value below prev means a reconnect/reset, so the new value is itself the
        # delta) rather than summing the running totals, which would grow with
        # sampling duration and double-count. packets_lost_delta and the delivery
        # `bytes` field are already per-window, so those are summed directly.
        self._conn_prev = {}  # connection_id -> {"bytes","qdrop","gaps"}
        self.ingest_bytes_delta = 0.0
        self.ingest_window_seconds = 0.0
        self.upload_bytes_total = 0.0  # per-request transport_bytes (not cumulative)
        self.upload_requests = 0
        self.output_bytes_window = []  # per delivery window (window bytes)
        # health (accumulated per-window deltas)
        self.lost_delta = 0
        self.qdrop = 0
        self.gaps200 = 0
        self.first_t = None
        self.last_t = None

    # ------------------------------------------------------------------ ingest
    def add(self, event, rec):
        t = (rec.get("recorded_at") or "")
        if t:
            self.first_t = self.first_t or t
            self.last_t = t
        handler = {
            RECEIVE: self._receive,
            RESOURCE: self._resource,
            SAM_HTTP: self._sam_http,
            SAM_COORD: self._sam_coord,
            DISPLAY: self._display,
            DELIVERY: self._delivery,
        }[event]
        handler(rec)

    def _receive(self, rec):
        self.lat["receive_decode_fps"].append(_num(rec.get("window_decoded_fps")))
        loop = (
            _find(rec, "stats_loop_lag")
            or _find(rec, "loop_lag")
            or {}
        )
        self.lat["loop_lag_p95"].append(_num(_find(loop, "sampled_p95_seconds")))
        totals = rec.get("totals") or {}
        rtp = rec.get("inbound_rtp") or []
        if isinstance(rtp, dict):
            rtp = [rtp]
        # packets_lost_delta is already a per-window delta from the harness.
        for s in rtp:
            self.lost_delta += int(s.get("packets_lost_delta") or 0)
        transport = rec.get("transport") or []
        if isinstance(transport, dict):
            transport = [transport]
        conn = rec.get("connection_id") or "default"
        cur_bytes = sum(_num(s.get("bytes_received")) for s in transport)
        cur_qdrop = int(totals.get("handoff_queue_dropped_frames") or 0)
        cur_gaps = int(totals.get("decoded_gaps_over_200ms") or 0)
        prev = self._conn_prev.get(conn, {"bytes": 0.0, "qdrop": 0, "gaps": 0})
        self._conn_prev[conn] = {
            "bytes": cur_bytes, "qdrop": cur_qdrop, "gaps": cur_gaps,
        }
        win = _num(rec.get("window_seconds")) or _num(rec.get("elapsed_seconds"))
        d_bytes = cur_bytes - prev["bytes"] if cur_bytes >= prev["bytes"] else cur_bytes
        if win > 0 and d_bytes > 0:
            self.ingest_bytes_delta += d_bytes
            self.ingest_window_seconds += win
        self.qdrop += (
            cur_qdrop - prev["qdrop"] if cur_qdrop >= prev["qdrop"] else cur_qdrop
        )
        self.gaps200 += (
            cur_gaps - prev["gaps"] if cur_gaps >= prev["gaps"] else cur_gaps
        )

    def _resource(self, rec):
        if self.camera_index is None:
            self.camera_index = rec.get("camera_index")
        self.resource_interval = (
            _num(rec.get("interval_seconds")) or self.resource_interval
        )
        if "cpu_percent" in rec:
            self.cpu_percent.append(_num(rec.get("cpu_percent")))
        if "rss_bytes" in rec:
            self.rss_bytes.append(_num(rec.get("rss_bytes")))
        if "num_threads" in rec:
            self.num_threads.append(_num(rec.get("num_threads")))
        if "system_cpu_percent_max" in rec:
            self.system_cpu_max.append(_num(rec.get("system_cpu_percent_max")))
        for stage, secs in (rec.get("thread_cpu_seconds") or {}).items():
            self.stage_cpu_seconds.setdefault(stage, []).append(_num(secs))

    def _sam_http(self, rec):
        self.lat["sam_http_total"].append(_num(rec.get("total_seconds")))
        tb = rec.get("transport_bytes")
        if tb is None:
            tb = _find(rec, "transport_bytes")
        if tb is not None:
            self.upload_bytes_total += _num(tb)
            self.upload_requests += 1

    def _sam_coord(self, rec):
        self.lat["sam_append"].append(_num(rec.get("append_seconds")))
        self.lat["sam_commit"].append(_num(rec.get("commit_seconds")))
        s2r = rec.get("source_to_result")
        if isinstance(s2r, list):
            for v in s2r:
                self.lat["source_to_result"].append(_num(v))
        elif s2r is not None:
            self.lat["source_to_result"].append(_num(s2r))

    def _display(self, rec):
        self.lat["render"].append(_num(rec.get("render_seconds")))
        self.lat["render_decode"].append(_num(rec.get("render_decode_seconds")))
        self.lat["render_draw"].append(_num(rec.get("render_draw_seconds")))
        self.lat["render_encode"].append(_num(rec.get("render_encode_seconds")))
        self.lat["publish"].append(_num(rec.get("publish_seconds")))

    def _delivery(self, rec):
        win = _num(rec.get("window_seconds"))
        by = _num(rec.get("bytes"))
        if win > 0:
            self.output_bytes_window.append((win, by))

    # ---------------------------------------------------------------- derived
    def _bw_ingest(self):
        return (
            self.ingest_bytes_delta / self.ingest_window_seconds
            if self.ingest_window_seconds > 0 else 0.0
        )

    def _bw_output(self):
        tot_w = sum(w for w, _ in self.output_bytes_window)
        tot_b = sum(b for _, b in self.output_bytes_window)
        return tot_b / tot_w if tot_w > 0 else 0.0

    def _bw_upload(self):
        # transport_bytes is per-request cumulative-of-batch; approximate rate by
        # spreading the total over the run window.
        elapsed = self._elapsed_seconds()
        return self.upload_bytes_total / elapsed if elapsed > 0 else 0.0

    def _elapsed_seconds(self):
        # Rough wall span from recorded_at timestamps (ISO8601, same day assumed).
        if not (self.first_t and self.last_t):
            return 0.0
        try:
            from datetime import datetime

            a = datetime.fromisoformat(self.first_t)
            b = datetime.fromisoformat(self.last_t)
            return max(0.0, (b - a).total_seconds())
        except ValueError:
            return 0.0

    def metrics(self):
        """Flat dict of headline numbers for the comparison table."""
        interval = self.resource_interval or 5.0
        stage_cpu_pct = {
            f"cpu_{stage}": (sum(v) / len(v) / interval * 100.0) if v else 0.0
            for stage, v in self.stage_cpu_seconds.items()
        }
        return {
            "cpu_proc_mean": (sum(self.cpu_percent) / len(self.cpu_percent))
            if self.cpu_percent else 0.0,
            "cpu_proc_max": max(self.cpu_percent, default=0.0),
            "sys_cpu_max": max(self.system_cpu_max, default=0.0),
            "rss_mb_max": max(self.rss_bytes, default=0.0) / 1e6,
            "threads_max": max(self.num_threads, default=0.0),
            "dec_fps": median(self.lat["receive_decode_fps"])
            if self.lat["receive_decode_fps"] else 0.0,
            "loop_p95_ms": _percentile(self.lat["loop_lag_p95"], 95) * 1000.0,
            "sam_http_p50_ms": _percentile(self.lat["sam_http_total"], 50) * 1000.0,
            "sam_http_p95_ms": _percentile(self.lat["sam_http_total"], 95) * 1000.0,
            "s2r_p50_ms": _percentile(self.lat["source_to_result"], 50) * 1000.0,
            "s2r_p95_ms": _percentile(self.lat["source_to_result"], 95) * 1000.0,
            "render_p95_ms": _percentile(self.lat["render"], 95) * 1000.0,
            "bw_ingest_mbps": self._bw_ingest() * 8 / 1e6,
            "bw_upload_mbps": self._bw_upload() * 8 / 1e6,
            "bw_output_mbps": self._bw_output() * 8 / 1e6,
            "lost_delta": self.lost_delta,
            "qdrop": self.qdrop,
            "gaps200": self.gaps200,
            **stage_cpu_pct,
        }


def _pctile_row(label, series_s):
    p50 = _percentile(series_s, 50) * 1000.0
    p95 = _percentile(series_s, 95) * 1000.0
    n = len([v for v in series_s if v])
    return f"  {label:22} p50={p50:8.1f}ms  p95={p95:8.1f}ms  (n={n})"


def print_stream(s):
    m = s.metrics()
    cam = f"camera {s.camera_index}" if s.camera_index is not None else "camera ?"
    print(f"\n=== {s.path}  ({cam}) ===")
    print("-- resources (CPU attribution) --")
    print(f"  process CPU:  mean={m['cpu_proc_mean']:6.1f}%  "
          f"max={m['cpu_proc_max']:6.1f}%   system CPU max={m['sys_cpu_max']:6.1f}%")
    print(f"  memory RSS max={m['rss_mb_max']:7.1f} MB   "
          f"threads max={m['threads_max']:.0f}")
    stage_cpu = {k[4:]: v for k, v in m.items() if k.startswith("cpu_") and k not in
                 ("cpu_proc_mean", "cpu_proc_max")}
    if stage_cpu:
        parts = "  ".join(f"{st}={pct:.0f}%" for st, pct in sorted(
            stage_cpu.items(), key=lambda kv: -kv[1]))
        print(f"  per-stage CPU (of 1 core): {parts}")
    print("-- per-stage latency --")
    print(_pctile_row("stage4 SAM http", s.lat["sam_http_total"]))
    print(_pctile_row("stage4 append", s.lat["sam_append"]))
    print(_pctile_row("stage6 commit", s.lat["sam_commit"]))
    print(_pctile_row("source->result", s.lat["source_to_result"]))
    print(_pctile_row("stage7 render", s.lat["render"]))
    print(_pctile_row("  render.decode", s.lat["render_decode"]))
    print(_pctile_row("  render.draw", s.lat["render_draw"]))
    print(_pctile_row("  render.encode", s.lat["render_encode"]))
    print(_pctile_row("stage7 publish", s.lat["publish"]))
    print("-- bandwidth --")
    print(f"  ingest(recv)={m['bw_ingest_mbps']:7.2f} Mbps   "
          f"upload(->SAM)={m['bw_upload_mbps']:7.2f} Mbps   "
          f"output={m['bw_output_mbps']:7.2f} Mbps")
    print("-- health --")
    print(f"  decode fps(med)={m['dec_fps']:.1f}  loop p95={m['loop_p95_ms']:.0f}ms  "
          f"lostΔ={m['lost_delta']}  qdrop={m['qdrop']}  gap>200ms={m['gaps200']}")


def print_comparison(streams, gpu=None):
    if len(streams) < 2 and gpu is None:
        return
    print("\n=== comparison (marginal cost per added stream) ===")
    cols = [
        ("cpu_proc_mean", "procCPU%", "%.0f"),
        ("sys_cpu_max", "sysCPUmax%", "%.0f"),
        ("rss_mb_max", "RSS MB", "%.0f"),
        ("bw_ingest_mbps", "ingest Mbps", "%.2f"),
        ("bw_upload_mbps", "upload Mbps", "%.2f"),
        ("bw_output_mbps", "output Mbps", "%.2f"),
        ("sam_http_p95_ms", "SAM p95ms", "%.0f"),
        ("s2r_p95_ms", "s2r p95ms", "%.0f"),
        ("loop_p95_ms", "loop p95ms", "%.0f"),
        ("lost_delta", "lostΔ", "%.0f"),
    ]
    header = f"{'stream':22} " + " ".join(f"{c[1]:>11}" for c in cols)
    print(header)
    print("-" * len(header))
    rows = []
    for s in streams:
        m = s.metrics()
        rows.append(m)
        name = Path(s.path).name[:22]
        cells = " ".join(f"{(fmt % m[key]):>11}" for key, _, fmt in cols)
        print(f"{name:22} {cells}")
    if gpu:
        print(f"{'remote GPU':22} util mean={gpu['util_mean']:.0f}% "
              f"max={gpu['util_max']:.0f}%  mem max={gpu['mem_max_mb']:.0f}MB")
    # crude growth hint: ratio last/first for each column
    if len(rows) >= 2:
        print("\ngrowth (last/first):")
        for key, label, _ in cols:
            first = rows[0][key] or 0.0
            last = rows[-1][key] or 0.0
            ratio = (last / first) if first else float("inf") if last else 0.0
            tag = ""
            if first and last:
                if ratio >= 1.6 * (len(rows) - 1) or ratio >= 1.6:
                    tag = "  <= grows with streams"
            print(f"  {label:14} {first:10.2f} -> {last:10.2f}  (x{ratio:.2f}){tag}")


def parse_gpu_csv(path):
    """nvidia-smi --query-gpu CSV (util.gpu, util.memory, memory.used ...)."""
    util, mem = [], []
    for raw in Path(path).read_text(errors="replace").splitlines():
        line = raw.strip()
        low = line.lower()
        if not line or low.startswith("timestamp") or "utilization" in low:
            continue
        parts = [p.strip() for p in line.split(",")]
        # expected: timestamp, util.gpu, util.memory, memory.used, memory.total, power
        nums = []
        for p in parts:
            tok = p.split()[0] if p else ""
            try:
                nums.append(float(tok))
            except ValueError:
                nums.append(None)
        # find first two numeric as util.gpu, and a larger one as mem.used
        numeric = [n for n in nums if n is not None]
        if not numeric:
            continue
        util.append(numeric[0])
        if len(numeric) >= 3:
            mem.append(numeric[2])
    if not util:
        return None
    return {
        "util_mean": sum(util) / len(util),
        "util_max": max(util),
        "mem_max_mb": max(mem) if mem else 0.0,
    }


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("logs", nargs="+", help="harness log file(s); one per stream")
    ap.add_argument("--gpu-csv", help="nvidia-smi CSV from diag/gpu_sampler.sh")
    args = ap.parse_args(argv)

    streams = []
    for path in args.logs:
        s = StreamSummary(path)
        for event, rec in _iter_events(path):
            s.add(event, rec)
        streams.append(s)
        print_stream(s)

    gpu = parse_gpu_csv(args.gpu_csv) if args.gpu_csv else None
    if gpu:
        print(f"\n=== remote GPU ({args.gpu_csv}) ===")
        print(f"  util mean={gpu['util_mean']:.1f}%  max={gpu['util_max']:.1f}%  "
              f"mem used max={gpu['mem_max_mb']:.0f} MB")
    print_comparison(streams, gpu)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
