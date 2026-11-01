#!/usr/bin/env python3
"""Summarize harness `network_receive_timing` diagnostics into a table.

Usage:
    python parse_receive_log.py harness.log
    python parse_receive_log.py harness_webrtc.log harness_rtmp.log   # compare

Reads the ~5s diagnostic windows the harness logs (stream_diagnostics.emit ->
log_timing) and prints one row per window with the numbers that decide
"snow vs lag": decode fps, RTP packet loss / jitter, decode stalls, event-loop
lag, and handoff-queue drops. Flags rows that look starved. Stdlib only.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

TOKEN = "network_receive_timing "


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


def _rtp_totals(rec):
    """Sum packets_lost (and its delta) and take max jitter across RTP streams."""
    rtp = rec.get("inbound_rtp") or _find(rec, "inbound_rtp") or []
    if isinstance(rtp, dict):
        rtp = [rtp]
    lost = lost_d = 0
    jitter = 0.0
    for s in rtp if isinstance(rtp, list) else []:
        lost += int(s.get("packets_lost") or 0)
        lost_d += int(s.get("packets_lost_delta") or 0)
        jitter = max(jitter, float(s.get("jitter") or 0.0))
    return lost, lost_d, jitter


def _num(v, default=0.0):
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def parse_file(path):
    rows = []
    for line in Path(path).read_text(errors="replace").splitlines():
        i = line.find(TOKEN)
        if i == -1:
            continue
        brace = line.find("{", i)
        if brace == -1:
            continue
        try:
            rec = json.loads(line[brace:])
        except json.JSONDecodeError:
            continue
        totals = rec.get("totals", {}) or {}
        timings = rec.get("timings", {}) or {}
        lost, lost_d, jitter = _rtp_totals(rec)
        # The harness emits this timing under "stats_loop_lag"
        # (stream_diagnostics.py), not "loop_lag"; the old exact-match key meant
        # loop_p95_ms was always 0 and the LOOP-LAG flag never fired.
        loop_lag = (
            _find(timings, "stats_loop_lag")
            or _find(rec, "stats_loop_lag")
            or _find(timings, "loop_lag")
            or {}
        )
        queue_wait = timings.get("handoff_queue_wait", {}) or {}
        rows.append({
            "t": (rec.get("recorded_at") or "")[11:19],
            "proto": rec.get("protocol", "?"),
            "dec_fps": _num(rec.get("window_decoded_fps")),
            "emit_fps": _num(rec.get("window_emitted_fps")),
            "tgt": _num(rec.get("target_fps")),
            "lost": lost,
            "lost_d": lost_d,
            "jitter_ms": jitter * 1000.0,
            "gaps200": int(totals.get("decoded_gaps_over_200ms") or 0),
            "loop_p95_ms": _num(_find(loop_lag, "sampled_p95_seconds")) * 1000.0,
            "q_drop": int(totals.get("handoff_queue_dropped_frames") or 0),
            # Only handoff_queue_dropped_frames is ever emitted; keep a column so
            # downstream code stays stable, but read the real key.
            "drop": int(totals.get("handoff_queue_dropped_frames") or 0),
            "q_wait_ms": _num(queue_wait.get("mean_seconds")) * 1000.0,
        })
    return rows


def _flag(r):
    marks = []
    if r["lost_d"] > 0:
        marks.append("LOSS→snow")
    if r["loop_p95_ms"] >= 100:
        marks.append("LOOP-LAG")
    if r["gaps200"] > 0:
        marks.append("DECODE-STALL")
    if r["tgt"] and r["dec_fps"] < 0.7 * r["tgt"]:
        marks.append("DEC<TGT")
    if r["q_drop"] > 0:
        marks.append("Q-DROP→lag")
    return ",".join(marks)


def print_table(path, rows):
    print(f"\n=== {path}  ({len(rows)} windows) ===")
    hdr = f"{'time':8} {'proto':6} {'dec/emit/tgt fps':17} {'lostΔ':6} {'jit ms':7} {'gap>200':7} {'loop p95ms':10} {'qdrop':6} {'flags'}"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        fps = f"{r['dec_fps']:.1f}/{r['emit_fps']:.1f}/{r['tgt']:.0f}"
        print(f"{r['t']:8} {r['proto']:6} {fps:17} {r['lost_d']:<6} {r['jitter_ms']:<7.0f} "
              f"{r['gaps200']:<7} {r['loop_p95_ms']:<10.0f} {r['q_drop']:<6} {_flag(r)}")
    if rows:
        loss = sum(r["lost_d"] for r in rows)
        qd = sum(r["q_drop"] for r in rows)
        loop_max = max((r["loop_p95_ms"] for r in rows), default=0)
        print(f"SUMMARY: total packet loss Δ={loss}  queue-drops={qd}  max loop p95={loop_max:.0f}ms")
        print("  → packet loss present  => SNOW is transport-side (WebRTC dropping)"
              if loss else "  → no packet loss      => snow, if any, is not from RTP loss")
        print("  → high loop lag        => receive loop starved by CPU (annotation contention)"
              if loop_max >= 100 else "  → loop lag ok")
        print("  → queue drops present  => LAG is analysis too slow (frames dropped downstream)"
              if qd else "  → no queue drops")


def main(argv):
    if not argv:
        print(__doc__)
        return 2
    for path in argv:
        print_table(path, parse_file(path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
