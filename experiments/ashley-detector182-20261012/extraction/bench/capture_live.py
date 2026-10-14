"""GPU-side capture: poll a running harness and write a capture JSON.

Runs on the colleague's GPU box while a harness session processes one video.
It polls `GET /v1/scene` fast and accumulates, keyed by (frame_sequence,
subject_id), every distinct tracked-person observation plus the scene event
episodes. The result is a `<video_id>.json` capture consumed by run_bench.

Field mapping (harness `active` scene snapshot -> capture schema):
  scene.entities[].subject_id                      -> subject.id
  scene.entities[].bbox {x1,y1,x2,y2}              -> subject.bbox (normalized)
  scene.entities[].last_observed_timestamp_seconds -> frame.t
  scene.entities[].frame_sequence                  -> frame.frame
  scene.events[].event_label                       -> event.event_label
  scene.events[].first_evidence_timestamp_seconds  -> event.first_evidence
  scene.events[].confirmed_since_timestamp_seconds -> event.confirmed_since
  scene.events[].ended_since_timestamp_seconds     -> event.ended_since

FIDELITY NOTE: /v1/scene updates at the scene snapshot cadence (~0.5 s over SSE;
polling is similar). That is coarse for 30 fps counting/location. For frame-exact
counting use the in-process hook instead (see benchmark/README.md "High-fidelity
capture") — record every canonical scene snapshot from `_on_tracking_snapshot`
into the same schema. The event answers (Q1/Q4) are unaffected by cadence.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any
from urllib.request import urlopen


def _get(url: str, timeout: float) -> dict[str, Any]:
    with urlopen(url, timeout=timeout) as response:  # noqa: S310 - trusted localhost
        return json.loads(response.read().decode("utf-8"))


def _bbox(box: Any) -> list[float]:
    if isinstance(box, dict):
        return [float(box[k]) for k in ("x1", "y1", "x2", "y2")]
    return [float(v) for v in box]


def _opt(src: dict, *keys: str) -> float | None:
    """First present, non-null numeric value among candidate keys."""
    for k in keys:
        v = src.get(k)
        if v:
            return float(v)
    return None


def _first(src: dict, *keys: str) -> Any:
    for k in keys:
        v = src.get(k)
        if v is not None:
            return v
    return None


def _session_id(status: dict, scene: dict) -> Any:
    return _first(status, "stream_session_id", "session_id") or \
        _first(scene, "stream_session_id", "session_id")


def capture(
    base_url: str,
    video_id: str,
    *,
    poll_hz: float,
    idle_timeout: float,
    request_timeout: float,
    require_session_change_from: Any = "__unset__",
) -> dict[str, Any]:
    """Poll a harness session into a capture.

    Session isolation: when reusing one base_url across many videos, the harness
    may still be serving the PREVIOUS video's session (or its retained scene) when
    this starts. Pass `require_session_change_from` = the session id observed
    before this video was published; capture then waits for a DIFFERENT running
    session id, binds to it, and accepts only that session's frames/events —
    never writing stale tracks from another video under this video_id.
    """
    frames: dict[int, dict[str, Any]] = {}
    events: dict[str, dict[str, Any]] = {}
    fps = 0.0
    duration: float | None = None
    width: float | None = None
    height: float | None = None
    interval = 1.0 / max(poll_hz, 0.1)
    last_change = time.monotonic()
    started = False
    bound_session: Any = None
    gate = require_session_change_from != "__unset__"

    while True:
        try:
            status = _get(f"{base_url}/v1/status", request_timeout)
            payload = _get(f"{base_url}/v1/scene", request_timeout)
        except Exception as exc:  # noqa: BLE001
            print(f"[capture] poll failed: {exc}", file=sys.stderr)
            time.sleep(interval)
            continue

        state = status.get("pipeline_state")
        scene = payload.get("scene") or {}
        sid = _session_id(status, scene)

        # Session gating: don't accept anything until a fresh running session
        # (id different from the pre-publish baseline) appears; then bind to it.
        if gate and bound_session is None:
            if state in {"running", "draining"} and sid is not None \
                    and sid != require_session_change_from:
                bound_session = sid
                started = True
                print(f"[capture] bound to session {sid}", file=sys.stderr)
            else:
                if time.monotonic() - last_change > idle_timeout:
                    print("[capture] timed out waiting for a fresh session", file=sys.stderr)
                    break
                time.sleep(interval)
                continue
        elif gate and sid is not None and sid != bound_session:
            print(f"[capture] session changed {bound_session}->{sid}; ending", file=sys.stderr)
            break
        elif not gate and state in {"running", "draining"}:
            started = True

        changed = False

        # Source metadata, best-effort across a few known field names. width/height
        # are needed only to normalize the pixel reference box for relative-
        # displacement location questions; fps/duration are convenience fields.
        for src in (status, scene, payload):
            fps = fps or float(src.get("fps") or src.get("source_fps") or 0.0)
            width = width or _opt(src, "width", "source_width", "frame_width")
            height = height or _opt(src, "height", "source_height", "frame_height")
            d = _opt(src, "duration_seconds", "source_duration_seconds")
            duration = duration if duration is not None else d

        for entity in scene.get("entities", ()):  # per-frame tracked people
            seq = int(entity.get("frame_sequence", -1))
            t = entity.get("last_observed_timestamp_seconds")
            if seq < 0 or t is None:
                continue
            frame = frames.setdefault(seq, {"t": float(t), "frame": seq, "subjects": {}})
            sid = str(entity["subject_id"])
            if sid not in frame["subjects"]:
                frame["subjects"][sid] = {"id": sid, "bbox": _bbox(entity["bbox"])}
                changed = True

        # Record an explicitly EMPTY frame when a tracking snapshot was observed
        # with no people. Without this, count_max_duration would stitch positive
        # frames across the (unrecorded) empty gap, and count_exact_time(n=0) could
        # never find a true no-person moment. NOTE: this depends on the scene
        # reporting the current tracking frame identity; if the harness instead
        # retains stale entities, only the in-process _on_tracking_snapshot hook
        # (see README "High-fidelity capture") observes the true empty frames.
        cur_seq = _first(scene, "frame_sequence", "frame_index", "current_frame_sequence")
        cur_t = _opt(scene, "timestamp_seconds", "current_timestamp_seconds",
                     "last_observed_timestamp_seconds")
        if cur_seq is not None and cur_t is not None:
            seq = int(cur_seq)
            if seq not in frames:  # observed, but no entity populated it -> empty
                frames[seq] = {"t": float(cur_t), "frame": seq, "subjects": {}}
                changed = True

        for event in scene.get("events", ()):  # scene event episodes
            key = str(event.get("event_id") or f"{event['event_label']}:{event.get('frame_sequence')}")
            record = {
                "event_label": str(event["event_label"]),
                "subject_ids": list(event.get("subject_ids", ())),
                "first_evidence": event.get("first_evidence_timestamp_seconds"),
                "confirmed_since": event.get("confirmed_since_timestamp_seconds"),
                "ended_since": event.get("ended_since_timestamp_seconds"),
            }
            if events.get(key) != record:
                events[key] = record
                changed = True

        if changed:
            last_change = time.monotonic()

        # Stop once the run finished (idle again after having started), or after a
        # long quiet stretch with no new observations.
        if started and state in {"idle", "ended", "stopped", None}:
            break
        if time.monotonic() - last_change > idle_timeout:
            print("[capture] idle timeout reached", file=sys.stderr)
            break
        time.sleep(interval)

    ordered = [
        {"t": f["t"], "frame": f["frame"], "subjects": list(f["subjects"].values())}
        for f in sorted(frames.values(), key=lambda f: f["frame"])
    ]
    return {
        "video_id": video_id,
        "fps": fps,
        "duration_seconds": duration,
        "width": width,
        "height": height,
        "frames": ordered,
        "events": list(events.values()),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:7900")
    parser.add_argument("--video-id", required=True, help="joins to media.jsonl")
    parser.add_argument("--out", type=Path, required=True, help="<video_id>.json")
    parser.add_argument("--poll-hz", type=float, default=10.0)
    parser.add_argument("--idle-timeout", type=float, default=30.0)
    parser.add_argument("--request-timeout", type=float, default=5.0)
    parser.add_argument(
        "--require-session-change-from", default=None,
        help="baseline stream_session_id observed before this video was published; "
             "capture waits for a different running session and binds to it, so "
             "stale tracks from the previous video are never written here. Use "
             "'none' when no prior session existed.")
    parser.add_argument(
        "--print-session", action="store_true",
        help="print the current stream_session_id (or 'none') and exit; used to "
             "snapshot the baseline before publishing the next video.")
    args = parser.parse_args(argv)

    if args.print_session:
        try:
            st = _get(f"{args.base_url.rstrip('/')}/v1/status", args.request_timeout)
            sc = (_get(f"{args.base_url.rstrip('/')}/v1/scene", args.request_timeout)
                  .get("scene") or {})
            print(_session_id(st, sc) or "none")
        except Exception:  # noqa: BLE001
            print("none")
        return 0

    gate: Any = "__unset__"
    if args.require_session_change_from is not None:
        gate = None if args.require_session_change_from.lower() == "none" \
            else args.require_session_change_from

    result = capture(
        args.base_url.rstrip("/"),
        args.video_id,
        poll_hz=args.poll_hz,
        idle_timeout=args.idle_timeout,
        request_timeout=args.request_timeout,
        require_session_change_from=gate,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"[capture] {args.video_id}: {len(result['frames'])} frames, "
        f"{len(result['events'])} events -> {args.out}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
