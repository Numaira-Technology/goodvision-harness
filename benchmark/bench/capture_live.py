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
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError
from urllib.request import Request, urlopen


def _get(url: str, timeout: float) -> dict[str, Any]:
    with urlopen(url, timeout=timeout) as response:  # noqa: S310 - trusted localhost
        return json.loads(response.read().decode("utf-8"))


def _post(url, payload, timeout=300):
    request = Request(url, data=json.dumps(payload).encode("utf-8"),
                      headers={"Content-Type": "application/json"}, method="POST")
    deadline = time.monotonic() + timeout
    while True:
        try:
            with urlopen(request, timeout=timeout) as response:
                return json.load(response)
        except HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            exc.close()
            # A completed answer can precede the native session's DELETE.
            # This specific 409 rejects the start before creating a session.
            if (url.endswith("/v1/sessions/start") and exc.code == 409
                    and "stop the active session and wait for cleanup before starting" in body
                    and time.monotonic() < deadline):
                time.sleep(0.5)
                continue
            raise RuntimeError(f"Harness HTTP {exc.code}: {body}") from exc


def capture_offline(
    base_url: str, video_id: str, video_path: str,
    questions: list[dict[str, Any]], output_dir: Path, *,
    observation_max_tokens: int | None = None,
    answer_max_tokens: int | None = None,
    reanswer: bool = False,
    features: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Run one full, file-backed Harness session and retain its artifacts."""
    base_url = base_url.rstrip("/")
    output_dir.mkdir(parents=True, exist_ok=True)
    if reanswer:
        started = json.loads((output_dir / "session.json").read_text(encoding="utf-8"))
        if started["base_url"] != base_url or started["video_id"] != video_id:
            raise ValueError("re-answer must use the original Harness server and video")
        previous = _get(f"{base_url}/v1/benchmark/{started['stream_session_id']}", 30)
        expected = [{key: value for key, value in row.items() if key != "video_id"}
                    for row in questions]
        if previous["metadata"]["questions"] != expected:
            raise ValueError("re-answer requires the original questions")
        _post(f"{base_url}/v1/benchmark/{started['stream_session_id']}/answer", {})
    else:
        payload = {
            "mode": "file", "video_path": video_path,
            "benchmark": {"video_id": video_id, "questions": questions,
                          "observation_max_tokens": observation_max_tokens,
                          "answer_max_tokens": answer_max_tokens},
        }
        if features is not None:
            payload["features"] = list(features)
        started = _post(f"{base_url}/v1/sessions/start", payload)
    session_id = started["stream_session_id"]
    endpoint = f"{base_url}/v1/benchmark/{session_id}"
    (output_dir / "session.json").write_text(json.dumps(
        {"base_url": base_url, "stream_session_id": session_id, "video_id": video_id},
        indent=2,
    ) + "\n", encoding="utf-8")
    print(f"[offline] {video_id}: session {session_id}", flush=True)
    # Keep previous attempts, including repeated final answers for one session.
    attempt = output_dir / session_id / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    attempt.mkdir(parents=True)
    try:
        while True:
            result = _get(endpoint, 30)
            if result["status"] in {"complete", "failed"} and result["processing_finished"]:
                break
            time.sleep(1)
    except KeyboardInterrupt:
        if not reanswer:
            _post(f"{base_url}/v1/sessions/stop", {"stream_session_id": session_id}, 30)
        raise
    result["artifacts"] = str(attempt)
    try:
        with urlopen(f"{endpoint}/records", timeout=60) as response:
            with (attempt / "records.jsonl").open("wb") as output:
                while chunk := response.read(1024 * 1024):
                    output.write(chunk)
        result["records_downloaded"] = True
    except Exception as exc:
        # Preserve completed answers even when downloading the trace fails.
        result["records_downloaded"] = False
        result["artifact_error"] = f"{type(exc).__name__}: {exc}"
        print(f"[warn] {video_id}: {result['artifact_error']}", file=sys.stderr)
    rendered = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    (attempt / "result.json").write_text(rendered, encoding="utf-8")
    (output_dir / "result.json").write_text(rendered, encoding="utf-8")
    print(f"[offline] {video_id}: {result['status']}", flush=True)
    return result


def _bbox(box: Any) -> list[float]:
    if isinstance(box, dict):
        return [float(box[k]) for k in ("x1", "y1", "x2", "y2")]
    return [float(v) for v in box]


def capture(
    base_url: str,
    video_id: str,
    *,
    poll_hz: float,
    idle_timeout: float,
    request_timeout: float,
) -> dict[str, Any]:
    frames: dict[int, dict[str, Any]] = {}
    events: dict[str, dict[str, Any]] = {}
    fps = 0.0
    duration: float | None = None
    interval = 1.0 / max(poll_hz, 0.1)
    last_change = time.monotonic()
    started = False

    while True:
        try:
            status = _get(f"{base_url}/v1/status", request_timeout)
            payload = _get(f"{base_url}/v1/scene", request_timeout)
        except Exception as exc:  # noqa: BLE001
            print(f"[capture] poll failed: {exc}", file=sys.stderr)
            time.sleep(interval)
            continue

        state = status.get("pipeline_state")
        if state in {"running", "draining"}:
            started = True
        scene = payload.get("scene") or {}
        changed = False

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
    args = parser.parse_args(argv)

    result = capture(
        args.base_url.rstrip("/"),
        args.video_id,
        poll_hz=args.poll_hz,
        idle_timeout=args.idle_timeout,
        request_timeout=args.request_timeout,
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
