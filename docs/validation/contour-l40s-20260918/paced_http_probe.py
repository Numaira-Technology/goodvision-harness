"""Bounded real-time JPEG replay through the actual Contour HTTP gateway.

Measures captured-sample to returned-contour delay, not RTSP/display/event delay.
Each lane has one active request and four waiting frames; overflow drops the new
sample explicitly. Every lane sees the same clip. Encoding is inside the clock.
"""

import argparse
import base64
import http.client
import io
import json
import math
import queue
import threading
import time
import uuid
from pathlib import Path

from PIL import Image


def quantiles(values):
    values = sorted(values)
    return (
        {
            "count": len(values),
            "p50": values[math.ceil(0.5 * len(values)) - 1],
            "p95": values[math.ceil(0.95 * len(values)) - 1],
            "max": max(values),
        }
        if values
        else {}
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fps", type=int, choices=(1, 3), required=True)
    parser.add_argument("--seconds", type=int, default=48)
    parser.add_argument("--lanes", type=int, default=6)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    key = Path("/probe/secrets/node-key").read_text().strip()
    report = {
        "ok": False,
        "fps": args.fps,
        "seconds": args.seconds,
        "lanes": args.lanes,
        "pending_capacity_per_lane": 4,
        "drop_policy": "drop_new_sample_when_queue_full",
        "transport_max_side": 512,
        "jpeg_quality": 55,
        "encoding_included": True,
        "rtsp_and_event_models_included": False,
        "all_lanes_same_clip": True,
        "results": [],
    }
    handle = args.output.open("x")
    channels = [queue.Queue(maxsize=4) for _ in range(args.lanes)]
    start = [None]
    ready = threading.Barrier(
        args.lanes + 1,
        action=lambda: start.__setitem__(0, time.monotonic() + 1),
    )
    errors = []

    def worker(lane):
        conn = http.client.HTTPConnection("gateway", 8093, timeout=90)
        stream = uuid.uuid4().hex
        session_path = None
        row = {"lane": lane + 1, "completed": [], "dropped_sequences": []}
        report["results"].append(row)

        def request(method, path, body=None):
            data = json.dumps(body).encode() if body is not None else None
            conn.request(
                method,
                path,
                data,
                {"Authorization": "Bearer " + key, "Content-Type": "application/json"},
            )
            response = conn.getresponse()
            value = json.loads(response.read())
            if not 200 <= response.status < 300:
                raise RuntimeError(f"HTTP {response.status}: {value}")
            return value

        try:
            prefix = f"/v1/cameras/camera-{lane + 1:02d}/contour/sessions"
            opened = request("POST", prefix, {"stream_session_id": stream})
            session_path = prefix + "/" + opened["video_session_id"]
            ready.wait(timeout=120)
            while True:
                item = channels[lane].get()
                if item is None:
                    break
                sequence, captured, jpeg = item
                begin = time.monotonic()
                result = request(
                    "POST",
                    session_path + "/append",
                    {
                        "stream_session_id": stream,
                        "request_id": uuid.uuid4().hex,
                        "is_last_batch": False,
                        "frames": [
                            {
                                "frame_sequence": sequence,
                                "timestamp_seconds": sequence / args.fps,
                                "jpeg_base64": base64.b64encode(jpeg).decode(),
                            }
                        ],
                    },
                )
                end = time.monotonic()
                frames = result["frames"]
                if len(frames) != 1 or frames[0]["frame_sequence"] != sequence:
                    raise RuntimeError("HTTP result changed source frame")
                row["completed"].append(
                    {
                        "sequence": sequence,
                        "request_seconds": end - begin,
                        "source_to_result_seconds": end - captured,
                        "client_wait_seconds": begin - captured,
                        "during_capture": end < start[0] + args.seconds,
                        "objects": len(frames[0]["objects"]),
                    }
                )
            request(
                "POST",
                session_path + "/append",
                {
                    "stream_session_id": stream,
                    "request_id": uuid.uuid4().hex,
                    "is_last_batch": True,
                    "frames": [],
                },
            )
        except Exception as error:
            errors.append(
                {"lane": lane + 1, "type": type(error).__name__, "message": str(error)}
            )
            ready.abort()
        finally:
            if session_path:
                try:
                    request("DELETE", session_path)
                except Exception:
                    pass
            conn.close()

    threads = [
        threading.Thread(target=worker, args=(lane,), daemon=True)
        for lane in range(args.lanes)
    ]
    try:
        for thread in threads:
            thread.start()
        ready.wait(timeout=120)
        for seq in range(args.fps * args.seconds):
            deadline = start[0] + seq / args.fps
            time.sleep(max(0, deadline - time.monotonic()))
            if errors:
                raise RuntimeError("lane failed")
            for lane in range(args.lanes):
                path = Path("/frames") / f"{1 + seq * (3 // args.fps):06d}.jpg"
                captured = deadline
                with Image.open(path) as source:
                    source.thumbnail((512, 512))
                    buffer = io.BytesIO()
                    source.save(buffer, format="JPEG", quality=55)
                try:
                    channels[lane].put_nowait((seq, captured, buffer.getvalue()))
                except queue.Full:
                    next(r for r in report["results"] if r["lane"] == lane + 1)[
                        "dropped_sequences"
                    ].append(seq)
        for channel in channels:
            channel.put(None, timeout=90)
        for thread in threads:
            thread.join(timeout=120)
        if any(t.is_alive() for t in threads) or errors:
            raise RuntimeError("lane failed or did not drain")
        elapsed = time.monotonic() - start[0]
        report.update(
            ok=True,
            elapsed_seconds=elapsed,
            eof_drain_seconds=max(0, elapsed - args.seconds),
        )
        for row in report["results"]:
            row["source_to_result"] = quantiles(
                [r["source_to_result_seconds"] for r in row["completed"]]
            )
            row["during_capture_fps"] = (
                sum(r["during_capture"] for r in row["completed"]) / args.seconds
            )
            row["captured_frames"] = args.seconds * args.fps
            if (
                len(row["completed"]) + len(row["dropped_sequences"])
                != row["captured_frames"]
            ):
                raise RuntimeError("frame accounting mismatch")
    except Exception as error:
        report["ok"] = False
        report["error"] = {"type": type(error).__name__, "message": str(error)}
    finally:
        report["lane_errors"] = errors
        json.dump(report, handle, indent=2)
        handle.close()
    print(
        json.dumps(
            {
                "ok": report["ok"],
                "output": str(args.output),
                "error": report.get("error"),
            }
        )
    )
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
