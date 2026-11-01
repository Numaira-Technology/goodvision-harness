"""Receive public result/scene streams in a separate process during real replay.

No model calls originate here. The receiver preserves timestamps and message IDs
so server publication can be distinguished from actual external receipt.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path

import httpx


def observation_alerts(payload: dict) -> list[dict]:
    """Interpret VLM positives without inventing tracking confirmation."""
    if payload.get("type") != "vlm_report" or payload.get("state") != "completed":
        return []
    return [
        {
            "message_id": payload["message_id"],
            "stream_session_id": payload["stream_session_id"],
            "frame": payload["frame"],
            "event_label": judgment["event_label"],
            "observed_detected": True,
            "confirmation_state": judgment.get("confirmation_state", "pending"),
            "localization_state": judgment.get("localization_state"),
            "emitted_at": payload.get("emitted_at"),
        }
        for judgment in payload.get("judgments", [])
        if judgment.get("observed_detected") is True
    ]


async def receive(base_url: str, output: Path, duration: float) -> None:
    output.mkdir(parents=True, exist_ok=True)
    stop_at = time.monotonic() + duration

    def write(name, **fields):
        with (output / f"{name}.jsonl").open("a", encoding="utf-8") as file:
            file.write(json.dumps({"received_at": time.time(), **fields},
                                  ensure_ascii=False) + "\n")

    async with httpx.AsyncClient(
        base_url=base_url, trust_env=False,
        limits=httpx.Limits(max_connections=24), timeout=10,
    ) as client:
        async def sse(name, path):
            cursor = None
            while time.monotonic() < stop_at and not (output / "STOP").exists():
                try:
                    headers = {"Last-Event-ID": cursor} if cursor else {}
                    async with client.stream("GET", path, headers=headers,
                                             timeout=None) as response:
                        if response.status_code == 404:
                            await asyncio.sleep(.2)
                            continue
                        response.raise_for_status()
                        write("connections", channel=name, status="connected")
                        data, event, next_cursor = [], None, None
                        async for line in response.aiter_lines():
                            if line.startswith("data:"):
                                data.append(line[5:].strip())
                            elif line.startswith("id:"):
                                next_cursor = line[3:].strip()
                            elif line.startswith("event:"):
                                event = line[6:].strip()
                            elif not line and data:
                                payload = json.loads("\n".join(data))
                                write(name, event=event, cursor=next_cursor,
                                      payload=payload)
                                if name == "vlm":
                                    for alert in observation_alerts(payload):
                                        write("observation_alerts", payload=alert)
                                cursor = next_cursor or cursor
                                data, event, next_cursor = [], None, None
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    write("errors", channel=name, error=type(exc).__name__)
                    await asyncio.sleep(.5)

        async def poll(name, path):
            while time.monotonic() < stop_at:
                started = time.monotonic()
                try:
                    response = await client.get(path)
                    write(name, status=response.status_code,
                          request_seconds=time.monotonic() - started,
                          payload=response.json())
                except Exception as exc:
                    write("errors", channel=name, error=type(exc).__name__)
                await asyncio.sleep(.5)

        async def video(name, path):
            frames, count = 0, 0
            try:
                async with client.stream("GET", path, timeout=None) as response:
                    response.raise_for_status()
                    write("connections", channel=name, status="connected")
                    buffer = b""
                    async for chunk in response.aiter_bytes():
                        count += len(chunk)
                        buffer += chunk
                        while b"\xff\xd9" in buffer:
                            end = buffer.index(b"\xff\xd9") + 2
                            frame, buffer = buffer[:end], buffer[end:]
                            if b"\xff\xd8" in frame:
                                frames += 1
                                headers = frame.split(b"\xff\xd8", 1)[0]
                                write(name, frame_number=frames, bytes=count,
                                      headers=headers.decode("ascii", errors="replace"))
                        if len(buffer) > 8 * 1024 * 1024:
                            raise ValueError("unbounded MJPEG frame")
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                write("errors", channel=name, error=type(exc).__name__)
            finally:
                write("video_totals", channel=name, frames=frames, bytes=count)

        tasks = [
            asyncio.create_task(sse("vlm", "/v1/vlm/results/events")),
            asyncio.create_task(sse("results", "/v1/results/events")),
            asyncio.create_task(sse("events", "/v1/events/stream")),
            asyncio.create_task(poll("faces", "/v1/faces/fast")),
            asyncio.create_task(poll("status", "/v1/status?lightweight=true")),
            asyncio.create_task(poll("scene", "/v1/scene")),
            asyncio.create_task(poll("associations", "/v1/vlm/associations")),
            asyncio.create_task(video("live_video", "/v1/video/live/stream.mjpeg")),
            asyncio.create_task(video(
                "annotated_video", "/v1/video/annotated/stream.mjpeg",
            )),
        ]
        (output / "READY").write_text(str(time.time()))
        try:
            while time.monotonic() < stop_at and not (output / "STOP").exists():
                await asyncio.sleep(.2)
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:18088")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--duration", type=float, default=240)
    args = parser.parse_args()
    if args.duration <= 0:
        parser.error("--duration must be positive")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("--output must be a new or empty directory")
    asyncio.run(receive(args.url, args.output, args.duration))
