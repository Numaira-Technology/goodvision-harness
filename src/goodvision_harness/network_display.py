"""Finite, source-identified display batches from the shared network decoder.

The bounded download is independent of analysis and uses the same contour
metadata as browser capture. No estimated geometry enters an analysis branch.
"""
from __future__ import annotations

import json
import struct
import time

from fastapi import HTTPException
from fastapi.responses import Response

MEDIA_TYPE = "application/vnd.goodvision.display"
MAX_BYTES = 8 * 1024 * 1024


def display_response(runtime, *, after=-1, connection_id="", revision=-1,
                     stream_session_id=""):
    started = time.monotonic()
    connection = runtime._video_connection
    if connection is None or not connection.status()["connected"]:
        raise HTTPException(404, "video connection has no preview yet")
    identity = runtime._camera_display_metadata.source_identity()
    active = (runtime._state.get("pipeline_state") in {"starting", "running", "draining"}
              and identity["generation"] is not None
              and getattr(runtime._session, "mode", None) != "camera"
              and runtime.config.rich_labeling.backend == "sam31_stream"
              and runtime._scene_mode == "active")
    config = runtime.video_display_config()
    config.update(
        source_mode="network", transport="frames", enabled=active,
        delay_seconds=runtime._display_settings.delay_seconds if active else 0,
        **(identity if active else dict(stream_session_id=connection.connection_id,
                                      generation=0, crop_version=0)),
    )
    config["max_history_seconds"] = max(8.0, config["delay_seconds"] + 4.0)
    metadata = runtime._camera_display_metadata.delta(
        revision if stream_session_id == config["stream_session_id"] and revision >= 0 else None,
        stream_session_id=config["stream_session_id"],
    ) if active else None
    frames = connection.display_frames(after if connection_id == connection.connection_id else -1)
    descriptors, images, size = [], [], 0
    for sequence, timestamp, width, height, jpeg in frames:
        if size + len(jpeg) > MAX_BYTES - 1024 * 1024:
            break
        descriptors.append(dict(source_sequence=sequence, media_timestamp=timestamp,
                                width=width, height=height, bytes=len(jpeg)))
        images.append(jpeg)
        size += len(jpeg)
    header = json.dumps(dict(protocol=1, connection_id=connection.connection_id,
                             config=config, metadata=metadata, frames=descriptors),
                        ensure_ascii=True, allow_nan=False, separators=(",", ":")).encode()
    if len(header) > 1024 * 1024:
        raise HTTPException(503, "display metadata capacity exceeded")
    body = b"GVD1" + struct.pack(">I", len(header)) + header + b"".join(images)
    try:
        connection.record_display_delivery(frames=descriptors, byte_count=len(body),
                                           prepare_seconds=time.monotonic()-started,
                                           stream_session_id=config['stream_session_id'])
    except Exception:
        pass  # Optional diagnostics must never interrupt displaying video.
    return Response(body, media_type=MEDIA_TYPE, headers={"Cache-Control": "no-store"})
