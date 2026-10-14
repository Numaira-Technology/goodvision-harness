"""Opt-in source-time file decoding for native research evaluations.

The serving pipeline does not import or select this reader. It keeps the
original frame identity and reports natural EOF separately from cancellation.
"""

from __future__ import annotations

import math
from pathlib import Path


def iter_native_frames(
    path, *, sample_fps=2.0, max_image_side=1280, jpeg_quality=90, record
):
    """Yield sampled VideoFrames; decode every source frame through natural EOF."""
    import av

    from .video_to_stream import encode_jpeg, probe_video
    from .video_tracking import VideoFrame

    if not math.isfinite(sample_fps) or not 0 < sample_fps <= 30:
        raise ValueError("sample_fps must be within (0,30]")
    count, emitted, first_pts, last_time, next_sample = 0, 0, None, None, 0.0
    path = Path(path)
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.codec_context.thread_type = "SLICE"
        avi = "avi" in container.format.name.split(",")
        info = probe_video(path) if avi else None
        fps = info.source_fps if info else float(stream.average_rate or 0)
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError("source frame rate is unavailable")
        basis = (
            "avi_original_frame_index_over_file_fps"
            if avi
            else "decoded_pts_or_decoder_dts_relative_to_first_frame"
        )
        record(
            "source_clock",
            timestamp_basis=basis,
            file_fps=fps,
            width=stream.width,
            height=stream.height,
            pyav_version=av.__version__,
            container_format=container.format.name,
            wall_clock_pacing=False,
        )
        for index, frame in enumerate(container.decode(video=0)):
            if avi:
                timestamp = index / fps
            else:
                tick = frame.pts if frame.pts is not None else frame.dts
                if tick is None or frame.time_base is None:
                    raise ValueError("source frame has no presentation timestamp")
                pts = float(tick * frame.time_base)
                if first_pts is None:
                    first_pts = pts
                timestamp = pts - first_pts
                if frame.pts is None:
                    record(
                        "source_timestamp_fallback",
                        frame=index,
                        basis="decoder_dts",
                        dts=frame.dts,
                    )
            if (
                not math.isfinite(timestamp)
                or timestamp < 0
                or (last_time is not None and timestamp < last_time)
            ):
                raise ValueError("source presentation timestamps decreased")
            if timestamp == last_time:
                record(
                    "source_timestamp_repeat", frame=index, timestamp_seconds=timestamp
                )
            count += 1
            if timestamp + 1e-9 >= next_sample and timestamp != last_time:
                jpeg, _, _ = encode_jpeg(
                    frame.to_ndarray(format="bgr24"),
                    max_image_side=max_image_side,
                    quality=jpeg_quality,
                )
                emitted += 1
                yield VideoFrame(index, timestamp, jpeg)
                next_sample = timestamp + 1.0 / sample_fps
            last_time = timestamp
        if not count or (info is not None and count != info.frame_count):
            raise ValueError("source frame count is empty or differs from AVI metadata")
        record(
            "source_eof",
            decoded_frames=count,
            sampled_frames=emitted,
            last_timestamp_seconds=last_time,
            file_fps=fps,
            timestamp_basis=basis,
            declared_frame_count=info.frame_count if info else None,
            wall_clock_pacing=False,
            dropped_due_to_lag=0,
        )
