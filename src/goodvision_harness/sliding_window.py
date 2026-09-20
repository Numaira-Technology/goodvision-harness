"""Frame and sliding-window primitives used by the streaming harness."""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable, Iterator
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SampledFrame:
    """One JPEG-encoded frame selected from a source video."""

    sample_index: int
    source_frame_index: int
    timestamp_seconds: float
    jpeg_bytes: bytes
    width: int
    height: int
    source_width: int | None = None
    source_height: int | None = None
    observed_at: str | None = None
    presentation_jpeg_bytes: bytes | None = None

    def __post_init__(self) -> None:
        if self.sample_index < 0 or self.source_frame_index < 0:
            raise ValueError("frame indexes must be non-negative")
        if self.timestamp_seconds < 0:
            raise ValueError("frame timestamps must be non-negative")
        if not self.jpeg_bytes:
            raise ValueError("jpeg_bytes must not be empty")
        if self.width < 1 or self.height < 1:
            raise ValueError("frame dimensions must be positive")
        if (self.source_width is None) != (self.source_height is None):
            raise ValueError("source_width and source_height must be set together")
        if self.source_width is not None and (
            self.source_width < 1
            or self.source_height is None
            or self.source_height < 1
        ):
            raise ValueError("source frame dimensions must be positive")
        if self.observed_at is not None and not self.observed_at.strip():
            raise ValueError("observed_at must not be empty")
        if (
            self.presentation_jpeg_bytes is not None
            and not self.presentation_jpeg_bytes
        ):
            raise ValueError("presentation_jpeg_bytes must not be empty")

    @property
    def original_dimensions(self) -> tuple[int, int]:
        return (
            self.source_width or self.width,
            self.source_height or self.height,
        )


@dataclass(frozen=True, slots=True)
class FrameWindow:
    """A chronologically ordered window of sampled frames."""

    index: int
    frames: tuple[SampledFrame, ...]
    is_partial: bool = False

    def __post_init__(self) -> None:
        if self.index < 0:
            raise ValueError("window index must be non-negative")
        if not self.frames:
            raise ValueError("a window must contain at least one frame")
        indexes = [frame.sample_index for frame in self.frames]
        if indexes != list(range(indexes[0], indexes[0] + len(indexes))):
            raise ValueError("window frames must have contiguous sample indexes")

    @property
    def start_index(self) -> int:
        return self.frames[0].sample_index

    @property
    def end_index(self) -> int:
        return self.frames[-1].sample_index

    @property
    def start_timestamp(self) -> float:
        return self.frames[0].timestamp_seconds

    @property
    def end_timestamp(self) -> float:
        return self.frames[-1].timestamp_seconds

    @property
    def newest_frame(self) -> SampledFrame:
        return self.frames[-1]

    @property
    def label(self) -> str:
        return f"W{self.index} · f{self.start_index}-f{self.end_index}"


def iter_sliding_windows(
    frames: Iterable[SampledFrame],
    *,
    size_frames: int,
    stride_frames: int,
    include_partial_final_window: bool = False,
) -> Iterator[FrameWindow]:
    """Yield bounded-memory sliding windows from sequential sampled frames.

    A size of four and stride of two produces f0-f3, f2-f5, f4-f7, and so on.
    Strides larger than the window are supported and intentionally leave gaps.
    """

    if size_frames < 1:
        raise ValueError("size_frames must be at least 1")
    if stride_frames < 1:
        raise ValueError("stride_frames must be at least 1")

    buffer: deque[SampledFrame] = deque()
    expected_sample_index = 0
    next_start = 0
    window_index = 0

    for frame in frames:
        if frame.sample_index != expected_sample_index:
            raise ValueError(
                "sample indexes must start at 0 and increase by 1; "
                f"expected {expected_sample_index}, got {frame.sample_index}"
            )
        expected_sample_index += 1

        if frame.sample_index < next_start:
            continue
        buffer.append(frame)

        required_end = next_start + size_frames - 1
        if frame.sample_index < required_end:
            continue

        window_frames = tuple(
            candidate
            for candidate in buffer
            if next_start <= candidate.sample_index <= required_end
        )
        if len(window_frames) != size_frames:
            raise RuntimeError("internal sliding-window buffer lost a required frame")

        yield FrameWindow(index=window_index, frames=window_frames)
        window_index += 1
        next_start += stride_frames
        while buffer and buffer[0].sample_index < next_start:
            buffer.popleft()

    if not include_partial_final_window:
        return

    while buffer and buffer[0].sample_index < next_start:
        buffer.popleft()
    partial_frames = tuple(buffer)
    if partial_frames and len(partial_frames) < size_frames:
        yield FrameWindow(
            index=window_index,
            frames=partial_frames,
            is_partial=True,
        )
