"""Frame and sliding-window primitives used by the streaming harness."""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field, replace


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
    # Local monotonic time the pipeline received this sample; diagnostics only.
    captured_monotonic: float | None = field(default=None, compare=False)

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


@dataclass(frozen=True, slots=True)
class BranchCadence:
    """Sampling rate and window shape for one independently scheduled branch."""

    fps: float
    size_frames: int = 1
    stride_frames: int = 1

    def __post_init__(self) -> None:
        if not isinstance(self.fps, (int, float)) or isinstance(self.fps, bool):
            raise TypeError("branch fps must be numeric")
        if not self.fps > 0:
            raise ValueError("branch fps must be positive")
        if self.size_frames < 1:
            raise ValueError("branch size_frames must be at least 1")
        if self.stride_frames < 1:
            raise ValueError("branch stride_frames must be at least 1")


@dataclass(frozen=True, slots=True)
class BranchSampling:
    """Per-branch cadences for a mixed safety + local-face session.

    The session samples the source once at ``base_fps`` (the faster branch) and
    each branch selects its own frames and builds its own windows, so selecting
    ``stranger`` no longer replaces the safety sampling rate or window size.
    """

    base_fps: float
    safety: BranchCadence
    stranger: BranchCadence

    def __post_init__(self) -> None:
        if not self.base_fps > 0:
            raise ValueError("base_fps must be positive")
        slowest_needed = max(self.safety.fps, self.stranger.fps)
        if self.base_fps + 1e-9 < slowest_needed:
            raise ValueError("base_fps must cover the fastest branch")

    def cadence(self, branch: str) -> BranchCadence:
        if branch == "safety":
            return self.safety
        if branch == "stranger":
            return self.stranger
        raise ValueError(f"unknown branch: {branch}")

    def as_dict(self) -> dict[str, object]:
        return {
            "base_fps": self.base_fps,
            "safety": {
                "fps": self.safety.fps,
                "size_frames": self.safety.size_frames,
                "stride_frames": self.safety.stride_frames,
            },
            "stranger": {
                "fps": self.stranger.fps,
                "size_frames": self.stranger.size_frames,
                "stride_frames": self.stranger.stride_frames,
            },
        }


class BranchWindowBuilder:
    """Select one branch's frames from a shared sample stream and window them.

    Frames are re-indexed per branch so every emitted window keeps contiguous
    sample indexes. A branch whose rate matches the shared stream keeps every
    frame, so its indexes equal the shared sample indexes.
    """

    def __init__(self, cadence: BranchCadence, *, base_fps: float) -> None:
        if not base_fps > 0:
            raise ValueError("base_fps must be positive")
        self.cadence = cadence
        self._select_all = cadence.fps + 1e-9 >= base_fps
        self._period = 1.0 / cadence.fps
        # Arrival jitter must not push a due frame into the next slot and halve
        # the branch rate; half a shared-sample period is the tolerance.
        self._tolerance = 0.5 / base_fps
        self._next_due: float | None = None
        self._buffer: deque[SampledFrame] = deque(maxlen=cadence.size_frames)
        self._local_index = 0
        self._window_index = 0
        self._frames_until_window = cadence.size_frames
        self.selected_frames = 0
        self.skipped_frames = 0

    def _due(self, timestamp: float) -> bool:
        if self._select_all:
            return True
        if self._next_due is None:
            self._next_due = timestamp
        if timestamp + self._tolerance < self._next_due:
            return False
        while self._next_due <= timestamp + self._tolerance:
            self._next_due += self._period
        return True

    def push(self, frame: SampledFrame) -> FrameWindow | None:
        if not self._due(frame.timestamp_seconds):
            self.skipped_frames += 1
            return None
        self.selected_frames += 1
        local = (
            frame
            if frame.sample_index == self._local_index
            else replace(frame, sample_index=self._local_index)
        )
        self._local_index += 1
        self._buffer.append(local)
        self._frames_until_window -= 1
        if self._frames_until_window > 0:
            return None
        self._frames_until_window = self.cadence.stride_frames
        window = FrameWindow(index=self._window_index, frames=tuple(self._buffer))
        self._window_index += 1
        return window
