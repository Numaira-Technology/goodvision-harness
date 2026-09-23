"""Pinned Supervision ByteTrack baseline for offline comparison, not native SAM.

Supervision 0.30.5 deprecates this API in favor of Roboflow Trackers. Keeping the
baseline behind one adapter makes that migration independent of the event engine.
"""

from .adapters.supervision import supervision


class ByteTrackBaseline:
    def __init__(self, *, frame_rate: float = 30) -> None:
        self._tracker = supervision().ByteTrack(frame_rate=frame_rate)
        self._last_sequence = -1

    def update(self, detections, *, frame_sequence: int):
        if type(frame_sequence) is not int or frame_sequence <= self._last_sequence:
            raise ValueError("tracker requires distinct increasing frame sequences")
        if detections.tracker_id is not None and len(detections):
            raise ValueError("native tracked detections must not be retracked")
        result = self._tracker.update_with_detections(detections)
        self._last_sequence = frame_sequence
        return result

    def reset(self) -> None:
        self._tracker.reset()
        self._last_sequence = -1
