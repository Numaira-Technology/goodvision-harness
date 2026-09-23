"""Apply commodity filtering at the existing non-native detector client boundary."""

import io

from PIL import Image

from ...observations.normalization import from_snapshot
from ..components import SupervisionPipeline
from .supervision import supervision


class SupervisionLabelClient:
    """Serial rich-label client decorator; native SAM sessions bypass this class."""

    def __init__(self, client, *, source_id, settings):
        supervision()  # A requested but missing dependency fails at configuration.
        self._client = client
        self._source_id = source_id
        self._settings = settings
        self._scope = None
        self._pipeline = None

    def __getattr__(self, name):
        return getattr(self._client, name)

    def infer(
        self, jpeg_bytes, *, stream_session_id, frame_sequence, timestamp_seconds
    ):
        snapshot = self._client.infer(
            jpeg_bytes,
            stream_session_id=stream_session_id,
            frame_sequence=frame_sequence,
            timestamp_seconds=timestamp_seconds,
        )
        if (
            snapshot.stream_session_id,
            snapshot.frame_sequence,
            snapshot.timestamp_seconds,
        ) != (stream_session_id, frame_sequence, timestamp_seconds):
            raise ValueError("detector returned a different source frame")
        with Image.open(io.BytesIO(jpeg_bytes)) as image:
            dimensions = image.size
        scope = stream_session_id, dimensions
        if self._scope != scope:
            self._pipeline = SupervisionPipeline(
                source_id=self._source_id,
                stream_session_id=stream_session_id,
                settings=self._settings,
                resolution_wh=dimensions,
            )
            self._scope = scope
        observation = from_snapshot(
            snapshot, source_id=self._source_id, model=type(self._client).__name__
        )
        return self._pipeline.process(observation).observation.snapshot
