"""Latest raw face match, independent of ordered identity and scene delivery."""

from __future__ import annotations

import threading
import time
from copy import deepcopy
from datetime import datetime, timezone

from .privacy import PUBLIC_FACE_ERROR


class FaceFastResults:
    """One bounded snapshot per run. Never take a service, scene or store lock."""

    def __init__(self, *, stream_session_id=None, source_id=None, generation=0,
                 display_fusion=None):
        self._lock = threading.Lock()
        self._display_fusion = display_fusion
        self._context = dict(
            stream_session_id=stream_session_id, source_id=source_id,
            generation=generation,
        )
        self._active = stream_session_id is not None
        self._latest = None
        self._published_at = None
        self._source_sequence = -1
        self._revision = 0
        self._gallery_version = None

    def stop(self):
        with self._lock:
            self._active = False

    def require_gallery(self, version):
        """Clear obsolete matches and reject work admitted with an old gallery."""
        with self._lock:
            self._gallery_version = version
            self._latest = None
            self._published_at = None
            self._revision += 1
            if self._display_fusion is not None:
                self._display_fusion.require_gallery(version)

    def publish(self, observation):
        frame = observation.window.newest_frame
        width, height = frame.original_dimensions
        # Explicit public allowlist: no JPEGs, embeddings, model internals or
        # provisional body/track identities from the slower fusion pipeline.
        attributes = (
            "status", "person_id", "name", "similarity", "threshold",
            "match_margin", "detection_score", "reason",
        )
        payload = {
            "frame": {
                "source_frame_index": frame.source_frame_index,
                "timestamp_seconds": frame.timestamp_seconds,
                "width": width, "height": height,
            },
            "emitted_at": datetime.now(timezone.utc).isoformat(),
            "gallery_version": observation.gallery.version,
            "error": PUBLIC_FACE_ERROR if observation.error else None,
            "faces": [
                {
                    "bbox": dict(x1=item.bbox.x1, y1=item.bbox.y1,
                                 x2=item.bbox.x2, y2=item.bbox.y2),
                    **{key: item.attributes[key] for key in attributes
                       if key in item.attributes},
                }
                for item in observation.objects
            ] if not observation.error else [],
        }
        with self._lock:
            if (not self._active
                    or (self._gallery_version is not None
                        and observation.gallery.version != self._gallery_version)):
                return
            if frame.source_frame_index > self._source_sequence:
                self._latest = payload
                self._source_sequence = frame.source_frame_index
                self._published_at = time.monotonic()
                self._revision += 1
        # Publication has completed. The display handoff only writes an
        # image-free mailbox; it never acquires the SAM/scene/service lock.
        if self._display_fusion is not None:
            self._display_fusion.offer_face(payload)

    def snapshot(self):
        with self._lock:
            latest = self._latest
            published_at = self._published_at
            context = {
                **self._context, "active": self._active,
                "revision": self._revision,
                "available": latest is not None,
                "analysis_kind": "face_raw",
            }
        # Published dictionaries are immutable; copying/HTTP serialization does
        # not keep inference waiting on a reader holding the cache lock.
        return {
            **context,
            "age_seconds": (time.monotonic() - published_at
                            if published_at is not None else None),
            "result": deepcopy(latest),
        }
