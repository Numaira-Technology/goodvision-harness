"""Fixed-size model inputs with boxes restored to source coordinates."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, TypeVar

import cv2
import numpy as np

from .contracts import NormalizedBBox
from .privacy import PUBLIC_ANALYSIS_ERROR
from .sliding_window import FrameWindow

if TYPE_CHECKING:
    from .grounding import GroundingResponse

_LOGGER = logging.getLogger(__name__)
_Item = TypeVar("_Item")


@dataclass(frozen=True)
class LetterboxGeometry:
    width: int
    height: int
    content_width: int
    content_height: int
    left: int
    top: int

    def restore_box(self, box: NormalizedBBox) -> NormalizedBBox:
        def clip(value: float) -> float:
            return max(0.0, min(1.0, value))

        # A box wholly in padding becomes empty and fails validation instead
        # of creating a false box at the source image's edge.
        return NormalizedBBox(
            clip((box.x1 * self.width - self.left) / self.content_width),
            clip((box.y1 * self.height - self.top) / self.content_height),
            clip((box.x2 * self.width - self.left) / self.content_width),
            clip((box.y2 * self.height - self.top) / self.content_height),
        )

    def restore_or_none(self, box: NormalizedBBox) -> NormalizedBBox | None:
        try:
            return self.restore_box(box)
        except ValueError:
            return None


def _geometry(
    source_width: int, source_height: int, width: int, height: int,
) -> LetterboxGeometry:
    scale = min(width / source_width, height / source_height)
    content_width = max(1, min(width, round(source_width * scale)))
    content_height = max(1, min(height, round(source_height * scale)))
    return LetterboxGeometry(
        width,
        height,
        content_width,
        content_height,
        (width - content_width) // 2,
        (height - content_height) // 2,
    )


def prepare_window(
    window: FrameWindow,
    size: tuple[int, int],
    quality: int,
) -> tuple[FrameWindow, LetterboxGeometry]:
    """Fit every frame into ``size``; the geometry is the newest frame's.

    Model boxes always refer to the newest (last) frame of a window.
    """

    width, height = size
    if width < 1 or height < 1 or not 1 <= quality <= 100:
        raise ValueError("invalid letterbox dimensions or JPEG quality")
    frames = []
    geometry = None
    for frame in window.frames:
        if (frame.width, frame.height) == size:
            # Already the model canvas (the usual 16:9 robot input): send the
            # sampled JPEG unchanged instead of decoding it again.
            geometry = _geometry(width, height, width, height)
            frames.append(frame)
            continue
        image = cv2.imdecode(
            np.frombuffer(frame.jpeg_bytes, dtype=np.uint8),
            cv2.IMREAD_COLOR,
        )
        if image is None:
            raise ValueError("could not decode letterbox input")
        source_height, source_width = image.shape[:2]
        geometry = _geometry(source_width, source_height, width, height)
        if (source_width, source_height) == size:
            frames.append(replace(frame, width=width, height=height))
            continue
        content_width, content_height = geometry.content_width, geometry.content_height
        resized = cv2.resize(
            image,
            (content_width, content_height),
            interpolation=(
                cv2.INTER_AREA
                if content_width < source_width
                else cv2.INTER_LINEAR
            ),
        )
        canvas = cv2.copyMakeBorder(
            resized,
            geometry.top,
            height - content_height - geometry.top,
            geometry.left,
            width - content_width - geometry.left,
            cv2.BORDER_CONSTANT,
            value=(0, 0, 0),
        )
        ok, encoded = cv2.imencode(
            ".jpg",
            canvas,
            [cv2.IMWRITE_JPEG_QUALITY, quality],
        )
        if not ok:
            raise ValueError("could not encode letterbox input")
        frames.append(
            replace(
                frame,
                jpeg_bytes=encoded.tobytes(),
                width=width,
                height=height,
            )
        )
    if geometry is None:
        raise ValueError("letterbox input has no frames")
    return replace(window, frames=tuple(frames)), geometry


def _restore_items(
    items: Sequence[_Item], geometry: LetterboxGeometry,
) -> tuple[tuple[_Item, ...], tuple[int, ...]]:
    """Restored items outside the padding, and the original index of each."""

    kept: list[Any] = []
    indexes: list[int] = []
    for index, item in enumerate(items):
        box = geometry.restore_or_none(item.bbox)  # type: ignore[attr-defined]
        if box is not None:
            kept.append(replace(item, bbox=box))  # type: ignore[type-var]
            indexes.append(index)
    return tuple(kept), tuple(indexes)


def _restore_classification(classification: Any, geometry: LetterboxGeometry) -> Any:
    """Map boxes back to the source frame, dropping boxes seen only in padding.

    Padding carries no image content, so a box wholly inside it is not
    evidence. A positive left without any box cannot be published: in a
    multi-event result only that event becomes missing (neither positive nor
    negative) and the other judgments are kept; a single-event result is
    unusable and raises ``ValueError``.
    """

    from .grounding import MultiGroundedClassification

    common = {
        "actors": _restore_items(classification.actors, geometry)[0],
        "face_observations": _restore_items(
            classification.face_observations, geometry,
        )[0],
    }
    if isinstance(classification, MultiGroundedClassification):
        judgments = []
        missing = list(classification.missing_event_labels)
        for judgment in classification.judgments:
            objects = _restore_items(judgment.objects, geometry)[0]
            if judgment.detected and not objects:
                _LOGGER.warning(
                    "letterbox dropped event=%s: every box was in padding",
                    judgment.event_label,
                )
                if judgment.event_label not in missing:
                    missing.append(judgment.event_label)
                continue
            judgments.append(replace(judgment, objects=objects))
        return replace(
            classification,
            judgments=tuple(judgments),
            missing_event_labels=tuple(missing),
            supporting_objects=_restore_items(
                classification.supporting_objects, geometry,
            )[0],
            **common,
        )

    objects, kept = _restore_items(classification.objects, geometry)
    indexes = classification.event_object_indexes
    if indexes is not None:
        position = {old: new for new, old in enumerate(kept)}
        indexes = tuple(position[index] for index in indexes if index in position)
    if classification.detected and classification.objects and (
        not objects or (classification.event_object_indexes and not indexes)
    ):
        raise ValueError("every event box was in the letterbox padding")
    return replace(
        classification,
        objects=objects,
        event_object_indexes=indexes,
        **common,
    )


def classify_letterboxed(
    classify: Callable[[FrameWindow], GroundingResponse],
    window: FrameWindow,
    size: tuple[int, int],
    quality: int,
    observe: Callable[[FrameWindow], None] | None = None,
) -> GroundingResponse:
    # Imported here to keep the geometry utility independent at import time.
    from .grounding import GroundingResponse, _record_analysis_failure

    started = time.monotonic()

    def failed(stage: str) -> GroundingResponse:
        _record_analysis_failure(
            window, branch="safety", stage=stage, code="invalid_response",
        )
        return GroundingResponse(
            window=window,
            classification=None,
            latency_seconds=time.monotonic() - started,
            error=PUBLIC_ANALYSIS_ERROR,
            branch_errors=(("safety", "invalid_response"),),
        )

    try:
        model_window, geometry = prepare_window(window, size, quality)
    except (ValueError, cv2.error):
        # Nothing was sent: an input problem, not a bad model reply.
        return failed("input")
    if observe is not None:
        observe(model_window)
    response = classify(model_window)
    classification = response.classification
    if classification is not None:
        try:
            classification = _restore_classification(classification, geometry)
        except (TypeError, ValueError):
            # The model answered, but nothing in the reply can be published.
            return failed("classification")
    return replace(
        response,
        window=window,
        classification=classification,
        latency_seconds=time.monotonic() - started,
    )
