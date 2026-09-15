"""Pluggable person route: SSDLite observations plus asynchronous SAM calibration.

Only this module owns its detector, tracker and calibration scheduler. SAM's
verified weights are borrowed from the video engine, never copied or reloaded.
Background work observes exact retained source frames and cannot emit an old
box as a current observation. One calibration runs and one latest frame waits.
"""

from __future__ import annotations

import hashlib
import logging
import math
import os
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

from fastapi import HTTPException

if __package__:
    from .hybrid_tracker import (
        PersonDetection,
        PersonTracker,
        TrackerCapacityError,
        describe_person,
    )
    from .sam31 import SourceFrame, VideoBatch, native_snapshot
else:
    from hybrid_tracker import (
        PersonDetection,
        PersonTracker,
        TrackerCapacityError,
        describe_person,
    )
    from sam31 import SourceFrame, VideoBatch, native_snapshot

LOGGER = logging.getLogger("goodvision.person_hybrid")
BACKEND = "person_hybrid"
MODEL_NAME = "ssdlite320_mobilenet_v3_large_coco_v1"
CHECKPOINT_URL = (
    "https://download.pytorch.org/models/"
    "ssdlite320_mobilenet_v3_large_coco-a79551df.pth"
)
CHECKPOINT_SHA256_PREFIX = "a79551df"
MAX_CHECKPOINT_BYTES = 32 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class HybridSettings:
    checkpoint_path: str = "/models/ssdlite320_mobilenet_v3_large_coco-a79551df.pth"
    score_threshold: float = 0.45
    low_score_threshold: float = 0.20
    calibration_seconds: float = 2.0
    track_ttl_seconds: float = 2.0

    def __post_init__(self):
        if not self.checkpoint_path.strip():
            raise ValueError("hybrid checkpoint path is required")
        for name, low, high in (
            ("score_threshold", 0.3, 0.95),
            ("low_score_threshold", 0.1, 0.5),
            ("calibration_seconds", 0.5, 30.0),
            ("track_ttl_seconds", 0.5, 5.0),
        ):
            value = getattr(self, name)
            if (
                type(value) not in (int, float)
                or not math.isfinite(value)
                or not low <= value <= high
            ):
                raise ValueError(f"invalid hybrid {name}")
        if self.low_score_threshold >= self.score_threshold:
            raise ValueError("hybrid low threshold must be below high threshold")

    @classmethod
    def from_env(cls):
        defaults = cls()
        values = {}
        for name in cls.__dataclass_fields__:
            value = os.environ.get("HYBRID_" + name.upper())
            if value is not None:
                values[name] = type(getattr(defaults, name))(value)
        return cls(**values)


def verify_checkpoint(path):
    """Verify the exact approved official download before weights-only loading."""
    path = Path(path)
    if not path.is_file() or not 0 < path.stat().st_size <= MAX_CHECKPOINT_BYTES:
        raise RuntimeError("approved local hybrid checkpoint is missing or too large")
    digest = hashlib.sha256()
    total = 0
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            total += len(chunk)
            if total > MAX_CHECKPOINT_BYTES:
                raise RuntimeError("hybrid checkpoint exceeds size limit")
            digest.update(chunk)
    if not digest.hexdigest().startswith(CHECKPOINT_SHA256_PREFIX):
        raise RuntimeError("hybrid checkpoint does not match official weights")
    return path


class TorchvisionPersonDetector:
    """The real COCO model, with explicit local weights and a CUDA warmup."""

    def __init__(self, settings):
        self.settings = settings
        self.model = None
        self.torch = None
        self.to_tensor = None
        self.gpu_ready = False

    def warmup(self):
        path = verify_checkpoint(self.settings.checkpoint_path)
        import torch
        from PIL import Image
        from torchvision.models.detection import ssdlite320_mobilenet_v3_large
        from torchvision.transforms.functional import to_tensor

        if not torch.cuda.is_available():
            raise RuntimeError("hybrid detector requires CUDA")
        self.torch, self.to_tensor = torch, to_tensor
        # Avoid any implicit ImageNet/COCO download. The complete local state
        # must strictly match the architecture before CUDA readiness is asserted.
        self.model = ssdlite320_mobilenet_v3_large(
            weights=None, weights_backbone=None, num_classes=91
        )
        weights = torch.load(str(path), map_location="cpu", weights_only=True)
        if not isinstance(weights, dict) or any(
            not isinstance(value, torch.Tensor)
            or (value.is_floating_point() and not torch.isfinite(value).all().item())
            for value in weights.values()
        ):
            raise RuntimeError("hybrid checkpoint contains invalid weights")
        self.model.load_state_dict(weights, strict=True)
        del weights
        self.model.score_thresh = self.settings.low_score_threshold
        self.model = self.model.eval().cuda()
        with Image.new("RGB", (320, 192)) as image:
            self.detect(image)
        torch.cuda.synchronize()
        self.gpu_ready = True

    def detect(self, image):
        if self.model is None or self.torch is None:
            raise RuntimeError("hybrid detector unavailable")
        with self.torch.inference_mode():
            tensor = self.to_tensor(image).cuda()
            raw = self.model([tensor])[0]
            labels = raw["labels"].detach().cpu().tolist()
            boxes = raw["boxes"].detach().cpu().tolist()
            scores = raw["scores"].detach().cpu().tolist()
            self.torch.cuda.synchronize()
        width, height = image.size
        result = []
        for label, box, score in zip(labels, boxes, scores, strict=True):
            if label != 1 or score < self.settings.low_score_threshold:
                continue
            if (
                not math.isfinite(score)
                or len(box) != 4
                or any(not math.isfinite(v) for v in box)
            ):
                raise RuntimeError("hybrid detector returned invalid output")
            bbox = tuple(
                min(1.0, max(0.0, value / dimension))
                for value, dimension in zip(
                    box, (width, height, width, height), strict=True
                )
            )
            if bbox[0] >= bbox[2] or bbox[1] >= bbox[3]:
                continue
            result.append(
                PersonDetection(bbox, float(score), describe_person(image, bbox))
            )
            if len(result) == 16:
                break
        return tuple(result)

    def close(self):
        self.gpu_ready = False
        self.model = None


class PriorityGpuGate:
    """Serialize shared CUDA model use; queued foreground work takes priority.

    An already running single-frame calibration is non-preemptible. It finishes
    before foreground inference, preventing overlapping mutation of SAM modules.
    """

    def __init__(self):
        self.condition = threading.Condition()
        self.active = False
        self.foreground_waiters = 0

    @contextmanager
    def use(self, *, foreground, cancelled):
        acquired = False
        with self.condition:
            if foreground:
                self.foreground_waiters += 1
            try:
                while self.active or (not foreground and self.foreground_waiters):
                    if cancelled():
                        raise InterruptedError("hybrid work cancelled")
                    self.condition.wait(0.05)
                if cancelled():
                    raise InterruptedError("hybrid work cancelled")
                self.active = acquired = True
            finally:
                if foreground:
                    self.foreground_waiters -= 1
                self.condition.notify_all()
        try:
            yield
        finally:
            if acquired:
                with self.condition:
                    self.active = False
                    self.condition.notify_all()


@dataclass
class HybridState:
    epoch: str
    tracker: PersonTracker
    cancel: threading.Event
    mutex: threading.RLock = field(default_factory=threading.RLock)
    closed: bool = False
    last_scheduled_timestamp: float = -math.inf
    last_calibration_source: float | None = None
    last_calibration_sequence: int | None = None
    detections: int = 0
    processed_frames: int = 0
    calibrations_scheduled: int = 0
    calibrations_completed: int = 0
    calibrations_replaced: int = 0
    calibration_failures: int = 0
    calibration_confirmed: int = 0
    calibration_uncertain: int = 0
    calibration_discovered: int = 0
    calibration_seconds: float = 0.0
    detector_seconds: float = 0.0

    def cancelled(self):
        return self.closed or self.cancel.is_set()


@dataclass(frozen=True, slots=True)
class CalibrationJob:
    state: HybridState
    epoch: str
    image: object
    observed: tuple
    source: SourceFrame


class HybridContinuation:
    """SessionEngine adapter, removable independently of native continuation."""

    def __init__(
        self,
        video_engine,
        session_settings,
        *,
        settings=None,
        detector=None,
        calibrator=None,
    ):
        self.video = video_engine
        self.session_settings = session_settings
        self.settings = settings or HybridSettings.from_env()
        self._production = detector is None and calibrator is None
        self.detector = detector or TorchvisionPersonDetector(self.settings)
        self.calibrator = calibrator or self._sam_calibrate
        self.gpu_ready = False
        self._gate = PriorityGpuGate()
        self._condition = threading.Condition()
        self._pending: CalibrationJob | None = None
        self._running: CalibrationJob | None = None
        self._closed = False
        self._thread = None
        self._states: list[HybridState] = []

    def warmup(self):
        if self._production and not self.video.gpu_ready:
            raise RuntimeError("hybrid SAM calibration requires verified CUDA model")
        self.detector.warmup()
        self.gpu_ready = self._production and self.detector.gpu_ready is True
        self._thread = threading.Thread(
            target=self._calibration_loop, name="hybrid-sam-calibration", daemon=True
        )
        self._thread.start()

    def _new_state(self, session):
        state = HybridState(
            session.predictor_epoch,
            PersonTracker(
                max_objects=self.video.settings.max_objects,
                max_ids=self.session_settings.max_cumulative_native_ids,
                ttl_seconds=self.settings.track_ttl_seconds,
                high_threshold=self.settings.score_threshold,
                low_threshold=self.settings.low_score_threshold,
            ),
            session.cancel,
        )
        with self._condition:
            if self._closed or self._states:
                raise RuntimeError("hybrid route already owns a session or is closed")
            self._states.append(state)
        session.native_state = state
        return state

    def run(self, session, images, *, final):
        state = session.native_state or self._new_state(session)
        if not isinstance(state, HybridState) or state.epoch != session.predictor_epoch:
            raise RuntimeError("hybrid session epoch mismatch")
        for offset, image in enumerate(images):
            if state.cancelled():
                raise InterruptedError("hybrid session cancelled")
            index = session.total_frames + offset
            source = session.sources[index]
            started = time.perf_counter()
            with self._gate.use(foreground=True, cancelled=state.cancelled):
                detections = self.detector.detect(image)
            elapsed = time.perf_counter() - started
            with state.mutex:
                if state.cancelled():
                    raise InterruptedError("hybrid session cancelled")
                try:
                    observed = state.tracker.update(
                        detections,
                        timestamp=source.timestamp_seconds,
                        frame_sequence=source.frame_sequence,
                    )
                except TrackerCapacityError:
                    raise HTTPException(409, "reset_required") from None
                state.detector_seconds += elapsed
                state.detections += len(observed)
                state.processed_frames += 1
            # The shared engine serializes these real detector boxes directly.
            # No rectangular masks are created or represented as SAM masks.
            yield index, observed
        if images:
            # Schedule only after this append's foreground detections finish.
            # Otherwise a calibration after frame 1 could block frames 2..4.
            self._schedule(state, images[-1], observed, source)

    def _schedule(self, state, image, observed, source):
        with state.mutex:
            if state.cancelled():
                return
            uncertain = any(not item.confirmed for item in observed)
            interval = (
                min(self.settings.calibration_seconds, 0.75)
                if uncertain
                else (self.settings.calibration_seconds)
            )
            if source.timestamp_seconds - state.last_scheduled_timestamp < interval:
                return
            # Copy only selected full frames, at most one pending plus one running.
            job = CalibrationJob(state, state.epoch, image.copy(), observed, source)
            with self._condition:
                if self._closed or state.cancelled():
                    job.image.close()
                    return
                if self._pending is not None:
                    self._pending.image.close()
                    state.calibrations_replaced += 1
                self._pending = job
                state.last_scheduled_timestamp = source.timestamp_seconds
                state.calibrations_scheduled += 1
                self._condition.notify_all()

    def _sam_calibrate(self, image, source):
        """Read SAM's prompted-frame output without future-frame confirmation.

        The pinned model's add_prompt returns its single-frame semantic detection
        directly. NativeSam31.infer additionally propagates and hides unconfirmed
        short masklets, which is inappropriate for this one-frame calibration.
        This observation is explicitly same-frame model corroboration, not a
        claim that SAM has established a temporally confirmed video trajectory.
        """
        batch = VideoBatch("hybrid-calibration", "calibration", (source,))
        native = self.video.backend
        model = native.predictor.model
        state = None
        with (
            native._torch.inference_mode(),
            native._torch.autocast(device_type="cuda", dtype=native._torch.bfloat16),
        ):
            try:
                state = model.init_state(
                    resource_path=[image],
                    offload_video_to_cpu=True,
                    async_loading_frames=False,
                )
                index, output = model.add_prompt(
                    inference_state=state,
                    frame_idx=0,
                    text_str="person",
                    output_prob_thresh=0.5,
                )
                if type(index) is not int or index != 0:
                    raise RuntimeError("SAM calibration source ordering mismatch")
                snapshot = native_snapshot(
                    output,
                    batch=batch,
                    index=0,
                    epoch="calibration",
                    image_size=image.size,
                    max_objects=self.video.settings.max_objects,
                )
            finally:
                if state is not None:
                    try:
                        model.reset_state(state)
                    finally:
                        state.clear()
                native._torch.cuda.synchronize()
        result = []
        for item in snapshot["objects"]:
            if item["confidence"] < 0.5:
                continue
            bbox = tuple(item["bbox"][key] for key in ("x1", "y1", "x2", "y2"))
            result.append(
                PersonDetection(bbox, item["confidence"], describe_person(image, bbox))
            )
        return tuple(result)

    def _calibration_loop(self):
        while True:
            with self._condition:
                while self._pending is None and not self._closed:
                    self._condition.wait()
                if self._closed:
                    return
                job, self._pending = self._pending, None
                self._running = job
            started = time.perf_counter()
            try:
                with self._gate.use(foreground=False, cancelled=job.state.cancelled):
                    detected = self.calibrator(job.image, job.source)
                with job.state.mutex:
                    state = job.state
                    if not state.cancelled() and state.epoch == job.epoch:
                        counts = state.tracker.calibrate(
                            job.observed,
                            detected,
                            frame_sequence=job.source.frame_sequence,
                            timestamp=job.source.timestamp_seconds,
                        )
                        state.calibrations_completed += 1
                        state.calibration_confirmed += counts["confirmed"]
                        state.calibration_uncertain += counts["uncertain"]
                        state.calibration_discovered += counts["discovered"]
                        state.last_calibration_source = job.source.timestamp_seconds
                        state.last_calibration_sequence = job.source.frame_sequence
                        state.calibration_seconds += time.perf_counter() - started
            except InterruptedError:
                pass
            except Exception:
                with job.state.mutex:
                    if not job.state.cancelled():
                        job.state.calibration_failures += 1
                        # Do not run another queued job against a failed shared
                        # CUDA model or leave the session silently fast-only.
                        job.state.cancel.set()
                        self.gpu_ready = False
                        # A failure in the shared SAM CUDA model must not leave
                        # its sibling native route attesting model readiness.
                        if self._production:
                            self.video.ready = False
                            self.video.error_code = "hybrid_calibration_failed"
                LOGGER.error("Hybrid SAM calibration failed; route is unavailable")
            finally:
                job.image.close()
                with self._condition:
                    self._running = None
                    self._condition.notify_all()

    @staticmethod
    def serialize_snapshot(output, *, batch, index, epoch, image_size, max_objects):
        source = batch.frames[index]
        if len(output) > max_objects or len({item.object_id for item in output}) != len(
            output
        ):
            raise RuntimeError("hybrid detector output exceeds person bounds")
        objects = []
        for observed in output:
            if type(observed.object_id) is not int or observed.object_id < 0:
                raise RuntimeError("hybrid person identifier invalid")
            object_id = str(observed.object_id)
            attributes = {
                "native_track": {
                    "backend": BACKEND,
                    "epoch": epoch,
                    "object_id": object_id,
                },
                "person_detection": {
                    "model": MODEL_NAME,
                    "geometry": "box",
                    "confirmed": observed.confirmed,
                },
            }
            if observed.calibration is not None:
                attributes["sam_calibration"] = observed.calibration
            objects.append(
                {
                    "track_id": f"hybrid-{epoch}-{object_id}",
                    "label": "person",
                    "bbox": dict(
                        zip(
                            ("x1", "y1", "x2", "y2"),
                            observed.detection.bbox,
                            strict=True,
                        )
                    ),
                    "confidence": observed.detection.confidence,
                    "attributes": attributes,
                    "observation_state": "observed",
                    "last_observed_frame_sequence": source.frame_sequence,
                    "last_observed_timestamp_seconds": source.timestamp_seconds,
                }
            )
        return {
            "stream_session_id": batch.stream_session_id,
            "frame_sequence": source.frame_sequence,
            "timestamp_seconds": source.timestamp_seconds,
            "objects": objects,
        }

    def route_metadata(self, session, snapshots):
        state = session.native_state
        # SessionEngine may request metadata immediately after EOF cleanup.
        diagnostics = getattr(session, "hybrid_final_diagnostics", {})
        if isinstance(state, HybridState):
            diagnostics = self._diagnostics(state)
        return {
            "confirmation_policy": "detector_observations_with_async_sam_calibration",
            "observations": [
                {
                    "frame_sequence": snapshot["frame_sequence"],
                    "provisional_object_ids": [
                        obj["attributes"]["native_track"]["object_id"]
                        for obj in snapshot["objects"]
                        if not obj["attributes"]["person_detection"]["confirmed"]
                    ],
                }
                for snapshot in snapshots
            ],
            "diagnostics": diagnostics,
        }

    def _diagnostics(self, state):
        with state.mutex:
            result = {
                name: getattr(state, name)
                for name in (
                    "processed_frames",
                    "detections",
                    "calibrations_scheduled",
                    "calibrations_completed",
                    "calibrations_replaced",
                    "calibration_failures",
                    "calibration_confirmed",
                    "calibration_uncertain",
                    "calibration_discovered",
                    "last_calibration_source",
                    "last_calibration_sequence",
                    "calibration_seconds",
                    "detector_seconds",
                )
            }
            result.update(
                {
                    "detector_model": MODEL_NAME,
                    "active_tracks": len(state.tracker.tracks),
                    "calibration_candidates": len(state.tracker.seeds),
                    "calibration_seed_matches": state.tracker.seed_matches,
                    "ambiguous_associations": state.tracker.ambiguous_count,
                    "gpu_ready": self.gpu_ready,
                }
            )
        with self._condition:
            result["calibration_running"] = (
                self._running is not None and self._running.state is state
            )
            result["calibration_pending"] = (
                self._pending is not None and self._pending.state is state
            )
        return result

    def prune(self, session):
        # There are no retained past images beyond the bounded calibration slots.
        pass

    def discard(self, session):
        state = session.native_state
        if not isinstance(state, HybridState):
            return
        with state.mutex:
            state.closed = True
        with self._condition:
            if self._pending is not None and self._pending.state is state:
                self._pending.image.close()
                self._pending = None
            # Do not allow another route to reuse shared SAM until its in-flight
            # generator has reset its state and released the CUDA gate.
            while self._running is not None and self._running.state is state:
                self._condition.wait(0.05)
            if state in self._states:
                self._states.remove(state)
        session.hybrid_final_diagnostics = self._diagnostics(state)
        session.native_state = None

    def close(self):
        self.gpu_ready = False
        for state in tuple(self._states):
            with state.mutex:
                state.closed = True
        with self._condition:
            self._closed = True
            if self._pending is not None:
                self._pending.image.close()
                self._pending = None
            self._condition.notify_all()
        if self._thread is not None:
            self._thread.join()
        self._states.clear()
        self.detector.close()
