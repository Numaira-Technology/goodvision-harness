"""Native SAM 3.1 video inference over finite, already-received frame batches.

Each batch has a fresh predictor epoch. No camera append API or cross-batch
native memory is claimed. Heavy model/image imports occur only during use.
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import json
import logging
import math
import os
import re
import threading
import time
import uuid
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import Any, Protocol

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

LOGGER = logging.getLogger("goodvision.sam31")
SAM3_COMMIT = "660a5e9e1b8b4c02c0ad97229b88a09a6e4ff5b7"
BACKEND = "sam3.1_video"


@dataclass(frozen=True, slots=True)
class VideoSettings:
    enabled: bool = False
    checkpoint_path: str = "/models/sam3.1_multiplex.pt"
    compile_mode: str = "off"
    fixed_person_text: bool = False
    gpu_diagnostics: bool = False
    max_frames: int = 16
    max_request_bytes: int = 32 * 1024 * 1024
    max_jpeg_bytes: int = 8 * 1024 * 1024
    max_total_jpeg_bytes: int = 24 * 1024 * 1024
    max_image_pixels: int = 3840 * 2160
    max_total_image_pixels: int = 64 * 1024 * 1024
    inference_max_side: int = 1280
    max_objects: int = 16
    request_timeout_seconds: float = 60.0

    def __post_init__(self) -> None:
        if type(self.enabled) is not bool or not self.checkpoint_path.strip():
            raise ValueError("invalid video backend configuration")
        for name in ("fixed_person_text", "gpu_diagnostics"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be boolean")
        if type(self.compile_mode) is not str or self.compile_mode not in {
            "off", "encoder"
        }:
            raise ValueError("compile_mode must be off or encoder")
        ceilings = {
            "max_frames": 16,
            "max_request_bytes": 32 * 1024 * 1024,
            "max_jpeg_bytes": 8 * 1024 * 1024,
            "max_total_jpeg_bytes": 24 * 1024 * 1024,
            "max_image_pixels": 3840 * 2160,
            "max_total_image_pixels": 64 * 1024 * 1024,
            "inference_max_side": 1280,
            "max_objects": 128,
        }
        for name, ceiling in ceilings.items():
            value = getattr(self, name)
            if type(value) is not int or not 1 <= value <= ceiling:
                raise ValueError(f"{name} must be in [1, {ceiling}]")
        if (
            type(self.request_timeout_seconds) not in (int, float)
            or not math.isfinite(self.request_timeout_seconds)
            or not 0 < self.request_timeout_seconds <= 120
        ):
            raise ValueError("request_timeout_seconds must be in (0, 120]")

    @classmethod
    def from_env(cls) -> VideoSettings:
        defaults = cls()
        values = {}
        for name in cls.__dataclass_fields__:
            value = os.environ.get("SAM31_" + name.upper())
            if value is None:
                continue
            default = getattr(defaults, name)
            if type(default) is bool:
                if value.lower() not in {"true", "false"}:
                    raise ValueError(f"SAM31_{name.upper()} must be true or false")
                values[name] = value.lower() == "true"
            elif type(default) is int:
                values[name] = int(value)
            elif type(default) is float:
                values[name] = float(value)
            else:
                values[name] = value
        return cls(**values)

    def limits(self) -> dict[str, Any]:
        return {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
            if name not in {
                "enabled", "checkpoint_path", "compile_mode",
                "fixed_person_text", "gpu_diagnostics",
            }
        } | {"max_inflight_requests": 1}


@dataclass(frozen=True, slots=True)
class SourceFrame:
    frame_sequence: int
    timestamp_seconds: float
    jpeg_bytes: bytes


@dataclass(frozen=True, slots=True)
class VideoBatch:
    stream_session_id: str
    batch_id: str
    frames: tuple[SourceFrame, ...]


def _unique_fields(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate field")
        result[key] = value
    return result


def _invalid_constant(_value):
    raise ValueError("nonfinite number")


def parse_batch(payload: bytes, settings: VideoSettings) -> VideoBatch:
    if not payload or len(payload) > settings.max_request_bytes:
        raise HTTPException(413, "video request size outside allowed bounds")
    try:
        raw = json.loads(
            payload, object_pairs_hook=_unique_fields, parse_constant=_invalid_constant
        )
        if type(raw) is not dict or set(raw) != {
            "stream_session_id",
            "batch_id",
            "frames",
        }:
            raise ValueError()
        for key in ("stream_session_id", "batch_id"):
            if type(raw[key]) is not str or not re.fullmatch(
                r"[A-Za-z0-9_.:-]{1,160}", raw[key]
            ):
                raise ValueError()
        if type(raw["frames"]) is not list:
            raise ValueError()
        if not 1 <= len(raw["frames"]) <= settings.max_frames:
            raise HTTPException(413, "video batch frame count outside allowed bounds")
        frames = []
        total_bytes = 0
        last_sequence, last_timestamp = -1, -1.0
        for frame in raw["frames"]:
            if type(frame) is not dict or set(frame) != {
                "frame_sequence",
                "timestamp_seconds",
                "jpeg_base64",
            }:
                raise ValueError()
            sequence, timestamp = frame["frame_sequence"], frame["timestamp_seconds"]
            if (
                type(sequence) is not int
                or not 0 <= sequence < 2**63
                or type(timestamp) not in (int, float)
                or not math.isfinite(timestamp)
                or timestamp < 0
            ):
                raise ValueError()
            if sequence <= last_sequence or timestamp <= last_timestamp:
                raise HTTPException(409, "video source frames must strictly increase")
            encoded = frame["jpeg_base64"]
            if type(encoded) is not str:
                raise ValueError()
            if not encoded or len(encoded) > 4 * ((settings.max_jpeg_bytes + 2) // 3):
                raise HTTPException(413, "JPEG size outside allowed bounds")
            jpeg = base64.b64decode(encoded, validate=True)
            total_bytes += len(jpeg)
            if (
                not jpeg
                or len(jpeg) > settings.max_jpeg_bytes
                or total_bytes > settings.max_total_jpeg_bytes
            ):
                raise HTTPException(413, "video JPEG budget exceeded")
            frames.append(SourceFrame(sequence, float(timestamp), jpeg))
            last_sequence, last_timestamp = sequence, timestamp
        return VideoBatch(raw["stream_session_id"], raw["batch_id"], tuple(frames))
    except (ValueError, TypeError, OverflowError, RecursionError, binascii.Error):
        raise HTTPException(400, "invalid video batch") from None


def verify_installed_source() -> None:
    """The deployment must be installed from the audited VCS commit, not latest."""
    direct_url = metadata.distribution("sam3").read_text("direct_url.json")
    record = json.loads(direct_url or "{}")
    if record.get("vcs_info", {}).get("commit_id") != SAM3_COMMIT:
        raise RuntimeError("SAM 3 source commit does not match the audited revision")


def strict_checkpoint_state(
    state: Mapping[str, Any], expected: Mapping[str, Any]
) -> dict[str, Any]:
    """Match the complete merged SAM3.1 detector/tracker state, never SAM3 image."""
    if not isinstance(state, Mapping) or not state or not expected:
        raise RuntimeError("invalid SAM3.1 checkpoint state")
    remapped = {}
    for key, value in state.items():
        if not isinstance(key, str):
            raise RuntimeError("invalid SAM3.1 checkpoint keys")
        if key.startswith("sam3_model."):
            key = "detector." + key.removeprefix("sam3_model.")
        elif key.startswith("sam2_predictor."):
            key = "tracker." + key.removeprefix("sam2_predictor.")
        if key in remapped:
            raise RuntimeError("duplicate SAM3.1 checkpoint keys after remapping")
        remapped[key] = value
    if remapped.keys() != expected.keys():
        raise RuntimeError("SAM3.1 checkpoint required or unexpected keys differ")
    for key, value in remapped.items():
        if getattr(value, "shape", None) != expected[key].shape:
            raise RuntimeError("SAM3.1 checkpoint tensor shapes differ")
    return remapped


def _native_bbox_without_speckles(
    mask: Any,
) -> tuple[tuple[int, int, int, int] | None, int, int]:
    bbox, _, ignored_count, ignored_pixels = native_mask_geometry(mask)
    return bbox, ignored_count, ignored_pixels


def native_mask_geometry(mask: Any) -> tuple[
    tuple[int, int, int, int] | None, list[list[list[float]]], int, int
]:
    """Ignore <=4-pixel 8-connected islands for exported geometry only.

    Four is an absolute area in the actual output mask's pixels, at every
    resolution. Retain every larger component, including disconnected visible
    body parts. Bound simplified external outlines to 16 polygons, 128 points
    per polygon and 1024 total points. The native mask is never modified.
    """
    import cv2
    import numpy as np

    if mask.ndim != 2 or mask.dtype != np.bool_ or not all(mask.shape):
        raise RuntimeError("native mask must be a nonempty boolean image")
    if not mask.any():
        return None, [], 0, 0
    _, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8
    )
    components = stats[1:]
    areas = components[:, cv2.CC_STAT_AREA]
    ignored = areas <= 4
    ignored_count = int(ignored.sum())
    ignored_pixels = int(areas[ignored].sum())
    retained = components[~ignored]
    if not len(retained):
        return None, [], ignored_count, ignored_pixels
    x = retained[:, cv2.CC_STAT_LEFT]
    y = retained[:, cv2.CC_STAT_TOP]
    right = x + retained[:, cv2.CC_STAT_WIDTH]
    bottom = y + retained[:, cv2.CC_STAT_HEIGHT]
    # One lookup keeps all retained components without allocating an image per
    # component. Exterior contours deliberately do not paint/fill mask holes.
    retained_labels = np.concatenate(([False], ~ignored))
    contours, _ = cv2.findContours(
        retained_labels[labels].astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    height, width = mask.shape
    polygons = []
    remaining = 1024
    for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:16]:
        limit = min(128, remaining)
        if limit < 3:
            break
        epsilon = max(0.5, cv2.arcLength(contour, True) * 0.002)
        simplified = cv2.approxPolyDP(contour, epsilon, True)
        while len(simplified) > limit:
            epsilon *= 1.5
            simplified = cv2.approxPolyDP(contour, epsilon, True)
        if len(simplified) < 3 or cv2.contourArea(simplified) <= 0:
            continue
        # OpenCV traces pixel centers; normalization preserves that geometry.
        polygons.append([
            [round(float((px + 0.5) / width), 6),
             round(float((py + 0.5) / height), 6)]
            for px, py in simplified[:, 0, :]
        ])
        remaining -= len(simplified)
    return (
        (int(x.min()), int(y.min()), int(right.max()), int(bottom.max())),
        polygons,
        ignored_count,
        ignored_pixels,
    )


def native_snapshot(
    output: Mapping[str, Any],
    *,
    batch: VideoBatch,
    index: int,
    epoch: str,
    image_size: tuple[int, int],
    max_objects: int,
) -> dict[str, Any]:
    """Validate native masks, derive normalized xyxy; never invent visible masks."""
    import numpy as np

    if not isinstance(output, Mapping):
        raise RuntimeError("invalid native video output")
    ids = np.asarray(output["out_obj_ids"])
    scores = np.asarray(output["out_probs"])
    masks = np.asarray(output["out_binary_masks"])
    width, height = image_size
    if (
        type(width) is not int
        or type(height) is not int
        or width <= 0
        or height <= 0
        or ids.ndim != 1
        or ids.dtype.kind not in "iu"
        or len(ids) > max_objects
        or len(set(ids.tolist())) != len(ids)
        or any(int(value) < 0 for value in ids)
        or scores.shape != (len(ids),)
        or scores.dtype.kind not in "fiu"
        or not np.isfinite(scores).all()
        or (scores < 0).any()
        or (scores > 1).any()
        or masks.dtype != np.bool_
        or masks.shape != (len(ids), height, width)
    ):
        raise RuntimeError(
            "native video masks or identifiers violate contract: "
            f"image_size={image_size}, max_objects={max_objects}, "
            f"ids={ids.shape}/{ids.dtype}, scores={scores.shape}/{scores.dtype}, "
            f"masks={masks.shape}/{masks.dtype}"
        )
    source = batch.frames[index]
    objects = []
    for object_id, score, mask in zip(
        ids.tolist(), scores.tolist(), masks, strict=True
    ):
        bbox, polygons, ignored_count, ignored_pixels = native_mask_geometry(mask)
        if bbox is None:
            continue
        x1, y1, x2, y2 = bbox
        native_id = str(object_id)
        objects.append(
            {
                "track_id": f"sam31-{epoch}-{native_id}",
                "label": "person",
                "bbox": {
                    "x1": float(x1 / width),
                    "y1": float(y1 / height),
                    "x2": float(x2 / width),
                    "y2": float(y2 / height),
                },
                "confidence": float(score),
                "attributes": {
                    "native_track": {
                        "backend": BACKEND,
                        "epoch": epoch,
                        "object_id": native_id,
                    },
                    "mask_geometry": {
                        "max_ignored_component_area_pixels": 4,
                        "ignored_component_count": ignored_count,
                        "ignored_component_pixels": ignored_pixels,
                    },
                    **({"mask_contours": {
                        "coordinate_system": "normalized",
                        "source_frame_sequence": source.frame_sequence,
                        "source_timestamp_seconds": source.timestamp_seconds,
                        "polygons": polygons,
                    }} if polygons else {}),
                },
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


class VideoBackend(Protocol):
    def warmup(self) -> None: ...
    def infer(self, images: Sequence[Any]) -> Iterator[tuple[int, Mapping]]: ...
    def close(self) -> None: ...


def compile_vision_encoder(torch, model):
    """Compile only the fixed-image ViT after strict loading, with no new weights.

    Wrapping the bound forward keeps state_dict names and shared Prod tensors
    unchanged. The official all-component compiler modifies global functions and
    suppresses compiler failures; this opt-in path does neither. Complex RoPE is
    retained, so performance and compiler support require real CUDA measurement.
    """
    try:
        encoder = model.detector.backbone.vision_backbone.trunk
        forward = encoder.forward
    except AttributeError:
        raise RuntimeError("pinned native vision encoder is unavailable") from None
    if not callable(forward) or getattr(model, "compile_model", False):
        raise RuntimeError("unsupported encoder compilation state")
    if getattr(encoder, "_goodvision_compiled_forward", False):
        raise RuntimeError("native vision encoder already compiled")
    expected = {
        key: (tuple(tensor.shape), tensor.data_ptr())
        for key, tensor in model.state_dict().items()
    }
    compiled = torch.compile(
        forward, backend="inductor", mode="default", fullgraph=True, dynamic=False
    )
    if not callable(compiled):
        raise RuntimeError("compiler did not produce an encoder callable")
    encoder.forward = compiled
    actual = {
        key: (tuple(tensor.shape), tensor.data_ptr())
        for key, tensor in model.state_dict().items()
    }
    if expected != actual:
        encoder.forward = forward
        raise RuntimeError("encoder compilation changed verified model tensors")
    encoder._goodvision_compiled_forward = True


def cache_person_text(backbone, torch):
    """Reuse exact fixed-prompt features and release the unused text weights.

    Install only after strict checkpoint validation. This worker's public video
    contract uses person exclusively; reject any future incompatible caller.
    CUDA warmup materializes the cache before any request is served.
    """
    if getattr(backbone, "_goodvision_person_text", False):
        return
    original = backbone.forward_text
    cached = None

    def forward(captions, input_boxes=None, additional_text=None, device="cuda"):
        nonlocal cached
        if (
            tuple(captions) != ("person", "visual", "geometric")
            or input_boxes is not None
            or additional_text is not None
            or str(device) not in {"cuda", "cuda:0"}
        ):
            raise RuntimeError("Contour supports the fixed person prompt only")
        if cached is None:
            result = original(captions, device=device)
            if set(result) != {"language_features", "language_mask", "language_embeds"}:
                raise RuntimeError("Unexpected text encoder output")
            cached = {key: value.detach().clone() for key, value in result.items()}
            encoder = backbone.language_backbone
            released = sum(p.numel() * p.element_size() for p in encoder.parameters())
            encoder.to(device="meta")
            # Once during warmup, never in the per-frame inference loop.
            torch.cuda.empty_cache()
            LOGGER.info("Fixed person text cached released_weight_bytes=%d", released)
        return {key: value.clone() for key, value in cached.items()}

    backbone.forward_text = forward
    backbone._goodvision_person_text = True


class NativeSam31:
    def __init__(self, settings: VideoSettings):
        self.settings = settings
        self.predictor: Any = None
        self.gpu_ready = False
        self._torch: Any = None

    def warmup(self) -> None:
        checkpoint = Path(self.settings.checkpoint_path)
        if not checkpoint.is_file():
            raise RuntimeError("approved local SAM3.1 checkpoint is missing")
        verify_installed_source()
        import torch
        from PIL import Image
        from sam3.model_builder import build_sam3_multiplex_video_predictor

        self._torch = torch
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA GPU unavailable")
        # The official merged-checkpoint builder loads non-strictly internally.
        # Recheck and strictly reload the complete model before any readiness.
        self.predictor = build_sam3_multiplex_video_predictor(
            checkpoint_path=str(checkpoint),
            max_num_objects=self.settings.max_objects,
            multiplex_count=16,
            use_fa3=False,
            # The official merged checkpoint stores complex freqs_cis only.
            # Real RoPE registers 64 additional persistent derived buffers that
            # are absent from it; use the native complex path with compile off.
            use_rope_real=False,
            compile=False,
            warm_up=False,
            async_loading_frames=False,
        )
        # Both upstream wrappers enter thread-local autocast contexts forever.
        # Warmup and HTTP execution may use different pool threads, so undo it
        # here and enter a fresh bounded context on each actual inference call.
        self._release_builder_autocast()
        model = self.predictor.model
        raw = torch.load(str(checkpoint), map_location="cpu", weights_only=True)
        state = raw.get("model", raw)
        verified = strict_checkpoint_state(state, model.state_dict())
        for tensor in verified.values():
            if tensor.is_floating_point() and not torch.isfinite(tensor).all().item():
                raise RuntimeError("SAM3.1 checkpoint contains nonfinite weights")
        model.load_state_dict(verified, strict=True)
        del verified, state, raw
        if self.settings.fixed_person_text:
            cache_person_text(model.detector.backbone, torch)
        # Bound masks/grounding buffers instead of precomputing 16 frames at once.
        model.postprocess_batch_size = 1
        model.use_batched_grounding = False
        if self.settings.compile_mode == "encoder":
            compile_vision_encoder(torch, model)
        blanks = [Image.new("RGB", (320, 192)) for _ in range(3)]
        probe = VideoBatch(
            "warmup",
            "warmup",
            tuple(
                SourceFrame(index, float(index), b"") for index in range(len(blanks))
            ),
        )
        try:
            seen = []
            for index, output in self.infer(blanks):
                if (
                    type(index) is not int
                    or not 0 <= index < len(blanks)
                    or index in seen
                ):
                    raise RuntimeError("native warmup frame ordering invalid")
                native_snapshot(
                    output,
                    batch=probe,
                    index=index,
                    epoch="warmup",
                    image_size=blanks[index].size,
                    max_objects=self.settings.max_objects,
                )
                seen.append(index)
            if seen != list(range(len(blanks))):
                raise RuntimeError("native warmup did not return all frames")
            torch.cuda.synchronize()
            self.gpu_ready = True
            LOGGER.info(
                "Native video CUDA warmup completed compile_mode=%s",
                self.settings.compile_mode,
            )
        finally:
            for image in blanks:
                image.close()

    def memory_stats(self):
        """No device synchronization; peak counters are since the last reset."""
        if not self.settings.gpu_diagnostics or self._torch is None:
            return {}
        cuda = self._torch.cuda
        free, total = cuda.mem_get_info()
        return {
            "cuda_allocated_bytes": cuda.memory_allocated(),
            "cuda_reserved_bytes": cuda.memory_reserved(),
            "cuda_peak_allocated_bytes": cuda.max_memory_allocated(),
            "cuda_peak_reserved_bytes": cuda.max_memory_reserved(),
            "cuda_device_free_bytes": free,
            "cuda_device_total_bytes": total,
        }

    def infer(self, images: Sequence[Any]) -> Iterator[tuple[int, Mapping]]:
        if self.predictor is None:
            raise RuntimeError("native video model unavailable")
        model = self.predictor.model
        state = None
        with (
            self._torch.inference_mode(),
            self._torch.autocast(device_type="cuda", dtype=self._torch.bfloat16),
        ):
            try:
                # Public wrapper's start_session passes offload_state_to_cpu to
                # a model signature that does not accept it at the pinned commit.
                # Call the documented model operations directly, with exact args.
                state = model.init_state(
                    resource_path=list(images),
                    offload_video_to_cpu=True,
                    async_loading_frames=False,
                )
                model.add_prompt(inference_state=state, frame_idx=0, text_str="person")
                yield from model.propagate_in_video(
                    inference_state=state,
                    start_frame_idx=0,
                    max_frame_num_to_track=len(images),
                    reverse=False,
                )
            finally:
                if state is not None:
                    try:
                        model.reset_state(state)
                    except Exception:
                        self.gpu_ready = False
                        raise
                    finally:
                        state.clear()

    def _release_builder_autocast(self) -> None:
        # Construction enters tracker first, then the outer video predictor.
        # Unwind in reverse order on the builder's thread. The Hydra parent
        # constructor is explicitly skipped by Sam3MultiplexPredictorWrapper.
        tracker = getattr(getattr(self.predictor, "model", None), "tracker", None)
        for owner in (self.predictor, tracker):
            context = getattr(owner, "bf16_context", None)
            if context is not None:
                context.__exit__(None, None, None)
                owner.bf16_context = None

    def close(self) -> None:
        self.gpu_ready = False
        if self.predictor is not None:
            self._release_builder_autocast()
            self.predictor = None
        if self._torch is not None:
            self._torch.cuda.empty_cache()


class VideoEngine:
    def __init__(
        self, settings: VideoSettings | None = None, backend: VideoBackend | None = None
    ):
        self.settings = settings or VideoSettings.from_env()
        self._production = backend is None
        self.backend = backend or NativeSam31(self.settings)
        self.ready = False
        self.error_code: str | None = None

    @property
    def gpu_ready(self) -> bool:
        return (
            self.settings.enabled
            and self._production
            and self.ready
            and getattr(self.backend, "gpu_ready", False) is True
        )

    def warmup(self) -> None:
        if self.settings.enabled:
            try:
                self.backend.warmup()
                self.ready = True
            except Exception:
                self.ready = False
                self.error_code = "model_initialization_failed"
                self.backend.close()
                raise

    def close(self) -> None:
        self.ready = False
        self.backend.close()

    def run(
        self, payload: bytes, *, deadline: float, decode_jpeg: Callable
    ) -> dict[str, Any]:
        batch = parse_batch(payload, self.settings)
        images = []
        total_pixels = 0
        source_size = None
        try:
            for frame in batch.frames:
                if time.monotonic() >= deadline:
                    raise TimeoutError()
                image = decode_jpeg(frame.jpeg_bytes, self.settings)
                try:
                    if source_size is not None and source_size != image.size:
                        raise HTTPException(400, "video batch frame dimensions changed")
                    source_size = image.size
                    total_pixels += image.width * image.height
                    if total_pixels > self.settings.max_total_image_pixels:
                        raise HTTPException(413, "video source pixel budget exceeded")
                    image.thumbnail((self.settings.inference_max_side,) * 2)
                    if images and image.size != images[0].size:
                        raise HTTPException(400, "video batch frame dimensions changed")
                    images.append(image)
                except BaseException:
                    image.close()
                    raise
            epoch = uuid.uuid4().hex
            snapshots = []
            outputs = self.backend.infer(images)
            try:
                for index, output in outputs:
                    if time.monotonic() >= deadline:
                        raise TimeoutError()
                    if (
                        type(index) is not int
                        or index != len(snapshots)
                        or index >= len(images)
                    ):
                        raise RuntimeError(
                            "native video frame order differs from source"
                        )
                    snapshots.append(
                        native_snapshot(
                            output,
                            batch=batch,
                            index=index,
                            epoch=epoch,
                            image_size=images[index].size,
                            max_objects=self.settings.max_objects,
                        )
                    )
                if len(snapshots) != len(images):
                    raise RuntimeError("native video omitted source frames")
            except TimeoutError:
                raise
            except Exception:
                self.ready = False
                self.error_code = "native_inference_failed"
                raise
            finally:
                if hasattr(outputs, "close"):
                    try:
                        outputs.close()
                    except Exception:
                        self.ready = False
                        self.error_code = "native_cleanup_failed"
                        raise
            if time.monotonic() >= deadline:
                raise TimeoutError()
            last = batch.frames[-1]
            return {
                "stream_session_id": batch.stream_session_id,
                "batch_id": batch.batch_id,
                "predictor_epoch": epoch,
                "backend": BACKEND,
                "memory_scope": "batch",
                "frames": snapshots,
                "completed_frame_sequence": last.frame_sequence,
                "completed_timestamp_seconds": last.timestamp_seconds,
            }
        finally:
            for image in images:
                image.close()


def register_video_routes(
    app: FastAPI, engine: VideoEngine, admission: threading.Lock, decode_jpeg: Callable
) -> None:
    app.state.video_engine = engine
    # Uvicorn does not configure the root logger. Keep this worker's phase
    # diagnostics visible without increasing third-party logging verbosity.
    if LOGGER.level == logging.NOTSET:
        LOGGER.setLevel(logging.INFO)
    if not LOGGER.hasHandlers():

        def needs_fallback(record):
            # SAM may configure an ancestor after our routes are registered.
            # Prefer that handler when it can emit this record; retain normal
            # propagation so test capture and configured application logs work.
            ancestor = LOGGER
            while ancestor.propagate and ancestor.parent is not None:
                ancestor = ancestor.parent
                if any(record.levelno >= item.level for item in ancestor.handlers):
                    return False
            return True

        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(levelname)s %(name)s %(message)s"))
        handler.addFilter(needs_fallback)
        LOGGER.addHandler(handler)

    @app.get("/health/video")
    async def health_video():
        return JSONResponse(
            {
                "ready": engine.gpu_ready,
                "gpu_ready": engine.gpu_ready,
                "enabled": engine.settings.enabled,
                "backend": BACKEND,
                "native_sam_video_tracking": True,
                "memory_scope": "batch",
                "sam3_commit": SAM3_COMMIT,
                "limits": engine.settings.limits(),
                "error_code": engine.error_code,
            },
            status_code=200 if engine.gpu_ready else 503,
        )

    @app.post("/v1/video/batches")
    async def video_batch(request: Request):
        if not engine.ready or (engine._production and not engine.gpu_ready):
            raise HTTPException(503, "native video inference unavailable")
        if not admission.acquire(blocking=False):
            raise HTTPException(429, "inference busy", headers={"Retry-After": "1"})
        session_engine = (
            getattr(app.state, "perception_sessions", None)
            or getattr(app.state, "session_engine", None)
        )
        if session_engine is not None and session_engine.has_active_session:
            admission.release()
            raise HTTPException(
                429, "video session active", headers={"Retry-After": "1"}
            )
        deadline = time.monotonic() + engine.settings.request_timeout_seconds
        started = time.perf_counter()
        stage = "receiving"
        received_bytes = 0
        task = None
        try:
            if (
                request.headers.get("content-type", "").split(";", 1)[0].lower()
                != "application/json"
            ):
                raise HTTPException(415, "expected application/json")
            content_length = request.headers.get("content-length")
            if content_length is not None:
                try:
                    length = int(content_length)
                except ValueError:
                    raise HTTPException(400, "invalid content length") from None
                if not 1 <= length <= engine.settings.max_request_bytes:
                    raise HTTPException(
                        413, "video request size outside allowed bounds"
                    )

            async def read_body():
                nonlocal received_bytes
                payload = bytearray()
                async for chunk in request.stream():
                    received_bytes += len(chunk)
                    if len(payload) + len(chunk) > engine.settings.max_request_bytes:
                        raise HTTPException(
                            413, "video request size outside allowed bounds"
                        )
                    payload.extend(chunk)
                return bytes(payload)

            payload = await asyncio.wait_for(
                read_body(), max(0.001, deadline - time.monotonic())
            )
            body_read_seconds = time.perf_counter() - started
            stage = "inference"
            inference_started = time.perf_counter()
            task = asyncio.create_task(
                asyncio.to_thread(
                    engine.run,
                    payload,
                    deadline=deadline,
                    decode_jpeg=decode_jpeg,
                )
            )
            result = await asyncio.wait_for(
                asyncio.shield(task), max(0.001, deadline - time.monotonic())
            )
            finished_at = time.perf_counter()
            # engine.run includes validation, JPEG decode, inference and cleanup.
            # No source identifiers, image bytes or model outputs enter logs.
            LOGGER.info(
                "Native video batch completed received_bytes=%d body_read_seconds=%.6f "
                "inference_cleanup_seconds=%.6f total_seconds=%.6f",
                received_bytes,
                body_read_seconds,
                finished_at - inference_started,
                finished_at - started,
            )
            return result
        except TimeoutError:
            LOGGER.warning(
                "Native video batch timed out stage=%s elapsed_seconds=%.6f "
                "received_bytes=%d",
                stage,
                time.perf_counter() - started,
                received_bytes,
            )
            raise HTTPException(
                504, "native video inference deadline exceeded"
            ) from None
        except HTTPException:
            raise
        except Exception:
            # Backend exception messages can include inputs; retain only the
            # failure category in the public worker log.
            LOGGER.error("Native video batch failed")
            raise HTTPException(503, "native video inference unavailable") from None
        finally:
            if task is not None and not task.done():

                def finished(background):
                    try:
                        background.result()
                    except (Exception, asyncio.CancelledError):
                        pass
                    admission.release()

                task.add_done_callback(finished)
            else:
                admission.release()
