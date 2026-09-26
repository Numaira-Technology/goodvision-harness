"""CUDA candidate worker; shared session protocol, no SAM model dependency."""

import argparse
import hashlib
import importlib.metadata
import json
import logging
import threading
import time
from collections import deque
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
from fastapi import FastAPI
from scipy.optimize import linear_sum_assignment

from tools.seg_benchmark.models import MODELS, Adapter, resolve_device, synchronize
from workers.sam_scene.app import decode_jpeg
from workers.sam_scene.continuous import (
    SessionEngine,
    SessionProfile,
    SessionSettings,
    register_session_routes,
)
from workers.sam_scene.sam31 import VideoSettings, native_snapshot

BACKEND = "seg_benchmark"
POLICY = "causal_iou_three_observations"
MIN_NATIVE_COMPONENT_PIXELS = 5


def has_serializable_component(mask):
    """Match native_snapshot's rule that components of four pixels are noise."""
    _, _, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8
    )
    return bool(
        len(stats) > 1
        and (stats[1:, cv2.CC_STAT_AREA] >= MIN_NATIVE_COMPONENT_PIXELS).any()
    )


class MaskTracker:
    """Conservative bbox-IoU association, real observed masks only; no ReID.

    The algorithm is identical for every candidate, but is not SAM's learned
    tracker. Report this difference rather than claiming tracking equivalence.
    Expiration uses source time, so dropped input frames do not prolong a track.
    """

    def __init__(self, iou=0.3, max_age=1.5):
        self.iou, self.max_age = iou, max_age
        self.tracks = {}
        self.next_id = 0

    def update(self, masks, timestamp):
        boxes = []
        for mask in masks:
            y, x = np.nonzero(mask)
            boxes.append((x.min(), y.min(), x.max() + 1, y.max() + 1))
        self.tracks = {
            key: value
            for key, value in self.tracks.items()
            if timestamp - value[1] <= self.max_age
        }
        old = list(self.tracks)
        overlaps = np.zeros((len(old), len(boxes)))
        for i, key in enumerate(old):
            a = self.tracks[key][0]
            for j, b in enumerate(boxes):
                intersection = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(
                    0, min(a[3], b[3]) - max(a[1], b[1])
                )
                union = (
                    (a[2] - a[0]) * (a[3] - a[1])
                    + (b[2] - b[0]) * (b[3] - b[1])
                    - intersection
                )
                overlaps[i, j] = intersection / union if union else 0
        # Prioritize valid match cardinality before aggregate IoU.
        cost = np.where(overlaps >= self.iou, 1 + overlaps / (len(old) + 1), 0)
        matched = {}
        if cost.size:
            rows, cols = linear_sum_assignment(-cost)
            matched = {
                j: old[i]
                for i, j in zip(rows, cols, strict=True)
                if overlaps[i, j] >= self.iou
            }
        ids, provisional = [], []
        for j, box in enumerate(boxes):
            key = matched.get(j)
            if key is None:
                key = self.next_id
                self.next_id += 1
                hits = 1
            else:
                hits = self.tracks[key][2] + 1
            self.tracks[key] = (box, timestamp, hits)
            ids.append(key)
            if hits < 3:
                provisional.append(str(key))
        return np.asarray(ids, dtype=np.int64), provisional


class CandidateBackend:
    def __init__(self, adapter, *, verified_cuda=False):
        self.adapter = adapter
        self.gpu_ready = False
        self.verified_cuda = verified_cuda
        self.timings = deque(maxlen=4096)

    def warmup(self):
        for _ in range(3):
            self.adapter.predict_scored(np.zeros((384, 640, 3), np.uint8))
        synchronize(self.adapter.device)
        self.gpu_ready = self.verified_cuda

    def run(self, session, images, *, final):
        if session.native_state is None:
            session.native_state = {"tracker": MaskTracker(), "observations": {}}
        state = session.native_state
        state["observations"].clear()
        for offset, image in enumerate(images):
            index = session.total_frames + offset
            source = session.sources[index]
            bgr = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
            synchronize(self.adapter.device)
            started = time.perf_counter()
            masks, scores = self.adapter.predict_scored(bgr)
            synchronize(self.adapter.device)
            inference_seconds = time.perf_counter() - started
            valid = (
                np.asarray(
                    [has_serializable_component(mask) for mask in masks], dtype=bool
                )
                if len(masks)
                else []
            )
            masks, scores = masks[valid], scores[valid]
            if len(masks) > 16:
                # Fail visibly rather than dropping people to improve coverage.
                raise ValueError("Candidate exceeded the baseline 16-person limit")
            ids, provisional = state["tracker"].update(masks, source.timestamp_seconds)
            state["observations"][source.frame_sequence] = provisional
            timing = {
                "stream_session_id": session.stream_session_id,
                "source_sequence": source.frame_sequence,
                "source_seconds": source.timestamp_seconds,
                "inference_seconds": inference_seconds,
                "inference_and_association_seconds": time.perf_counter() - started,
                "people": len(ids),
            }
            self.timings.append(timing)
            logging.info("candidate_frame %s", json.dumps(timing))
            yield (
                index,
                {"out_obj_ids": ids, "out_probs": scores, "out_binary_masks": masks},
            )

    @staticmethod
    def serialize_snapshot(output, **kwargs):
        result = native_snapshot(output, **kwargs)
        for obj in result["objects"]:
            native = obj["attributes"]["native_track"]
            native["backend"] = BACKEND
            obj["track_id"] = f"seg-{native['epoch']}-{native['object_id']}"
        return result

    @staticmethod
    def route_metadata(session, snapshots):
        return {
            "confirmation_policy": POLICY,
            "observations": [
                {
                    "frame_sequence": row["frame_sequence"],
                    "provisional_object_ids": session.native_state["observations"].get(
                        row["frame_sequence"], []
                    ),
                }
                for row in snapshots
            ],
        }

    def prune(self, session):
        pass

    def discard(self, session):
        session.native_state = None

    def close(self):
        self.gpu_ready = False


def make_engine(backend):
    video = SimpleNamespace(
        settings=VideoSettings(enabled=True),
        gpu_ready=backend.verified_cuda,
    )
    return SessionEngine(
        video,
        SessionSettings(enabled=True, h264_enabled=True),
        backend,
        profile=SessionProfile(BACKEND, BACKEND, native_sam_video_tracking=False),
        production=True,
    )


def create_app(backend, metadata):
    engine = make_engine(backend)

    @asynccontextmanager
    async def lifespan(app):
        engine.warmup()
        try:
            yield
        finally:
            engine.close()

    app = FastAPI(lifespan=lifespan)
    register_session_routes(
        app,
        engine,
        threading.Lock(),
        decode_jpeg,
        legacy=False,
        api_prefix=f"/v1/perception/{BACKEND}",
        health_path=f"/health/perception/{BACKEND}",
    )

    @app.get("/benchmark/info")
    async def info():
        return {
            **metadata,
            "ready": engine.gpu_ready,
            "backend": BACKEND,
            "protocol_limits": engine.limits(),
            "tracker": POLICY,
            "tracker_iou": 0.3,
            "tracker_max_age_seconds": 1.5,
            "native_sam_video_tracking": False,
        }

    @app.get("/benchmark/timings")
    async def timings():
        return {"frames": list(backend.timings), "capacity": 4096}

    return app


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=MODELS)
    parser.add_argument("--port", type=int, default=8092)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--confidence", type=float, default=0.25)
    args = parser.parse_args()
    if not 0 < args.confidence < 1:
        parser.error("confidence must be in (0,1)")
    from tools.seg_benchmark.__main__ import setup_cache

    setup_cache()
    device = resolve_device(args.device)
    if not device.startswith("cuda"):
        parser.error("The comparison worker requires CUDA; use the offline tool on Mac")
    import torch
    import uvicorn

    adapter = Adapter(
        args.model, device, Path(".models/seg-benchmark"), args.confidence
    )
    with adapter.weight.open("rb") as stream:
        weight_hash = hashlib.file_digest(stream, "sha256").hexdigest()
    metadata = {
        "model": args.model,
        "model_size": adapter.size,
        "confidence": args.confidence,
        "weight_sha256": weight_hash,
        "gpu": torch.cuda.get_device_name(device),
        "vram_bytes": torch.cuda.get_device_properties(device).total_memory,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "dependencies": {
            name: importlib.metadata.version(name)
            for name in ("rfdetr", "ultralytics", "numpy", "scipy")
        },
    }
    logging.basicConfig(level=logging.INFO)
    uvicorn.run(
        create_app(CandidateBackend(adapter, verified_cuda=True), metadata),
        host="127.0.0.1",
        port=args.port,
    )


if __name__ == "__main__":
    main()
