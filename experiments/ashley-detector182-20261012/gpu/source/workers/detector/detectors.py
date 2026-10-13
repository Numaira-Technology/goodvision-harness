"""真实评测帧触发固定模型加载；无 health、dummy、warmup 或 CPU 回退。"""
from __future__ import annotations
import hashlib
import importlib.metadata
import json
import os
import platform
import time
from pathlib import Path
from typing import Protocol
import numpy as np
from tracker import Detection, valid_pixel_bbox

MODEL_REGISTRY = {"yolo26-s": "yolo26s.pt", "rfdetr-l": "rf-detr-large.pth"}
CONFIDENCE = 0.35

class Detector(Protocol):
    provenance: dict
    def detect(self, image_bgr: np.ndarray) -> tuple[Detection, ...]: ...

def _emit(event, values):
    record = {"event": event, "recorded_epoch": time.time(), **values}
    line = json.dumps(record, ensure_ascii=False, sort_keys=True)
    print("DETECTOR_RUNTIME " + line, flush=True)
    path = Path(os.environ.get("DETECTOR_RUNTIME_RECORD", "../records/runtime.jsonl"))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as output:
        output.write(line + "\n")
        output.flush()
        os.fsync(output.fileno())

def _metadata(model):
    directory = Path(os.environ.get("DETECTOR_WEIGHTS_DIR", "../models"))
    manifest = json.loads((directory / "weights.json").read_text())
    weight = directory / MODEL_REGISTRY[model]
    declared = manifest["models"][model]
    digest = hashlib.sha256()
    with weight.open("rb") as stream:
        while chunk := stream.read(4 * 1024 * 1024):
            digest.update(chunk)
    if digest.hexdigest() != declared["sha256"] or weight.stat().st_size != declared["bytes"]:
        raise RuntimeError("权重内容与本轮下载冻结记录不一致")
    packages = {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()}
    return weight, {
        "detector_model": model, "weights": declared, "packages": packages,
        "python": platform.python_version(), "confidence": CONFIDENCE,
        "dummy_warmup": False, "requested_device": "cuda",
    }

class RealDetector:
    def __init__(self, key):
        self.key = key
        self.model = None
        self.provenance = {}
        self._first_completed = False

    def _load_for_real_frame(self):
        if self.model is not None:
            return
        weight, provenance = _metadata(self.key)
        import torch
        if self.key == "yolo26-s":
            from ultralytics import YOLO
            from ultralytics.models.yolo.detect import DetectionPredictor
            class NoDummyPredictor(DetectionPredictor):
                def __init__(self, *args, **kwargs):
                    super().__init__(*args, **kwargs)
                    self.done_warmup = True
            self.predictor_class = NoDummyPredictor
            model = YOLO(str(weight), task="detect")
            model.to("cuda:0")
            device = next(model.model.parameters()).device
            provenance["input_size"] = 640
            provenance["person_class_id"] = 0
        else:
            from rfdetr import RFDETRLarge
            model = RFDETRLarge(pretrain_weights=str(weight), device="cuda")
            device = next(model.model.model.parameters()).device
            provenance["input_size"] = model.model.resolution
            provenance["person_class_id"] = 1
            provenance["rfdetr_config"] = model.model_config.model_dump(mode="json")
        if device.type != "cuda":
            raise RuntimeError("实际模型参数未放置CUDA，禁止继续")
        provenance.update({
            "actual_parameter_device": str(device), "torch_cuda_version": torch.version.cuda,
            "cuda_device_name": torch.cuda.get_device_name(device),
        })
        self.model = model
        self.provenance = provenance
        _emit("model_loaded_for_real_request", provenance)

    def detect(self, image_bgr):
        self._load_for_real_frame()
        import torch
        height, width = image_bgr.shape[:2]
        output = []
        if self.key == "yolo26-s":
            results = self.model.predict(
                image_bgr, predictor=self.predictor_class, imgsz=640,
                conf=CONFIDENCE, classes=[0], device="cuda:0", verbose=False,
                save=False, half=False, compile=False,
            )
            actual_device = self.model.predictor.device
            first = next(iter(results), None)
            if first is not None and first.boxes is not None:
                for box in first.boxes:
                    coords = tuple(float(v) for v in box.xyxy[0].tolist())
                    score = float(box.conf[0])
                    if score >= CONFIDENCE and valid_pixel_bbox(coords, width, height):
                        output.append(Detection("person", score, coords))
        else:
            image_rgb = np.ascontiguousarray(image_bgr[:, :, ::-1])
            result = self.model.predict(image_rgb, threshold=CONFIDENCE)
            actual_device = self.model.model.device
            for index, box in enumerate(result.xyxy):
                if int(result.class_id[index]) != 1:
                    continue
                coords = tuple(float(v) for v in box)
                score = float(result.confidence[index])
                if score >= CONFIDENCE and valid_pixel_bbox(coords, width, height):
                    output.append(Detection("person", score, coords))
        if torch.device(actual_device).type != "cuda":
            raise RuntimeError("实际推理设备不是CUDA")
        if not self._first_completed:
            _emit("first_real_inference_completed", {
                "detector_model": self.key, "actual_inference_device": str(actual_device),
                "weights_sha256": self.provenance["weights"]["sha256"],
                "frame_shape": list(image_bgr.shape), "person_count": len(output),
            })
            self._first_completed = True
        return tuple(output)

def build_detector(model_key):
    if model_key not in MODEL_REGISTRY:
        raise ValueError("只允许yolo26-s或rfdetr-l，禁止替换模型")
    return RealDetector(model_key)

