"""Frozen RF parameter comparison, loaded by real evaluation requests only."""
from __future__ import annotations
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import time
from pathlib import Path
from typing import Protocol
import numpy as np
from tracker import Detection, valid_pixel_bbox

MODEL_REGISTRY = {"yolo26-s": "yolo26s.pt", "yolo26-l": "yolo26l.pt", "rfdetr-l": "rf-detr-large.pth"}
CONFIDENCE = 0.10

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
        raise RuntimeError("GLOBAL_WEIGHT_HASH_MISMATCH")
    packages = {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()}
    return weight, {
        "detector_model": model, "weights": declared, "packages": packages,
        "python": platform.python_version(), "default_confidence": CONFIDENCE,
        "dummy_warmup": False, "requested_device": "cuda",
    }

class RealDetector:
    def __init__(self, key):
        self.key = key
        self.model = None
        self.provenance = {}
        self._first_completed = False
        self.parameters = {'imgsz':1288,'confidence':0.10,'max_det':300}

    def configure(self, parameters):
        candidate=dict(parameters or {})
        if (set(candidate)!={'imgsz','confidence','max_det'} or
                candidate['imgsz'] != 1288 or
                candidate['confidence'] != 0.10 or candidate['max_det']!=300):
            raise ValueError('GLOBAL_UNREGISTERED_DETECTOR_PARAMETERS')
        if self.key=='rfdetr-l' and candidate!={'imgsz':1288,'confidence':0.10,'max_det':300}:
            raise ValueError('GLOBAL_RF_PARAMETERS_CHANGED')
        self.parameters=candidate
        self.provenance['confidence']=candidate['confidence']
        if self.key.startswith('yolo26-'):
            self.provenance['input_size']=candidate['imgsz']

    def _load_for_real_frame(self):
        if self.model is not None:
            return
        weight, provenance = _metadata(self.key)
        import torch
        if self.key.startswith("yolo26-"):
            from ultralytics import YOLO
            from ultralytics.models.yolo.detect import DetectionPredictor
            class NoDummyPredictor(DetectionPredictor):
                def __init__(self, *args, **kwargs):
                    super().__init__(*args, **kwargs)
                    self.done_warmup = True
                def preprocess(self, im):
                    tensor = super().preprocess(im)
                    self.last_actual_tensor_shape = list(tensor.shape)
                    return tensor
            self.predictor_class = NoDummyPredictor
            model = YOLO(str(weight), task="detect")
            model.to("cuda:0")
            device = next(model.model.parameters()).device
            provenance["input_size"] = self.parameters['imgsz']
            provenance["person_class_id"] = 0
            provenance['end2end'] = bool(getattr(model.model,'end2end',False))
            provenance['detection_head_max_det'] = int(model.model.model[-1].max_det)
        else:
            from rfdetr import RFDETRLarge
            if importlib.metadata.version("rfdetr") != "1.3.0":
                raise RuntimeError("GLOBAL_RF_PACKAGE_VERSION_MISMATCH")
            model = RFDETRLarge(pretrain_weights=str(weight), device="cuda",
                               resolution=1288, num_select=300)
            if model.model.resolution != 1288 or model.model_config.num_select != 300:
                raise RuntimeError("GLOBAL_RF_EFFECTIVE_PARAMETERS_MISMATCH")
            device = next(model.model.model.parameters()).device
            provenance["input_size"] = model.model.resolution
            provenance["person_class_id"] = 1
            provenance["rfdetr_config"] = model.model_config.model_dump(mode="json")
        if device.type != "cuda":
            raise RuntimeError("GLOBAL_PARAMETERS_NOT_CUDA")
        provenance.update({
            "confidence": self.parameters['confidence'],
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
        candidates = []
        if self.key.startswith("yolo26-"):
            results = self.model.predict(
                image_bgr, predictor=self.predictor_class, imgsz=self.parameters['imgsz'],
                conf=self.parameters['confidence'], max_det=self.parameters['max_det'],
                classes=[0], device="cuda:0", verbose=False,
                save=False, half=False, compile=False,
            )
            actual_device = self.model.predictor.device
            first = next(iter(results), None)
            if first is not None and first.boxes is not None:
                for box in first.boxes:
                    coords = tuple(float(v) for v in box.xyxy[0].tolist())
                    score = float(box.conf[0])
                    candidates.append((0, score, coords))
        else:
            image_rgb = np.ascontiguousarray(image_bgr[:, :, ::-1])
            result = self.model.predict(image_rgb, threshold=self.parameters["confidence"])
            actual_device = self.model.model.device
            for index, box in enumerate(result.xyxy):
                coords = tuple(float(v) for v in box)
                score = float(result.confidence[index])
                candidates.append((int(result.class_id[index]), score, coords))
        stages = []
        person_class = 0 if self.key.startswith('yolo26-') else 1
        for category, score, coords in candidates:
            clipped, reason = None, None
            if category != person_class:
                reason = 'not_person'
            elif not math.isfinite(score) or score < self.parameters['confidence']:
                reason = 'confidence'
            elif not all(math.isfinite(v) for v in coords):
                reason = 'nonfinite_coordinates'
            elif coords[2] <= coords[0] or coords[3] <= coords[1]:
                reason = 'nonpositive_raw_area'
            else:
                clipped = (max(0.0, min(width, coords[0])), max(0.0, min(height, coords[1])),
                           max(0.0, min(width, coords[2])), max(0.0, min(height, coords[3])))
                if not valid_pixel_bbox(clipped, width, height):
                    reason = 'zero_area_after_clipping'
                else:
                    output.append(Detection('person', score, clipped))
            stages.append({'class_id': category, 'score': score if math.isfinite(score) else repr(score),
                'raw_xyxy': [v if math.isfinite(v) else repr(v) for v in coords], 'clipped_xyxy': clipped,
                'accepted': reason is None, 'rejection': reason,
                'clipped': clipped is not None and clipped != coords})
        self.last_audit = {'stage': 'sdk_output_before_worker_filter', 'candidates': stages,
            'parameters':self.parameters,'end2end':self.provenance.get('end2end'),
            'sdk_count': len(candidates), 'accepted_person_count': len(output),
            'native_sdk_filtering_earlier_than_this_stage_not_recovered': True,
            'candidate_capacity': 300, 'candidate_capacity_hit': len(candidates) >= 300}
        if self.key.startswith('yolo26-'):
            self.last_audit['actual_tensor_shape'] = self.model.predictor.last_actual_tensor_shape
            self.last_audit['predictor_effective_args'] = {k: getattr(self.model.predictor.args,k)
                for k in ('imgsz','conf','max_det','classes','half','rect')}
        else:
            self.last_audit['effective_parameters'] = {'confidence':self.parameters['confidence'],
                'native_resolution':self.model.model.resolution,
                'num_select':self.model.model_config.num_select,
                'capacity_semantics':'all_class_top_k_before_person_filter'}
        if torch.device(actual_device).type != "cuda":
            raise RuntimeError("GLOBAL_INFERENCE_NOT_CUDA")
        if not self._first_completed:
            _emit("first_real_inference_completed", {
                "detector_model": self.key, "actual_inference_device": str(actual_device),
                "weights_sha256": self.provenance["weights"]["sha256"],
                "frame_shape": list(image_bgr.shape), "person_count": len(output),
            })
            self._first_completed = True
        return tuple(output)

def build_detector(model_key):
    if model_key != "rfdetr-l":
        raise ValueError("Only the approved RF-DETR-L comparison is permitted")
    return RealDetector(model_key)


