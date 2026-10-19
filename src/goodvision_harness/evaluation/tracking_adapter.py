"""Adapt the count service to Harness evidence contracts without changing models or filtering boxes.

SAM retains its frozen native causal worker. Build completion watermarks from
frame receipts and reuse production sam_live metadata semantics. Keep ordinary
detectors distinct from SAM.
"""
import base64
import httpx

from ..perception import PerceptionSessionHTTPClient
from ..rich_labeling import RichLabelSnapshot
from ..video_tracking_session import VideoSessionTrackingResult


class CountProtocolTracker:
    def __init__(self, endpoint, *, model, sid, parameters, expected_weights_path=None):
        if model not in {"sam3.1", "yolo26-s", "yolo26-l", "rfdetr-l"}:
            raise ValueError("unsupported fixed model")
        self.model, self.sid, self.parameters = model, sid, parameters
        self.weights_path = expected_weights_path
        self.client = httpx.Client(timeout=75, trust_env=False, follow_redirects=False)
        self.endpoint = endpoint
        self.last_response, self.last_stamp, self.opened, self.drained = None, None, False, False
        self.parser = (PerceptionSessionHTTPClient(endpoint, method="sam_live", max_response_objects=256)
                       if model == "sam3.1" else None)

    def _request(self, operation, *, timeout, **extra):
        response = self.client.post(self.endpoint, json={"operation": operation, "model": self.model,
                                    "sid": self.sid, **extra}, timeout=timeout)
        response.raise_for_status()
        result = response.json()
        if result.get("sid") != self.sid or result.get("model") != self.model:
            raise ValueError("formal tracking response identity mismatch")
        self.last_response = result
        return result

    def open(self):
        result = self._request("open", timeout=300, detector_parameters=self.parameters)
        self.opened = True
        evidence = result["provenance"]
        if ("RTX 4090" not in evidence.get("actual_gpu_name", "")
                or not 23 * 1024**3 <= evidence.get("actual_gpu_memory_bytes", 0) <= 26 * 1024**3
                or evidence.get("sam_internal_patch") is not False
                or not evidence.get("actual_parameter_device", "").startswith("cuda")):
            raise ValueError("actual CUDA/standard4090/stockSAM mismatch")
        weight = evidence.get("weights", {})
        actual_path = weight.get("path") if isinstance(weight, dict) else None
        actual_path = actual_path or evidence.get("weights_path")
        if self.weights_path is not None and actual_path != self.weights_path:
            raise ValueError("actual weight path differs from configured model")
        if self.model == "sam3.1":
            if (evidence.get("dummy_warmup") is not False
                    or evidence.get("causal_tracking") is not True):
                raise ValueError("stock causal SAM source identity mismatch")
        elif evidence.get("parameters") != self.parameters:
            raise ValueError("actual detector parameters differ from locked configuration")
        return result

    def infer(self, data, *, stream_session_id, frame_sequence, timestamp_seconds):
        if not self.opened or self.drained or stream_session_id != self.sid:
            raise ValueError("tracking session not open")
        result = self._request("frame", timeout=75, frame_sequence=frame_sequence,
                               timestamp_seconds=timestamp_seconds,
                               jpeg_base64=base64.b64encode(data).decode("ascii"))
        if (result.get("frame_sequence") != frame_sequence
                or result.get("timestamp_seconds") != timestamp_seconds
                or not result.get("actual_inference_device", "").startswith("cuda")):
            raise ValueError("tracking frame identity or actual device mismatch")
        snapshot = RichLabelSnapshot.from_dict({"stream_session_id": self.sid,
                    "frame_sequence": frame_sequence, "timestamp_seconds": timestamp_seconds,
                    "objects": result["objects"]})
        self.last_stamp = (frame_sequence, timestamp_seconds)
        if self.model != "sam3.1":
            return snapshot
        for obj in snapshot.objects:
            native = obj.attributes.get("native_track", {})
            if (native.get("backend") != "sam3.1_video" or native.get("epoch") != self.sid
                    or obj.observation_state != "observed"):
                raise ValueError("causal native SAM frame provenance missing")
        audit = result["audit"]
        prepared = self.parser.prepare_snapshots({"route_metadata": {
            "route": "sam_live", "confirmation_policy": audit["confirmation_policy"],
            "observations": audit["observations"]}}, (snapshot,))
        # The frozen worker requires one output at the session's processed index, not acceptance alone.
        return VideoSessionTrackingResult(
            video_session_id=self.sid, stream_session_id=self.sid,
            request_id=f"{self.sid}:tracking:{frame_sequence}", predictor_epoch=self.sid,
            state="open", frames=tuple(prepared), accepted_frame_sequence=frame_sequence,
            accepted_timestamp_seconds=timestamp_seconds, processed_frame_sequence=frame_sequence,
            processed_timestamp_seconds=timestamp_seconds, emitted_frame_sequence=frame_sequence,
            emitted_timestamp_seconds=timestamp_seconds, pending_frame_count=0,
            memory_scope="session", backend="sam3.1_video")

    def finish(self):
        result = self._request("close", timeout=75)
        if (result.get("tracking_drained") is not True
                or tuple(result.get("last_stamp") or ()) != self.last_stamp):
            raise ValueError("actual final tracking drain mismatch")
        self.drained = True
        return result

    def discard(self):
        if self.opened and not self.drained:
            # End this SID without marking an incomplete source complete; resource recovery owns reset.
            self.finish()

    def close(self):
        try:
            self.client.close()
        finally:
            if self.parser:
                self.parser.close()
