"""正式计数服务到最新Harness证据契约的薄适配，不改变模型或过滤框。

SAM仍运行已冻结的原生因果worker。这里仅根据真实逐帧回执构造完成水位，
并复用生产sam_live的确认元数据解释；不把普通检测器伪装成SAM。
"""
import base64
import hashlib
import httpx

from ..perception import PerceptionSessionHTTPClient
from ..rich_labeling import RichLabelSnapshot
from ..video_tracking_session import VideoSessionTrackingResult


class CountProtocolTracker:
    def __init__(self, endpoint, *, model, sid, parameters, expected_weights_sha256,
                 expected_sam_runtime_sources=None):
        if model not in {"sam3.1", "yolo26-s", "yolo26-l", "rfdetr-l"}:
            raise ValueError("unsupported fixed model")
        self.model, self.sid, self.parameters = model, sid, parameters
        self.weights_sha256, self.sam_sources = expected_weights_sha256, expected_sam_runtime_sources
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
        # 两种旧worker的权重元数据字段保留原样，只核对已冻结的SHA。
        weight = evidence.get("weights", {})
        actual_sha = weight.get("sha256") if isinstance(weight, dict) else None
        actual_sha = actual_sha or evidence.get("weight_sha256") or evidence.get("weights_sha256")
        if actual_sha != self.weights_sha256:
            raise ValueError("actual weight differs from frozen model")
        if self.model == "sam3.1":
            if (not self.sam_sources or evidence.get("sam_runtime_sources") != self.sam_sources
                    or evidence.get("dummy_warmup") is not False
                    or evidence.get("causal_tracking") is not True):
                raise ValueError("stock causal SAM source identity mismatch")
        elif evidence.get("parameters") != self.parameters:
            raise ValueError("actual detector parameters differ from locked configuration")
        return result

    def infer(self, data, *, stream_session_id, frame_sequence, timestamp_seconds):
        if not self.opened or self.drained or stream_session_id != self.sid:
            raise ValueError("tracking session not open")
        digest = hashlib.sha256(data).hexdigest()
        result = self._request("frame", timeout=75, frame_sequence=frame_sequence,
                               timestamp_seconds=timestamp_seconds, jpeg_sha256=digest,
                               jpeg_base64=base64.b64encode(data).decode("ascii"))
        if (result.get("jpeg_sha256") != digest or result.get("frame_sequence") != frame_sequence
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
        # 冻结worker强制len(outputs)==1且index==本会话已处理数；非accepted-only。
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
            # 只是结束本SID，并不把不完整源标为完成；失败后由资源层reset。
            self.finish()

    def close(self):
        try:
            self.client.close()
        finally:
            if self.parser:
                self.parser.close()
