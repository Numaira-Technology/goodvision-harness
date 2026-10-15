"""正式观察的持久化出口；不调用模型、不读题目、不重新做关联。

先保存原始记录再适配；逐条文件不可覆盖，IO/身份错误必须交给控制器处理。
不能用有限ResultBus历史或最后scene快照替代本出口的完整序列。
"""
from __future__ import annotations

import copy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile
from threading import RLock
import time

from .evidence import EvidenceCapture
from .privacy import reject_private


def canonical_bytes(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True,
                       separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")


def publish_once(path, value):
    """原子发布且拒绝覆盖；目录由调用方明确创建。"""
    path = Path(path)
    raw = canonical_bytes(value)
    fd, temporary = tempfile.mkstemp(prefix=".publishing-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        os.unlink(temporary)
    return hashlib.sha256(raw).hexdigest()


class ProductionEvidenceRecorder:
    """每个新SID一个出口；调用者持有模型、资源和实际EOF/排空责任。"""

    def __init__(self, directory, *, video_id, stream_session_id, source_id, source,
                 frame_plan, window_plan, provenance):
        manifest = dict(video_id=video_id, stream_session_id=stream_session_id,
                        source_id=source_id, source=source, frame_plan=frame_plan,
                        window_plan=window_plan, provenance=provenance)
        reject_private(manifest, observation=True)
        self.capture = EvidenceCapture(**manifest)
        self.root = Path(directory).resolve()
        self.root.mkdir(parents=True, exist_ok=False)
        (self.root / "records").mkdir()
        self._lock = RLock()
        self._sequence = 0
        self._previous_sha256 = None
        self._poisoned = False
        self._closed = False
        self.clock_domain = f"{stream_session_id}:pid-{os.getpid()}:monotonic"
        publish_once(self.root / "manifest.json", {
            "schema_version": "production-evidence-recording-v1", **manifest,
            "clock_domain": self.clock_domain, "face_enabled": False,
            "question_conditioned": False,
        })
        self._window_keys = {}
        for window in self.capture.window_plan:
            key = (float(window["start"]), float(window["end"]))
            if key in self._window_keys:
                raise ValueError("ambiguous planned window source interval")
            self._window_keys[key] = window["window_id"]

    def _open(self):
        if self._closed or self._poisoned:
            raise RuntimeError("recording is sealed or failed; use a new independent attempt")

    def _record(self, kind, payload):
        self._open()
        self._sequence += 1
        relative = f"records/{self._sequence:08d}-{kind}.json"
        row = {"sequence": self._sequence, "kind": kind,
               "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
               "recorded_monotonic": time.monotonic(), "clock_domain": self.clock_domain,
               "previous_record_sha256": self._previous_sha256, "payload": payload}
        digest = publish_once(self.root / relative, row)
        self._previous_sha256 = digest
        return str(self.root / relative), self._sequence

    def _apply(self, kind, payload, operation):
        with self._lock:
            self._open()
            try:
                reference, cursor = self._record(kind, payload)
                return operation(reference, cursor)
            except Exception:
                # 原始记录即使不合契约也保留。禁止继续使用已部分更新的capture。
                self._poisoned = True
                raise

    def tracking_frame(self, *, frame, timestamp_seconds, subjects, jpeg_sha256,
                       raw_response):
        payload = dict(frame=frame, timestamp_seconds=timestamp_seconds, subjects=subjects,
                       jpeg_sha256=jpeg_sha256, raw_response=raw_response)
        return self._apply("tracking", payload, lambda reference, cursor:
                           self.capture.tracking_frame(
                               frame=frame, timestamp_seconds=timestamp_seconds,
                               subjects=subjects, jpeg_sha256=jpeg_sha256, raw_reference=reference))

    def independent_report(self, report):
        """接IndependentVLMBranch的发布后observer，information/grounding都记录。"""
        def adapt(reference, cursor):
            stamps = [float(frame["timestamp_seconds"]) for frame in report.get("evidence", [])]
            if not stamps:
                raise ValueError("published report has no source frames")
            key = (min(stamps), max(stamps))
            if key not in self._window_keys:
                raise ValueError("published report is outside preregistered window plan")
            self.capture.independent_report(report, window_id=self._window_keys[key],
                                            raw_reference=reference, publication_cursor=cursor)
        return self._apply("independent", copy.deepcopy(report), adapt)

    def rejected_report(self, report, *, reason):
        """身份不符的原始报告单独留档，不发布成有效观察，也不再使用该capture。"""
        def reject(reference, cursor):
            self.capture.request_failure(stage="identity", reference=reference, reason=reason)
            self._poisoned = True
        return self._apply("rejected-independent", {"reason": reason, "report": copy.deepcopy(report)}, reject)

    def scene_snapshot(self, snapshot, *, association_links):
        return self._apply("scene", {"snapshot": snapshot, "association_links": association_links},
                           lambda reference, cursor: self.capture.scene_snapshot(
                               snapshot, association_links=association_links, raw_reference=reference))

    def request_failure(self, *, stage, reason, detail=None):
        return self._apply("failure", dict(stage=stage, reason=reason, detail=detail),
                           lambda reference, cursor: self.capture.request_failure(
                               stage=stage, reference=reference, reason=reason))

    def timing(self, row):
        if row.get("clock_domain") != self.clock_domain:
            raise ValueError("timing must use this recorder's client clock domain")
        return self._apply("timing", row, lambda reference, cursor: reference)

    def operational_record(self, kind, payload):
        """正式入口的请求元数据/会话回执；不得携带凭据、题干或私有答案。"""
        if kind not in {"tracking-request", "tracking-session", "tracking-receipt", "source-eof"}:
            raise ValueError("unknown operational record kind")
        reject_private(payload, observation=True)
        return self._apply(kind, payload, lambda reference, cursor: reference)

    def finish(self, *, source_eof, tracking_drained, events_drained, fusion_drained):
        """仅用调用者实测的终态，未完成可封存但不得伪造true。"""
        states = dict(source_eof=source_eof, tracking_drained=tracking_drained,
                      events_drained=events_drained, fusion_drained=fusion_drained)
        def seal(reference, cursor):
            evidence = self.capture.seal(**states)
            capture_sha = publish_once(self.root / "capture.json", evidence)
            publish_once(self.root / "completion.json", {
                "schema_version": "production-evidence-completion-v1", **states,
                "video_id": self.capture.video_id, "stream_session_id": self.capture.session_id,
                "records": self._sequence, "last_record_sha256": self._previous_sha256,
                "capture_file_sha256": capture_sha, "evidence_sha256": evidence["evidence_sha256"],
                "manifest_complete": evidence["observation_manifest"]["complete"],
                "failures": len(evidence["failures"]),
            })
            self._closed = True
            return evidence
        return self._apply("completion", states, seal)
