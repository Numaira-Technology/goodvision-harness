"""Bounded canvas adapters over the existing camera and inference runtimes.

Applied plans never rewrite legacy settings. A workflow start uses optimistic
preflight followed by a reserved admission; Stop invalidates that admission.
"""

from __future__ import annotations

import copy
import io
from uuid import uuid4

from PIL import Image

from .model_api import MissingAPIKeyError
from .privacy import (
    PUBLIC_ANALYSIS_ERROR,
    PUBLIC_FACE_ERROR,
    PUBLIC_MISSING_API_KEY_ERROR,
    public_safe_value,
)


class WorkflowRuntimeMixin:
    _workflow_plan = None
    _workflow_binding = None
    _workflow_summaries = None
    _workflow_test_manager = None
    _workflow_testing = False
    _workflow_starting = False
    _workflow_start_operation = None
    _workflow_operation = 0
    _workflow_binding_revision = 0
    _workflow_result_context = None
    _workflow_raw_session_id = None

    def configure_workflow(self, payload):
        from .workflows import _identifier, compile_graph

        if not isinstance(payload, dict) or set(payload) != {"binding"}:
            raise ValueError("workflow control requires a binding")
        binding = payload["binding"]
        plan = None
        if binding is not None:
            if not isinstance(binding, dict) or set(binding) != {
                "workflow_id",
                "group_id",
                "graph",
                "enabled",
                "block_enabled",
                "camera_override",
            }:
                raise ValueError("invalid workflow binding")
            _identifier(binding["workflow_id"], "workflow id")
            _identifier(binding["group_id"], "group id")
            plan = compile_graph(
                binding["graph"],
                enabled=binding["enabled"],
                block_enabled=binding["block_enabled"],
                camera_override=binding["camera_override"],
            )
        with self._control_lock:
            if self._processing_lifecycle_busy():
                raise RuntimeError("stop this camera before applying its workflow")
            self._workflow_operation += 1
            self._workflow_binding_revision += 1
            self._workflow_plan = copy.deepcopy(plan)
            self._workflow_binding = copy.deepcopy(binding)
            self._workflow_result_context = None
            self._workflow_summaries = None
            self._workflow_test_manager = None
        return {"configured": True, "managed": plan is not None}

    def _workflow_busy(self, *, include_pending_start=True):
        return (
            self._workflow_testing
            or (include_pending_start and self._workflow_starting)
            or self._workflow_raw_session_id is not None
            or any(
                manager is not None and manager.busy
                for manager in (
                    self._workflow_summaries,
                    self._workflow_test_manager,
                )
            )
        )

    def _workflow_choices(self):
        plan = self._workflow_plan
        return copy.deepcopy(
            self._analysis_settings
            if plan is None
            else {
                "features": plan["features"],
                "event_sensitivities": plan["event_sensitivities"],
            }
        )

    def _start_connected_with_workflow(self):
        with self._control_lock:
            if self._processing_lifecycle_busy():
                raise RuntimeError("stop the active session before starting analysis")
            self._workflow_operation += 1
            operation = self._workflow_operation
            revision = self._workflow_binding_revision
            config, dry_run, key = self.config, self.dry_run, self._aiclub_api_key
        # May warm native models; no workflow/service lock surrounds that work.
        self.prepare_analysis()
        admitted = False
        try:
            with self._control_lock:
                if (
                    operation != self._workflow_operation
                    or revision != self._workflow_binding_revision
                    or config is not self.config
                    or dry_run != self.dry_run
                    or key != self._aiclub_api_key
                    or self._closed
                ):
                    raise RuntimeError(
                        "analysis settings changed or start was cancelled; retry"
                    )
                if self._processing_lifecycle_busy():
                    raise RuntimeError("another analysis operation is already running")
                connection = self._video_connection
                if connection is None or not connection.status()["connected"]:
                    raise RuntimeError(
                        "video connection changed; reconnect before starting"
                    )
                self._workflow_starting = True
                self._workflow_start_operation = operation
                admitted = True
                choices = self._workflow_choices()
                layout = self._connection_camera_layout
                self._workflow_result_context = None
                self._workflow_summaries = None
            if choices["features"]:
                self.start_stream(
                    connection.stream_url,
                    camera_layout=layout,
                    _connection_id=connection.connection_id,
                    **choices,
                )
            with self._control_lock:
                if (
                    operation != self._workflow_operation
                    or self._video_connection is not connection
                    or not connection.status()["connected"]
                    or self._closed
                ):
                    raise RuntimeError(
                        "analysis start was cancelled or the source changed"
                    )
                if not choices["features"]:
                    with self._state_lock:
                        self._generation += 1
                        self._state["generation"] = self._generation
                session = self._session if choices["features"] else None
                session_id = (
                    session.stream_session_id if session is not None else uuid4().hex
                )
                self._workflow_result_context = {
                    "session_id": session_id,
                    "generation": self._generation,
                    "binding_revision": revision,
                    "connection_id": connection.connection_id,
                    "result_epoch": self.results.epoch,
                }
                self._start_workflow_summaries(connection, session_id=session_id)
                plan = self._workflow_plan
                if (
                    plan is not None
                    and plan["enabled"]
                    and not choices["features"]
                    and not plan["summaries"]
                ):
                    self._workflow_raw_session_id = session_id
                self._workflow_starting = False
            return self.status()
        except Exception:
            if admitted:
                self.stop_session()
            raise
        finally:
            if admitted:
                with self._control_lock:
                    if self._workflow_start_operation == operation:
                        self._workflow_starting = False
                        self._workflow_start_operation = None

    def _make_summary_manager(self, specs, connection, *, session_id=None):
        from .workflow_summary import SummaryManager

        return SummaryManager.from_config(
            specs,
            self.config,
            source_id=self.source_id,
            stream_session_id=session_id or uuid4().hex,
            generation=self._generation,
            connection=connection,
            api_key=self._aiclub_api_key or "",
            dry_run=self.dry_run,
        )

    def _start_workflow_summaries(self, connection, *, session_id=None):
        plan = self._workflow_plan
        if plan is None or not plan["summaries"]:
            self._workflow_summaries = None
            return
        previous = self._workflow_summaries
        if previous is not None and previous.busy:
            raise RuntimeError("wait for the previous summary worker to drain")
        manager = self._make_summary_manager(
            plan["summaries"], connection, session_id=session_id
        )
        self._workflow_summaries = manager
        manager.start()

    def _stop_workflow_summaries(self):
        # Stop cannot wait for network work while holding a service lock.
        with self._control_lock:
            self._workflow_operation += 1
            self._workflow_raw_session_id = None
            managers = (self._workflow_summaries, self._workflow_test_manager)
        for manager in managers:
            if manager is not None:
                manager.stop(timeout=0.2)

    def _workflow_status(self, payload):
        manager, test = self._workflow_summaries, self._workflow_test_manager
        if self._workflow_testing or (test is not None and test.busy):
            payload["pipeline_state"] = (
                "stopping"
                if test is not None and test.state == "stopping"
                else "starting"
            )
            payload["workflow_test_running"] = True
        elif self._workflow_starting:
            payload["pipeline_state"] = "starting"
        elif self._workflow_raw_session_id is not None or (
            manager is not None and manager.busy
        ):
            if payload.get("pipeline_state") in {
                None,
                "idle",
                "ended",
                "stopped",
                "error",
                "configuration_error",
            }:
                if payload.get("pipeline_state") in {"error", "configuration_error"}:
                    payload["native_pipeline_state"] = payload["pipeline_state"]
                payload["pipeline_state"] = (
                    "stopping"
                    if manager is not None and manager.state == "stopping"
                    else "running"
                )
        if self._workflow_plan is not None:
            payload["workflow"] = {
                "workflow_id": self._workflow_binding["workflow_id"],
                "group_id": self._workflow_binding["group_id"],
                "enabled": self._workflow_plan["enabled"],
                "generation": (self._workflow_result_context or {}).get("generation"),
                "stream_session_id": (self._workflow_result_context or {}).get(
                    "session_id"
                ),
                "summaries": manager.status() if manager is not None else None,
            }
        return payload

    def _workflow_context_valid(self, context):
        connection = self._video_connection
        return bool(
            context
            and context["binding_revision"] == self._workflow_binding_revision
            and connection is not None
            and context["connection_id"] == connection.connection_id
        )

    def workflow_summary_results(self, *, limit=100, node_id=None):
        if type(limit) is not int or not 1 <= limit <= 1000:
            raise ValueError("summary history limit must be within 1 through 1000")
        with self._control_lock:
            manager = (
                self._workflow_summaries
                if self._workflow_context_valid(self._workflow_result_context)
                else None
            )
        return {
            "camera_id": f"camera{self._fleet_camera_index:02d}",
            "status": manager.status() if manager else {"state": "idle"},
            "items": manager.history(node_id=node_id, limit=limit) if manager else [],
        }

    def _workflow_feature_results(self, labels, context):
        from .service import _feature_result_payload

        if (
            not self._workflow_context_valid(context)
            or context["generation"] != self._generation
            or self.results.epoch != context["result_epoch"]
        ):
            return []
        values = []
        for label in labels:
            record = self.results.latest(feature=label)
            if (
                record is None
                or record.payload.get("stream_session_id") != context["session_id"]
                or record.payload.get("source_id") != self.source_id
            ):
                continue
            value = _feature_result_payload(record.payload, label)
            if value is not None:
                # The generic record's aggregate description may describe a face
                # branch. This projection describes only its selected judgment.
                value["description"] = value["judgments"][0].get("description", "")
                values.append(value)
        return values

    def workflow_results(self):
        from .workflow_summary import snapshot_from_connection

        with self._control_lock:
            plan = copy.deepcopy(self._workflow_plan)
            context = copy.deepcopy(self._workflow_result_context)
            manager, connection = self._workflow_summaries, self._video_connection
        outputs = []
        if plan is not None:
            for output in plan["outputs"]:
                values = []
                for item in output["inputs"]:
                    port, value = item["port"], None
                    if (
                        port == "summary"
                        and manager is not None
                        and self._workflow_context_valid(context)
                    ):
                        value = manager.latest(node_id=item["node_id"])
                    elif (
                        port == "faces"
                        and self._workflow_context_valid(context)
                        and context["generation"] == self._generation
                    ):
                        fast = self.face_fast_results.snapshot()
                        if (
                            fast.get("stream_session_id") == context["session_id"]
                            and fast.get("generation") == context["generation"]
                            and fast.get("available")
                        ):
                            value = fast
                        else:
                            records = self._workflow_feature_results(
                                ("stranger",), context
                            )
                            value = (
                                {"node_id": item["node_id"], "results": records}
                                if records
                                else None
                            )
                    elif port == "events":
                        records = self._workflow_feature_results(
                            [
                                label
                                for label in plan["features"]
                                if label != "stranger"
                            ],
                            context,
                        )
                        value = (
                            {"node_id": item["node_id"], "results": records}
                            if records
                            else None
                        )
                    elif port == "frames" and connection is not None:
                        frame = snapshot_from_connection(connection)
                        if frame is not None:
                            value = {
                                "jpeg_url": "/v1/video/source/latest.jpg",
                                "connection_id": frame.connection_id,
                                "source_frame_sequence": frame.source_frame_sequence,
                                "timestamp_seconds": frame.timestamp_seconds,
                            }
                    values.append({**item, "value": value})
                outputs.append({"node_id": output["node_id"], "inputs": values})
        return public_safe_value(
            {
                "camera_id": f"camera{self._fleet_camera_index:02d}",
                "managed": plan is not None,
                "outputs": outputs,
            }
        )

    @staticmethod
    def _workflow_test_response(response, *, node_id, identity, context):
        classification = response.classification
        judgments = getattr(classification, "judgments", ()) if classification else ()
        result = {
            **context,
            "node_id": node_id,
            "type": "face" if identity else "events",
            "error": (PUBLIC_FACE_ERROR if identity else PUBLIC_ANALYSIS_ERROR)
            if response.error
            else None,
            "observations": [
                {
                    "event_label": item.event_label,
                    "detected": item.detected,
                    "confidence": item.confidence,
                    "description": item.description,
                }
                for item in judgments
                if (item.event_label == "stranger") == identity
            ],
        }
        if identity:
            faces = (
                (
                    getattr(classification, "face_observations", ())
                    or getattr(classification, "objects", ())
                )
                if classification
                else ()
            )
            result["faces"] = [
                {
                    "label": item.label,
                    "bbox": item.bbox.as_dict(),
                    "confidence": item.confidence,
                    "attributes": dict(item.attributes),
                }
                for item in faces
            ]
        return result

    def test_workflow(self, graph):
        """Real one-frame observations; never write live result/history stores."""
        from .face_recognition import FaceRecognitionSession
        from .grounding import build_selected_features_grounding_client
        from .sliding_window import FrameWindow, SampledFrame
        from .workflow_summary import snapshot_from_connection
        from .workflows import compile_graph

        plan = compile_graph(graph)
        with self._control_lock:
            if self._processing_lifecycle_busy():
                raise RuntimeError("stop this camera before testing a draft")
            connection = self._video_connection
            if connection is None or not connection.status()["connected"]:
                raise RuntimeError(
                    "connect a camera and wait for its preview before testing"
                )
            snapshot = snapshot_from_connection(connection)
            if snapshot is None:
                raise RuntimeError("wait for a camera frame before testing")
            self._workflow_operation += 1
            operation = self._workflow_operation
            self._workflow_testing = True
        face, manager = None, None
        session_id = uuid4().hex
        context = {
            "source_id": self.source_id,
            "stream_session_id": session_id,
            "generation": self._generation,
            "connection_id": snapshot.connection_id,
            "source_frame_sequence": snapshot.source_frame_sequence,
            "timestamp_seconds": snapshot.timestamp_seconds,
            "synthetic": self.dry_run,
            "test": True,
        }

        def check_active():
            with self._control_lock:
                if operation != self._workflow_operation or self._closed:
                    raise RuntimeError("workflow test was cancelled")

        try:
            with Image.open(io.BytesIO(snapshot.jpeg_bytes)) as image:
                width, height = image.size
            window = FrameWindow(
                0,
                (
                    SampledFrame(
                        0,
                        snapshot.source_frame_sequence,
                        snapshot.timestamp_seconds,
                        snapshot.jpeg_bytes,
                        width,
                        height,
                    ),
                ),
            )
            results = []
            for node_id, identity in (
                (plan["face_node_id"], True),
                (plan["events_node_id"], False),
            ):
                if not node_id:
                    continue
                check_active()
                try:
                    if identity and not self.dry_run:
                        face = FaceRecognitionSession(self._ensure_face_runtime())
                    features = (
                        ("stranger",)
                        if identity
                        else tuple(
                            label for label in plan["features"] if label != "stranger"
                        )
                    )
                    client = build_selected_features_grounding_client(
                        self.config.inference,
                        self.config.model_api,
                        features=features,
                        stranger_client=face,
                        dry_run=self.dry_run,
                        api_key=self._aiclub_api_key,
                        event_sensitivities=plan["event_sensitivities"],
                    )
                    response = (
                        client.classify_stranger_window(window)
                        if identity
                        else client.classify_safety_window(window)
                    )
                    result = self._workflow_test_response(
                        response, node_id=node_id, identity=identity, context=context
                    )
                except Exception:
                    result = {
                        **context,
                        "node_id": node_id,
                        "type": "face" if identity else "events",
                        "error": PUBLIC_FACE_ERROR
                        if identity
                        else PUBLIC_ANALYSIS_ERROR,
                        "observations": [],
                    }
                    if identity:
                        result["faces"] = []
                results.append(result)
            check_active()
            if plan["summaries"]:
                preparation_error = None
                try:
                    with self._control_lock:
                        check_active()
                        manager = self._make_summary_manager(
                            plan["summaries"], connection, session_id=session_id
                        )
                        self._workflow_test_manager = manager
                        manager.start(scheduled=False)
                except Exception as error:
                    check_active()
                    preparation_error = {
                        "code": "missing_api_key"
                        if isinstance(error, MissingAPIKeyError)
                        else "request_failed",
                        "message": PUBLIC_MISSING_API_KEY_ERROR
                        if isinstance(error, MissingAPIKeyError)
                        else "Summary inference could not be prepared.",
                    }
                for spec in plan["summaries"]:
                    check_active()
                    if preparation_error is None:
                        result = manager.run_once(spec, snapshot, start_if_idle=False)
                    else:
                        result = {
                            **context,
                            "type": "workflow_summary",
                            "schema_version": 1,
                            "node_id": spec["id"],
                            "status": "error",
                            "summary": None,
                            "error": preparation_error,
                        }
                    results.append(
                        {"node_id": spec["id"], "type": "summary", "result": result}
                    )
            check_active()
            return public_safe_value(
                {
                    **context,
                    "dry_run": self.dry_run,
                    "camera_id": f"camera{self._fleet_camera_index:02d}",
                    "results": results,
                    "note": (
                        "Single-frame observations; tracking, temporal confirmation "
                        "and delivery are exercised by live analysis."
                    ),
                }
            )
        finally:
            try:
                if face is not None:
                    face.close(wait=True)
            finally:
                if manager is not None:
                    manager.stop(timeout=0.2)
                with self._control_lock:
                    self._workflow_testing = False
                    if (
                        manager is not None
                        and not manager.busy
                        and self._workflow_test_manager is manager
                    ):
                        self._workflow_test_manager = None
