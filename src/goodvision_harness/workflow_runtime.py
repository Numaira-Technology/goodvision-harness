"""Bounded canvas adapters over the existing camera and inference runtimes.

Applied plans never rewrite legacy settings. A workflow start uses optimistic
preflight followed by a reserved admission; Stop invalidates that admission.
"""

from __future__ import annotations

import copy
import io
import time
from collections.abc import Mapping
from dataclasses import replace
from uuid import uuid4

from PIL import Image

from .model_api import MissingAPIKeyError, inference_error_detail
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
    _workflow_tracking_baseline = None
    _workflow_tracking_engine = None
    _workflow_event_baseline = None

    def _workflow_summary_configuration(self):
        return self.config

    def _workflow_event_configuration(self, configuration=None):
        from .model_api import ModelAPISettings
        from .workflow_modules import MODELS, native_endpoints

        configuration = configuration or self.config
        plan = self._workflow_plan or {}
        selected = plan.get("events_engine")
        if selected not in MODELS:
            return configuration
        return replace(
            configuration,
            inference=replace(
                configuration.inference, model=MODELS[selected], request_parameters={}
            ),
            aiclub=ModelAPISettings(
                provider="local",
                endpoint=native_endpoints(plan["events_service_url"])[0],
                max_retries=0,
            ),
        )

    def _workflow_bbox_probe(self, endpoint):
        if not endpoint or self.config is None:
            raise ValueError("Configure the Bbox service URL before starting.")
        from .rich_label_models import DETECTOR_MODEL_HEADER

        rich = replace(self.config.rich_labeling, endpoint=endpoint)
        probe = self._create_rich_http_client(
            rich, headers={DETECTOR_MODEL_HEADER: "yolo26-s"}
        )
        try:
            if not probe.gpu_ready():
                raise ValueError("The Bbox service is unavailable.")
        finally:
            probe.close()

    @staticmethod
    def _workflow_event_probe(plan):
        import httpx

        from .workflow_modules import MODELS, native_endpoints

        endpoint = plan.get("events_service_url")
        if not endpoint:
            raise ValueError("Configure the recognition service URL before starting.")
        try:
            with httpx.Client(
                timeout=5, trust_env=False, follow_redirects=False
            ) as client:
                response = client.get(native_endpoints(endpoint)[1])
                response.raise_for_status()
                value = response.json()
            if (
                not isinstance(value, dict)
                or value.get("model") != MODELS[plan["events_engine"]]
                or value.get("ready") is not True
            ):
                raise ValueError("model unavailable")
        except Exception:
            raise ValueError(
                "The recognition service is unavailable or has a different model."
            ) from None

    def workflow_tracking_capability(self):
        baseline = self._workflow_tracking_baseline
        rich = (
            baseline["rich_labeling"]
            if baseline
            else (self.config.rich_labeling if self.config is not None else None)
        )
        configured = rich is not None and rich.backend == "sam31_stream"
        return {
            "configured": configured,
            "processing_method": "sam_live",
            "reason": None
            if configured
            else "Person tracking is not configured for this deployment.",
        }

    @staticmethod
    def _workflow_tracking_configuration(configuration, plan, baseline):
        """Change only runtime tracking policy, retaining all inference settings."""
        rich = baseline["rich_labeling"]
        scene_mode = baseline["scene_mode"]
        if plan is not None:
            bbox = plan.get("tracking_engine") == "bbox"
            if bbox and plan["tracking_enabled"]:
                endpoint = plan.get("tracking_service_url")
                if not endpoint:
                    raise ValueError("Configure the Bbox service URL before starting.")
                rich = replace(
                    rich, backend="paddle", endpoint=endpoint, model="yolo26-s"
                )
            tracking = plan["tracking_enabled"] and (
                bbox or rich.backend == "sam31_stream"
            )
            rich = replace(
                rich,
                enabled=False,
                tracking_enabled=tracking,
                processing_method="sam_live" if tracking and not bbox else None,
            )
            if tracking:
                scene_mode = "active"
            elif not configuration.runtime.scene_independent_vlm:
                # Native safety inference remains available with tracking off.
                scene_mode = "disabled"
        runtime = replace(configuration.runtime, scene_mode=scene_mode)
        if (plan or {}).get("events_engine") in {"system1", "neural"}:
            runtime = replace(
                runtime,
                scene_mode="active",
                scene_vlm_output_mode="events",
                scene_independent_vlm=True,
            )
        return replace(configuration, rich_labeling=rich, runtime=runtime)

    def _set_workflow_tracking(self, plan):
        if self.config is None:
            if plan is not None and plan["tracking_enabled"]:
                raise ValueError(self.workflow_tracking_capability()["reason"])
            return
        baseline = self._workflow_tracking_baseline or {
            "rich_labeling": self.config.rich_labeling,
            "scene_mode": self.config.runtime.scene_mode,
        }
        if (
            plan is not None
            and plan["tracking_enabled"]
            and plan.get("tracking_engine") != "bbox"
        ):
            capability = self.workflow_tracking_capability()
            if not capability["configured"]:
                raise ValueError(capability["reason"])
        configuration = self.config
        event_baseline = self._workflow_event_baseline
        if event_baseline is not None:
            configuration = replace(
                configuration,
                runtime=replace(
                    configuration.runtime,
                    scene_vlm_output_mode=event_baseline[0],
                    scene_independent_vlm=event_baseline[1],
                ),
            )
        selected = (plan or {}).get("events_engine", "vlm")
        if selected in {"system1", "neural"}:
            if not plan.get("events_service_url"):
                raise ValueError(
                    "Configure the recognition service URL before starting."
                )
            event_baseline = (
                configuration.runtime.scene_vlm_output_mode,
                configuration.runtime.scene_independent_vlm,
            )
        configuration = self._workflow_tracking_configuration(
            configuration, plan, baseline
        )
        # Client construction can precede installing the new plan. Publish only
        # the selected tracking engine here, so its first probe uses that engine.
        previous_engine = self._workflow_tracking_engine
        self._workflow_tracking_engine = (plan or {}).get("tracking_engine")
        try:
            if configuration != self.config:
                self._replace_processing_method(
                    configuration.rich_labeling.processing_method,
                    config=configuration,
                    suppress_remote_config=True,
                )
        except Exception:
            self._workflow_tracking_engine = previous_engine
            raise
        self._workflow_tracking_baseline = baseline if plan is not None else None
        self._workflow_event_baseline = (
            event_baseline if selected in {"system1", "neural"} else None
        )

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
                camera_id=f"camera{self._fleet_camera_index:02d}",
                enabled=binding["enabled"],
                block_enabled=binding["block_enabled"],
                camera_override=binding["camera_override"],
                vocabulary=self._event_vocabulary(),
            )
        with self._control_lock:
            if self._processing_lifecycle_busy():
                raise RuntimeError("stop this camera before applying its workflow")
            self._set_workflow_tracking(plan)
            self._workflow_operation += 1
            self._workflow_binding_revision += 1
            self._workflow_plan = copy.deepcopy(plan)
            self._workflow_binding = copy.deepcopy(binding)
            self._workflow_result_context = None
            self._workflow_summaries = None
            self._workflow_test_manager = None
            operation = self._workflow_operation
            revision = self._rich_config_revision
            binding_revision = self._workflow_binding_revision
            retiring = self._retiring_rich_labels
        # Global Start installs bindings immediately before preflight. Finish
        # the old coordinator's retirement here so switching tracking off does
        # not require a second Start, without blocking status or cancellation.
        deadline = (
            time.monotonic()
            + max(5.0, min(60.0, self.config.rich_labeling.timeout_seconds + 2.0))
            if self.config is not None
            else time.monotonic() + 5.0
        )
        while retiring is not None:
            with self._control_lock:
                if (
                    operation != self._workflow_operation
                    or revision != self._rich_config_revision
                    or binding_revision != self._workflow_binding_revision
                    or self._closed
                ):
                    raise RuntimeError("workflow change was cancelled; retry")
                if self._retiring_rich_labels is not retiring:
                    break
            if time.monotonic() >= deadline:
                raise RuntimeError("previous tracking cleanup has not finished; retry")
            self._rich_probe_stop.wait(0.02)
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
            native = bool(
                choices["features"]
                or (
                    self._workflow_plan is not None
                    and self._workflow_plan["tracking_enabled"]
                )
            )
            if native:
                self.start_stream(
                    connection.stream_url,
                    camera_layout=layout,
                    _connection_id=connection.connection_id,
                    _tracking_only=not choices["features"],
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
                if not native:
                    with self._state_lock:
                        self._generation += 1
                        self._state["generation"] = self._generation
                session = self._session if native else None
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
                    and not plan["tracking_enabled"]
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

        configuration = self._workflow_summary_configuration()
        return SummaryManager.from_config(
            specs,
            configuration,
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
                "tracking_enabled": self._workflow_plan["tracking_enabled"],
                "tracking_node_id": self._workflow_plan["tracking_node_id"],
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
                    elif (
                        port == "tracks"
                        and self._workflow_context_valid(context)
                        and context["generation"] == self._generation
                    ):
                        metadata = self._camera_display_metadata.snapshot()
                        bbox = plan.get("tracking_engine") == "bbox"
                        if bbox:
                            latest = (
                                self._rich_labels.latest(context["session_id"])
                                if self._rich_labels
                                else None
                            )
                            metadata = (
                                latest.as_dict()
                                if latest
                                else {"stream_session_id": context["session_id"]}
                            )
                        if metadata.get("stream_session_id") == context["session_id"]:
                            value = {
                                "node_id": item["node_id"],
                                "type": "tracking",
                                "source_id": self.source_id,
                                "stream_session_id": context["session_id"],
                                "generation": context["generation"],
                                "processing_method": "bbox" if bbox else "sam_live",
                                "state": "running"
                                if metadata.get("anchors") or "objects" in metadata
                                else ("waiting_for_tracks"),
                                "metadata": metadata,
                            }
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
                outputs.append(
                    {
                        "node_id": output["node_id"],
                        "inputs": values,
                        "display": output.get("display", {}),
                    }
                )
        return public_safe_value(
            {
                "camera_id": f"camera{self._fleet_camera_index:02d}",
                "managed": plan is not None,
                "outputs": outputs,
            }
        )

    def workflow_output_preview(self, node_id, language="zh", *, workflow_id=None):
        from .workflow_render import render_output
        from .workflow_summary import SummarySnapshot, snapshot_from_connection

        with self._control_lock:
            if workflow_id is not None and workflow_id != (
                self._workflow_binding or {}
            ).get("workflow_id"):
                raise ValueError("This camera is running a different workflow.")
            connection, context = self._video_connection, self._workflow_result_context
            operation = self._workflow_operation
            generation = self._generation
            running = (
                self._workflow_raw_session_id is not None
                or (self._session is not None and self._session.running)
                or (
                    self._workflow_summaries is not None
                    and self._workflow_summaries.state == "running"
                )
            )
            if not running or not self._workflow_context_valid(context):
                raise ValueError("Start the applied workflow to preview its output.")
            snapshot = snapshot_from_connection(connection)
            if snapshot is None:
                raise ValueError("No video frame is available.")
            outputs = self.workflow_results()["outputs"]
            selected = next(
                (item for item in outputs if item["node_id"] == node_id), None
            )
            if selected is None:
                raise ValueError("Output is not active in this workflow.")
            for item in selected["inputs"]:
                if item["port"] == "tracks":
                    if item.get("value"):
                        item["value"] = item["value"].get("metadata")
                elif item["port"] == "summary" and item.get("value"):
                    spec = next(
                        (
                            s
                            for s in self._workflow_plan["summaries"]
                            if s["id"] == item["node_id"]
                        ),
                        {},
                    )
                    item["value"]["display_max_age"] = 2 * spec.get(
                        "interval_seconds", 60
                    )
            # Use the tracked source frame, never draw old geometry on a new person.
            for item in selected["inputs"]:
                if item["port"] != "tracks" or not item.get("value"):
                    continue
                value = item["value"]
                anchors = value.get("anchors", [])
                tracked_time = (
                    anchors[-1].get("media_timestamp")
                    if anchors
                    else value.get("timestamp_seconds")
                )
                tracked_sequence = (
                    anchors[-1].get("source_sequence")
                    if anchors
                    else value.get("frame_sequence")
                )
                # The history cursor excludes sequence zero. Include the current
                # snapshot explicitly so a stream's first frame can carry tracks.
                frames = (
                    *connection.display_frames(after=0, max_age_seconds=2, limit=120),
                    (
                        snapshot.source_frame_sequence,
                        snapshot.timestamp_seconds,
                        None,
                        None,
                        snapshot.jpeg_bytes,
                    ),
                )
                matched = next(
                    (
                        f
                        for f in reversed(frames)
                        if f[0] == tracked_sequence
                        and tracked_time is not None
                        and abs(f[1] - tracked_time) < 0.001
                    ),
                    None,
                )
                if matched and selected.get("display", {}).get("tracks", {}).get(
                    "visible", True
                ):
                    sequence, timestamp, _, _, jpeg = matched
                    snapshot = SummarySnapshot(
                        jpeg, sequence, timestamp, connection.connection_id
                    )
                else:
                    item["value"] = None
        rendered = render_output(
            snapshot.jpeg_bytes, selected, snapshot.timestamp_seconds, language=language
        )
        with self._control_lock:
            if (
                operation != self._workflow_operation
                or not self._workflow_context_valid(context)
                or generation != self._generation
            ):
                raise ValueError("The workflow changed; retry the preview.")
        return rendered

    @staticmethod
    def _workflow_test_response(response, *, node_id, identity, context):
        classification = response.classification
        judgments = getattr(classification, "judgments", ()) if classification else ()
        branch = "stranger" if identity else "safety"
        branch_errors = dict(getattr(response, "branch_errors", ()))
        failed = bool(response.error or branch in branch_errors)
        detail = (getattr(response, "branch_error_details", None) or {}).get(branch)
        detail = detail if isinstance(detail, Mapping) else {}
        # Rebuild the bounded message rather than forwarding provider text, even
        # when the branch already supplied diagnostics.
        error_detail = (
            inference_error_detail(
                code=detail.get("code", branch_errors.get(branch, "request_failed")),
                http_status=detail.get("http_status"),
            )
            if failed
            else None
        )
        result = {
            **context,
            "node_id": node_id,
            "type": "face" if identity else "events",
            "error": (PUBLIC_FACE_ERROR if identity else PUBLIC_ANALYSIS_ERROR)
            if failed
            else None,
            "error_detail": error_detail,
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

    def _workflow_native_event_test(self, plan, connection, snapshot, context, preview_frames):
        """Run the existing native inspector on distinct, already-captured preview frames."""
        from .model_api import ModelAPISettings
        from .neural_model import NeuralInspector
        from .runtime.scene import TemporalInspection
        from .runtime.scheduling import EvidenceFrame, InspectionRequest
        from .videojev_model import VideoJevInspector
        from .vlm_independent import VLMProcessConfig
        from .workflow_modules import MODELS, native_endpoints

        result = {
            **context,
            "node_id": plan["events_node_id"],
            "type": "events",
            "state": "live_analysis_required",
            "error": None,
            "observations": [],
        }
        inspector = None
        try:
            status = connection.status()
            if (not status.get("connected")
                    or status.get("connection_id") != snapshot.connection_id):
                raise RuntimeError("The preview connection changed; retry the workflow test.")
            candidates = {
                sequence: (sequence, timestamp, jpeg)
                for sequence, timestamp, _width, _height, jpeg in preview_frames
                if sequence < snapshot.source_frame_sequence
                and 0 <= timestamp < snapshot.timestamp_seconds
            }
            result["input"] = {
                "kind": "distinct_source_preview_frames",
                "source_frame_sequences": [snapshot.source_frame_sequence],
                "source_timestamps_seconds": [snapshot.timestamp_seconds],
            }
            if not candidates:
                result["note"] = (
                    "Only one distinct preview frame is available. Start live analysis "
                    "or wait for another source frame, then retry. No model request was sent."
                )
                return result
            # Match the native live route's half-second context when available;
            # retain the actual times even when the preview history is shorter.
            earlier = min(candidates.values(), key=lambda item: (
                abs(item[1] - (snapshot.timestamp_seconds - 0.5)), -item[0],
            ))
            selected = (earlier, (
                snapshot.source_frame_sequence, snapshot.timestamp_seconds, snapshot.jpeg_bytes,
            ))
            result["input"].update(
                source_frame_sequences=[item[0] for item in selected],
                source_timestamps_seconds=[item[1] for item in selected],
                model_frame_usage=(
                    "base labels: last preview frame; PPE derived: both supplied source frames"
                    if plan["events_engine"] == "neural"
                    else "last preview frame with the earlier frame as context"
                ),
            )
            if self.dry_run:
                result["note"] = (
                    "Synthetic test mode does not run the native recognition model. "
                    "Use real analysis mode to test the configured service."
                )
                return result
            endpoint = plan.get("events_service_url")
            if not endpoint:
                raise ValueError("Configure the recognition service URL before testing.")
            labels = tuple(label for label in plan["features"] if label != "stranger")
            delta = selected[-1][1] - selected[0][1]
            settings = ModelAPISettings(
                provider="local", endpoint=native_endpoints(endpoint)[0],
                timeout_seconds=self.config.runtime.scene_timeout_seconds, max_retries=0,
            )
            config = VLMProcessConfig(
                context["stream_session_id"], self.source_id, self._generation,
                labels, settings, MODELS[plan["events_engine"]],
                sensitivities=tuple(plan["event_sensitivities"].items()),
                output_mode="events", observation_window_frames=2,
                observation_frame_delta_seconds=delta,
                vocabulary=self._event_vocabulary(),
            )
            frames = tuple(EvidenceFrame(
                context["stream_session_id"], sequence, timestamp, jpeg,
            ) for sequence, timestamp, jpeg in selected)
            job = TemporalInspection(
                context["stream_session_id"], self.source_id,
                InspectionRequest(
                    f"preview-{self._workflow_operation}", (), "workflow_preview", labels,
                    frames[0].timestamp_seconds, frames[-1].timestamp_seconds,
                ),
                frames, event_sensitivities=config.sensitivities,
                vocabulary=config.vocabulary,
            )
            cls = NeuralInspector if plan["events_engine"] == "neural" else VideoJevInspector
            inspector = cls(config)
            verdicts = inspector.inspect(job)
            result.update(
                state="completed",
                observations=[{
                    "event_label": verdict.event_label,
                    "detected": verdict.detected,
                    "confidence": verdict.confidence,
                } for verdict in verdicts],
                note=(
                    "Actual native recognition on the listed preview evidence. "
                    "This does not exercise tracking, temporal confirmation or delivery."
                ),
            )
        except Exception as error:
            result.update(
                state="unavailable", error=PUBLIC_ANALYSIS_ERROR,
                error_detail=inference_error_detail(error),
            )
        finally:
            if inspector is not None:
                result["transport"] = dict(inspector.last_http_timing)
                try:
                    inspector.close()
                except Exception as error:
                    result["cleanup_error"] = inference_error_detail(error)
        return result

    def test_workflow(self, graph):
        """Use real preview observations; never write live result/history stores."""
        from .face_recognition import FaceRecognitionSession
        from .grounding import build_selected_features_grounding_client
        from .sliding_window import FrameWindow, SampledFrame
        from .workflow_summary import snapshot_from_connection
        from .workflows import compile_graph

        plan = compile_graph(
            graph,
            camera_id=f"camera{self._fleet_camera_index:02d}",
            vocabulary=self._event_vocabulary(),
        )
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
            # Pin the native evidence before face or tracking work can age it
            # out of the live connection's presentation history.
            native_preview_frames = (
                connection.display_frames(after=None, max_age_seconds=2, limit=120)
                if plan.get("events_engine") in {"system1", "neural"} else ()
            )
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
            if plan["tracking_enabled"]:
                bbox = plan.get("tracking_engine") == "bbox"
                capability = (
                    {"configured": True, "reason": None}
                    if bbox
                    else self.workflow_tracking_capability()
                )
                error = capability["reason"]
                if capability["configured"]:
                    probe = None
                    try:
                        baseline = self._workflow_tracking_baseline
                        rich = (
                            baseline["rich_labeling"]
                            if baseline
                            else (self.config.rich_labeling)
                        )
                        if bbox:
                            self._workflow_bbox_probe(plan.get("tracking_service_url"))
                        else:
                            probe = self._create_tracking_client(
                                replace(
                                    rich,
                                    enabled=False,
                                    tracking_enabled=True,
                                    processing_method="sam_live",
                                )
                            )
                        if probe is not None and not probe.ready_for_new_session():
                            error = (
                                "The configured person tracking service is unavailable."
                            )
                    except Exception:
                        error = "The configured person tracking service is unavailable."
                    finally:
                        if probe is not None:
                            try:
                                probe.close()
                            except Exception:
                                pass
                check_active()
                results.append(
                    {
                        **context,
                        "node_id": plan["tracking_node_id"],
                        "type": "tracking",
                        "state": "unavailable" if error else "live_analysis_required",
                        "processing_method": "bbox" if bbox else "sam_live",
                        "error": error,
                    }
                )
            for node_id, identity in (
                (plan["face_node_id"], True),
                (plan["events_node_id"], False),
            ):
                if not node_id:
                    continue
                check_active()
                if not identity and plan.get("events_engine") in {"system1", "neural"}:
                    results.append(self._workflow_native_event_test(
                        plan, connection, snapshot, context, native_preview_frames,
                    ))
                    check_active()
                    continue
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
                    test_config = self._workflow_summary_configuration()
                    client = build_selected_features_grounding_client(
                        test_config.inference,
                        test_config.model_api,
                        features=features,
                        stranger_client=face,
                        dry_run=self.dry_run,
                        api_key=self._aiclub_api_key,
                        event_sensitivities=plan["event_sensitivities"],
                        vocabulary=self._event_vocabulary(),
                    )
                    response = (
                        client.classify_stranger_window(window)
                        if identity
                        else client.classify_safety_window(window)
                    )
                    result = self._workflow_test_response(
                        response, node_id=node_id, identity=identity, context=context
                    )
                except Exception as error:
                    result = {
                        **context,
                        "node_id": node_id,
                        "type": "face" if identity else "events",
                        "error": PUBLIC_FACE_ERROR
                        if identity
                        else PUBLIC_ANALYSIS_ERROR,
                        "error_detail": (
                            {
                                "code": "missing_api_key",
                                "http_status": None,
                                "message": PUBLIC_MISSING_API_KEY_ERROR,
                            }
                            if isinstance(error, MissingAPIKeyError)
                            else inference_error_detail(error)
                        ),
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
                        "API and face observations use the current frame; native recognition "
                        "uses the listed distinct preview frames. Tracking, temporal "
                        "confirmation and delivery are exercised by live analysis."
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
