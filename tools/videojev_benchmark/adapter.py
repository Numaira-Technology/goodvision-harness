"""Use native choice scoring while retaining shared scheduling and ID contracts."""

import asyncio
import base64
import json
import time
from dataclasses import replace
from functools import partial

from goodvision_harness.model_api import ModelAPIError
from goodvision_harness.vlm_transport import create_vlm_http_client
from tools.benchmark_runtime.person_grounding import (
    TrackedPersonInspector,
    parse_verdicts,
)
from tools.benchmark_runtime.vlm_compact_model import EventJSONInspector
from tools.benchmark_runtime.vlm_event_policy import event_policy_snapshot
from tools.benchmark_runtime.vlm_independent_model import IndependentVerdict
from tools.videojev_probe.contract import (
    LABELS,
    MODEL,
    POLICY_SHA,
    REVISION,
    parse_choices,
    questions,
)

ENDPOINT = "http://127.0.0.1:8093/predict"


def native_questions(job, messages, person):
    window = {
        "window_id": job.stream_session_id,
        "frames": [{"media_seconds": f.timestamp_seconds} for f in job.frames],
    }
    snapshots = None
    if person:
        metadata = [
            json.loads(x["text"]) for x in messages[1]["content"] if x["type"] == "text"
        ][1:]
        snapshots = [
            dict(
                state="ready",
                stream_session_id=job.stream_session_id,
                frame_sequence=i,
                timestamp_seconds=f.timestamp_seconds,
                people=[
                    dict(
                        track_id=p["subject_id"],
                        bbox=dict(
                            zip(("x1", "y1", "x2", "y2"), p["bbox"], strict=True)
                        ),
                    )
                    for p in m["subject_boxes"]
                ],
            )
            for i, (f, m) in enumerate(zip(job.frames, metadata, strict=True))
        ]
    qs = questions(window, snapshots, ppe_variant="focused", max_people=64)
    if person:
        for q in qs:
            if q["id"].startswith("scene/") and not q["id"].endswith("/ground_litter"):
                # Same shared grounding contract: person votes + unassigned vote.
                q["query"] += (
                    "\nFor THIS scene question assess only visible people "
                    "or occurrences "
                    "NOT assigned to any supplied LAST-frame ID; exclude occurrences "
                    "fully accounted for by those IDs. If no IDs exist, assess the "
                    "whole scene. Detector omissions never prove absence. "
                    "This narrows the target, not the six-event criteria."
                )
    return qs


class NativeInspector(EventJSONInspector):
    def __init__(self, config, broker_url):
        super().__init__(
            replace(config.model_settings, timeout_seconds=20, max_retries=0),
            model=MODEL,
            api_key="",
            max_image_side=720,
            jpeg_quality=90,
        )
        self.person = config.output_mode == "compact"
        self.broker_url = broker_url
        self.boxes, self.detector_timing = {}, {}

    def _ensure_transport(self):
        if self._loop is None:
            self._loop = asyncio.new_event_loop()
        if self._http is None:
            self._http = create_vlm_http_client(trust_env=False)

    def _inspect_locked(self, job, *, timing):
        self.job = job
        self.boxes, self.detector_timing = {}, {}
        try:
            return super()._inspect_locked(job, timing=timing)
        finally:
            timing.update(self.detector_timing)

    def build_messages(self, job, *, encoded_images=None):
        if (
            tuple(job.request.event_labels) != LABELS
            or event_policy_snapshot(LABELS, dict(job.event_sensitivities))["sha256"]
            != POLICY_SHA
        ):
            raise ValueError("six-class policy mismatch")
        cls = TrackedPersonInspector if self.person else EventJSONInspector
        messages = cls.build_messages(self, job, encoded_images=encoded_images)
        self.questions = native_questions(job, messages, self.person)
        return messages

    async def _request(self, payload, headers, *, timing):
        # The parent resets this before every inspection. Keep the live metrics
        # mapping visible for success, HTTP errors and timeout paths alike.
        self.last_http_timing = timing
        body = {
            "model": MODEL,
            "revision": REVISION,
            "request_id": self.job.request.request_id,
            "session": self.job.stream_session_id,
            "lane": "grounding" if self.person else "events",
            "policy_sha256": POLICY_SHA,
            "fps": 2.5,
            "questions": self.questions,
            "images": [
                base64.b64encode(f.jpeg_bytes).decode() for f in self.job.frames
            ],
        }
        timing["http_request_attempted"] = True
        start = time.monotonic()
        response = await self._http.post(
            ENDPOINT,
            json=body,
            headers={"Content-Type": "application/json"},
            timeout=20,
        )
        timing.update(
            http_status=response.status_code,
            http_elapsed_seconds=time.monotonic() - start,
            response_bytes=len(response.content),
        )
        if response.status_code != 200:
            raise ModelAPIError(
                "native VideoJev request failed",
                code="http_error",
                status_code=response.status_code,
            )
        result = response.json()
        if (
            result.get("request_id") != body["request_id"]
            or result.get("model") != MODEL
            or result.get("revision") != REVISION
        ):
            raise ValueError("native response identity mismatch")
        timing.update(
            {
                f"videojev_{k}": v
                for k, v in result["timings"].items()
                if isinstance(v, (int, float))
            }
        )
        return result

    def _parse(self, job, response):
        decoded = parse_choices(response["raw"], self.questions)
        if not self.person:
            return tuple(
                IndependentVerdict(
                    label, (), decoded[f"scene/{label}"], None, (), ("scene_only",)
                )
                for label in LABELS
            )
        raw = {"v": 1, "e": {label: [] for label in LABELS}}
        for key, value in decoded.items():
            sid, label = key.rsplit("/", 1)
            raw["e"][label].append(
                {"p": [] if sid == "scene" else [sid], "d": value, "c": None}
            )
        return parse_verdicts(raw, job, self.boxes)


def inspector(config, *, broker_url):
    if (
        config.model_settings.provider != "local"
        or config.model != MODEL
        or config.model_settings.endpoint != ENDPOINT
        or config.output_mode not in {"events", "compact"}
    ):
        raise ValueError("native VideoJev route mismatch")
    return NativeInspector(config, broker_url)


def factory(broker_url, location):
    return partial(inspector, broker_url=broker_url)
