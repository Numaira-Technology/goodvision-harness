"""Adapter preserving original images and historical benchmark parsers."""

import asyncio
import copy
import time
from dataclasses import replace
from pathlib import Path

import httpx

from goodvision_harness.model_api import ModelAPISettings
from goodvision_harness.vlm_transport import create_vlm_http_client
from tools.benchmark_runtime.scene_model import _ResponseValidationError
from tools.benchmark_runtime.vlm_compact_model import EventJSONInspector
from tools.benchmark_runtime.vlm_lean_grounding import LeanGroundingInspector
from tools.vlm_benchmark import modes

from . import profile as p
from .runtime import sha, write_json
from .stream import StreamingMixin


class LocalTransport(StreamingMixin):
    def _ensure_transport(self):
        if self._loop is None:
            self._loop = asyncio.new_event_loop()
        if self._http is None:
            self._http = create_vlm_http_client(trust_env=False)

    async def _request(self, payload, headers, *, timing):
        payload = {**payload, "stream_options": {"include_usage": True}}
        # The shared inspector may discover a local key at call time. This
        # loopback-only baseline has no authentication and forwards no key.
        headers = {k: v for k, v in headers.items() if k.lower() != "authorization"}
        result = await super()._request(payload, headers, timing=timing)
        if result.get("model") != p.SERVED_MODEL:
            raise _ResponseValidationError("unexpected_model", "model")
        return result


class LocalEvents(LocalTransport, EventJSONInspector):
    # Reuse SSE accounting, not the Cerebras-specific two-image packing.
    build_messages = EventJSONInspector.build_messages


class LocalGrounding(LocalTransport, LeanGroundingInspector):
    build_messages = LeanGroundingInspector.build_messages


def inspector(config, *, grounding_class=LocalGrounding, extra=None):
    if (
        config.model_settings.provider != "local"
        or config.model != p.SERVED_MODEL
        or config.model_settings.endpoint != p.BASE_URL + "/v1/chat/completions"
        or config.output_mode not in {"events", "compact"}
        or config.grounding_format != "lean-v1"
        or tuple(config.labels) != tuple(modes.LABELS)
    ):
        raise ValueError("self-hosted inspector requires the frozen local route")
    options = dict(config.options)
    options.update(max_image_side=720, jpeg_quality=90)
    options["max_tokens"] = 160 if config.output_mode == "events" else 1600
    parameters = copy.deepcopy(dict(config.parameters))
    parameters.update(
        response_format={"type": "json_object"},
        chat_template_kwargs={"enable_thinking": False},
        temperature=0,
    )
    if "provider" in parameters or "reasoning" in parameters:
        raise ValueError("cloud routing fields must never reach vLLM")
    cls = LocalEvents if config.output_mode == "events" else grounding_class
    return cls(
        replace(config.model_settings, timeout_seconds=20, max_retries=0),
        model=config.model,
        api_key="",
        request_parameters=parameters,
        **options,
        **(extra or {}),
    )


def arguments(video=None, output=None, real=False):
    args = modes.parser().parse_args([])
    for name, value in p.FROZEN.items():
        setattr(args, name, copy.deepcopy(value))
    args.video = [Path(video)] if video else []
    args.output = Path(output) if output else None
    args.real = real
    modes.validate(args)
    return args


def plan():
    args = arguments()
    return {
        "version": p.VERSION,
        "model": p.MODEL_REPO,
        "revision": p.MODEL_REVISION,
        "image": p.IMAGE,
        "vllm_version": p.ENGINE_VERSION,
        "hardware": "one dedicated >=110 GiB GPU; H200 candidate, unvalidated",
        "frozen": p.FROZEN,
        "policy": p.policy_check(),
        "video_sha256": p.VIDEO_SHA256,
        "request_budget": modes.request_budget(args),
        "additional_four_frame_probe_requests": 2,
        "total_request_upper_including_probe": modes.request_budget(args)["total_upper"]
        + 2,
        "recordings": 6,
        "execute": False,
        "readiness": "GPU/runtime verification pending; no capacity claim",
    }


def check_video(video):
    if sha(video) != p.VIDEO_SHA256:
        raise ValueError("video differs from the frozen output.mp4")


def readiness(timeout=120):
    """Read-only HTTP checks; bounded total wait, not a GPU-capacity assertion."""
    end = time.monotonic() + timeout
    with httpx.Client(trust_env=False, follow_redirects=False, timeout=2) as client:
        while True:
            try:
                health = client.get(p.BASE_URL + "/health")
                models = client.get(p.BASE_URL + "/v1/models")
                if health.status_code == models.status_code == 200:
                    if {x["id"] for x in models.json()["data"]} != {p.SERVED_MODEL}:
                        raise ValueError("served model alias differs")
                    metrics = client.get(p.BASE_URL + "/metrics")
                    if metrics.status_code != 200 or "vllm:" not in metrics.text:
                        raise ValueError("vLLM metrics unavailable")
                    return {"healthy": True, "model": p.SERVED_MODEL}
            except httpx.HTTPError:
                pass
            if time.monotonic() >= end:
                raise TimeoutError("model did not become ready")
            time.sleep(min(1, max(0, end - time.monotonic())))


def preflight(video):
    """Two bounded real four-frame calls using actual prompts and strict parsers."""
    from goodvision_harness.runtime.scene import TemporalInspection
    from goodvision_harness.runtime.scheduling import EvidenceFrame, InspectionRequest
    from tools.benchmark_runtime.vlm_independent import VLMProcessConfig
    from tools.vlm_multistream_benchmark import LABELS, decode

    p.policy_check()
    check_video(video)
    readiness()
    frames = decode(Path(video), 2.5, 720, 90)
    selected = (10, 11, 12, 13)
    evidence = tuple(
        EvidenceFrame("vast-probe", i, frames.media_seconds[i], frames[i])
        for i in selected
    )
    request = InspectionRequest(
        "vast-probe",
        (),
        "independent_source",
        LABELS,
        evidence[0].timestamp_seconds,
        evidence[-1].timestamp_seconds,
    )
    job = TemporalInspection("vast-probe", "camera01", request, evidence)
    results = []
    for mode in ("events", "compact"):
        cfg = VLMProcessConfig(
            "vast-probe",
            "camera01",
            1,
            LABELS,
            ModelAPISettings(provider="local"),
            p.SERVED_MODEL,
            output_mode=mode,
            grounding_format="lean-v1",
        )
        client = inspector(cfg)
        try:
            # inspect() is the benchmark sync interface with its own timeout.
            verdicts = client.inspect(job)
            if set(v.event_label for v in verdicts) != set(LABELS):
                raise ValueError("preflight missed event labels")
            results.append({"mode": mode, "complete": True})
        finally:
            client.close()
    return results


def run(video, output, packet, *, deployment=None):
    from tools.vlm_benchmark.admission_storage import memory_root
    from tools.vlm_benchmark.model_comparison import load_reference

    check_video(video)
    memory_root()  # Fail before model calls on non-Linux or missing tmpfs.
    reference = load_reference(packet, video)
    if reference["policy"]["sha256"] != p.POLICY_SHA256:
        raise ValueError("Astra reference policy differs")
    output = Path(output)
    if output.exists():
        raise ValueError("choose a new output directory; never repeat into an old run")
    args = arguments(video, output, real=True)
    probe = []
    try:
        # Use the runner's shared suite lock for probes as well as measurements.
        modes.run(args, before_run=lambda: probe.extend(preflight(video)))
    finally:
        if output.exists():
            write_json(output / "self-hosted-plan.json", plan())
            write_json(output / "self-hosted-preflight.json", probe)
            if deployment:
                write_json(output / "deployment.json", deployment)
    from .report import analyze

    analyze(output, reference)
