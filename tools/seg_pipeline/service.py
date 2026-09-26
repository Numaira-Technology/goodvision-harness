"""Isolated full Harness service with only the tracking client substituted."""

import argparse
import json
import math
import os
from dataclasses import asdict, replace
from pathlib import Path

import httpx
from fastapi.responses import FileResponse

from goodvision_harness.config import load_config
from goodvision_harness.local_gpu import LocalGPUDeployment
from goodvision_harness.perception import PerceptionSessionHTTPClient
from goodvision_harness.service import HarnessRuntime, create_app
from goodvision_harness.service_locations import ServiceLocations

ROOT = Path(__file__).resolve().parents[2]
BACKEND = "seg_benchmark"
POLICY = "causal_iou_three_observations"


class CandidateClient(PerceptionSessionHTTPClient):
    def __init__(self, endpoint, **kwargs):
        # Reuse the source validation, replay and transport protocol. The route,
        # backend and confirmation metadata explicitly identify the candidate.
        super().__init__(endpoint, method="sam_live", **kwargs)
        self.method = replace(
            self.method,
            id=BACKEND,
            backend=BACKEND,
            confirmation_policy=POLICY,
            native_sam_video_tracking=False,
        )
        self.backend = BACKEND
        self.native_sam_video_tracking = False
        self.session_path = f"/v1/perception/{BACKEND}/sessions"
        self.health_path = f"/health/perception/{BACKEND}"
        base = httpx.URL(endpoint)
        self.endpoint = str(base.copy_with(path=self.session_path))
        self._health = str(base.copy_with(path=self.health_path))


class CandidateRuntime(HarnessRuntime):
    def _create_tracking_client(self, rich):
        return CandidateClient(
            rich.endpoint,
            timeout_seconds=rich.video_timeout_seconds,
            transport_max_side=rich.video_transport_max_side,
            transport_jpeg_quality=rich.video_transport_jpeg_quality,
            transport_codec=rich.video_transport_codec,
            transport_crf=rich.video_transport_crf,
        )


def main_defaults(provider=None):
    """Read repository defaults before starting threads, without local overrides."""
    saved = dict(os.environ)
    try:
        os.environ.clear()
        os.environ["PYTHON_DOTENV_DISABLED"] = "1"
        if provider is not None:
            os.environ["INFERENCE_PROVIDER"] = provider
        return load_config(ROOT / "config.yaml")
    finally:
        os.environ.clear()
        os.environ.update(saved)


def display_delay(value):
    delay = float(value)
    if not math.isfinite(delay) or not 0 <= delay <= 8:
        raise ValueError("Display delay must be finite and between 0 and 8 seconds")
    return delay


def benchmark_config(job):
    # Use main's OpenRouter preset for the candidate comparison. The repository
    # default provider is AICLUB; local environment must not select the provider.
    base = main_defaults(job.get("inference_provider", "openrouter"))
    display = base.display
    if job.get("display_delay_seconds") is not None:
        display = replace(
            display,
            sam_live_delay_seconds=display_delay(job["display_delay_seconds"]),
        )
    state = Path(job["output"]) / "state"
    # Activate the existing continuous route and isolate private output paths.
    # Only an explicit delay experiment may override main's display buffer.
    # All other sampling, transport and face options remain main's.
    return replace(
        base,
        memory=None,
        display=display,
        rich_labeling=replace(
            base.rich_labeling,
            enabled=True,
            tracking_enabled=True,
            backend="sam31_stream",
            processing_method="sam_live",
            endpoint=job["endpoint"],
        ),
        runtime=replace(base.runtime, scene_mode="active"),
        face=replace(
            base.face,
            model_root=Path(job["face_model_root"]),
            allow_model_download=job["allow_face_download"],
            gallery_path=state / "empty-gallery.json",
        ),
        events=replace(base.events, directory=state / "events"),
        service=replace(
            base.service,
            source_id="segmentation-benchmark",
            webhook=replace(base.service.webhook, url=None),
        ),
    )


def fixed_protocol(config):
    # HarnessRuntime applies this same route configuration. In particular,
    # tracking_sample_fps overrides rich_labeling.max_fps even in legacy mode.
    effective = ServiceLocations().configure(config, LocalGPUDeployment(), 1)
    rich = asdict(effective.rich_labeling)
    rich.pop("endpoint")
    return {
        "display": asdict(effective.display),
        "effective_display_delay_seconds": effective.display.sam_live_delay_seconds,
        "video": asdict(effective.video),
        "tracking": rich,
        "runtime": asdict(effective.runtime),
        "inference_provider": effective.model_api.provider,
        "inference_model": effective.inference.model,
        "inference_parameters": effective.inference.request_parameters,
        "face": {
            k: v
            for k, v in asdict(effective.face).items()
            if k not in {"model_root", "gallery_path", "allow_model_download"}
        },
    }


def app_for_job(job):
    config = benchmark_config(job)
    if job.get("fixed_main") != fixed_protocol(config):
        raise ValueError("Repository configuration changed after the run was frozen")
    configure_runtime_environment(job, config)
    app = create_app(runtime_factory=lambda: CandidateRuntime(config))

    @app.get("/experiment-video/original", include_in_schema=False)
    async def original_video():
        # Starlette FileResponse implements real Range requests. Never transcode
        # or use a basename supplied in a URL to locate arbitrary user files.
        return FileResponse(job["video"], media_type="video/mp4")

    @app.get("/benchmark/config", include_in_schema=False)
    async def benchmark_info():
        return {
            key: job[key]
            for key in ("model", "delay", "sample_fps", "queue_policy", "video_sha256")
        } | {"scope": "full_pipeline", "fixed_main": fixed_protocol(config)}

    return app


def configure_runtime_environment(job, config):
    """Activate the production browser route in this isolated service process."""
    state = Path(job["output"]) / "state"
    state.mkdir(parents=True, exist_ok=True)
    os.environ.update(
        PYTHON_DOTENV_DISABLED="1",
        HARNESS_AUTOSTART="false",
        HARNESS_DRY_RUN="false",
        LOCAL_INFERENCE_ENABLED="false",
        LOCAL_TRACKING_ENABLED="false",
        INFERENCE_EXECUTION_MODE="cloud",
        TRACKING_EXECUTION_MODE="legacy",
        GOODVISION_SAM_SPOOL_DIRECTORY=str(state / "tracking"),
        HARNESS_DEPLOYMENT_SETTINGS_PATH="",
        LOCAL_CAMERA_DISPLAY_ENABLED="true",
        # The browser has a separate six-second fallback. Bind it to the same
        # frozen main delay used by the server and the acceptance report.
        LOCAL_CAMERA_DISPLAY_DELAY_SECONDS=str(
            fixed_protocol(config)["effective_display_delay_seconds"]
        ),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", type=Path, required=True)
    parser.add_argument("--port", type=int, required=True)
    args = parser.parse_args()
    job = json.loads(args.job.read_text())
    import uvicorn

    uvicorn.run(app_for_job(job), host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()
