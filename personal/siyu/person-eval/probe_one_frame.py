import os
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

from goodvision_harness.config import load_config
from goodvision_harness.grounding import AIClubMultiGroundingClient
from goodvision_harness.sliding_window import FrameWindow, SampledFrame
from low_resolution_model_eval import prepare_frame

os.environ["INFERENCE_PROVIDER"] = os.environ.get("PROBE_PROVIDER", "aiclub")
config = load_config(ROOT / "config.yaml")
client = AIClubMultiGroundingClient(
    config.inference,
    replace(config.model_api, timeout_seconds=90, max_retries=0),
)
path = ROOT / "tmp" / "person-eval-videos" / "Walk1.mpg"
jpeg, w, h, orig, ts, idx = prepare_frame(path, 4.0, 0, 75)
print("model", config.inference.model, "provider", config.model_api.provider, "jpeg", len(jpeg))
response = client.classify_window(
    FrameWindow(0, (SampledFrame(0, idx, ts, jpeg, w, h),))
)
print("error", response.error, "latency", round(response.latency_seconds, 2))
print("judgments", getattr(response.classification, "judgments", None))
