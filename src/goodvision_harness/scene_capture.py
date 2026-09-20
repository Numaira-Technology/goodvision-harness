"""Source-only session adapter for the continuous scene pipeline."""

from .config import InferenceSettings


class SceneCaptureClient:
    """Keep source decoding alive without generating per-frame verdicts."""

    scene_source_only = True

    def __init__(self, settings: InferenceSettings, event_labels: tuple[str, ...]):
        self.settings = settings
        self.event_labels = event_labels

    def classify_window(self, window):
        raise RuntimeError("scene capture must not schedule frame classification")
