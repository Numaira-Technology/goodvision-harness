"""Stable canvas engine IDs, independent of vendor/model names."""

from urllib.parse import urlsplit

ENGINES = {"tracking": ("contour", "bbox"), "events": ("vlm", "system1", "neural")}
NAMES = {
    "contour": "Contour",
    "bbox": "Bbox",
    "vlm": "VLM semantic recognition",
    "system1": "System 1 recognition",
    "neural": "Neural network recognition",
}
MODELS = {"system1": "ZhuYuhan/VideoJev-4B", "neural": "openai/clip-vit-large-patch14"}
NEURAL_EVENTS = ("fight", "smoking", "fall", "weapon", "ppe_violation", "ground_litter")
POSITIONS = ("top-left", "top-right", "bottom-left", "bottom-right")


def display_settings(value):
    if not isinstance(value, dict) or set(value) - {"tracks", "events", "summary"}:
        raise ValueError("invalid output display settings")
    result = {}
    for kind, default in (
        ("tracks", "top-left"),
        ("events", "top-left"),
        ("summary", "top-right"),
    ):
        item = value.get(kind, {})
        if not isinstance(item, dict) or set(item) - {"visible", "position"}:
            raise ValueError("invalid output layer")
        visible, position = item.get("visible", True), item.get("position", default)
        if type(visible) is not bool or position not in POSITIONS:
            raise ValueError("invalid output layer visibility or position")
        result[kind] = {"visible": visible, "position": position}
    return result


def engine_for(kind, config):
    return config.get("engine", ENGINES[kind][0]) if kind in ENGINES else kind


def native_endpoints(endpoint):
    base = endpoint.rstrip("/").removesuffix("/predict")
    return base + "/predict", base + "/health"


def service_url(value):
    if not isinstance(value, str) or len(value) > 2048:
        raise ValueError("Service URL must be an HTTP(S) address without credentials.")
    value = value.strip()
    parsed = urlsplit(value)
    if value and (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or "\\" in value
        or any(ord(c) <= 32 or ord(c) == 127 for c in value)
    ):
        raise ValueError("Service URL must be an HTTP(S) address without credentials.")
    if value:
        _ = parsed.port  # Validate malformed or out-of-range ports before persisting.
    return value
